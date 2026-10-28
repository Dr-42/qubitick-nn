import math
from collections import deque
import torch, torch.nn as nn, torch.nn.functional as F
from .config import MemoryConfig
from .short import ShortTermBuffer
from .long import LongTermStore
from .retrieval import Retriever
from .consolidate import Consolidator

BOUNDARIES = ("flash_short", "short_long", "short_discard", "long_evict")

def _zl(i, o):
    l = nn.Linear(i, o); nn.init.zeros_(l.weight); nn.init.zeros_(l.bias); return l

class MemoryTree(nn.Module):
    """Three read paths (flash handled by the loop; short = windowed attention; long = top-k cosine retrieval)."""
    def __init__(self, config: MemoryConfig):
        super().__init__()
        c = self.cfg = config; d, kd = c.d_model, c.key_dim
        self.short = ShortTermBuffer(max(c.W_short, 0), d, True, c.batch_size, c.write_per_seq)
        self.long = LongTermStore(max(c.N_long, 0), d, kd)
        self.retriever = Retriever(kd, c.k_retrieval)
        self.consolidator = Consolidator(c.importance, c.M_consolidate)
        if c.W_short > 0:   # path 1
            self.qs, self.ks = nn.Linear(d, 64), nn.Linear(d, 64)
            self.so = _zl(d, d); self.beta = nn.Parameter(torch.zeros(()))
        if c.N_long > 0:    # path 2
            self.Wq, self.key_proj = nn.Linear(d, kd), nn.Linear(d, kd)
            self.uh, self.uv = nn.Linear(d, 1), nn.Linear(d, 1)
            self.lo = _zl(d, d)
        self.tick = 0
        self.hist = deque(maxlen=max(c.janus_window, 1))
        self._prev = self._cum()

    # ---- device / state ----
    def reset(self, device=None):
        device = device or next(self.parameters()).device
        self.short.reset(device) if self.short.W else None
        self.long.reset(device) if self.long.N else None
        self.tick = 0; self.hist.clear(); self._prev = self._cum()

    def get_state(self): return {"short": self.short.get_state() if self.short.W else None, "long": self.long.get_state() if self.long.N else None, "tick": self.tick, "hist": list(self.hist), "prev": dict(self._prev)}
    def set_state(self, s):
        if s["short"] is not None: self.short.set_state(s["short"])
        if s["long"] is not None: self.long.set_state(s["long"])
        self.tick = s["tick"]; self.hist = deque(s["hist"], maxlen=self.hist.maxlen); self._prev = dict(s["prev"])

    # ---- read ----
    def read(self, h):
        c = self.cfg; B, S, d = h.shape
        out = {"short": torch.zeros_like(h), "combined": torch.zeros_like(h), "retrieved": None, "scores": None, "util": None, "next_tok": None}
        if c.W_short > 0 and self.short.n > 0:
            st = self.short.states().to(h.dtype)                                  # (B,n,d)
            q, k = self.qs(h), self.ks(st)
            imp = self.short.importance_weights()
            z = ((imp - imp.mean(1, keepdim=True)) / (imp.std(1, keepdim=True) + 1e-6)).clamp(-3, 3) if imp.shape[1] > 1 else torch.zeros_like(imp)
            bias = (self.beta * z)[:, None, :].to(h.dtype)                        # (B,1,n)
            a = torch.softmax(((q @ k.transpose(1, 2)) / 8.0 + bias).float(), -1).to(h.dtype)
            out["short"] = self.so(a @ st)
        if c.N_long > 0 and self.long.n > 0:
            keys = self.key_proj(self.long.values[:self.long.n].to(h.dtype))      # live keys -> grads reach key_proj
            r = self.long.retrieve_full(self.Wq(h), c.k_retrieval, keys)
            vals = r["values"].to(h.dtype)                                        # (B,S,k,d)
            util = (self.uh(h).unsqueeze(2) + self.uv(vals)).squeeze(-1).float()               # = Linear([h; v])
            w = torch.softmax(r["sims"] / c.retr_temp, -1) * torch.sigmoid(util)
            out["combined"] = self.lo((w.to(h.dtype).unsqueeze(-1) * vals).sum(2))
            out.update(retrieved=vals, scores=r["sims"], util=util, next_tok=r["next_tok"])
        return out

    # ---- write ----
    @torch.no_grad()
    def write(self, h_final, halts, importance, tick, next_tok=None):
        c = self.cfg; info = {}
        if c.flat:   # single-tier ablation: every write goes straight into the one store
            n = min(c.write_per_seq, h_final.shape[1])
            idx = importance.float().topk(n, dim=1).indices
            hs = h_final.gather(1, idx[..., None].expand(-1, -1, h_final.shape[-1])).reshape(-1, h_final.shape[-1])
            im = importance.float().gather(1, idx).reshape(-1)
            nt = next_tok.gather(1, idx).reshape(-1) if next_tok is not None else None
            self.long.add(self.key_proj(hs.float()), hs, im, nt); return {"flat_added": hs.shape[0]}
        if c.W_short > 0:
            self.short.append(h_final, tick, importance, next_tok)
        if c.W_short > 0 and c.N_long > 0 and tick > 0 and tick % c.K_consolidate == 0:
            info = self.consolidator.consolidate(self.short, self.long, self.key_proj, c.long_decay, c.p_dissolve)
        return info

    def advance_tick(self):
        cur = self._cum(); d = {k: cur[k] - self._prev[k] for k in cur}; self._prev = cur
        self.hist.append(d); self.tick += 1

    def _cum(self):
        return {"appended": self.short.appended, "overwritten": self.short.overwritten,
                "long_added": self.long.added, "long_evicted": self.long.evicted + self.long.dissolved}

    def boundary_stats(self):
        c = self.cfg; sc = max(c.batch_size * c.W_short, 1); lc = max(c.N_long, 1)
        h = list(self.hist)
        g = lambda k: [x[k] for x in h]
        n = lambda xs, cap: [x / cap for x in xs]
        promoted = [x["long_added"] for x in h]
        return {"flash_short": {"writes": n(g("appended"), sc), "evicts": n(g("overwritten"), sc)},
                "short_long": {"writes": n(promoted, lc), "evicts": n(g("long_evicted"), lc)},
                "short_discard": {"writes": n(g("overwritten"), sc), "evicts": n(promoted, sc)},
                "long_evict": {"writes": n(promoted, lc), "evicts": n(g("long_evicted"), lc)}}

    def nbytes(self): return (self.short.nbytes() if self.short.W else 0) + (self.long.nbytes() if self.long.N else 0)
    def stats(self): return {"tick": self.tick, "short": self.short.stats() if self.short.W else None, "long": self.long.stats() if self.long.N else None, "mb": self.nbytes() / 2**20}
