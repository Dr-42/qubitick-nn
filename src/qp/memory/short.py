from dataclasses import dataclass
import torch

@dataclass
class MemoryEntry:
    h: torch.Tensor
    next_tok: int
    importance: float
    tick: int

class ShortTermBuffer:
    """Per-row ring buffer (B parallel contiguous streams). Written on halt, FIFO eviction."""
    def __init__(self, capacity=2048, d_model=384, fp16=True, batch_size=4, n_write=1, device="cpu"):
        self.W, self.d, self.B, self.n_write = capacity, d_model, batch_size, n_write
        self.dtype = torch.float16 if fp16 else torch.float32
        self.reset(device)

    def reset(self, device=None):
        dev = device or self.h.device
        W, B = self.W, self.B
        self.h = torch.zeros(B, W, self.d, dtype=self.dtype, device=dev)
        self.imp = torch.zeros(B, W, device=dev)
        self.nxt = torch.zeros(B, W, dtype=torch.long, device=dev)
        self.tk = torch.zeros(B, W, dtype=torch.long, device=dev)
        self.n = 0; self.ptr = 0; self.appended = 0; self.overwritten = 0

    @torch.no_grad()
    def append(self, h, tick, importance, next_tok=None):
        if self.W == 0: return
        B, S, d = h.shape
        n = min(self.n_write, S)
        idx = importance.float().topk(n, dim=1).indices
        hs = h.gather(1, idx[..., None].expand(-1, -1, d)).to(self.dtype)
        im = importance.float().gather(1, idx)
        nt = next_tok.gather(1, idx) if next_tok is not None else torch.zeros_like(idx)
        for j in range(n):
            s = self.ptr
            self.h[:, s] = hs[:, j]; self.imp[:, s] = im[:, j]; self.nxt[:, s] = nt[:, j]; self.tk[:, s] = tick
            self.ptr = (s + 1) % self.W
            if self.n == self.W: self.overwritten += B
            else: self.n += 1
            self.appended += B

    def states(self): return self.h[:, :self.n]
    def importance_weights(self): return self.imp[:, :self.n]
    def next_tokens(self): return self.nxt[:, :self.n]

    @torch.no_grad()
    def select_top_m(self, m):
        if self.n == 0: return []
        flat = self.imp[:, :self.n].reshape(-1)
        v, i = flat.topk(min(m, flat.numel()))
        b, s = i // self.n, i % self.n
        return [MemoryEntry(self.h[bb, ss], int(self.nxt[bb, ss]), float(vv), int(self.tk[bb, ss]))
                for bb, ss, vv in zip(b.tolist(), s.tolist(), v.tolist())]

    def evict(self, n):  # drop the n oldest entries per row (FIFO)
        n = min(n, self.n)
        if n == 0: return
        order = (torch.arange(self.n, device=self.h.device) - (self.ptr - self.n)) % self.W if self.n == self.W else torch.arange(self.n, device=self.h.device)
        keep = order >= n
        for name in ("h", "imp", "nxt", "tk"):
            t = getattr(self, name); k = t[:, :self.n][:, keep]
            t[:, :k.shape[1]] = k
        self.n -= n; self.ptr = self.n % self.W; self.overwritten += n * self.B

    def nbytes(self): return sum(t.numel() * t.element_size() for t in (self.h, self.imp, self.nxt, self.tk))
    def stats(self): return {"n": self.n, "capacity": self.W, "fill": self.n / max(self.W, 1),
                              "appended": self.appended, "overwritten": self.overwritten, "mb": self.nbytes() / 2**20}
    def get_state(self): return {k: getattr(self, k).clone() for k in ("h", "imp", "nxt", "tk")} | {"n": self.n, "ptr": self.ptr, "appended": self.appended, "overwritten": self.overwritten}
    def set_state(self, s):
        for k in ("h", "imp", "nxt", "tk"): getattr(self, k).copy_(s[k])
        self.n, self.ptr, self.appended, self.overwritten = s["n"], s["ptr"], s["appended"], s["overwritten"]
