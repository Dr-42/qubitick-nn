import torch
from .retrieval import Retriever

class LongTermStore:
    """Content-addressed shared store. Stores fp16 values (+ keys at write time, + next-token labels)."""
    def __init__(self, capacity=8192, d_model=384, key_dim=384, device="cpu"):
        self.N, self.d, self.kd = capacity, d_model, key_dim
        self.retriever = Retriever(key_dim)
        self.reset(device)

    def reset(self, device=None):
        dev = device or self.values.device
        N = self.N
        self.values = torch.zeros(N, self.d, dtype=torch.float16, device=dev)
        self.keys = torch.zeros(N, self.kd, dtype=torch.float16, device=dev)
        self.imp = torch.zeros(N, device=dev)
        self.nxt = torch.zeros(N, dtype=torch.long, device=dev)
        self.n = 0; self.added = 0; self.evicted = 0; self.dissolved = 0

    @torch.no_grad()
    def add(self, keys, values, importance, next_tok=None):
        M = values.shape[0]
        if self.N == 0 or M == 0: return
        M = min(M, self.N); keys, values, importance = keys[:M], values[:M], importance[:M]
        if next_tok is None: next_tok = torch.zeros(M, dtype=torch.long, device=values.device)
        free = self.N - self.n
        if M <= free:
            slots = torch.arange(self.n, self.n + M, device=values.device)
        else:
            nv = M - free
            victims = self.imp[:self.n].argsort()[:nv]
            slots = torch.cat([torch.arange(self.n, self.N, device=values.device), victims])
            self.evicted += nv
        self.values[slots] = values.to(torch.float16); self.keys[slots] = keys.to(torch.float16)
        self.imp[slots] = importance.float(); self.nxt[slots] = next_tok
        self.n = min(self.N, self.n + M); self.added += M

    @torch.no_grad()
    def decay(self, f): self.imp[:self.n] *= f

    @torch.no_grad()
    def evict(self, victims):
        if not victims: return
        keep = torch.ones(self.n, dtype=torch.bool, device=self.values.device)
        keep[torch.as_tensor(victims, device=keep.device)] = False
        self._compact(keep); self.evicted += len(victims)

    @torch.no_grad()
    def dissolve(self, p):
        if p <= 0 or self.n == 0: return
        keep = torch.rand(self.n, device=self.values.device) >= p
        d = int((~keep).sum())
        if d: self._compact(keep); self.dissolved += d

    def _compact(self, keep):
        for name in ("values", "keys", "imp", "nxt"):
            t = getattr(self, name); k = t[:self.n][keep]; t[:k.shape[0]] = k
        self.n = int(keep.sum())

    def retrieve_full(self, q, k=8, keys=None):
        """q:(B,S,kd). keys: optional (n,kd) live (grad-carrying) keys. Returns sims, idx, values, next_tok."""
        keys = self.keys[:self.n] if keys is None else keys
        sims = self.retriever.score(q, keys)
        sv, si = self.retriever.top_k(sims, k)
        return {"sims": sv, "idx": si, "values": self.values[si], "next_tok": self.nxt[si]}

    def retrieve(self, q, k=8, keys=None): return self.retrieve_full(q, k, keys)["values"]
    def nbytes(self): return sum(t.numel() * t.element_size() for t in (self.values, self.keys, self.imp, self.nxt))
    def stats(self): return {"n": self.n, "capacity": self.N, "util": self.n / max(self.N, 1), "added": self.added,
                              "evicted": self.evicted, "dissolved": self.dissolved, "mean_importance": float(self.imp[:self.n].mean()) if self.n else 0.0,
                              "mb": self.nbytes() / 2**20}
    def get_state(self): return {k: getattr(self, k).clone() for k in ("values", "keys", "imp", "nxt")} | {"n": self.n, "added": self.added, "evicted": self.evicted, "dissolved": self.dissolved}
    def set_state(self, s):
        for k in ("values", "keys", "imp", "nxt"): getattr(self, k).copy_(s[k])
        self.n, self.added, self.evicted, self.dissolved = s["n"], s["added"], s["evicted"], s["dissolved"]
