import math, torch

class FlashCap:
    """Per loop iteration, channels split into C_flash groups; only the ceil(G*(1-max_frozen_ratio))
    groups with the largest update norm may change, the rest are frozen (never > max_frozen_ratio frozen)."""
    def __init__(self, C_flash=16, max_frozen_ratio=0.5):
        self.G, self.ratio = C_flash, max_frozen_ratio
        self.n_active = max(1, math.ceil(C_flash * (1 - max_frozen_ratio)))
        self._last = None

    def enforce(self, h, h_prev=None):
        if h_prev is None: return h
        B, S, d = h.shape; G = self.G
        assert d % G == 0, "d_model must be divisible by C_flash"
        delta = (h - h_prev).view(B, S, G, d // G)
        mag = delta.detach().float().abs().mean(-1)
        idx = mag.topk(self.n_active, dim=-1).indices
        mask = torch.zeros_like(mag).scatter_(-1, idx, 1.0)
        self._last = 1.0 - mask.mean().detach()
        assert (G - self.n_active) / G <= self.ratio + 1e-9
        return h_prev + (delta * mask.unsqueeze(-1).to(h.dtype)).view(B, S, d)

    def stats(self): return {"frozen_frac": float(self._last) if self._last is not None else 0.0, "n_active_groups": self.n_active}
