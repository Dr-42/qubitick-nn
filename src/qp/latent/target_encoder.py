import copy, torch, torch.nn.functional as F

class EMATarget:
    """EMA copy of the recurrent block; target = LayerNorm(E(h^{t+1})). NOTE: kept fp32 (see NOTES.md)."""
    def __init__(self, rec_block, alpha=0.99):
        self.block = copy.deepcopy(rec_block).eval(); self.alpha = alpha
        for p in self.block.parameters(): p.requires_grad_(False)
        self.src = rec_block
    def to(self, *a, **k): self.block.to(*a, **k); return self
    @torch.no_grad()
    def update(self):
        e = list(self.block.parameters()); s = [p.detach() for p in self.src.parameters()]
        torch._foreach_mul_(e, self.alpha); torch._foreach_add_(e, s, alpha=1 - self.alpha)
    @torch.no_grad()
    def encode(self, h):
        return F.layer_norm(self.block.layer(h).float(), (h.shape[-1],)).to(h.dtype)
