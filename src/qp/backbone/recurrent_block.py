import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from .qubit_layer import QubitLayer

class RecurrentBlock(nn.Module):
    """h_{t+1} = A*h_t + (1-A)*F(h_t + eps), A = exp(-softplus(a)) in (0,1) diagonal  =>  rho(A) < 1 by construction."""
    def __init__(self, d, n_heads, ffn_mult=4, sigma=0.02, tau_halt=1e-3, cap=None):
        super().__init__()
        self.layer = QubitLayer(d, n_heads, ffn_mult)
        self.a_raw = nn.Parameter(torch.zeros(d))
        self.sigma, self.tau, self.cap = sigma, tau_halt, cap
        self.last_alive = []

    def A(self): return torch.exp(-F.softplus(self.a_raw.float()))
    def spectral_radius(self): return float(self.A().detach().max())

    def step(self, h, alive, active=None):
        x = h + self.sigma * torch.randn_like(h) if self.training and self.sigma > 0 else h
        hn = self.layer(x, active)
        A = self.A().to(h.dtype)
        hn = A * h + (1 - A) * hn
        if self.cap is not None: hn = self.cap.enforce(hn, h)
        hn = torch.where(alive.unsqueeze(-1), hn, h)
        rel = (hn - h).float().norm(dim=-1) / (h.float().norm(dim=-1) + 1e-6)
        return hn, alive & (rel >= self.tau)

    @torch.no_grad()
    def halt_mask(self, h, tau):
        rel = (self.layer(h) - h).float().norm(dim=-1) / (h.float().norm(dim=-1) + 1e-6)
        return rel < tau

    def forward(self, h, T, active=None, use_ckpt=False):
        alive = torch.ones(h.shape[:2], dtype=torch.bool, device=h.device)
        hs, alives = [h], []
        for _ in range(T):
            alives.append(alive)
            if use_ckpt and self.training and torch.is_grad_enabled():
                h, alive = checkpoint(self.step, h, alive, active, use_reentrant=False)
            else:
                h, alive = self.step(h, alive, active)
            hs.append(h)
            if not self.training and not bool(alive.any()): break
        self.last_alive = alives
        return h, hs
