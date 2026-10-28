import torch.nn as nn
class Predictor(nn.Module):
    """P_phi: small bottleneck MLP (~49K params at d=384)."""
    def __init__(self, d, hidden=64): super().__init__(); self.net = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d))
    def forward(self, h): return self.net(h)
