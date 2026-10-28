import torch
from qp.cap.flash_cap import FlashCap

def test_flash_cap_only():
    c = FlashCap(16, 0.5); hp = torch.zeros(2, 3, 48); h = torch.randn(2, 3, 48)
    o = c.enforce(h, hp); frozen = ((o - hp).view(2, 3, 16, 3).abs().sum(-1) == 0).float().mean()
    assert abs(float(frozen) - 0.5) < 1e-6 and c.stats()["frozen_frac"] <= 0.5
