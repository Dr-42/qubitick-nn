import torch
from qp.memory.short import ShortTermBuffer
from qp.memory.long import LongTermStore
from qp.memory.consolidate import Consolidator

def test_topm_promotion():
    s = ShortTermBuffer(32, 8, True, 2, 1); L = LongTermStore(64, 8, 8)
    for t in range(30): s.append(torch.randn(2, 4, 8), t, torch.rand(2, 4), torch.randint(0, 5, (2, 4)))
    info = Consolidator("latent_loss", 8).consolidate(s, L, torch.nn.Linear(8, 8))
    assert info["promoted"] == 8 and L.n == 8
    assert float(L.imp[:8].min()) >= float(torch.cat([s.imp[0], s.imp[1]]).sort(descending=True).values[7]) - 1e-6
