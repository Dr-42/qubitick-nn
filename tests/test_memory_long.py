import torch
from qp.memory.long import LongTermStore

def test_add_retrieve_evict_exact_cosine():
    L = LongTermStore(16, 8, 8); torch.manual_seed(0)
    v = torch.randn(10, 8); L.add(v, v, torch.arange(10.), torch.arange(10))
    q = v[3:4].unsqueeze(0).float()                              # (1,1,8)
    r = L.retrieve_full(q, 3)
    assert int(r["idx"][0, 0, 0]) == 3 and abs(float(r["sims"][0, 0, 0]) - 1) < 1e-3
    ref = torch.nn.functional.normalize(q[0], dim=-1) @ torch.nn.functional.normalize(L.keys[:10].float(), dim=-1).t()
    assert torch.allclose(r["sims"][0, 0], ref[0].topk(3).values, atol=1e-5)
    L.add(v, v, torch.arange(10.) + 100); assert L.n == 16 and L.evicted == 4   # lowest-importance victims
    assert float(L.imp[:16].min()) >= 4
    L.evict([0, 1]); assert L.n == 14
