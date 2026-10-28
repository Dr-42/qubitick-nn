import torch
from qp.memory.short import ShortTermBuffer

def test_fifo_overflow_and_importance():
    s = ShortTermBuffer(8, 4, True, batch_size=2, n_write=1)
    for t in range(20):
        imp = torch.rand(2, 5); h = torch.full((2, 5, 4), float(t))
        s.append(h, t, imp, torch.zeros(2, 5, dtype=torch.long))
    assert s.n == 8 and s.states().shape == (2, 8, 4) and s.overwritten == 12 * 2
    assert set(s.tk[0].tolist()) == set(range(12, 20))           # only the newest 8 ticks survive
    top = s.select_top_m(3); assert len(top) == 3 and top[0].importance >= top[1].importance >= top[2].importance
    s.evict(2); assert s.n == 6
