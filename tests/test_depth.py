import torch
from qp.train.trainer import Trainer

def test_depth_extrapolation_finite(tiny):
    t = Trainer(tiny)
    for _ in range(5): b = t.stream.next(); t.step({"x": b[:, :-1], "y": b[:, 1:]})
    r = t.depth_extrapolate(T_eval=(4, 8, 16, 32))
    assert all(r[T]["finite"] for T in (4, 8, 16, 32))
