import torch
from qp.train.trainer import Trainer

def test_latent_decreases_and_ema_moves(tiny):
    t = Trainer(tiny.replace(lr=3e-3)); e0 = [p.clone() for p in t.model.target.block.parameters()]
    L = []
    for _ in range(40):
        b = t.stream.next(); L.append(float(t.step({"x": b[:, :-1], "y": b[:, 1:]})["latent"]))
    assert all(map(lambda v: v == v, L)) and sum(L[-5:]) < sum(L[:5])
    assert any((a != b).any() for a, b in zip(e0, t.model.target.block.parameters()))
    assert all(not p.requires_grad for p in t.model.target.block.parameters())
