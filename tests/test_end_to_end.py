import pytest, torch
from qp.train.trainer import Trainer

def test_tiers_fill_and_retrieve(tiny):
    t = Trainer(tiny.replace(K_consolidate=3, lr=2e-3)); seen = False
    for _ in range(25):
        b = t.stream.next(); r = t.step({"x": b[:, :-1], "y": b[:, 1:]}); assert torch.isfinite(r["loss"])
        seen |= "retrieval" in r
    st = t.model.memory.stats(); assert st["short"]["n"] > 0 and st["long"]["n"] > 0 and seen
    assert st["mb"] < 20

@pytest.mark.parametrize("ov", [dict(use_loop=False), dict(use_short=False), dict(use_long=False), dict(use_latent=False), dict(use_janus=False), dict(flat=True)])
def test_ablations_run(tiny, ov):
    t = Trainer(tiny.replace(K_consolidate=3, **ov))
    for _ in range(8): b = t.stream.next(); assert torch.isfinite(t.step({"x": b[:, :-1], "y": b[:, 1:]})["loss"])
    assert t.evaluate()["finite"]

def test_checkpoint_resume(tiny, tmp_path):
    from qp.train.checkpoint import load_checkpoint
    t = Trainer(tiny)
    for _ in range(6): b = t.stream.next(); t.step({"x": b[:, :-1], "y": b[:, 1:]})
    p = t.save(tmp_path / "c.pt"); t2 = Trainer(tiny); load_checkpoint(p, t2)
    assert t2.tick == 6 and all(torch.equal(a, b) for a, b in zip(t.model.parameters(), t2.model.parameters()))

@pytest.mark.slow
def test_full_spec_10_ticks():
    from qp.train.config import Config
    t = Trainer(Config(dataset="synthetic", device="cpu", amp="off"))
    for _ in range(10): b = t.stream.next(); assert torch.isfinite(t.step({"x": b[:, :-1], "y": b[:, 1:]})["loss"])
