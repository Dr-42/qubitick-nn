import torch
from qp.model import QPModel

def test_forward_finite_no_nan(tiny):
    m = QPModel(tiny); x = torch.randint(0, 65, (2, 32))
    for _ in range(20):
        out = m(torch.randint(0, 65, (2, 32)))
        assert torch.isfinite(out["logits"]).all()
    assert out["logits"].shape == (2, 32, 65) and len(out["h_loop"]) >= 2

def test_lti_stable(tiny):
    m = QPModel(tiny); assert m.backbone.rec.spectral_radius() < 1
    m.backbone.rec.a_raw.data.fill_(-20)       # pathological param still < 1
    assert m.backbone.rec.spectral_radius() < 1 + 1e-6 and m.backbone.rec.A().max() <= 1

def test_causal(tiny):
    m = QPModel(tiny).eval(); a = torch.randint(0, 65, (1, 32)); b = a.clone(); b[0, 20:] = (b[0, 20:] + 1) % 65
    with torch.no_grad(): la, lb = m(a, use_memory=False)["logits"], m(b, use_memory=False)["logits"]
    assert torch.allclose(la[:, :20], lb[:, :20], atol=1e-4)

def test_psd_projection(tiny):
    m = QPModel(tiny); l = m.backbone.prelude[0]; l.bulk.data[:, 0] = -1.0     # negative bulk -> clamped PSD
    M = l.bulk_matrix(); assert torch.isfinite(M).all()
