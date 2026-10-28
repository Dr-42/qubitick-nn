import argparse, json, sys
from .config import Config
from .trainer import Trainer

def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None); ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--ticks", type=int, default=-1); ap.add_argument("--device", default=None)
    ap.add_argument("--out-dir", default=None); ap.add_argument("--resume", default=None)
    a = ap.parse_args(argv)
    cfg = Config.load(a.config, device=a.device, out_dir=a.out_dir)
    if cfg.match_vram_mb > 0:
        from ..perf.vram import match_vram; cfg = match_vram(cfg, cfg.match_vram_mb); print("matched d_model =", cfg.d_model)
    tr = Trainer(cfg)
    if a.resume:
        from .checkpoint import load_checkpoint; load_checkpoint(a.resume, tr)
    print("params", json.dumps(tr.model.param_counts()), "| device", tr.device, "| amp", tr.amp, "| rho(A) =", tr.model.backbone.rec.spectral_radius())
    from ..perf.vram import vram_profile
    p = vram_profile(tr.model, tr.stream.next(), cfg.seq_len, cfg.T); tr.stream.off = 0
    print("vram_profile", json.dumps(p))
    if p["verified"]: assert p["peak_mb"] <= 1900, f"VRAM {p['peak_mb']:.0f} MB exceeds 1.9 GB ceiling"
    path = tr.train(a.ticks, dry_run=a.dry_run); print("done", path)
main()
