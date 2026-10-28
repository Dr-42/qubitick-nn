from ..train.config import Config

def memory_sweep(model, stream, configs, ticks=300, **kw):
    """Retrain short runs under varied W_short/N_long/k_retrieval (memory dims are structural, so each is a fresh run)."""
    from ..train.trainer import Trainer
    out = []
    for ov in configs:
        cfg = model.cfg.replace(**ov, n_ticks=ticks, eval_every=0, out_dir=f"{model.cfg.out_dir}/msweep"); t = Trainer(cfg)
        t.train(ticks); out.append({"overrides": ov, **t.evaluate()})
    return out
