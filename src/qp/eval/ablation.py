import json
from ..train.config import Config

def ablation(config_overrides, base=None, ticks=None):
    """Train + evaluate one config; if match_vram_mb>0 (CUDA) d_model is first scaled to the target peak."""
    from ..train.trainer import Trainer
    from ..perf.vram import match_vram
    cfg = (base or Config()).replace(**config_overrides)
    if ticks: cfg = cfg.replace(n_ticks=ticks)
    if cfg.match_vram_mb > 0: cfg = match_vram(cfg, cfg.match_vram_mb)
    t = Trainer(cfg); t.train(cfg.n_ticks)
    return {"d_model": cfg.d_model, "params": t.model.param_counts(), **t.evaluate(), "depth": t.depth_extrapolate((4, 8, 16, 32))}
