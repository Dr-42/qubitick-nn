from dataclasses import dataclass, fields, asdict
import yaml

@dataclass
class Config:
    # --- spec (DESIGN §1.3) ---
    d_model: int = 384
    n_heads: int = 6
    n_phys_blocks: int = 5          # 2 prelude + 1 recurrent + 2 coda
    n_prelude: int = 2
    n_coda: int = 2
    loop_T: int = 8
    seq_len: int = 256
    batch_size: int = 4
    ffn_mult: int = 4
    vocab_size: int = 65
    C_flash: int = 16
    W_short: int = 2048
    N_long: int = 8192
    k_retrieval: int = 8
    K_flash: int = 1
    K_consolidate: int = 500
    M_consolidate: int = 64
    importance: str = "latent_loss"
    lambda_latent: float = 0.5
    gamma_retrieval: float = 0.1
    mu_janus: float = 0.1
    janus_window: int = 100
    ema_alpha: float = 0.99
    p_dissolve: float = 0.01
    # --- implementation knobs not fixed by the spec ---
    tau_halt: float = 1e-3
    sigma_noise: float = 0.02
    max_frozen_ratio: float = 0.5
    flash_cap: bool = True
    write_per_seq: int = 1
    retr_temp: float = 0.1
    long_decay: float = 0.99
    bank_rates: tuple = (5.0, 3.0, 2.0, 2.0)   # Poisson rates: bulk, rot, create, kill
    lr: float = 1e-3
    min_lr_ratio: float = 0.1
    warmup: int = 200
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    n_ticks: int = 10000
    eval_every: int = 500
    eval_batches: int = 50
    log_every: int = 50
    ckpt_every: int = 500
    seed: int = 0
    device: str = "auto"
    amp: str = "auto"
    grad_checkpoint: bool = True
    dataset: str = "tinyshakespeare"
    data_dir: str = "data"
    out_dir: str = "runs/base"
    # --- ablation switches (§5 M4) ---
    use_loop: bool = True
    use_short: bool = True
    use_long: bool = True
    use_latent: bool = True
    use_janus: bool = True
    flat: bool = False
    match_vram_mb: float = 0        # >0: scale d_model to this peak (needs CUDA)

    @property
    def T(self):  # effective train loop count
        return self.loop_T if self.use_loop else 1

    @classmethod
    def load(cls, path=None, **over):
        d = yaml.safe_load(open(path)) if path else {}
        d.update({k: v for k, v in over.items() if v is not None})
        names = {f.name for f in fields(cls)}
        bad = set(d) - names
        if bad: raise KeyError(f"unknown config keys: {bad}")
        if "bank_rates" in d: d["bank_rates"] = tuple(d["bank_rates"])
        return cls(**d)

    def replace(self, **kw):
        d = asdict(self); d.update(kw); return Config(**d)
