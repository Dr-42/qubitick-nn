from dataclasses import dataclass

@dataclass
class MemoryConfig:
    d_model: int = 384
    key_dim: int = 384
    batch_size: int = 4
    W_short: int = 2048
    N_long: int = 8192
    k_retrieval: int = 8
    K_consolidate: int = 500
    M_consolidate: int = 64
    importance: str = "latent_loss"
    write_per_seq: int = 1
    retr_temp: float = 0.1
    long_decay: float = 0.99
    p_dissolve: float = 0.01
    janus_window: int = 100
    use_short: bool = True
    use_long: bool = True
    flat: bool = False

    @classmethod
    def from_cfg(cls, c):
        use_short = c.use_short and not c.flat
        return cls(d_model=c.d_model, key_dim=c.d_model, batch_size=c.batch_size,
                   W_short=c.W_short if use_short else 0,
                   N_long=(c.W_short + c.N_long) if c.flat else (c.N_long if c.use_long else 0),
                   k_retrieval=c.k_retrieval, K_consolidate=c.K_consolidate, M_consolidate=c.M_consolidate,
                   importance=c.importance, write_per_seq=c.write_per_seq, retr_temp=c.retr_temp,
                   long_decay=c.long_decay, p_dissolve=c.p_dissolve, janus_window=c.janus_window,
                   use_short=use_short, use_long=(c.use_long or c.flat), flat=c.flat)
