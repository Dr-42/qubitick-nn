import numpy as np, torch

class PoissonClock:
    """Each sector bank fires with Poisson(rate) events per tick; a bank is active this tick if count>0."""
    def __init__(self, rates=(5.0, 3.0, 2.0, 2.0), seed=0):
        self.rates = np.asarray(rates, dtype=float); self.rng = np.random.default_rng(seed); self.tick = 0
    def step(self):
        a = (self.rng.poisson(self.rates) > 0).astype(np.float32); self.tick += 1
        return {"tick": self.tick, "active_banks": torch.from_numpy(a)}
