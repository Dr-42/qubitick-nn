import torch

def tier_janus_loss(boundary_stats, window=100):
    """sum_b (E[writes_b] - E[evicts_b])^2 over the last `window` ticks.
    NOTE: counters are non-differentiable, so this is a monitored regularizer value (zero gradient)."""
    tot = 0.0
    for b, s in boundary_stats.items():
        w, e = s["writes"][-window:], s["evicts"][-window:]
        if w: tot += (sum(w) / len(w) - sum(e) / len(e)) ** 2
    return torch.tensor(float(tot))
