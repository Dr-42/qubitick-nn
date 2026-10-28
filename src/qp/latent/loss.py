import torch

def latent_loss(z_hat_list, z_tgt_list, alive, var_floor=0.5, var_coef=0.05):
    """mean_t ||P(h_t) - E(h_{t+1})||^2 (mean over channels) per token over its T_eff alive steps.
    Returns (scalar, per-token (B,S)). Variance-floor hinge on predictor outputs guards identity collapse."""
    n = len(z_hat_list)
    zh, zt = torch.stack(z_hat_list).float(), torch.stack(z_tgt_list).float()
    err = ((zh - zt) ** 2).mean(-1)
    m = torch.stack(alive[:n]).float()
    tok = (err * m).sum(0) / m.sum(0).clamp(min=1)
    std = zh.reshape(-1, zh.shape[-1]).std(0)
    return tok.mean() + var_coef * torch.relu(var_floor - std).mean(), tok.detach()
