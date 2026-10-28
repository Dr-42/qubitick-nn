import torch, torch.nn.functional as F
from ..latent.loss import latent_loss
from ..janus.tier_balance import tier_janus_loss

def combined_loss(logits, target_tokens, z_hat_list, z_tgt_list, halt_mask,
                  retrieval_scores, retrieval_targets, boundary_stats, cfg):
    """CE + lambda*latent + gamma*retrieval + mu*tier_janus. halt_mask = list of (B,S) alive masks per loop step.
    retrieval_scores = utility logits (B,S,k); retrieval_targets = +/-1 (retrieved next-token == true next token)."""
    V = logits.shape[-1]
    tok_ce = F.cross_entropy(logits.float().reshape(-1, V), target_tokens.reshape(-1), reduction="none").view_as(target_tokens)
    ce = tok_ce.mean(); total = ce; m = {"ce": ce.detach()}
    tok_imp = tok_ce.detach()
    if z_hat_list and cfg.use_latent:
        lat, tok_lat = latent_loss(z_hat_list, z_tgt_list, halt_mask)
        total = total + cfg.lambda_latent * lat; m["latent"] = lat.detach(); tok_imp = tok_lat
    if retrieval_scores is not None:
        ret = F.softplus(-retrieval_targets.float() * retrieval_scores.float()).mean()   # -log sigmoid(sgn * u)
        total = total + cfg.gamma_retrieval * ret; m["retrieval"] = ret.detach()
    jan = tier_janus_loss(boundary_stats, cfg.janus_window).to(logits.device)
    mu = cfg.mu_janus if cfg.use_janus else 0.0
    total = total + mu * jan; m["janus"] = jan
    m["token_importance"] = tok_imp
    return total, m
