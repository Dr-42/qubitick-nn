import torch

class Consolidator:
    def __init__(self, importance="latent_loss", m=64):
        if importance != "latent_loss":
            raise NotImplementedError("importance scorers 'attn_centrality'/'learned' are M4 alternatives, not implemented; use 'latent_loss'")
        self.importance, self.m = importance, m

    def select(self, short_term): return short_term.select_top_m(self.m)

    @torch.no_grad()
    def consolidate(self, short_term, long_term, key_proj, decay=1.0, p_dissolve=0.0):
        ents = self.select(short_term)
        if not ents: return {"promoted": 0}
        hs = torch.stack([e.h for e in ents]); imp = torch.tensor([e.importance for e in ents], device=hs.device)
        nt = torch.tensor([e.next_tok for e in ents], device=hs.device)
        keys = key_proj(hs.float())
        long_term.decay(decay); long_term.dissolve(p_dissolve)
        long_term.add(keys, hs, imp, nt)
        return {"promoted": len(ents), "top_importance": float(imp.max())}
