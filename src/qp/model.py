import torch, torch.nn as nn
from .backbone.stack import Backbone
from .lm.head import LMHead
from .latent.predictor import Predictor
from .latent.target_encoder import EMATarget
from .memory.tree import MemoryTree
from .memory.config import MemoryConfig

class QPModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg; d = cfg.d_model
        self.emb = nn.Embedding(cfg.vocab_size, d)
        self.pos = nn.Parameter(torch.randn(cfg.seq_len, d) * 0.02)
        self.backbone = Backbone(cfg)
        self.head = LMHead(d, cfg.vocab_size)
        self.predictor = Predictor(d) if cfg.use_latent else None
        self.memory = MemoryTree(MemoryConfig.from_cfg(cfg))
        self.target = EMATarget(self.backbone.rec, cfg.ema_alpha) if cfg.use_latent else None
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear) and m.weight.abs().sum() != 0:
            nn.init.normal_(m.weight, std=0.02); nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding): nn.init.normal_(m.weight, std=0.02)

    def to(self, *a, **k):
        super().to(*a, **k)
        if self.target is not None: self.target.to(*a, **k)
        dev = next(self.parameters()).device
        self.memory.reset(dev)
        return self

    def forward(self, tokens, T=None, active=None, use_memory=True):
        S = tokens.shape[1]
        x = self.emb(tokens) + self.pos[:S].to(self.emb.weight.dtype)
        out = self.backbone(x, T or self.cfg.T, self.memory if use_memory else None, active)
        out["logits"] = self.head(out["h_final"])
        return out

    def latent_pairs(self, out):
        hs = out["h_loop"]; n = len(hs) - 1
        if n < 1 or self.predictor is None: return [], []
        zh = [self.predictor(h) for h in hs[:-1]]
        with torch.no_grad(): zt = list(self.target.encode(torch.cat([h.detach() for h in hs[1:]], 0)).chunk(n, 0))
        return zh, zt

    def param_counts(self):
        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        ema = sum(p.numel() for p in self.target.block.parameters()) if self.target else 0
        return {"trainable": tr, "ema_target": ema, "predictor": sum(p.numel() for p in self.predictor.parameters()) if self.predictor else 0}
