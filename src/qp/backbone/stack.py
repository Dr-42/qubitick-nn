import torch, torch.nn as nn
from .qubit_layer import QubitLayer
from .recurrent_block import RecurrentBlock
from ..cap.flash_cap import FlashCap

class Backbone(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d, H, m = cfg.d_model, cfg.n_heads, cfg.ffn_mult
        assert cfg.n_prelude + 1 + cfg.n_coda == cfg.n_phys_blocks
        self.cfg = cfg
        self.prelude = nn.ModuleList(QubitLayer(d, H, m) for _ in range(cfg.n_prelude))
        self.flashcap = FlashCap(cfg.C_flash, cfg.max_frozen_ratio) if cfg.flash_cap else None
        self.rec = RecurrentBlock(d, H, m, cfg.sigma_noise, cfg.tau_halt, self.flashcap)
        self.coda = nn.ModuleList(QubitLayer(d, H, m) for _ in range(cfg.n_coda))
        self.ln_f = nn.LayerNorm(d)

    def forward(self, x, T, memory_tree=None, active=None):
        h = x
        for l in self.prelude: h = l(h, active)
        h, hs = self.rec(h, T, active, self.cfg.grad_checkpoint)
        alive = self.rec.last_alive
        mem = memory_tree.read(h) if memory_tree is not None else None
        hm = h + mem["short"] + mem["combined"] if mem is not None else h
        for l in self.coda: hm = l(hm, active)
        return {"h_final": self.ln_f(hm), "h_halt": h, "h_loop": hs, "alive": alive,
                "halts": ~alive[-1] if alive else torch.zeros_like(h[..., 0], dtype=torch.bool),
                "retrieved": None if mem is None else mem["retrieved"], "mem": mem}
