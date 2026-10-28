import json, math, time, random
from pathlib import Path
import numpy as np, torch
from .config import Config
from .checkpoint import save_checkpoint, load_checkpoint
from ..model import QPModel
from ..data import get_stream
from ..clock.poisson import PoissonClock
from ..loss.combined import combined_loss
from ..eval.evaluator import evaluate
from ..eval.depth_sweep import depth_sweep

class Trainer:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)
        self.device = torch.device("cuda" if cfg.device == "auto" and torch.cuda.is_available() else ("cpu" if cfg.device == "auto" else cfg.device))
        self.amp = (self.device.type == "cuda") if cfg.amp == "auto" else cfg.amp in (True, "true", "on")
        self.model = QPModel(cfg).to(self.device)
        decay = [p for p in self.model.parameters() if p.requires_grad and p.ndim >= 2]
        nodecay = [p for p in self.model.parameters() if p.requires_grad and p.ndim < 2]
        self.opt = torch.optim.AdamW([{"params": decay, "weight_decay": cfg.weight_decay}, {"params": nodecay, "weight_decay": 0.0}], lr=cfg.lr, betas=(0.9, 0.95))
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=self.amp)
        self.stream = get_stream(cfg.dataset, cfg.seq_len, cfg.batch_size, "train", cfg.data_dir, cfg.vocab_size)
        self.clock = PoissonClock(cfg.bank_rates, cfg.seed)
        self.tick = 0; self.reverts = 0; self.out = Path(cfg.out_dir); self.log_f = None
        self._good = None

    def _lr(self, t):
        c = self.cfg
        if t < c.warmup: return c.lr * (t + 1) / c.warmup
        p = min(1.0, (t - c.warmup) / max(1, c.n_ticks - c.warmup))
        return c.lr * (c.min_lr_ratio + (1 - c.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * p)))

    def step(self, batch):
        cfg, m = self.cfg, self.model; m.train()
        x, y = batch["x"].to(self.device), batch["y"].to(self.device)
        for g in self.opt.param_groups: g["lr"] = self._lr(self.tick)
        active = self.clock.step()["active_banks"].to(self.device)
        with torch.autocast(self.device.type, dtype=torch.float16, enabled=self.amp):
            out = m(x, active=active)
            zh, zt = m.latent_pairs(out)
            util = tgt = None
            mem = out["mem"]
            if mem is not None and mem["util"] is not None:
                util = mem["util"]; tgt = (mem["next_tok"] == y.unsqueeze(-1)).float() * 2 - 1
            loss, met = combined_loss(out["logits"], y, zh, zt, out["alive"], util, tgt, m.memory.boundary_stats(), cfg)
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        gn = torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad], cfg.grad_clip)
        self.scaler.step(self.opt); self.scaler.update()
        if m.target is not None: m.target.update()
        with torch.no_grad():
            m.memory.write(out["h_halt"].detach(), out["halts"], met["token_importance"], self.tick, next_tok=y)
            m.memory.advance_tick()
        self.tick += 1
        return {"loss": loss.detach(), "grad_norm": gn.detach(), **{k: v for k, v in met.items() if k != "token_importance"}}

    def _finite(self, loss):
        return bool(torch.isfinite(loss)) and all(bool(torch.isfinite(p).all()) for p in self.model.parameters())

    def train(self, n_ticks=-1, dry_run=False):
        cfg = self.cfg; n = cfg.n_ticks if n_ticks < 0 else n_ticks
        self.out.mkdir(parents=True, exist_ok=True)
        self.log_f = None if dry_run else open(self.out / "log.jsonl", "a")
        vs = get_stream(cfg.dataset, cfg.seq_len, cfg.batch_size, "val", cfg.data_dir, cfg.vocab_size)
        t0 = time.time(); last = None
        while self.tick < n:
            b = self.stream.next(); r = self.step({"x": b[:, :-1], "y": b[:, 1:]}); last = r
            bad = not bool(torch.isfinite(r["loss"]))
            if bad and not dry_run and self._good is not None:     # NaN detector -> revert to last good ckpt
                self.reverts += 1; load_checkpoint(self._good, self)
                if self.reverts > 10: raise RuntimeError("too many NaN reverts")
                continue
            if self.tick % cfg.log_every == 0 or self.tick == n or dry_run:
                rec = {"tick": self.tick, "tps": self.tick / (time.time() - t0 + 1e-9), "reverts": self.reverts,
                       **{k: float(v) for k, v in r.items()}, "mem": self.model.memory.stats()}
                print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items() if k != "mem"}), flush=True)
                if self.log_f: self.log_f.write(json.dumps(rec) + "\n"); self.log_f.flush()
            if not dry_run and cfg.eval_every and self.tick % cfg.eval_every == 0:
                ev = self.evaluate(vs); print("EVAL", json.dumps(ev), flush=True)
                if self.log_f: self.log_f.write(json.dumps({"tick": self.tick, **ev}) + "\n")
            if not dry_run and cfg.ckpt_every and self.tick % cfg.ckpt_every == 0 and bool(torch.isfinite(r["loss"])):
                p = self.out / "last_good.pt"; self.save(p); self._good = p
        return self.save(self.out / "final.pt") if not dry_run else self.out

    def evaluate(self, stream=None, T=None):
        stream = stream or get_stream(self.cfg.dataset, self.cfg.seq_len, self.cfg.batch_size, "val", self.cfg.data_dir, self.cfg.vocab_size)
        return evaluate(self.model, stream, T=T, n_batches=self.cfg.eval_batches, device=self.device, amp=self.amp)

    def depth_extrapolate(self, stream=None, T_eval=(4, 8, 16, 32)):
        stream = stream or get_stream(self.cfg.dataset, self.cfg.seq_len, self.cfg.batch_size, "val", self.cfg.data_dir, self.cfg.vocab_size)
        return depth_sweep(self.model, stream, T_eval, device=self.device, amp=self.amp)

    def save(self, path):
        save_checkpoint(path, self); return Path(path)
