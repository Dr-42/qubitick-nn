import dataclasses, torch
from pathlib import Path

def save_checkpoint(path, tr):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"cfg": dataclasses.asdict(tr.cfg), "model": tr.model.state_dict(), "opt": tr.opt.state_dict(), "scaler": tr.scaler.state_dict(),
                "ema": tr.model.target.block.state_dict() if tr.model.target else None, "memory": tr.model.memory.get_state(),
                "tick": tr.tick, "data_off": tr.stream.off, "clock_rng": tr.clock.rng.bit_generator.state}, path)

def load_checkpoint(path, tr):
    s = torch.load(path, map_location=tr.device, weights_only=False)
    tr.model.load_state_dict(s["model"]); tr.opt.load_state_dict(s["opt"]); tr.scaler.load_state_dict(s["scaler"])
    if s["ema"] is not None: tr.model.target.block.load_state_dict(s["ema"])
    tr.model.memory.set_state(s["memory"]); tr.tick = s["tick"]; tr.stream.off = s["data_off"]; tr.clock.rng.bit_generator.state = s["clock_rng"]
    return s
