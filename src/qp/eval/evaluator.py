import math, torch

@torch.no_grad()
def evaluate(model, val_iter, T=None, n_batches=50, device=None, amp=False):
    """Val CE/ppl. Continues from the live memory state (train-data memories only) and restores it afterwards."""
    dev = device or next(model.parameters()).device
    snap = model.memory.get_state(); was = model.training; model.eval()
    tot, n, ok = 0.0, 0, True
    try:
        for _ in range(n_batches):
            b = val_iter.next().to(dev); x, y = b[:, :-1], b[:, 1:]
            with torch.autocast(dev.type, dtype=torch.float16, enabled=amp):
                out = model(x, T=T)
            ce = torch.nn.functional.cross_entropy(out["logits"].float().reshape(-1, out["logits"].shape[-1]), y.reshape(-1))
            ok &= bool(torch.isfinite(ce)); tot += float(ce); n += 1
            imp = (out["logits"].float().log_softmax(-1).gather(-1, y[..., None]).squeeze(-1)).neg()
            model.memory.write(out["h_halt"], out["halts"], imp, model.memory.tick, next_tok=y); model.memory.advance_tick()
    finally:
        model.memory.set_state(snap); model.train(was)
    ce = tot / max(n, 1)
    return {"val_ce": ce, "val_ppl": math.exp(ce) if ok and ce < 50 else float("inf"), "finite": ok}
