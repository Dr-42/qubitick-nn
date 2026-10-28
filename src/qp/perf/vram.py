import time, torch
from ..train.config import Config

def _opt_state_bytes(model): return 2 * 4 * sum(p.numel() for p in model.parameters() if p.requires_grad)

def vram_profile(model, batch, seq_len, T):
    """Peak VRAM of fwd+bwd (+analytic AdamW state). CPU => analytic only, verified=False."""
    dev = next(model.parameters()).device
    pb = sum(p.numel() * p.element_size() for p in model.parameters())
    if model.target is not None: pb += sum(p.numel() * p.element_size() for p in model.target.block.parameters())
    tree = model.memory.nbytes()
    if dev.type != "cuda":
        return {"peak_mb": None, "verified": False, "breakdown": {"params": pb / 2**20, "optimizer_state": _opt_state_bytes(model) / 2**20, "memory_tree": tree / 2**20}}
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    base = torch.cuda.memory_allocated()
    x, y = batch[:, :-1].to(dev), batch[:, 1:].to(dev)
    with torch.autocast("cuda", dtype=torch.float16):
        out = model(x, T=T); loss = torch.nn.functional.cross_entropy(out["logits"].float().reshape(-1, out["logits"].shape[-1]), y.reshape(-1))
    loss.backward(); torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated(); resv = torch.cuda.max_memory_reserved()
    model.zero_grad(set_to_none=True)
    ctx = 450 * 2**20   # CUDA context + runtime (not visible to the allocator; spec budget 400-500 MB)
    total = resv + _opt_state_bytes(model) + ctx
    return {"peak_mb": total / 2**20, "verified": True, "allocator_peak_mb": peak / 2**20, "reserved_mb": resv / 2**20,
            "breakdown": {"params": pb / 2**20, "optimizer_state": _opt_state_bytes(model) / 2**20, "memory_tree": tree / 2**20,
                          "cuda_ctx_assumed": ctx / 2**20, "activations_and_frag": (resv - base) / 2**20}}

def throughput(model, stream, n_steps, T):
    dev = next(model.parameters()).device; model.eval(); t0 = None
    with torch.no_grad():
        for i in range(n_steps + 2):
            if i == 2:
                if dev.type == "cuda": torch.cuda.synchronize()
                t0 = time.time()
            b = stream.next().to(dev); model(b[:, :-1], T=T)
    if dev.type == "cuda": torch.cuda.synchronize()
    return n_steps / (time.time() - t0)

def match_vram(cfg, target_mb, lo=96, hi=1536, step=12):
    """Largest d_model (multiple of lcm(step,C_flash)) whose measured peak <= target_mb. Needs CUDA."""
    import math
    from ..model import QPModel
    from ..data import get_stream
    assert torch.cuda.is_available(), "match_vram needs CUDA"
    q = math.lcm(step, cfg.C_flash); cands = list(range(max(q, (lo // q) * q), hi + 1, q)); best = cands[0]
    lo_i, hi_i = 0, len(cands) - 1
    while lo_i <= hi_i:
        mid = (lo_i + hi_i) // 2; c = cfg.replace(d_model=cands[mid]); torch.cuda.empty_cache()
        try:
            m = QPModel(c).cuda(); b = get_stream(c.dataset, c.seq_len, c.batch_size, data_dir=c.data_dir).next()
            peak = vram_profile(m, b, c.seq_len, c.T)["peak_mb"]
        except torch.cuda.OutOfMemoryError: peak = float("inf")
        del m; torch.cuda.empty_cache()
        if peak <= target_mb: best = cands[mid]; lo_i = mid + 1
        else: hi_i = mid - 1
    return cfg.replace(d_model=best, match_vram_mb=0)
