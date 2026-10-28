from .evaluator import evaluate

def depth_sweep(model, stream, T_values=(4, 8, 16, 32), n_batches=20, **kw):
    res = {T: evaluate(model, stream, T=T, n_batches=n_batches, **kw) for T in T_values}
    t8 = res.get(8, {}).get("val_ppl")
    if t8 and 16 in res: res["T16_within_20pct_of_T8"] = res[16]["val_ppl"] <= 1.2 * t8
    return res
