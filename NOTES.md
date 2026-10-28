# Interpretations & deviations (read before trusting results)
Config values are exactly DESIGN §1.3. Where the doc was underspecified or self-contradictory I chose the following.

## Unverified (no GPU in build sandbox)
- VRAM (<=1.9 GB, vram_profile <=1400 MB) is NOT measured. CPU analytic: params 30 MB + AdamW 52 MB + memory tree 18.3 MB. `vram_profile` + `--dry-run` measure/assert on CUDA.
- fp16 autocast + GradScaler path never executed (CPU runs are fp32). First CUDA run: watch for NaNs.
- Only 10-tick runs were done at full spec; M3 metrics (latent x3 drop, retrieval learning, ppl<3.5) are untested.

## Deviations
1. EMA target stored fp32, not fp16 (+~5 MB): fp16 with alpha=0.99 stalls updates. Spec says fp16.
2. Latent loss is the mean over channels (not sum) of squared error; target = LayerNorm(E(h)); plus a small variance-floor hinge (risk register "collapse").
3. tier_janus is computed from integer counters => zero gradient. It is a logged balance metric, not an optimizer force.

## Interpretations
- Param count: actual 6.77M trainable + 1.19M EMA + 49.6K predictor. §1.3's "~3.5M" is inconsistent with 5 blocks x 1.18M FFN.
- Qubit layer: d/2 qubit pairs, 16 real params each (7/3/3/3) = ~3K/layer. Banks defined in qubit_layer.py docstring; bracket routing = composition order (rot->Q, bulk->V, kill gates output, create adds). No dense Q/K/V projections.
- Short-term buffer is per batch row (B contiguous streams) and gets `write_per_seq=1` top-importance token per row per tick, so W=2048 spans 2048 ticks and each 500-tick consolidation sees 500 candidates/row. Spec's "all tokens append" would fill W in 2 ticks.
- Long store holds fp16 values + next-token labels; keys are recomputed live via key_proj each read so key_proj/W_q get gradients. Retrieval loss = softplus(-sgn*u), sgn=+1 if retrieved next-token == true next token; u also gates the retrieval weight.
- FlashCap: C_flash=16 = channel GROUPS; top 8 groups by update norm may change per iteration (16 active channels contradicts max_frozen_ratio=0.5).
- p_dissolve = per-consolidation probability that an existing long-term entry is dropped. Long eviction = lowest importance (importance decays x0.99 per consolidation).
- Poisson clock: bank b active this tick if Poisson(rate_b)>0 (rates 5,3,2,2); all banks on at eval.
- Halting: relative change < tau_halt=1e-3, frozen after halt; spec's halt-entropy regularizer not implemented. Importance scorer: only `latent_loss` (others raise NotImplementedError).
- `flat` ablation: one store (W+N slots), written directly every tick, top-k retrieval only.
- Eval continues from live train-memory then restores it. Val ppl<3.5 (CE 1.25) in 10K ticks (~10M tokens) is ambitious for char-level; expect it to fail unless training is longer.
