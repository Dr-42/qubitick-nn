# qubit-poisson (DESIGN v2.0, full spec)
```
pip install -e .[test]
pytest -m "not slow"                                   # 21 tests, CPU
python -m qp.train --config configs/base.yaml --dry-run --ticks 10   # prints vram_profile, asserts <=1900 MB on CUDA
python -m qp.train --config configs/base.yaml                         # 10K ticks, eval every 500, ckpt in runs/base
python -m qp.train --config configs/ablation_no_long.yaml             # match_vram_mb=1700 scales d_model first (CUDA only)
```
Logs: runs/<name>/log.jsonl. Resume: --resume runs/base/last_good.pt. Config is never altered by the code.
