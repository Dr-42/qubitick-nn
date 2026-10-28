import sys, pathlib, pytest
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
from qp.train.config import Config

@pytest.fixture
def tiny():
    return Config(d_model=48, n_heads=6, seq_len=32, batch_size=2, loop_T=4, W_short=32, N_long=64, k_retrieval=4,
                  K_consolidate=5, M_consolidate=8, dataset="synthetic", device="cpu", amp="off", out_dir="/tmp/qp_test",
                  warmup=2, n_ticks=20, log_every=1000, eval_batches=2, grad_checkpoint=True)
