import inspect, torch
from qp.backbone.qubit_layer import QubitLayer
from qp.backbone.recurrent_block import RecurrentBlock
from qp.backbone.stack import Backbone
from qp.memory.short import ShortTermBuffer
from qp.memory.long import LongTermStore
from qp.memory.retrieval import Retriever
from qp.memory.consolidate import Consolidator
from qp.memory.tree import MemoryTree
from qp.lm.tokenizer import CharTokenizer
from qp.lm.head import LMHead
from qp.lm.sampler import Sampler
from qp.cap.flash_cap import FlashCap
from qp.janus.tier_balance import tier_janus_loss
from qp.clock.poisson import PoissonClock
from qp.loss.combined import combined_loss
from qp.train.trainer import Trainer
from qp.eval import evaluate, depth_sweep, memory_sweep, ablation
from qp.data import get_stream
from qp.perf import vram_profile, throughput

def test_signatures():
    sig = lambda f: list(inspect.signature(f).parameters)
    assert sig(ShortTermBuffer.append)[:4] == ["self", "h", "tick", "importance"]
    assert sig(LongTermStore.add)[:4] == ["self", "keys", "values", "importance"]
    assert sig(Retriever.score) == ["self", "q", "keys"] and sig(Retriever.top_k) == ["self", "scores", "k"]
    assert sig(MemoryTree.write)[:5] == ["self", "h_final", "halts", "importance", "tick"]
    assert sig(combined_loss) == ["logits", "target_tokens", "z_hat_list", "z_tgt_list", "halt_mask", "retrieval_scores", "retrieval_targets", "boundary_stats", "cfg"]
    assert sig(RecurrentBlock.forward)[:3] == ["self", "h", "T"] and sig(Backbone.forward)[:4] == ["self", "x", "T", "memory_tree"]
    assert sig(get_stream)[:2] == ["name", "seq_len"] and sig(vram_profile) == ["model", "batch", "seq_len", "T"]
    for n in ("step", "train", "save", "depth_extrapolate"): assert hasattr(Trainer, n)

def test_tokenizer_roundtrip():
    t = CharTokenizer.from_text("hello world"); assert t.decode(t.encode("low")) == "low"
