# Qubit-Poisson Net — Design Document (Standalone)

**Version 2.0 · Target: Autonomous agent swarm build+train · Launch budget: 2GB VRAM**

---

## 0 · Core idea

Build a **recurrent-depth transformer** on a Poisson clock, with **tiered memory** (flash / short-term / long-term) and **dual losses** (token CE + latent self-prediction). The recurrent block reuses weights across $T$ loop iterations, decoupling effective depth from parameter count. The tiered memory decouples **immediate computation** (flash) from **recent context** (short-term) from **consolidated history** (long-term), giving the model an explicit accessibility hierarchy that replaces the monolithic monotone ledger with three structurally distinct storage-and-retrieval paths.

The design draws on three validated threads:

- **Recurrent-depth transformers**: Ultron (75M params, effective depth 36), HelixLM (small-scale recurrent columns), Looped GPT-BERT 4x12 (12M params).
- **Joint-embedding predictive architecture**: I-JEPA and V-JEPA 1 (~300M), VL-JEPA (1.6B, matched token-space VLMs with ~50% fewer trainable params), V-JEPA 2 (1.2B, SOTA after LLM alignment).
- **Tiered memory hierarchies**: classic cognitive architectures, HTM, MemGPT, RETRO, kNN-LM — all separate working memory from consolidated store.

The open question this design tests: **do recurrent depth, latent self-prediction, and tiered memory compose at small scale**, using a recurrent hidden state as the latent prediction target and a content-addressed store as the long-term tier.

**Success =** runs end-to-end at ≤1.9 GB VRAM, all losses decrease over 10K ticks, val perplexity < 3.5, depth extrapolation works (train on 8 loops, evaluate on 16), and ablating any tier measurably degrades performance at matched VRAM.

**Non-goals:** competitive LLM capability, multi-GPU, inference optimization, production deployment, multimodal.

---

## 1 · Constraints

### 1.1 VRAM envelope (hard ceiling 1.9 GB)

| Component | Budget |
|---|---|
| CUDA context + PyTorch runtime | 400–500 MB |
| Params (backbone + embeddings + head + predictor + EMA target + retrieval keys) | 100–210 MB |
| Activations (grad checkpointing ON, per loop iteration) | 100–300 MB |
| Memory tree (short-term buffer + long-term store + retrieval index) | ~15 MB |
| Allocator fragmentation buffer | 200–300 MB |
| Total | ~0.9–1.4 GB |

### 1.2 Precision policy

- fp16 storage + matmuls; fp32 tensor-core accumulate.
- **fp32 mandatory** for `eigh`, `log`, `exp`, softmax normalizer, PSD projection, retrieval cosine similarity on fp16 values.
- GradScaler for fp16 underflow protection.
- No fp16 spectral ops — silent NaN source.

### 1.3 Base configuration

```yaml
d_model: 384
n_heads: 6
n_phys_blocks: 5             # 2 prelude + 1 recurrent + 2 coda
loop_T: 8                    # train-time loop count
seq_len: 256
batch_size: 4
ffn_mult: 4
vocab_size: 65               # char-level

# Memory tiers
C_flash: 16                  # active channels per loop iteration
W_short: 2048                # short-term window (ticks)
N_long: 8192                 # long-term slots
k_retrieval: 8               # top-k per tick
K_flash: 1                   # write to short-term every tick (on halt)
K_consolidate: 500           # consolidate short→long every 500 ticks
M_consolidate: 64            # candidates per consolidation pass
importance: "latent_loss"    # one of: latent_loss, attn_centrality, learned

# Losses
lambda_latent: 0.5
mu_janus: 0.1
janus_window: 100
ema_alpha: 0.99

# Capacity
p_dissolve: 0.01
```

Param count: ~3.5M trainable + ~3.5M EMA target (no grad, fp16) + ~50K predictor + retrieval keys (~6 MB fp16 for the store) ≈ **~7M total in memory**. Memory tree adds ~15 MB.

**Note on params.** Qubit-attention layers are parameter-efficient (~3K real params per layer). FFN after each block restores standard per-layer param count. The qubit structure does routing/interpretation work; FFN does representation work.

---

## 2 · Architecture

```
                        POISSON(Λ)
                             │
                             ▼
        ┌───────────────────────────────────────────────┐
        │                MEMORY TREE                    │
        │  ┌─────────┐  ┌─────────┐  ┌──────────────┐   │
        │  │  FLASH  │→ │  SHORT  │→ │     LONG     │   │
        │  │ (loops) │  │(window) │  │ (retrieval)  │   │
        │  └────┬────┘  └────┬────┘  └──────┬───────┘   │
        │       │            │              │           │
        │       └────────────┴──────────────┘           │
        │       accessibility tree (three read paths)   │
        └───────────────────────┬───────────────────────┘
                                │
              ┌─────────────────┼──────────────────┐
              ▼                 ▼                  ▼
        ┌──────────┐     ┌──────────┐      ┌──────────┐
        │ FORWARD  │     │ BACKWARD │      │  TIER    │
        │ prelude  │     │ latent   │      │  JANUS   │
        │ → loop   │     │ predict  │      │ balance  │
        │ → coda   │     │ h_{t+1}  │      │ at all   │
        │ → LM     │     │ vs EMA   │      │ 4 bounds │
        └────┬─────┘     └────┬─────┘      └────┬─────┘
             │                │                  │
             └────────────────┴──────────────────┘
                              ▼
                    ┌──────────────────┐
                    │ COMBINED LOSS    │
                    │ CE + λ·latent    │
                    │ + μ·tier_janus   │
                    └────────┬─────────┘
                             ▼
                    ┌──────────────────┐
                    │ OPT + EMA update │
                    └──────────────────┘
```

### 2.1 The three tiers

**Flash (F-tier).** The recurrent loop's working state, per token. Ephemeral — discarded at halting. This is *not* storage; it's the activation buffer the loop iterates in. Access: zero-latency, exact, per loop iteration. Capacity: $C_{\text{flash}} = 16$ active channels per iteration.

**Short-term (S-tier).** A windowed buffer of the last $W$ halted hidden states. Written on halt, read via attention. Lifetime: until aged out of the window. Access: zero-latency (in memory), exact, attention-weighted. Capacity: $W = 2048$ entries.

**Long-term (L-tier).** A content-addressed store of consolidated representations. Written on consolidation triggers, read via top-$k$ retrieval. Lifetime: persistent (with eviction policy). Access: retrieval latency, approximate, top-$k$. Capacity: $N = 8192$ slots.

### 2.2 The accessibility tree

At tick $\tau$, loop iteration $t$, the recurrent block has access to three parallel paths:

| Path | Source | Latency | Fidelity | Content |
|---|---|---|---|---|
| 0 | Flash | 0 | exact | $\{h_\tau^{(j)} : j \le t\}$ — current tick's loop states |
| 1 | Short-term | 0 | exact | $\{h_{\tau'}^{\text{final}} : \tau - W \le \tau' < \tau\}$ — last $W$ halts |
| 2 | Long-term | $O(\log N)$ | top-$k$ | $\text{retrieve}(q, k=8)$ from content-addressed store |

The block reads from all three paths every tick. The **query** for path 2 is a learned projection of the current flash state: $q = W_q h_\tau^{(t)}$.

### 2.3 Backbone: prelude / loop / coda

- **Prelude** (2 physical blocks): run once. Builds initial representation. Standard QubitLayers with sector-banked attention + FFN.
- **Recurrent block** (1 physical block): run $T$ times with LTI-stable injection:
  $$h_{t+1} = \mathcal{F}_\theta(h_t + \epsilon_t), \quad \epsilon_t \sim \mathcal{N}(0, \sigma^2 I)$$
  with $\rho(A) < 1$ required for stability.
- **Coda** (2 physical blocks): run once. Produces final representation for LM head.

Effective depth = $2 + T + 2$. With $T=8$: effective depth 12 from 5 physical layers.

**Adaptive compute (ACT):** each token halts its loop when $\|h_{t+1} - h_t\| < \tau_{\text{halt}}$. Training adds an entropy regularizer to discourage halting too early.

### 2.4 Attention: qubit-matrix weights with sector banks

Each QubitLayer contains four **sector banks** of 2×2 Hermitian weights:

| Bank | Sector | Real params/block | Role |
|---|---|---|---|
| Bulk | $\mathcal{E}_{++}$ | 7 | context integration |
| Rotation | $\mathcal{E}_{+-}$ | 3 | angular mixing (attention) |
| Create | $\mathcal{E}_{-+}$ | 3 | feature injection |
| Kill | $\mathcal{E}_{--}$ | 3 | feature removal |

Bracket routing:
$$
[W, J] \to J, \quad [J, K] \to \text{bulk}, \quad [W, K] \to K
$$

Each physical block has all four banks with learned mixing.

### 2.5 FFN

Standard 2-layer MLP with GELU after each QubitLayer: $d_{ff} = 4 d_{\text{model}}$.

### 2.6 Forward pass (per tick)

```python
def forward_tick(x, memory_tree, T):
    h = prelude(x)
    for t in range(T):
        h = recurrent_block(h)
        if should_halt(h, tau=t): break
    # Read all three tiers
    h_short = attention(h, memory_tree.short_term.states())      # Path 1
    q = W_q(h)                                                    # Path 2 query
    retrieved = memory_tree.long_term.retrieve(q, k=8)
    h_mem = h + h_short + combine(retrieved)
    # Coda + LM head
    h_out = coda(h_mem)
    logits = W_LM(h_out)
    return logits, h, retrieved
```

### 2.7 Backward pass (per tick, latent self-prediction)

At the same tick:

$$
\mathcal{L}_{\text{latent}} = \frac{1}{T_{\text{eff}}} \sum_{t=1}^{T_{\text{eff}}} \left\| \mathcal{P}_\phi(h^{(t)}) - \mathcal{E}_{\bar\theta}(h^{(t+1)}) \right\|^2
$$

- $\mathcal{P}_\phi$: small MLP applied to $h^{(t)}$.
- $\mathcal{E}_{\bar\theta}$: EMA copy of the recurrent block, applied to $h^{(t+1)}$.
- $T_{\text{eff}}$: effective loop count (accounting for early halts).

Plus a **retrieval utility loss** (new with tiering):

$$
\mathcal{L}_{\text{retrieval}} = -\log \sigma\bigl(W_r \, [h; \text{combine}(\text{retrieved})]\bigr) \cdot \text{sgn}(t_{\text{next}} \text{ from retrieved})
$$

The retrieval loss asks: "does this retrieved entry help predict the next token?" Entries that don't help get low importance scores and are evicted at the next consolidation.

### 2.8 Loss composition

$$
\mathcal{L} = \mathcal{L}_{\text{CE}} + \lambda \cdot \mathcal{L}_{\text{latent}} + \gamma \cdot \mathcal{L}_{\text{retrieval}} + \mu \cdot \mathcal{L}_{\text{tier\_janus}}
$$

Default: $\lambda = 0.5$, $\gamma = 0.1$, $\mu = 0.1$.

### 2.9 Tiered capacity caps

Each tier has its own budget:

| Tier | Cap | Eviction policy |
|---|---|---|
| Flash | $C_{\text{flash}} = 16$ channels per loop iteration | Auto (soft threshold at next iteration) |
| Short-term | $W = 2048$ entries | FIFO by tick, importance-weighted priority |
| Long-term | $N = 8192$ slots | Importance-weighted, or LRU |

The flash capacity cap is the previous design's `C_cap`. It is now *only* about flash compute budget, not the whole system.

### 2.10 Write paths (per tick)

| Boundary | Frequency | Selection |
|---|---|---|
| Flash → Short-term | Every halt (`K_flash = 1`) | All tokens append; drop if window full |
| Short-term → Long-term | Every `K_consolidate = 500` ticks | Top-$M_{\text{consolidate}} = 64$ by importance |
| Short-term → discard | When window full | FIFO or importance-weighted |
| Long-term → evict | When bank full | Importance-weighted or LRU |

**Importance scoring** (initial implementation): per-token latent loss magnitude, averaged over the loop. High latent loss = the model "worked hard" on this token = important. Alternatives at M4: attention centrality, learned scorer.

### 2.11 Tiered Janus regularizer

Balances write and evict rates at *each* tier boundary:

$$
\mathcal{L}_{\text{tier\_janus}} = \sum_{b \in \text{boundaries}} \left( \mathbb{E}[\text{writes}_b] - \mathbb{E}[\text{evicts}_b] \right)^2
$$

Four boundaries: flash→short, short→long, short→discard, long→evict. Each tier should neither hoard (writes ≫ evicts) nor churn (evicts ≫ writes). The Janus regularizer prevents both failure modes.

### 2.12 Continuous learning, structurally

Tiering makes the continuous-learning story structural rather than bolted-on:

| Standard CL | Tiered Qubit-Poisson |
|---|---|
| Replay buffer of past data | Long-term store *is* the replay, content-addressed |
| EWC on weights | EMA target + tiered decay resist drift at each tier |
| Sleep phase | Consolidation (short→long) is the sleep phase, every 500 ticks |
| Catastrophic forgetting | Short-term FIFO bounds recent memory; long-term eviction bounds archive |
| Knowledge accumulation | Long-term consolidates useful content indefinitely |

The importance scorer is the gate: only tokens the model *worked hard on* (high latent loss, high attention centrality) get promoted from short to long. Content that didn't matter gets evicted from short-term and never reaches long-term.

---

## 3 · Component contracts

Each agent owns one module. All exports must match signatures exactly. A1 (Architect) signs off before M1.

### A2 — `qp.backbone`

```python
class QubitLayer(nn.Module):
    def forward(self, x: Tensor[B, S, d]) -> Tensor[B, S, d]:
        """Sector-banked qubit attention + FFN."""

class RecurrentBlock(nn.Module):
    def forward(self, h: Tensor[B, S, d], T: int) -> tuple[Tensor, list[Tensor]]:
        """Loop T times with LTI injection; return final h + per-step h list."""
    def halt_mask(self, h: Tensor, tau: float) -> BoolTensor[B, S]: ...

class Backbone(nn.Module):
    def forward(self, x: Tensor[B, S, d], T: int, memory_tree: "MemoryTree") -> dict:
        """Returns {'h_final', 'h_loop', 'retrieved', 'halts'}"""
```

### A20 — `qp.memory.short` (new, P0)

```python
class ShortTermBuffer:
    def __init__(self, capacity: int = 2048, d_model: int = 384, fp16: bool = True): ...
    def append(self, h: Tensor[B, S, d], tick: int, importance: Tensor[B, S]) -> None:
        """Write to buffer; evict if full per policy."""
    def states(self) -> Tensor[B, W, d]:
        """Return windowed buffer for attention."""
    def importance_weights(self) -> Tensor[B, W]:
        """Per-entry importance, for attention weighting and consolidation selection."""
    def select_top_m(self, m: int) -> list[MemoryEntry]:
        """Top-m entries by importance, for promotion to long-term."""
    def evict(self, n: int) -> None: ...
    def stats(self) -> dict: ...
```

### A21 — `qp.memory.long` (new, P0)

```python
class LongTermStore:
    def __init__(self, capacity: int = 8192, d_model: int = 384, key_dim: int = 384): ...
    def add(self, keys: Tensor[M, key_dim], values: Tensor[M, d_model], importance: Tensor[M]) -> None:
        """Write entries; evict if full."""
    def retrieve(self, q: Tensor[B, S, key_dim], k: int = 8) -> Tensor[B, S, k, d_model]:
        """Top-k retrieval by cosine similarity."""
    def evict(self, victims: list[int]) -> None: ...
    def stats(self) -> dict: ...
```

### A22 — `qp.memory.retrieval` (new, P0)

```python
class Retriever:
    def __init__(self, key_dim: int = 384, k: int = 8): ...
    def score(self, q: Tensor[B, S, key_dim], keys: Tensor[N, key_dim]) -> Tensor[B, S, N]:
        """Cosine similarity. fp32 mandatory."""
    def top_k(self, scores: Tensor[B, S, N], k: int) -> tuple[Tensor[B, S, k], Tensor[B, S, k]]: ...
```

### A23 — `qp.memory.consolidate` (new, P0)

```python
class Consolidator:
    def __init__(self, importance: str = "latent_loss", m: int = 64): ...
    def select(self, short_term: ShortTermBuffer) -> list[MemoryEntry]: ...
    def consolidate(self, short_term: ShortTermBuffer, long_term: LongTermStore,
                    key_proj: nn.Module) -> dict:
        """Promote top-M short-term entries to long-term."""
```

### A24 — `qp.memory.tree` (new, P0)

```python
class MemoryTree:
    def __init__(self, config: MemoryConfig): ...
    def read(self, h: Tensor[B, S, d]) -> dict:
        """Returns {'short': h_short, 'retrieved': retrieved, 'scores': sim_scores}"""
    def write(self, h_final: Tensor[B, S, d], halts: BoolTensor, importance: Tensor[B, S],
              tick: int) -> dict:
        """Flash→short write on halt; consolidate on schedule."""
    def advance_tick(self) -> None: ...
    def stats(self) -> dict: ...
```

### A4 — `qp.lm`

```python
class CharTokenizer:
    def encode(self, s: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...

class LMHead(nn.Module):
    def forward(self, h: Tensor[B, S, d]) -> Tensor[B, S, vocab]: ...

class Sampler:
    def sample(self, logits: Tensor, temperature: float = 1.0, top_k: int = None) -> int: ...
```

### A5 — `qp.cap` (repurposed as FlashCap)

```python
class FlashCap:
    def __init__(self, C_flash: int = 16, max_frozen_ratio: float = 0.5): ...
    def enforce(self, h: Tensor[B, S, d]) -> Tensor[B, S, d]:
        """Freeze channels exceeding C_flash per loop iteration."""
    def stats(self) -> dict: ...
```

### A6 — `qp.janus` (repurposed as TierJanus)

```python
def tier_janus_loss(boundary_stats: dict, window: int = 100) -> Tensor:
    """Sum over boundaries of (E[writes] - E[evicts])²."""
```

### A7 — `qp.clock`

```python
class PoissonClock:
    def step(self) -> dict: ...  # {"tick", "active_banks"}
```

### A8 — `qp.loss`

```python
def combined_loss(logits, target_tokens, z_hat_list, z_tgt_list, halt_mask,
                  retrieval_scores, retrieval_targets, boundary_stats, cfg) -> tuple[Tensor, dict]:
    """CE + λ·latent + γ·retrieval + μ·tier_janus. Returns loss + metrics."""
```

### A9 — `qp.train`

```python
class Trainer:
    def step(self, batch: dict) -> dict: ...
    def train(self, stream, n_ticks: int = -1) -> Path: ...
    def save(self, path: Path) -> None: ...
    def depth_extrapolate(self, stream, T_eval: list[int]) -> dict: ...
```

### A10 — `qp.eval`

```python
def evaluate(model, val_iter) -> dict: ...
def depth_sweep(model, stream, T_values: list[int]) -> dict: ...
def memory_sweep(model, stream, configs: list[dict]) -> dict:
    """Vary W_short, N_long, k_retrieval; measure ppl."""
def ablation(config_overrides: dict) -> dict: ...
```

### A11 — `qp.data`

```python
class Stream(Protocol):
    def next(self) -> Tensor: ...
    def __iter__(self) -> "Stream": ...

def get_stream(name: str, seq_len: int) -> Stream: ...
```

### A12 — `qp.perf`

```python
def vram_profile(model, batch, seq_len, T: int) -> dict:
    """{peak_mb, breakdown: {params, activations, memory_tree, cuda, frag}}"""
def throughput(model, stream, n_steps: int, T: int) -> float: ...
```

### A13 — `tests/`

- `test_interfaces.py`, `test_backbone.py`, `test_latent.py`
- `test_memory_short.py` — FIFO, importance-weighted, window overflow
- `test_memory_long.py` — add, retrieve, evict, cosine exactness on fp16→fp32
- `test_consolidate.py` — top-M selection, promotion rate
- `test_cap.py` — flash cap only, not total
- `test_janus.py` — four-boundary balance converges
- `test_depth.py` — depth extrapolation
- `test_end_to_end.py` — 10 ticks without OOM

---

## 4 · Task graph

```
M0 (A1) — 2 days
  ├─ interface stubs
  ├─ pyproject.toml
  └─ CI green
       │
       ├────────────────────────────────────┐
       │                                    │
  M1a: Backbone (A2)   M1b: LM (A4)   M1c: Data (A11)   M1d: Clock (A7)
       │                                    │
       ├─ M2a: FlashCap (A5)  ← A2
       ├─ M2b: Latent (A3)    ← A2
       ├─ M2c: Retriever (A22)  ← A21 (dep on memory long)
       ├─ M2d: Short (A20)    ← A2
       ├─ M2e: Long (A21)     ← A2, A22
       └─ M2f: Consolidate (A23) ← A20, A21
             │
             └─ M3a: MemoryTree (A24)  ← A20, A21, A22, A23
                   │
                   └─ M3b: Loss (A8)    ← A2, A3, A5, A6, A24
                         │
                         └─ M4a: Trainer (A9) ← A8
                               │
                               ├─ M4b: Eval (A10)  ← A9
                               ├─ M4c: Perf (A12)  ← A9
                               └─ M4d: QA (A13)    ← A9
                               │
                               └─ M5: Integration (A1)
                                     │
                                     └─ M6: Full training + ablations (A10)
```

**Critical path:** A2 → A20/A21 → A24 → A8 → A9 → M5.

**Parallel starts after M0:** A4, A7, A11, A22.

---

## 5 · Milestones & acceptance criteria

### M0 — Spec freeze (day 1–2)

- All Protocol stubs committed.
- `pyproject.toml` locked: torch ≥ 2.1, transformers ≥ 4.35, datasets, pyyaml.
- `test_interfaces.py` passes on stubs.
- CI green.

### M1 — Minimal end-to-end (day 3–6)

- `python -m qp.train --config base.yaml --dry-run --ticks 10` succeeds.
- `vram_profile` reports ≤1400 MB.
- Zero NaN in 100 random-input ticks.
- LTI stability verified: $\rho(A) < 1$.

### M2 — Memory tiers functional (day 7–11)

- Short-term: append, evict, window overflow without leak.
- Long-term: add, retrieve (top-k cosine exactness), evict.
- Consolidator: top-M selection works.
- MemoryTree: read path returns all three tiers.
- **Memory overhead ≤ 20 MB.**
- 1000 ticks clean.

### M3 — Depth + tiered memory validation (day 12–16)

Four checks:

- **Latent loss decreases**: 5K-tick latent loss < 100-tick latent loss by factor ≥ 3.
- **Depth extrapolation**: trained at $T=8$, evaluated at $T \in \{4,8,16,32\}$. Finite at $T=32$; not worse than $T=8$ by more than 20% at $T=16$.
- **Retrieval utility**: retrieval loss decreases; retrieved entries at 5K ticks have higher average importance than at 100 ticks (the store is *learning* what to keep).
- **Task doesn't regress**: val ppl < 3.5.

If any fail, the composition hypothesis is weak at this scale; fall back to non-recurrent + flat memory (documented variant).

### M4 — Ablations (day 17–22)

Seven configs, **matched VRAM (~1.7 GB each)**. When a feature is removed, model is scaled up via `d_model` or `n_phys_blocks` until it hits the same peak — otherwise comparison is feature-vs-more-params, not feature-vs-no-feature.

| Config | What's removed | What it tests | Pass criterion |
|---|---|---|---|
| `base` | Nothing | Reference | — |
| `no_flash` (no_loop) | Loop unrolled to fixed depth 5 | Recurrence | ppl ≥ base + 0.05 |
| `no_short` | $W = 0$ (no short-term) | Recent context | ppl ≥ base + 0.05 |
| `no_long` | $N = 0$ (no long-term) | Consolidated history | ppl ≥ base + 0.05 |
| `no_latent` | No latent loss | Self-prediction | ppl ≥ base + 0.05 |
| `no_janus` | No tier balance | Tier discipline | Freeze drift > 2× base |
| `flat` | Collapse to single tier (base design v1) | Value of tiering | ppl ≥ base + 0.03 |

**If `no_flash`, `no_short`, `no_long`, and `no_latent` all match `base` at matched VRAM**, the composition claim collapses entirely. **If any single one regresses**, that mechanism contributes.

**Rich ablation:** the `flat` variant tests *tiering* specifically, on top of the individual tier tests.

### M5 — Integration (day 23–25)

- All tests green.
- Reproducibility: same seed → same curve (±5%).
- `REPORT.md` with M3/M4 tables, depth-sweep curve, memory sweep.

### M6 — Scale validation (day 26–30)

- 24GB GPU: `d_model=512`, `loop_T=12`, `W_short=4096`, `N_long=16384`, batch 8.
- Confirm no regression; document scale-up.

---

## 6 · Repo structure

```
qubit-poisson/
├── pyproject.toml
├── README.md
├── DESIGN.md
├── REPORT.md
├── configs/
│   ├── base.yaml
│   ├── ablation_no_flash.yaml
│   ├── ablation_no_short.yaml
│   ├── ablation_no_long.yaml
│   ├── ablation_no_latent.yaml
│   ├── ablation_no_janus.yaml
│   └── ablation_flat.yaml
├── src/qp/
│   ├── __init__.py
│   ├── backbone/{qubit_layer,recurrent_block,stack}.py
│   ├── lm/{head,tokenizer,sampler}.py
│   ├── latent/{predictor,target_encoder,loss}.py
│   ├── memory/
│   │   ├── flash.py          # stability docs + FlashCap
│   │   ├── short.py          # ShortTermBuffer
│   │   ├── long.py           # LongTermStore
│   │   ├── retrieval.py      # Retriever (top-k cosine)
│   │   ├── consolidate.py    # Consolidator (importance selection)
│   │   ├── evict.py          # EvictionPolicy per tier
│   │   └── tree.py           # MemoryTree coordinator
│   ├── cap/flash_cap.py
│   ├── janus/tier_balance.py
│   ├── clock/{poisson,rates}.py
│   ├── loss/{combined,scaling}.py
│   ├── train/{trainer,config,checkpoint}.py
│   ├── eval/{evaluator,ablation,depth_sweep,memory_sweep}.py
│   ├── data/{stream,dataset}.py
│   └── perf/{vram,profiler}.py
├── tests/
│   ├── test_interfaces.py
│   ├── test_backbone.py
│   ├── test_latent.py
│   ├── test_memory_short.py
│   ├── test_memory_long.py
│   ├── test_consolidate.py
│   ├── test_cap.py
│   ├── test_janus.py
│   ├── test_depth.py
│   └── test_end_to_end.py
└── scripts/{train,eval,profile}.sh
```

---

## 7 · Risk register

| Risk | Likelihood | Impact | Mitigation | Owner |
|---|---|---|---|---|
| Recurrent loop unstable at $T > 8$ | Medium | High | LTI $\rho(A)<1$; test $T=32$ at M1 | A2 |
| Latent loss collapses to identity | Medium | High | Variance floor on predictor; increase λ if collapse | A3 |
| **Retrieval useless at small scale** | High | Medium | Initialize retrieval as additive with learned weight; ablation tests; ship with γ=0 if retrieval hurts | A22, A10 |
| **Consolidation writes garbage** | Medium | High | Importance starts as latent loss; ablation at M4; log store precision@k against canary | A23 |
| **Short-term window too small** | Medium | Medium | W tunable; memory sweep at M4 | A20 |
| **Janus over-constrains** | Medium | Medium | Ship with μ=0; add only flash→short and short→long initially | A6 |
| **Tiered memory slows ticks** | Medium | Medium | Async retrieval (prefetch during loop); profile at M2 | A24, A12 |
| fp16 spectral instability | High | High | fp32 forced for spectral + cosine | A12, A9 |
| 2GB OOM at base | Low | High | Base ~1.4 GB; tested margin; backup d=256 | A12 |
| Freeze runaway (flash) | Medium | High | Hard cap `n_frozen ≤ 0.5 d_model`; assert in CI | A5 |
| Silent NaN after ~5K ticks | Medium | High | NaN detector; auto-revert to last good ckpt | A9 |

**The two tiering-specific risks** (retrieval useless, consolidation writes garbage) are the honest unknowns. Both are addressed by the ablation structure: if `no_long` matches `base`, retrieval is a null result and the design ships without it (or with γ=0). The `flat` ablation tests whether *any* tiering helps, independent of which tiers work.

---

## 8 · Success metrics

### Primary (must-have)

- End-to-end train at **≤1.9 GB VRAM** peak.
- All four losses decrease over 10K ticks (CE, latent, retrieval, tier Janus).
- Val ppl **< 3.5** on tinyshakespeare.
- **Depth extrapolation passes**: $T=16$ perplexity finite, not worse than $T=8$ by more than 20%.
- **At least three of four** ablation regressions pass: `no_flash`, `no_short`, `no_long`, `no_latent` all show ppl ≥ base + 0.05 at matched VRAM.

### Secondary (nice-to-have)

- Val ppl < 3.0.
- Throughput > 40 ticks/sec on 2GB.
- Tier Janus balance < 0.005 after warmup.
- Retrieval precision@8 on canary set > 0.3 (retrieved entries actually help).
- Long-term store utilization > 30% after 10K ticks (not hoarding, not starving).

### Learning signal (informational)

- Depth-sweep curve shape.
- Memory sweep: optimal `W_short` and `N_long` at this scale.
- Importance scorer accuracy: does latent loss predict future relevance?
- Consolidation quality: does top-M selection beat random?

---

## 9 · Scale-up path (post-M6)

| Stage | d_model | phys_blocks | loop_T | W_short | N_long | batch | VRAM | Target ppl |
|---|---|---|---|---|---|---|---|---|
| M6.1 | 512 | 7 | 12 | 4096 | 16384 | 8 | ~6 GB | 2.8 |
| M6.2 | 768 | 9 | 16 | 8192 | 32768 | 12 | ~14 GB | 2.5 |
| M6.3 | 1024 | 11 | 24 | 16384 | 65536 | 16 | ~24 GB | 2.2 |

Scale by **effective depth** (`loop_T`) and `d_model` jointly. Memory tiers scale linearly with `d_model` and log-linearly with `N_long` (retrieval index).

---

## 10 · Handoff protocol

**Phase 1 (M0):** A1 alone, produces interfaces + `pyproject.toml` + stub tests.

**Phase 2 (M1):** A2, A4, A7, A11, A22 in parallel (A22 has no deps).

**Phase 3 (M2):** A3, A5, A20, A21 in parallel (A21 depends on A22). A23 fires when A20+A21 land. A6 when A5 lands.

**Phase 4 (M3):** A24 (MemoryTree) depends on A20, A21, A22, A23. A8 (loss) depends on A2, A3, A5, A6, A24.

**Phase 5 (M4):** A9 (trainer). Then A10, A12, A13 in parallel.

**Phase 6 (M5/M6):** A1 integrates. A10 runs ablations, depth sweep, memory sweep. A12 produces scale report.

Every agent commits on its own branch; CI blocks merge on interface mismatch. No agent runs the full training loop except A9 and A10.

---

## 11 · The claim being tested

> **A recurrent-depth transformer on a Poisson clock, with tiered memory (flash / short-term / long-term) and dual (token + latent) losses, tests whether iterative refinement composes with hierarchical memory at small scale. Recurrence is validated (Ultron, HelixLM). Latent prediction is validated (VL-JEPA, V-JEPA 2). Tiered memory is validated (classic LLM memory hierarchies). The open question is whether these three compose at 7M params in a 2GB budget, and which tier contributes what.**

This claim is falsified at M3/M4 if:

- Depth extrapolation fails (recurrence not useful here),
- `no_flash` matches `base` at matched VRAM (recurrence not contributing),
- `flat` matches `base` at matched VRAM (tiering not contributing),
- or all four individual-tier ablations match `base` (composition claim collapses entirely).

If any subset fails, we fall back to the documented working subset (e.g., tiering + loop without long-term retrieval, if `no_long` matches). The document is a hypothesis, not a promise.

---

## 12 · Empirical grounding

Three published threads:

1. **Recurrent-depth**: Ultron (Hugging Face) — 770M looped matches 1.3B standard; `ultron_small` at 75M with effective depth 36. HelixLM — brain-inspired recurrent columns at small scale. Looped GPT-BERT 4x12 — 12M params on BabyLM Strict-Small.

2. **Latent prediction (JEPA)**: I-JEPA and V-JEPA 1 (~300M). VL-JEPA (1.6B, matched token-space VLMs with ~50% fewer params). V-JEPA 2 (1.2B, SOTA video QA after LLM alignment).

3. **Tiered memory**: classic cognitive architectures (Atkinson-Shiffrin, Baddeley), HTM hierarchical memory, MemGPT (virtual context), RETRO (retrieval-augmented), kNN-LM (nearest-neighbor retrieval). The design uses the standard tiered structure, applied for the first time (to our knowledge) to recurrent-depth transformers.

---

**Status:** Ready for handoff. A1 to begin M0.
