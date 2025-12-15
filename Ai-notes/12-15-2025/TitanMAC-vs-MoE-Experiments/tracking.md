# TitanMAC vs MoE A/B Experiments

**Created**: 2025-12-15
**Status**: Implementation Complete - Ready for Execution
**Goal**: Surgical A/B comparison of TitanMAC architecture and DeepNestedOptimizer against 5-dollar-llm baseline

---

## Overview

Two experiments to evaluate your TitanMAC work against the 5-dollar-llm MoE baseline:

| Experiment | Variable Tested | Held Constant |
|------------|-----------------|---------------|
| **Exp 1** | Architecture (TitanMAC vs MoE) | Optimizer (Muon+AdamW), Data, Training Pipeline |
| **Exp 2** | Optimizer (DeepNested vs Muon) | Architecture (MoE), Data, Training Pipeline |

---

## Experiment 1: Architecture Comparison

### TitanMAC Configuration

**Philosophy**: Let TitanMAC fully express itself - neural memory + sliding window attention. The whole point of Titans is to be attention-free via learned memory.

**Key Features to Enable**:
- `use_neural_memory=True` - Core Titans feature
- `use_block_sparse=True` - O(T*w) sliding window attention
- `titans_variant="MAC"` or `"MAG"` - Memory as Context or Gate dataflow

**What Makes TitanMAC Different**:
1. **No full attention** - Uses sliding window (O(T*w) vs O(T²))
2. **Neural Memory** - Deep MLP that acts as learned long-term memory, updated via gradient descent
3. **Memory dataflow variants**:
   - MAC: Memory tokens concatenated with input segments
   - MAG: Memory gates the attention output
   - MAL: Memory replaces attention in alternating layers

### Baseline MoE Configuration (GPU24GBMoEModelConfig)

```python
d_model=512, n_heads=8, n_layers=8, d_ff=2048
num_experts=8, expert_top_k=2
batch_size=16, max_seq_len=1024, max_steps=800
muon_lr=0.04, adamw_lr=0.006
```

### Integration Requirements

| Component | MoE Baseline | TitanMAC Adaptation |
|-----------|--------------|---------------------|
| Output format | `(logits, aux_loss)` tuple | Wrapper returns tuple; aux_loss = memory_loss |
| Tokenizer | SmolLM-135M | Must use SmolLM-135M |
| Data | smollm-corpus | Same |
| Optimizer | Muon+AdamW | Same (NOT DeepNested) |
| Config interface | `MoEModelConfig` dataclass | Create compatible `TitanMACGPU24GBConfig` |

### Files to Create

```
models/
└── titanmac_wrapper.py       # TitanMAC with MoE-compatible interface

configs/
└── titanmac_config.py        # TitanMACGPU24GBConfig

train_titanmac.py             # Training script reusing train_moe infrastructure
```

### TitanMAC Wrapper Interface

```python
class TitanMACWrapper(nn.Module):
    """Wraps TitanMAC to match MoEMinimalLLM interface."""

    def forward(self, x, return_aux_loss=True):
        # x: input_ids [B, T]
        output = self.model(x, labels=x)  # TitanMAC returns dict
        logits = output["logits"]
        aux_loss = output.get("memory_loss", torch.tensor(0.0))

        if return_aux_loss:
            return logits, aux_loss
        return logits
```

---

## Experiment 2: Optimizer Comparison

### DeepNestedOptimizer Overview

The nested optimizer from TitanMAC combines:

1. **L2RegressionMomentum** - Learned momentum via MLP that predicts optimal gradient transform
2. **NestedController** - Learned per-group LR multipliers based on gradient statistics
3. **ContinuumMemorySystem (CMS)** - Multi-frequency updates (fast/medium/slow adaptation)
4. **Meta-learning** - Components trained via k-step unrolled differentiation

### Simple vs Explicit Mode

| Mode | Behavior | Use Case |
|------|----------|----------|
| **Simple** | Auto meta-updates every N steps using training loss as proxy | Quick experiments, less principled |
| **Explicit** | Manual `meta_update(val_batch)` calls with validation data | Proper meta-learning, more control |

**Explicit mode selected** because:
- More principled: uses actual validation loss for meta-objective
- Better control: we decide when/how often to meta-update
- Clearer comparison: we can match meta-update frequency to eval frequency

### Explicit Mode Usage

```python
optimizer = DeepNestedOptimizer(model, mode='explicit', ...)

for step, batch in enumerate(train_loader):
    loss = compute_loss(model, batch)
    loss.backward()
    optimizer.step(loss.item())

    # Manual meta-update on eval steps
    if step % eval_every == 0:
        val_batch = next(val_iter)
        optimizer.meta_update(
            val_batch=val_batch,
            train_batches=recent_train_batches,  # k batches for unrolling
            use_unrolled=True
        )
```

### Files to Create

```
optimizers/
└── nested_optimizer/
    ├── __init__.py
    ├── deep_nested_optimizer.py    # Main optimizer class
    ├── nested_controller.py        # LR multiplier network
    ├── param_groups.py             # Parameter grouping for MoE
    └── meta_trainer.py             # Unrolled meta-learning

train_moe_nested.py                 # Training script with nested optimizer
```

### MoE Parameter Grouping

Need to adapt `group_titans_params()` for MoE architecture:

```python
def group_moe_params(model):
    """Group MoE parameters for nested optimizer."""
    core_params = []    # Attention, experts
    embed_params = []   # Embeddings, norms, router

    for name, param in model.named_parameters():
        if 'embedding' in name or 'norm' in name or 'router' in name:
            embed_params.append(param)
        else:
            core_params.append(param)

    return core_params, embed_params
```

---

## Implementation Checklist

### Phase 0: Baseline
- [ ] Run baseline MoE training to establish reference metrics
- [ ] Verify checkpoint format and metrics output
- [ ] Document exact baseline configuration

### Phase 1: Experiment 1 Setup
- [x] Calculate TitanMAC param count with neural memory enabled (~52M params)
- [x] Create `configs/titanmac_config.py` with `TitanMACGPU24GBConfig`
- [x] Create `models/titanmac_wrapper.py` with MoE-compatible interface
- [x] Verify TitanMAC loads SmolLM tokenizer correctly
- [x] Create `train_titanmac.py` using train_moe infrastructure

### Phase 2: Experiment 1 Execution
- [ ] Run TitanMAC training with identical hyperparameters
- [ ] Monitor: val_loss, val_accuracy, perplexity, memory_loss
- [ ] Save checkpoint in same format as baseline
- [ ] Compare metrics

### Phase 3: Experiment 2 Setup
- [x] Create `optimizers/nested_optimizer/` directory structure
- [x] Port DeepNestedOptimizer from TitanMAC
- [x] Port supporting files (controller, param_groups, meta_trainer)
- [x] Create `group_moe_params()` function
- [x] Create `train_moe_nested.py`

### Phase 4: Experiment 2 Execution
- [ ] Run MoE with DeepNestedOptimizer (explicit mode)
- [ ] Match meta-update frequency to eval_every
- [ ] Monitor: val_loss, val_accuracy, perplexity, lr_multipliers
- [ ] Compare metrics to baseline

---

## Metrics Comparison Template

| Metric | MoE Baseline | TitanMAC (Exp1) | MoE+Nested (Exp2) |
|--------|--------------|-----------------|-------------------|
| Val Loss | 4.0977 | TBD | TBD |
| Val Accuracy | 31.90% | TBD | TBD |
| Perplexity | 60.20 | TBD | TBD |
| Aux Loss | (load_bal) | (memory) | (load_bal) |
| Training Time | TBD | TBD | TBD |
| Peak VRAM | TBD | TBD | TBD |

---

## Open Questions / Decisions Made

1. **TitanMAC memory**: YES - use neural memory, it's the core of Titans
2. **Attention type**: Sliding window (block-sparse) - this is what makes Titans efficient
3. **Tokenizer**: SmolLM-135M for both
4. **Param matching**: Not strict - let TitanMAC be larger due to memory module
5. **Nested mode**: Explicit - more principled meta-learning
6. **File organization**: Nested optimizer files in `optimizers/nested_optimizer/`

---

## Implementation Notes (2025-12-15)

### Experiment 1: TitanMAC

**Files Created**:
- `models/titanmac_wrapper.py` (158 lines) - Wrapper with MoE-compatible interface
- `configs/titanmac_config.py` (243 lines) - Three configs: base, GPU24GB, debug
- `train_titanmac.py` (302 lines) - Training script

**Key Implementation Details**:
1. **AMP Disabled**: Neural memory's `torch.autograd.grad` calls fail with mixed precision. AMP is disabled by default.
2. **Memory Compensation**: Smaller batch (4) + 4x gradient accumulation to match effective batch size
3. **Gradient Checkpointing**: Enabled by default to reduce VRAM
4. **Default Variant**: MAG (Memory as Gate) - more efficient than MAC
5. **Param Count**: ~52M parameters

**Usage**:
```bash
python train_titanmac.py                           # Full training
python train_titanmac.py --debug --max_steps 10    # Quick test
python train_titanmac.py --variant MAC             # Use MAC variant
```

### Experiment 2: Nested Optimizer

**Files Created**:
- `optimizers/nested_optimizer/__init__.py` (37 lines)
- `optimizers/nested_optimizer/deep_nested_optimizer.py` (724 lines)
- `optimizers/nested_optimizer/nested_controller.py` (119 lines)
- `optimizers/nested_optimizer/param_groups.py` (172 lines)
- `optimizers/nested_optimizer/meta_trainer.py` (352 lines)
- `train_moe_nested.py` (702 lines)

**Key Implementation Details**:
1. **Parameter Grouping**: Core (2D matrices) vs Embed (embeddings, norms, router) - mirrors Muon/AdamW split
2. **Explicit Mode**: Manual `meta_update()` on eval steps with validation batch
3. **Batch Buffer**: Recent k training batches stored in deque for unrolled meta-learning
4. **Extra Metrics**: Tracks lr_multipliers_core, lr_multipliers_embed, meta_losses

**Usage**:
```bash
python train_moe_nested.py                                    # Full training
python train_moe_nested.py --base_lr 3e-4 --meta_lr 1e-4     # Custom LRs
python train_moe_nested.py --k_unroll 5                       # Unroll steps
```

---

## References

- TitanMAC source: `111TitanMAC-Standalone/titans_core/`
- Baseline config: `configs/moe_config.py:GPU24GBMoEModelConfig`
- Training loop: `training/trainer.py:train_model()`
- Muon optimizer: `optimizers/muon.py`
