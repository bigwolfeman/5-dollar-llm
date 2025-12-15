# TitanMAC vs MoE A/B Experiments

**Created**: 2025-12-15
**Status**: Planning
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
- [ ] Calculate TitanMAC param count with neural memory enabled
- [ ] Create `configs/titanmac_config.py` with `TitanMACGPU24GBConfig`
- [ ] Create `models/titanmac_wrapper.py` with MoE-compatible interface
- [ ] Verify TitanMAC loads SmolLM tokenizer correctly
- [ ] Create `train_titanmac.py` using train_moe infrastructure

### Phase 2: Experiment 1 Execution
- [ ] Run TitanMAC training with identical hyperparameters
- [ ] Monitor: val_loss, val_accuracy, perplexity, memory_loss
- [ ] Save checkpoint in same format as baseline
- [ ] Compare metrics

### Phase 3: Experiment 2 Setup
- [ ] Create `optimizers/nested_optimizer/` directory structure
- [ ] Port DeepNestedOptimizer from TitanMAC
- [ ] Port supporting files (controller, param_groups, meta_trainer)
- [ ] Create `group_moe_params()` function
- [ ] Create `train_moe_nested.py`

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

## References

- TitanMAC source: `111TitanMAC-Standalone/titans_core/`
- Baseline config: `configs/moe_config.py:GPU24GBMoEModelConfig`
- Training loop: `training/trainer.py:train_model()`
- Muon optimizer: `optimizers/muon.py`
