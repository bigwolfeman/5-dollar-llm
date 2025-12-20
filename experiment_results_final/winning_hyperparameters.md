# Winning Hyperparameters from Grid Search

## Overview

This document records the winning hyperparameters discovered through grid search fuzzing of the DeepNestedOptimizer's network depth parameters. The grid search tested 72 configurations (36 per model) across a 6x6 grid of momentum and controller network depths.

## Grid Search Configuration

| Parameter | Values Tested |
|-----------|---------------|
| momentum_num_layers (M) | 1, 2, 3, 4, 5, 6 |
| controller_num_layers (C) | 1, 2, 3, 4, 5, 6 |
| Total Experiments | 72 (36 MoE + 36 TitanMAC) |
| Steps per Experiment | 500 |
| Evaluation Metric | Validation Loss |

## Fixed Hyperparameters (from prior LR fuzzing)

These parameters were held constant during the depth grid search:

| Parameter | Value | Description |
|-----------|-------|-------------|
| base_lr | 3e-4 | Base learning rate for model parameters |
| meta_lr | 1e-4 | Learning rate for meta-learner updates |
| k_unroll | 5 | Unrolling steps for meta-gradient |
| momentum_hidden_dim | 64 | Hidden dimension of momentum network |
| controller_hidden_dim | 32 | Hidden dimension of controller network |
| weight_decay | 0.2 | Weight decay regularization |
| max_grad_norm | 1.0 | Gradient clipping threshold |
| mode | explicit | Optimization mode |
| use_unrolled | False | Use SimplifiedMetaTrainer (memory efficient) |
| use_cms_updates | False | Disable CMS updates for MoE |

---

## Winning Configurations

### MoE + DeepNestedOptimizer

**Best Configuration: M=3, C=2**

| Metric | Value |
|--------|-------|
| momentum_num_layers | **3** |
| controller_num_layers | **2** |
| Val Loss (500 steps) | 5.1725 |
| Val Accuracy (500 steps) | 24.02% |
| Peak VRAM | 18.68 GB |

**Grid Search Top 5 (MoE):**

| Rank | Config | Val Loss | Val Accuracy |
|------|--------|----------|--------------|
| 1 | M3_C2 | 5.1725 | 24.02% |
| 2 | M5_C4 | 5.1777 | 23.99% |
| 3 | M4_C1 | 5.1919 | 23.78% |
| 4 | M2_C1 | 5.1921 | 23.83% |
| 5 | M2_C2 | 5.1961 | 23.81% |

### TitanMAC + DeepNestedOptimizer

**Best Configuration: M=4, C=5**

| Metric | Value |
|--------|-------|
| momentum_num_layers | **4** |
| controller_num_layers | **5** |
| Val Loss (500 steps) | 5.2697 |
| Val Accuracy (500 steps) | 23.28% |
| Peak VRAM | 19.52 GB |

**Grid Search Top 5 (TitanMAC):**

| Rank | Config | Val Loss | Val Accuracy |
|------|--------|----------|--------------|
| 1 | M4_C5 | 5.2697 | 23.28% |
| 2 | M6_C1 | 5.2942 | 23.01% |
| 3 | M5_C2 | 5.3059 | 22.77% |
| 4 | M1_C6 | 5.3088 | 22.79% |
| 5 | M5_C5 | 5.3082 | 23.02% |

---

## Extended Training Results (Winners vs Baselines)

The winning nested configurations were trained for extended steps alongside Muon+AdamW baselines.

### At 4800 Steps

| Model | Optimizer | Val Loss | Val Accuracy | Perplexity | VRAM |
|-------|-----------|----------|--------------|------------|------|
| TitanMAC | Muon+AdamW | **2.7262** | **45.45%** | **15.28** | 10.94 GB |
| MoE | Muon+AdamW | 3.1741 | 40.78% | 23.91 | 10.74 GB |
| MoE | Nested (M3_C2) | 3.7870 | 34.22% | 44.12 | 19.35 GB |

### TitanMAC Nested Winner (Extended to 6400 Steps)

| Metric | Value |
|--------|-------|
| Configuration | M=4, C=5 |
| Total Steps | 6400 |
| Val Loss | 3.5016 |
| Val Accuracy | 36.85% |
| Val Perplexity | 33.17 |
| Peak VRAM | 20.23 GB |
| Training Time | 58.56 min |

---

## Key Findings

### 1. Optimal Depth Ranges
- **MoE**: Shallower networks work best (M=3, C=2)
- **TitanMAC**: Deeper networks preferred (M=4, C=5)

### 2. Performance vs Baseline
The Muon+AdamW baseline significantly outperforms the DeepNestedOptimizer on both architectures at equivalent training steps. However, the nested optimizer shows consistent improvement with more training.

### 3. VRAM Usage
DeepNestedOptimizer uses approximately 2x more VRAM than Muon+AdamW due to:
- Additional momentum network states
- Controller network parameters
- Meta-gradient computation overhead

### 4. Training Stability
Many depth configurations (especially M=1 or M=6 with certain C values) showed instability with loss values near 6.9+ (close to random initialization). The winning configurations demonstrated stable convergence.

---

## Reproduction Commands

### MoE Nested Winner
```bash
python train_moe_nested.py \
    --config 168m \
    --max_steps 4800 \
    --momentum_num_layers 3 \
    --controller_num_layers 2 \
    --base_lr 0.0003 \
    --meta_lr 0.0001 \
    --experiment_name moe_nested_winner
```

### TitanMAC Nested Winner
```bash
python train_titanmac_nested.py \
    --config 168m \
    --max_steps 4800 \
    --momentum_num_layers 4 \
    --controller_num_layers 5 \
    --base_lr 0.0003 \
    --meta_lr 0.0001 \
    --experiment_name titanmac_nested_winner
```

---

## Files

- Grid Search Results: `grid_results_168m/grid_results.json`
- Winner Checkpoints: `checkpoints_168m/{model}_nested_winner/`
- Training Curves: `checkpoints_168m/{model}_nested_winner/training_curves.png`
- Heatmaps: `experiment_results_final/charts/`

---

*Document generated: 2024-12-19*
*Grid search completed with 72/72 successful experiments*
