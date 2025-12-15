# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

5-Dollar LLM is an open research project for training language models from scratch on consumer hardware. The goal is to train the best possible LLM for approximately $5 in compute costs. It uses a Mixture of Experts (MoE) architecture with the Muon optimizer.

## Common Commands

```bash
# Full training run (requires 24GB+ VRAM, e.g., RTX 3090/4090)
python train_moe.py

# Debug/test run (works on any hardware including CPU)
python debug_moe.py

# Run benchmarks
python benchmarks/arc_challenge.py --checkpoint path/to/model.pt
python benchmarks/hellaswag.py --checkpoint path/to/model.pt
python benchmarks/compare_models.py checkpoint1.pt checkpoint2.pt

# Install dependencies
pip install -r requirements.txt
```

## Architecture

### Model (`models/`)
- **MoEMinimalLLM** (`moe_llm.py`): Main model class combining token embeddings, MoE transformer blocks, and output head with tied embeddings
- **MoETransformerBlock** (`layers.py`): Transformer block with attention + MoE feed-forward. Supports both standard MultiHeadAttention and MultiHeadLatentAttention (MLA)
- **MixtureOfExperts** (`components.py`): Top-k expert routing with load balancing loss. Each expert is a SiLU-activated FFN

### Training (`training/`)
- **train_model()** (`trainer.py`): Generic training loop with gradient accumulation, AMP, early stopping, and metrics tracking
- **train_moe_model()**: Convenience wrapper that sets up Muon+AdamW hybrid optimizer with cosine LR schedule

### Optimizer (`optimizers/`)
- **Muon** (`muon.py`): Momentum-based optimizer using Newton-Schulz orthogonalization. Used for 2D weight matrices (excluding embeddings/norms)
- Training uses a hybrid approach: Muon for weight matrices, AdamW for embeddings and normalization layers

### Configs (`configs/`)
- **MoEModelConfig**: Base config for ~3B parameter model (H100 scale)
- **GPU24GBMoEModelConfig**: Reduced config for single 24GB GPU (512 d_model, 8 layers, 8 experts)
- **DebugMoEConfig**: Tiny config for testing (128 d_model, 2 layers, 4 experts)

### Data (`data/`)
- Uses HuggingFaceTB/smollm-corpus (cosmopedia-v2 subset) with SmolLM-135M tokenizer
- Streaming mode by default to handle large datasets
- Dataset caching with invalidation based on config changes

## Key Patterns

### Training Flow
1. `train_moe.py` loads config, prepares datasets with caching, creates dataloaders
2. `train_moe_model()` initializes model, sets up Muon+AdamW optimizers with cosine schedule
3. Training loop handles gradient accumulation, AMP, periodic evaluation, and checkpointing

### MoE Routing
- TopKRouter selects top-2 experts per token using learned gating with noise for exploration
- Load balancing auxiliary loss encourages uniform expert utilization
- Weighted expert outputs are combined per token

### Config Hierarchy
All configs inherit from `MoEModelConfig` and override specific parameters. Override via CLI args:
```bash
python train_moe.py --muon_lr 0.03 --adamw_lr 0.005 --max_steps 1000
```

## Baselines

Current 24GB GPU baseline (GPU24GBMoEModelConfig):
- Val Loss: 4.0977
- Val Accuracy: 31.90%
- Perplexity: 60.20

Results stored in `baselines/gpu_24gb/`. New experiments should aim to surpass these metrics.
