"""
Training script for MoE model with DeepNestedOptimizer.

This is Experiment 2 of the A/B comparison:
- Experiment 1 (baseline): train_moe.py with Muon+AdamW
- Experiment 2 (this file): train_moe_nested.py with DeepNestedOptimizer

The nested optimizer uses explicit mode with manual meta_update calls
aligned with evaluation steps.

Usage:
    python train_moe_nested.py --experiment_name nested_exp1

Key differences from baseline:
- Single optimizer (DeepNestedOptimizer) instead of Muon+AdamW split
- Meta-updates during evaluation steps using buffered training batches
- Learned LR multipliers per parameter group
"""

import argparse
import time
import os
import math
import json
import torch
import torch.nn.functional as F
import logging
from collections import deque
from pathlib import Path
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Optional, Dict, Any
import matplotlib.pyplot as plt

# Fix tokenizer parallelism warning when using DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from configs.moe_config import MoEModelConfig, GPU24GBMoEModelConfig
from configs.dataset_config import DataConfig
from models.moe_llm import MoEMinimalLLM
from optimizers.nested_optimizer import DeepNestedOptimizer, group_moe_params
from training.evaluation import evaluate_model
from utils.helpers import set_seed
from utils.logger import setup_logging


def print_system_info():
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    print(f"PyTorch: {torch.__version__}\n")


def prepare_datasets(data_cfg, tokenizer, cache_dir="./processed_data"):
    """Prepare train and validation datasets with caching."""
    import json
    import shutil
    from datasets import load_from_disk, load_dataset, Dataset
    from data.loader import tokenize_and_chunk, finalize_dataset

    train_cache = os.path.join(cache_dir, "train")
    val_cache = os.path.join(cache_dir, "val")
    info_path = os.path.join(cache_dir, "dataset_info.json")

    # Define what config parameters invalidate the cache
    config_state = {
        "dataset_path": data_cfg.dataset_path,
        "dataset_name": data_cfg.dataset_name,
        "tokenizer_name": data_cfg.tokenizer_name,
        "seq_length": data_cfg.seq_length,
        "num_samples": data_cfg.num_samples,
    }

    # Try to load valid cache
    if os.path.exists(train_cache) and os.path.exists(val_cache) and os.path.exists(info_path):
        try:
            with open(info_path, "r") as f:
                if json.load(f) == config_state:
                    print(f"Loading cached datasets from {cache_dir}...")
                    return load_from_disk(train_cache), load_from_disk(val_cache)
            print("Cache configuration mismatch. Rebuilding...")
        except Exception as e:
            print(f"Cache check failed ({e}). Rebuilding...")

    # Rebuild cache
    if os.path.exists(cache_dir):
        print(f"Cleaning old cache at {cache_dir}...")
        shutil.rmtree(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    # Load and split
    print("Loading raw dataset and splitting documents...")
    raw_dataset = load_dataset(
        data_cfg.dataset_path,
        data_cfg.dataset_name,
        split=data_cfg.split,
        cache_dir=data_cfg.cache_dir,
        streaming=True,
    )

    raw_samples = list(raw_dataset.take(data_cfg.num_samples))
    num_val = int(len(raw_samples) * 0.1)
    num_train = len(raw_samples) - num_val

    raw_train = Dataset.from_list(raw_samples[:num_train])
    raw_val = Dataset.from_list(raw_samples[num_train:])
    print(f"Split into {len(raw_train):,} train docs and {len(raw_val):,} val docs")

    # Tokenize and save
    print("Tokenizing train set...")
    train_ds = finalize_dataset(tokenize_and_chunk(raw_train, tokenizer, data_cfg), data_cfg)
    train_ds.save_to_disk(train_cache)

    print("Tokenizing validation set...")
    val_ds = finalize_dataset(tokenize_and_chunk(raw_val, tokenizer, data_cfg), data_cfg)
    val_ds.save_to_disk(val_cache)

    # Save cache info
    with open(info_path, "w") as f:
        json.dump(config_state, f, indent=2)
    print("Saved dataset cache info.")

    return train_ds, val_ds


def create_loss_fn(config: MoEModelConfig):
    """
    Create a loss function compatible with meta_update interface.

    Returns a function that takes (model, batch) and returns total loss.
    """
    def loss_fn(model, batch):
        device = next(model.parameters()).device

        # Handle different batch formats
        if isinstance(batch, dict):
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
        elif isinstance(batch, (list, tuple)):
            if len(batch) == 3:
                x, _, y = batch
            else:
                x, y = batch
            x = x.to(device)
            y = y.to(device)
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        logits, aux_loss = model(x, return_aux_loss=True)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = y[:, 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, config.vocab_size),
            shift_labels.view(-1)
        )

        total_loss = ce_loss
        if aux_loss is not None:
            total_loss = total_loss + aux_loss

        return total_loss

    return loss_fn


def train_moe_nested(
    config: MoEModelConfig,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: Optional[str] = None,
    experiment_name: Optional[str] = None,
    nested_config: Optional[Dict[str, Any]] = None,
):
    """
    Train MoE model with DeepNestedOptimizer.

    Args:
        config: Model configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        output_dir: Optional directory to save outputs
        experiment_name: Optional experiment name for logging
        nested_config: Optional config dict for nested optimizer

    Returns:
        model, final_metrics, metrics_history
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Default nested optimizer config
    if nested_config is None:
        nested_config = {
            'base_lr': 3e-4,
            'meta_lr': 1e-4,
            'k_unroll': 5,
            'momentum_hidden_dim': 64,
            'controller_hidden_dim': 32,
            'mode': 'explicit',
            'meta_update_freq': config.eval_every,
            'weight_decay': config.weight_decay,
            'max_grad_norm': config.grad_clip,
        }

    print(f"\n[Nested Optimizer] Training MoE model with DeepNestedOptimizer")
    print(f"  Base LR: {nested_config['base_lr']}")
    print(f"  Meta LR: {nested_config['meta_lr']}")
    print(f"  K-unroll: {nested_config['k_unroll']}")
    print(f"  Mode: {nested_config['mode']}")

    # Initialize model
    set_seed(42)
    model = MoEMinimalLLM(config)
    model = model.to(device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    core_params, embed_params = group_moe_params(model)
    core_numel = sum(p.numel() for p in core_params)
    embed_numel = sum(p.numel() for p in embed_params)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Core parameters: {core_numel:,}")
    print(f"  Embed parameters: {embed_numel:,}")

    # Create nested optimizer
    optimizer = DeepNestedOptimizer(
        model=model,
        base_lr=nested_config['base_lr'],
        meta_lr=nested_config['meta_lr'],
        k_unroll=nested_config['k_unroll'],
        momentum_hidden_dim=nested_config['momentum_hidden_dim'],
        controller_hidden_dim=nested_config['controller_hidden_dim'],
        mode=nested_config['mode'],
        meta_update_freq=nested_config['meta_update_freq'],
        weight_decay=nested_config['weight_decay'],
        max_grad_norm=nested_config['max_grad_norm'],
    )

    # Create loss function for meta-updates
    loss_fn = create_loss_fn(config)

    # Buffer for recent training batches (for k-step unrolling)
    k_unroll = nested_config['k_unroll']
    train_batch_buffer = deque(maxlen=k_unroll + 2)

    # Mixed precision
    scaler = GradScaler() if config.use_amp else None

    # Reset peak memory stats for accurate tracking
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # Training metrics tracking
    train_start_time = time.time()
    metrics_history = {
        'steps': [],
        'val_losses': [],
        'val_aux_losses': [],
        'val_accuracies': [],
        'val_perplexities': [],
        'elapsed_times': [],
        'learning_rates': [],
        'lr_multipliers_core': [],
        'lr_multipliers_embed': [],
        'meta_losses': [],
    }

    # Training loop
    model.train()
    step = 0
    desc = f"Training {experiment_name}" if experiment_name else "Training (Nested)"
    pbar = tqdm(total=config.max_steps, desc=desc)

    # Get a validation batch iterator for meta-updates
    val_iter = iter(val_loader)

    def get_val_batch():
        nonlocal val_iter
        try:
            return next(val_iter)
        except StopIteration:
            val_iter = iter(val_loader)
            return next(val_iter)

    while step < config.max_steps:
        for batch_idx, batch in enumerate(train_loader):
            if step >= config.max_steps:
                break

            # Handle different batch formats
            if isinstance(batch, dict):
                x = batch["input_ids"]
                y = batch["labels"]
                attention_mask = batch.get("attention_mask")
            elif isinstance(batch, (list, tuple)):
                if len(batch) == 3:
                    x, attention_mask, y = batch
                elif len(batch) == 2:
                    x, y = batch
                    attention_mask = None
                else:
                    raise ValueError(f"Unexpected batch structure with {len(batch)} elements.")
            else:
                raise TypeError(f"Unsupported batch type: {type(batch)}")

            x, y = x.to(device), y.to(device)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            # Store batch in buffer for meta-updates
            train_batch_buffer.append({'input_ids': x, 'labels': y})

            # Forward pass
            optimizer.zero_grad()

            if config.use_amp:
                with autocast('cuda', dtype=torch.float16):
                    logits, aux_loss = model(x, return_aux_loss=True)
                    shift_logits = logits[:, :-1, :].contiguous()
                    shift_labels = y[:, 1:].contiguous()
                    ce_loss = F.cross_entropy(
                        shift_logits.view(-1, config.vocab_size),
                        shift_labels.view(-1)
                    )

                    total_loss = ce_loss
                    if aux_loss is not None:
                        total_loss = total_loss + aux_loss

                    loss = total_loss / config.gradient_accumulation_steps

                scaler.scale(loss).backward()
            else:
                logits, aux_loss = model(x, return_aux_loss=True)
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = y[:, 1:].contiguous()
                ce_loss = F.cross_entropy(
                    shift_logits.view(-1, config.vocab_size),
                    shift_labels.view(-1)
                )

                total_loss = ce_loss
                if aux_loss is not None:
                    total_loss = total_loss + aux_loss

                loss = total_loss / config.gradient_accumulation_steps
                loss.backward()

            # Optimizer step
            if (step + 1) % config.gradient_accumulation_steps == 0:
                if config.use_amp:
                    scaler.unscale_(optimizer.base_optimizer)

                    # Check for inf/nan gradients before stepping (GradScaler safety)
                    found_inf = False
                    for group in optimizer.base_optimizer.param_groups:
                        for p in group['params']:
                            if p.grad is not None and (torch.isnan(p.grad).any() or torch.isinf(p.grad).any()):
                                found_inf = True
                                break
                        if found_inf:
                            break

                    if not found_inf:
                        # Step with loss value for controller
                        optimizer.step(loss_value=total_loss.item())
                    else:
                        # Skip step, just zero grads (mimics scaler.step() behavior)
                        optimizer.zero_grad()

                    scaler.update()
                else:
                    # Step with loss value for controller
                    optimizer.step(loss_value=total_loss.item())

            # Logging
            if step % 100 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == y).float().mean().item()
                    current_loss = ce_loss.item()
                    perplexity = math.exp(min(current_loss, 20))
                    lr_mults = optimizer.get_lr_multipliers()

                pbar.set_postfix({
                    'loss': f'{current_loss:.4f}',
                    'aux': f'{aux_loss.item() if aux_loss is not None else 0:.4f}',
                    'acc': f'{accuracy:.3f}',
                    'ppl': f'{perplexity:.1f}',
                    'lr_core': f'{lr_mults[0].item():.3f}',
                    'lr_embed': f'{lr_mults[1].item():.3f}',
                })

            # Evaluation and meta-update
            if step % config.eval_every == 0 and step > 0:
                # Evaluation
                eval_metrics = evaluate_model(model, val_loader, config)
                elapsed_time = (time.time() - train_start_time) / 60
                lr_mults = optimizer.get_lr_multipliers()
                effective_lrs = optimizer.get_effective_lrs()

                # Meta-update using buffered train batches and a val batch
                if nested_config['mode'] == 'explicit' and len(train_batch_buffer) >= k_unroll:
                    val_batch = get_val_batch()
                    if isinstance(val_batch, dict):
                        val_batch_dict = {
                            'input_ids': val_batch['input_ids'].to(device),
                            'labels': val_batch['labels'].to(device),
                        }
                    else:
                        vx, vy = val_batch[0], val_batch[-1]
                        val_batch_dict = {
                            'input_ids': vx.to(device),
                            'labels': vy.to(device),
                        }

                    train_batches = list(train_batch_buffer)[:k_unroll]

                    # Perform meta-update
                    # Note: use_unrolled=True is VERY memory intensive (clones all params k times)
                    # For large models, use_unrolled=False uses SimplifiedMetaTrainer instead
                    optimizer.meta_update(
                        val_batch=val_batch_dict,
                        train_batches=train_batches,
                        loss_fn=loss_fn,
                        use_unrolled=nested_config.get('use_unrolled', False),
                    )

                # Track metrics
                metrics_history['steps'].append(step)
                metrics_history['val_losses'].append(eval_metrics['val_loss'])
                metrics_history['val_aux_losses'].append(eval_metrics['val_aux_loss'])
                metrics_history['val_accuracies'].append(eval_metrics['val_accuracy'])
                metrics_history['val_perplexities'].append(eval_metrics['val_perplexity'])
                metrics_history['elapsed_times'].append(elapsed_time)
                metrics_history['learning_rates'].append(effective_lrs[0])
                metrics_history['lr_multipliers_core'].append(lr_mults[0].item())
                metrics_history['lr_multipliers_embed'].append(lr_mults[1].item())
                metrics_history['meta_losses'].append(optimizer.last_meta_loss if optimizer.last_meta_loss else 0.0)

                print(f"\nStep {step}: Val Loss: {eval_metrics['val_loss']:.4f}, "
                      f"Val Aux Loss: {eval_metrics['val_aux_loss']:.4f}, "
                      f"Val Acc: {eval_metrics['val_accuracy']:.4f}, "
                      f"Val PPL: {eval_metrics['val_perplexity']:.2f}, "
                      f"LR mult [core: {lr_mults[0].item():.3f}, embed: {lr_mults[1].item():.3f}]")

                model.train()

            step += 1
            if step % 20 == 0:
                pbar.update(20)

    pbar.close()

    # Final evaluation
    final_eval = evaluate_model(model, val_loader, config)
    elapsed_time = (time.time() - train_start_time) / 60
    lr_mults = optimizer.get_lr_multipliers()
    effective_lrs = optimizer.get_effective_lrs()

    metrics_history['steps'].append(step)
    metrics_history['val_losses'].append(final_eval['val_loss'])
    metrics_history['val_aux_losses'].append(final_eval['val_aux_loss'])
    metrics_history['val_accuracies'].append(final_eval['val_accuracy'])
    metrics_history['val_perplexities'].append(final_eval['val_perplexity'])
    metrics_history['elapsed_times'].append(elapsed_time)
    metrics_history['learning_rates'].append(effective_lrs[0])
    metrics_history['lr_multipliers_core'].append(lr_mults[0].item())
    metrics_history['lr_multipliers_embed'].append(lr_mults[1].item())
    metrics_history['meta_losses'].append(optimizer.last_meta_loss if optimizer.last_meta_loss else 0.0)

    total_time = (time.time() - train_start_time) / 60

    # Track peak VRAM usage
    if torch.cuda.is_available():
        peak_vram_gb = torch.cuda.max_memory_allocated() / 1e9
    else:
        peak_vram_gb = 0.0

    # Add to final metrics
    final_eval['peak_vram_gb'] = peak_vram_gb

    print(f"\n[Nested Optimizer] Final Results:")
    print(f"   Val Loss: {final_eval['val_loss']:.4f}")
    print(f"   Val Aux Loss: {final_eval['val_aux_loss']:.4f}")
    print(f"   Val Accuracy: {final_eval['val_accuracy']:.4f}")
    print(f"   Val Perplexity: {final_eval['val_perplexity']:.2f}")
    print(f"   Peak VRAM: {peak_vram_gb:.2f} GB")
    print(f"   Total Time: {total_time:.2f} min")
    print(f"   Final LR multipliers: [core: {lr_mults[0].item():.3f}, embed: {lr_mults[1].item():.3f}]")

    # Save outputs if directory specified
    if output_dir:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save metrics
        metrics_file = output_path / "metrics.json"
        metrics_data = {
            'final_metrics': final_eval,
            'total_time_minutes': total_time,
            'peak_vram_gb': peak_vram_gb,
            'actual_steps': step,
            'history': metrics_history,
            'nested_config': nested_config,
            'optimizer_type': 'DeepNestedOptimizer',
        }

        with open(metrics_file, 'w') as f:
            json.dump(metrics_data, f, indent=2)
        print(f"   Metrics saved to {metrics_file}")

        # Plot metrics
        plot_nested_training_metrics(metrics_history, output_path)

        # Save model checkpoint
        checkpoint_path = output_path / "model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': config,
            'nested_config': nested_config,
            'metrics': final_eval,
            'step': step,
        }, checkpoint_path)
        print(f"   Model saved to {checkpoint_path}")

    return model, final_eval, metrics_history


def plot_nested_training_metrics(metrics_history: Dict, output_path: Path):
    """Plot training metrics for nested optimizer experiment."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Nested Optimizer Training Metrics', fontsize=14, fontweight='bold')

    # Plot 1: Val Loss vs Time
    ax = axes[0, 0]
    ax.plot(metrics_history['elapsed_times'], metrics_history['val_losses'], 'b-o', linewidth=2, markersize=4)
    ax.set_xlabel('Time (minutes)')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss vs Time')
    ax.grid(True, alpha=0.3)
    if metrics_history['val_losses']:
        best_idx = metrics_history['val_losses'].index(min(metrics_history['val_losses']))
        ax.plot(metrics_history['elapsed_times'][best_idx],
                metrics_history['val_losses'][best_idx],
                'r*', markersize=15, label=f'Best: {metrics_history["val_losses"][best_idx]:.4f}')
        ax.legend()

    # Plot 2: Val Loss vs Steps
    ax = axes[0, 1]
    ax.plot(metrics_history['steps'], metrics_history['val_losses'], 'g-o', linewidth=2, markersize=4)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Validation Loss')
    ax.set_title('Validation Loss vs Steps')
    ax.grid(True, alpha=0.3)

    # Plot 3: Val Accuracy vs Steps
    ax = axes[0, 2]
    ax.plot(metrics_history['steps'], metrics_history['val_accuracies'], 'purple', linewidth=2, marker='o', markersize=4)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Validation Accuracy')
    ax.set_title('Validation Accuracy vs Steps')
    ax.grid(True, alpha=0.3)

    # Plot 4: LR Multipliers vs Steps
    ax = axes[1, 0]
    ax.plot(metrics_history['steps'], metrics_history['lr_multipliers_core'], 'b-', linewidth=2, label='Core')
    ax.plot(metrics_history['steps'], metrics_history['lr_multipliers_embed'], 'r-', linewidth=2, label='Embed')
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('LR Multiplier')
    ax.set_title('Learned LR Multipliers')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # Plot 5: Effective Learning Rate vs Steps
    ax = axes[1, 1]
    ax.plot(metrics_history['steps'], metrics_history['learning_rates'], 'orange', linewidth=2)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Effective Learning Rate (Core)')
    ax.set_title('Effective Learning Rate')
    ax.grid(True, alpha=0.3)

    # Plot 6: Meta Loss vs Steps
    ax = axes[1, 2]
    ax.plot(metrics_history['steps'], metrics_history['meta_losses'], 'cyan', linewidth=2)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Meta Loss')
    ax.set_title('Meta-Learning Loss')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = output_path / "metrics_plot.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   Plots saved to {plot_path}")


def main():
    logger = setup_logging(log_dir="./logs")
    logger.info("Starting MoE training with DeepNestedOptimizer")

    print_system_info()
    set_seed(42)

    parser = argparse.ArgumentParser(description="Train MoE Model with Nested Optimizer")
    parser.add_argument("--base_lr", type=float, default=3e-4, help="Base learning rate")
    parser.add_argument("--meta_lr", type=float, default=1e-4, help="Meta learning rate")
    parser.add_argument("--k_unroll", type=int, default=5, help="K-step unrolling for meta-update")
    parser.add_argument("--momentum_hidden_dim", type=int, default=64, help="Hidden dimension for momentum network")
    parser.add_argument("--controller_hidden_dim", type=int, default=32, help="Hidden dimension for controller network")
    parser.add_argument("--use_unrolled", action="store_true",
                        help="Use full k-step unrolled meta-learning (WARNING: very memory intensive, ~3x VRAM)")
    parser.add_argument("--steps", "--max_steps", type=int, dest="max_steps", help="Override max_steps")
    parser.add_argument("--experiment_name", type=str, default="moe_nested", help="Name of the experiment")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    args = parser.parse_args()

    # Use GPU24GB config for smaller GPU
    config = GPU24GBMoEModelConfig()

    # Override config with args
    if args.max_steps is not None:
        config.max_steps = args.max_steps

    experiment_name = args.experiment_name
    output_dir = os.path.join(args.output_dir, experiment_name)

    # Nested optimizer config
    nested_config = {
        'base_lr': args.base_lr,
        'meta_lr': args.meta_lr,
        'k_unroll': args.k_unroll,
        'use_unrolled': args.use_unrolled,  # False by default - SimplifiedMetaTrainer is much cheaper
        'momentum_hidden_dim': args.momentum_hidden_dim,
        'controller_hidden_dim': args.controller_hidden_dim,
        'mode': 'explicit',
        'meta_update_freq': config.eval_every,
        'weight_decay': config.weight_decay,
        'max_grad_norm': config.grad_clip,
    }

    print("Loading dataset with Hugging Face Datasets API...")
    data_cfg = DataConfig(
        seq_length=config.max_seq_len,
        num_samples=config.num_documents,
        cache_dir="./hf_cache",
    )

    from data.loader import setup_tokenizer

    # Setup tokenizer first to get vocab size
    tokenizer = setup_tokenizer(data_cfg)
    config.vocab_size = tokenizer.vocab_size

    # Prepare datasets (handles caching automatically)
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)

    logger.info(f"Train sequences: {len(train_ds):,}, Val sequences: {len(val_ds):,}")

    # Check for sufficient data
    total_needed = config.max_steps * config.batch_size
    if len(train_ds) < total_needed:
        msg = (
            f"Insufficient training data! "
            f"Need {total_needed} sequences (max_steps={config.max_steps} * batch_size={config.batch_size}) "
            f"but only have {len(train_ds)} sequences. "
            f"The model will overfit if data repeats. "
            f"To fix: increase num_documents (currently {config.num_documents}) "
            f"or reduce max_steps."
        )
        logger.error(msg)
        raise ValueError(msg)

    loader_args = dict(
        batch_size=config.batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    print("\nModel configuration")
    print("-" * 70)
    print(f"d_model: {config.d_model}, layers: {config.n_layers}, heads: {config.n_heads}")
    print(f"ff dim: {config.d_ff}")
    print(f"experts: {config.num_experts}, top-k: {config.expert_top_k}")
    print(f"steps: {config.max_steps}, batch size: {config.batch_size}")
    print(f"vocab size: {config.vocab_size}")
    print(f"\nNested Optimizer configuration")
    print("-" * 70)
    print(f"base_lr: {nested_config['base_lr']}, meta_lr: {nested_config['meta_lr']}")
    print(f"k_unroll: {nested_config['k_unroll']}, mode: {nested_config['mode']}")
    meta_mode = "UnrolledMetaTrainer (memory intensive)" if nested_config['use_unrolled'] else "SimplifiedMetaTrainer (memory efficient)"
    print(f"meta-learning: {meta_mode}")
    print()
    logger.info(f"Model configuration: {vars(config)}")
    logger.info(f"Nested optimizer configuration: {nested_config}")

    print("Starting training with DeepNestedOptimizer...")
    print("-" * 70)
    start = time.time()

    model, metrics, history = train_moe_nested(
        config, train_loader, val_loader,
        output_dir=output_dir,
        experiment_name=experiment_name,
        nested_config=nested_config,
    )

    elapsed = (time.time() - start) / 60
    logger.info("Training complete")

    print("\nResults")
    print("-" * 70)
    print(f"Training time: {elapsed:.2f} min")
    print(f"Val loss:       {metrics['val_loss']:.4f}")
    print(f"Val accuracy:   {metrics['val_accuracy']:.4f}")
    print(f"Val perplexity: {metrics['val_perplexity']:.2f}")
    logger.info(f"Final metrics: {metrics}")

    ckpt_path = os.path.join(output_dir, "final_model.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(
        {"model_state_dict": model.state_dict(),
         "config": config,
         "nested_config": nested_config,
         "metrics": metrics},
        ckpt_path,
    )
    print(f"Model checkpoint saved to {ckpt_path}")
    logger.info(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
