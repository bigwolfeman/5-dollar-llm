"""
TitanMAC Training Script for 5-dollar-llm.

This script trains the TitanMAC model (neural memory + sliding window attention)
using the same infrastructure as the MoE baseline:
    - Same SmolLM-135M tokenizer
    - Same smollm-corpus dataset
    - Same Muon+AdamW optimizer setup
    - Same evaluation metrics

Usage:
    python train_titanmac.py
    python train_titanmac.py --muon_lr 0.03 --adamw_lr 0.005
    python train_titanmac.py --max_steps 500 --experiment_name titanmac_test
"""

import argparse
import time
import os
import sys
import math
import torch
import logging
from torch.utils.data import DataLoader

# Fix tokenizer parallelism warning when using DataLoader workers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path for imports
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from configs.titanmac_config import (
    TitanMACModelConfig,
    TitanMACGPU24GBConfig,
    DebugTitanMACConfig,
)
from configs.dataset_config import DataConfig
from models.titanmac_wrapper import TitanMACWrapper
from training.trainer import train_model, setup_muon_optimizer
from utils.helpers import set_seed
from utils.logger import setup_logging

# Import prepare_datasets from train_moe to reuse data pipeline
from train_moe import prepare_datasets


def print_system_info():
    """Print system and GPU information."""
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    print(f"PyTorch: {torch.__version__}\n")


def train_titanmac_model(
    config,
    train_loader: DataLoader,
    val_loader: DataLoader,
    output_dir: str = None,
    experiment_name: str = None,
):
    """
    Train the TitanMAC model with Muon+AdamW optimizer setup.

    This mirrors train_moe_model but uses TitanMACWrapper.

    Args:
        config: TitanMAC configuration
        train_loader: Training data loader
        val_loader: Validation data loader
        output_dir: Optional output directory
        experiment_name: Optional experiment name for logging

    Returns:
        Tuple of (model, final_metrics)
    """
    variant = getattr(config, 'titans_variant', 'MAG')
    memory_enabled = getattr(config, 'use_neural_memory', True)

    print(f"\n{'='*70}")
    print(f"Training TitanMAC model")
    print(f"  Variant: {variant}")
    print(f"  Neural Memory: {'Enabled' if memory_enabled else 'Disabled'}")
    print(f"  Window Size: {config.window_size}")
    print(f"  Persistent Tokens: {config.n_persistent}")
    print(f"{'='*70}\n")

    # Initialize model
    set_seed(42)
    model = TitanMACWrapper(config)

    # Enable gradient checkpointing if configured
    if getattr(config, 'use_gradient_checkpointing', False):
        model.enable_gradient_checkpointing()
        print("  Gradient checkpointing: ENABLED")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")

    # Setup Muon+AdamW optimizers (same as MoE baseline)
    optimizers = setup_muon_optimizer(model, config)

    # Learning rate schedule with cosine decay (same as MoE baseline)
    schedulers = []
    warmup_steps = max(1, int(config.max_steps * config.warmup_ratio))

    for optimizer in optimizers:
        def lr_lambda(step):
            if step < warmup_steps:
                return step / warmup_steps
            else:
                progress = (step - warmup_steps) / (config.max_steps - warmup_steps)
                return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        schedulers.append(scheduler)

    # Use the generic training function from trainer.py
    extra_config = {
        "model_type": "TitanMAC",
        "titans_variant": variant,
        "use_neural_memory": memory_enabled,
        "window_size": config.window_size,
        "n_persistent": config.n_persistent,
        "d_model": config.d_model,
        "n_layers": config.n_layers,
        "n_heads": config.n_heads,
        "d_ff": config.d_ff,
    }

    model, final_eval, metrics_history = train_model(
        model=model,
        config=config,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizers=optimizers,
        schedulers=schedulers,
        early_stopper=None,
        output_dir=output_dir,
        experiment_name=experiment_name,
        plot_fn=None,
        extra_config=extra_config,
    )

    return model, final_eval


def main():
    """Main entry point for TitanMAC training."""
    logger = setup_logging(log_dir="./logs")
    logger.info("Starting TitanMAC training")

    print_system_info()
    set_seed(42)

    parser = argparse.ArgumentParser(description="Train TitanMAC Model")
    parser.add_argument("--muon_lr", type=float, help="Override Muon learning rate")
    parser.add_argument("--adamw_lr", type=float, help="Override AdamW learning rate")
    parser.add_argument("--steps", "--max_steps", type=int, dest="max_steps", help="Override max_steps")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="titanmac_training",
        help="Name of the experiment",
    )
    parser.add_argument(
        "--output_dir", type=str, default="./checkpoints", help="Output directory"
    )
    parser.add_argument(
        "--debug", action="store_true", help="Use debug config for quick testing"
    )
    parser.add_argument(
        "--variant",
        type=str,
        choices=["MAC", "MAG", "MAL"],
        default=None,
        help="TitanMAC variant (default: MAG)",
    )
    args = parser.parse_args()

    # Select configuration
    if args.debug:
        config = DebugTitanMACConfig()
        print("Using DEBUG configuration (tiny model for quick testing)")
    else:
        # Use GPU24GB config for 4090/similar GPUs
        config = TitanMACGPU24GBConfig()
        print("Using GPU24GB configuration")

    # Override config with args
    if args.muon_lr is not None:
        config.muon_lr = args.muon_lr
    if args.adamw_lr is not None:
        config.adamw_lr = args.adamw_lr
    if args.max_steps is not None:
        config.max_steps = args.max_steps
    if args.variant is not None:
        config.titans_variant = args.variant

    experiment_name = args.experiment_name
    output_dir = os.path.join(args.output_dir, experiment_name)

    # Setup data configuration (same as MoE baseline)
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

    # Prepare datasets (reuses caching from train_moe)
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

    # Create data loaders
    loader_args = dict(
        batch_size=config.batch_size,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=True,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_args)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_args)

    # Print configuration
    print("\nModel configuration")
    print("-" * 70)
    print(f"d_model: {config.d_model}, layers: {config.n_layers}, heads: {config.n_heads}")
    print(f"ff dim: {config.d_ff}")
    print(f"window_size: {config.window_size}, persistent tokens: {config.n_persistent}")
    print(f"neural memory: {config.use_neural_memory}, variant: {config.titans_variant}")
    print(f"steps: {config.max_steps}, batch size: {config.batch_size}")
    print(f"vocab size: {config.vocab_size}\n")
    logger.info(f"Model configuration: {vars(config)}")

    # Train model
    print("Starting training...")
    print("-" * 70)
    start = time.time()

    model, metrics = train_titanmac_model(
        config, train_loader, val_loader, output_dir=output_dir, experiment_name=experiment_name
    )

    elapsed = (time.time() - start) / 60
    logger.info("Training complete")

    # Print results
    print("\nResults")
    print("-" * 70)
    print(f"Training time: {elapsed:.2f} min")
    print(f"Val loss:       {metrics['val_loss']:.4f}")
    print(f"Val accuracy:   {metrics['val_accuracy']:.4f}")
    print(f"Val perplexity: {metrics['val_perplexity']:.2f}")
    logger.info(f"Final metrics: {metrics}")

    # Save final checkpoint
    ckpt_path = os.path.join(output_dir, "final_model.pt")
    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": config,
            "metrics": metrics,
        },
        ckpt_path,
    )
    print(f"Model checkpoint saved to {ckpt_path}")
    logger.info(f"Model saved to {ckpt_path}")


if __name__ == "__main__":
    main()
