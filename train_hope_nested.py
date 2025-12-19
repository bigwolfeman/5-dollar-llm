"""
Training script for HOPE model with DeepNestedOptimizer.

Usage:
    python train_hope_nested.py --experiment_name hope_nested

    # 168M config
    python train_hope_nested.py --config 168m --experiment_name hope_nested_168m

    # Grid search depth
    python train_hope_nested.py --momentum_num_layers 4 --controller_num_layers 4
"""

import argparse
import time
import os
import math
import json
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from collections import deque
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Dict, Any, List

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add TitanMAC path for nested optimizer imports
_titan_path = os.path.join(os.path.dirname(__file__), "111TitanMAC-Standalone")
if _titan_path not in sys.path:
    sys.path.insert(0, _titan_path)

from configs.hope_config import HOPEModelConfig, HOPEGPU24GBConfig, HOPE168MConfig, DebugHOPEConfig
from configs.dataset_config import DataConfig
from titans_core.opt import DeepNestedOptimizer
from utils.helpers import set_seed
from utils.logger import setup_logging


# Import HOPE model from train_hope
from train_hope import HOPE, count_parameters, prepare_datasets, print_system_info


def get_config(config_name: str) -> HOPEModelConfig:
    """Get config by name."""
    configs = {
        "default": HOPEGPU24GBConfig,
        "24gb": HOPEGPU24GBConfig,
        "168m": HOPE168MConfig,
        "debug": DebugHOPEConfig,
    }
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Choose from: {list(configs.keys())}")
    return configs[config_name]()


def group_hope_params(model: nn.Module):
    """Group HOPE parameters into core and embed groups."""
    core_params = []
    embed_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue

        is_embed = False
        if 'tok_emb' in name or 'pos_emb' in name:
            is_embed = True
        elif 'norm' in name.lower():
            is_embed = True
        elif param.ndim != 2:
            is_embed = True

        if is_embed:
            embed_params.append(param)
        else:
            core_params.append(param)

    return core_params, embed_params


def train_hope_nested(
    config: HOPEModelConfig,
    data_cfg: DataConfig,
    nested_config: Dict[str, Any],
    experiment_name: str = "hope_nested",
    output_dir: str = "./checkpoints",
    max_steps: Optional[int] = None,
):
    """Train HOPE model with DeepNestedOptimizer."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining {experiment_name} on {device}")

    if max_steps is not None:
        config.max_steps = max_steps

    # Setup tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(data_cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config.vocab_size = len(tokenizer)

    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(data_cfg, tokenizer, cache_dir=f"./processed_data/{experiment_name}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    # Create model
    model = HOPE(config).to(device)
    num_params = count_parameters(model)
    print(f"HOPE Parameters: {num_params / 1e6:.2f}M")

    # Create DeepNestedOptimizer
    optimizer = DeepNestedOptimizer(
        model=model,
        base_lr=nested_config['base_lr'],
        meta_lr=nested_config['meta_lr'],
        k_unroll=nested_config['k_unroll'],
        momentum_hidden_dim=nested_config['momentum_hidden_dim'],
        momentum_num_layers=nested_config['momentum_num_layers'],
        controller_hidden_dim=nested_config['controller_hidden_dim'],
        controller_num_layers=nested_config['controller_num_layers'],
        mode='simple',
        meta_update_freq=config.eval_every,
        weight_decay=config.weight_decay,
        low_memory=True,
        use_preprocessing=True,
    )

    print(f"Optimizer: DeepNestedOptimizer")
    print(f"  base_lr: {nested_config['base_lr']}")
    print(f"  meta_lr: {nested_config['meta_lr']}")
    print(f"  momentum_layers: {nested_config['momentum_num_layers']}")
    print(f"  controller_layers: {nested_config['controller_num_layers']}")

    # Training loop
    model.train()
    step = 0
    start_time = time.time()
    train_batch_buffer = deque(maxlen=nested_config['k_unroll'])

    metrics_history = {
        'steps': [],
        'val_losses': [],
        'val_accuracies': [],
        'elapsed_times': [],
        'lr_multipliers_core': [],
        'lr_multipliers_embed': [],
        'meta_losses': [],
    }

    def loss_fn(model, batch):
        """Loss function for meta-updates."""
        input_ids = batch['input_ids']
        labels = batch['labels']
        logits = model(input_ids)
        return F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=tokenizer.pad_token_id,
        )

    print(f"\nStarting training for {config.max_steps} steps...")
    pbar = tqdm(total=config.max_steps, desc="Training")

    for epoch in range(1000):
        for batch in train_loader:
            if step >= config.max_steps:
                break

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            # Buffer batch for meta-updates
            train_batch_buffer.append({
                'input_ids': input_ids,
                'labels': labels,
            })

            optimizer.zero_grad()

            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=tokenizer.pad_token_id,
            )

            loss.backward()
            optimizer.step(loss.item())

            # Logging
            if step % 100 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == labels).float().mean().item()
                    lr_mults = optimizer.get_lr_multipliers()
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{accuracy:.3f}',
                        'lr_core': f'{lr_mults[0].item():.3f}',
                        'lr_embed': f'{lr_mults[1].item():.3f}',
                    })

            # Evaluation and meta-update
            if step % config.eval_every == 0 and step > 0:
                model.eval()
                val_losses = []
                val_accs = []
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_input = val_batch['input_ids'].to(device)
                        val_labels = val_batch['labels'].to(device)
                        val_logits = model(val_input)
                        val_loss = F.cross_entropy(
                            val_logits.view(-1, val_logits.size(-1)),
                            val_labels.view(-1),
                            ignore_index=tokenizer.pad_token_id,
                        )
                        val_losses.append(val_loss.item())
                        val_preds = val_logits.argmax(dim=-1)
                        val_accs.append((val_preds == val_labels).float().mean().item())

                avg_val_loss = sum(val_losses) / len(val_losses)
                avg_val_acc = sum(val_accs) / len(val_accs)
                elapsed = (time.time() - start_time) / 60
                lr_mults = optimizer.get_lr_multipliers()

                metrics_history['steps'].append(step)
                metrics_history['val_losses'].append(avg_val_loss)
                metrics_history['val_accuracies'].append(avg_val_acc)
                metrics_history['elapsed_times'].append(elapsed)
                metrics_history['lr_multipliers_core'].append(lr_mults[0].item())
                metrics_history['lr_multipliers_embed'].append(lr_mults[1].item())
                metrics_history['meta_losses'].append(
                    optimizer.last_meta_loss.item() if hasattr(optimizer.last_meta_loss, 'item') else float(optimizer.last_meta_loss or 0)
                )

                print(f"\nStep {step}: Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}, "
                      f"LR mult [core: {lr_mults[0].item():.3f}, embed: {lr_mults[1].item():.3f}]")
                model.train()

            step += 1
            pbar.update(1)

        if step >= config.max_steps:
            break

    pbar.close()

    # Final evaluation
    model.eval()
    final_val_losses = []
    with torch.no_grad():
        for val_batch in val_loader:
            val_input = val_batch['input_ids'].to(device)
            val_labels = val_batch['labels'].to(device)
            val_logits = model(val_input)
            val_loss = F.cross_entropy(
                val_logits.view(-1, val_logits.size(-1)),
                val_labels.view(-1),
                ignore_index=tokenizer.pad_token_id,
            )
            final_val_losses.append(val_loss.item())

    final_val_loss = sum(final_val_losses) / len(final_val_losses)
    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Final Val Loss: {final_val_loss:.4f}")
    print(f"Total Time: {total_time/60:.1f} minutes")
    print(f"Parameters: {num_params/1e6:.2f}M")
    print(f"{'='*60}")

    # Save
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_path / "model.pt")
    with open(output_path / "metrics.json", "w") as f:
        json.dump({
            "final_metrics": {"val_loss": final_val_loss},
            "history": metrics_history,
            "config": {
                "d_model": config.d_model,
                "n_layers": config.n_layers,
                "n_heads": config.n_heads,
                "params_m": num_params / 1e6,
            },
            "nested_config": nested_config,
            "total_time_s": total_time,
        }, f, indent=2)

    print(f"Saved to {output_path}")
    return final_val_loss


def main():
    parser = argparse.ArgumentParser(description="Train HOPE with DeepNestedOptimizer")
    parser.add_argument("--config", type=str, default="24gb", help="Config name: 24gb, 168m, debug")
    parser.add_argument("--experiment_name", type=str, default="hope_nested", help="Experiment name")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--max_steps", "--steps", type=int, default=None, dest="max_steps", help="Max training steps")

    # Nested optimizer params
    parser.add_argument("--base_lr", type=float, default=3e-4, help="Base learning rate")
    parser.add_argument("--meta_lr", type=float, default=1e-4, help="Meta learning rate")
    parser.add_argument("--k_unroll", type=int, default=5, help="K-step unrolling")
    parser.add_argument("--momentum_hidden_dim", type=int, default=64, help="Momentum MLP hidden dim")
    parser.add_argument("--momentum_num_layers", type=int, default=2, help="Momentum MLP layers")
    parser.add_argument("--controller_hidden_dim", type=int, default=32, help="Controller MLP hidden dim")
    parser.add_argument("--controller_num_layers", type=int, default=2, help="Controller MLP layers")

    args = parser.parse_args()

    print_system_info()
    set_seed(42)

    config = get_config(args.config)
    data_cfg = DataConfig()

    nested_config = {
        'base_lr': args.base_lr,
        'meta_lr': args.meta_lr,
        'k_unroll': args.k_unroll,
        'momentum_hidden_dim': args.momentum_hidden_dim,
        'momentum_num_layers': args.momentum_num_layers,
        'controller_hidden_dim': args.controller_hidden_dim,
        'controller_num_layers': args.controller_num_layers,
    }

    train_hope_nested(
        config=config,
        data_cfg=data_cfg,
        nested_config=nested_config,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
