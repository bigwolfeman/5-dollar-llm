"""
Detailed profiling script for TitanMAC training with DeepNestedOptimizer.

Measures time breakdown for:
- DataLoader iteration
- Forward pass
- Backward pass
- Optimizer step (including controller, momentum MLP, etc.)
- Memory operations
"""

import argparse
import time
import os
import math
import torch
import torch.nn.functional as F
from collections import defaultdict
from pathlib import Path
from torch.utils.data import DataLoader
from typing import Dict, Any, List
import sys

# Fix tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from configs.titanmac_config import DebugTitanMACConfig
from configs.dataset_config import DataConfig
from models.titanmac_wrapper import create_titanmac_model

# Add TitanMAC path for nested optimizer imports
_titan_path = os.path.join(os.path.dirname(__file__), "111TitanMAC-Standalone")
if _titan_path not in sys.path:
    sys.path.insert(0, _titan_path)
from titans_core.opt import DeepNestedOptimizer, group_titanmac_params
from utils.helpers import set_seed


class Timer:
    """Context manager for precise GPU-synchronized timing."""

    def __init__(self, name: str, timings: Dict[str, List[float]], sync: bool = True):
        self.name = name
        self.timings = timings
        self.sync = sync

    def __enter__(self):
        if self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        if self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - self.start) * 1000  # ms
        self.timings[self.name].append(elapsed)


def profile_training(num_steps: int = 100, warmup_steps: int = 10, use_speedy: bool = True):
    """Profile training loop with detailed timing breakdown."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")

    # Setup
    set_seed(42)
    config = DebugTitanMACConfig()

    # Timings storage
    timings = defaultdict(list)

    # Load data
    data_cfg = DataConfig(
        seq_length=config.max_seq_len,
        num_samples=config.num_documents,
        cache_dir="./hf_cache",
    )

    from data.loader import setup_tokenizer
    tokenizer = setup_tokenizer(data_cfg)
    config.vocab_size = tokenizer.vocab_size

    # Use prepare_datasets from train_titanmac_nested
    from train_titanmac_nested import prepare_datasets
    train_ds, val_ds = prepare_datasets(data_cfg, tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    # Create model
    with Timer("model_creation", timings):
        model = create_titanmac_model(config)
        model = model.to(device)

    # Apply torch.compile if speedy mode
    if use_speedy:
        print("Applying torch.compile...")
        with Timer("torch_compile", timings):
            model = torch.compile(model, mode='reduce-overhead')

    # Create optimizer
    nested_config = {
        'base_lr': 3e-4,
        'meta_lr': 1e-4,
        'k_unroll': 5,
        'use_unrolled': False,
        'use_cms_updates': False,  # AdamW mode
        'momentum_hidden_dim': 64,
        'momentum_num_layers': 4,
        'controller_hidden_dim': 32,
        'controller_num_layers': 5,
        'mode': 'explicit',
        'meta_update_freq': config.eval_every,
        'weight_decay': config.weight_decay,
        'max_grad_norm': config.grad_clip,
        'use_torch_compile': False,  # Already compiled model
        'use_amp': use_speedy,
        'controller_update_freq': 10 if use_speedy else 1,
    }

    with Timer("optimizer_creation", timings):
        optimizer = DeepNestedOptimizer(
            model=model,
            base_lr=nested_config['base_lr'],
            meta_lr=nested_config['meta_lr'],
            k_unroll=nested_config['k_unroll'],
            momentum_hidden_dim=nested_config['momentum_hidden_dim'],
            momentum_num_layers=nested_config['momentum_num_layers'],
            controller_hidden_dim=nested_config['controller_hidden_dim'],
            controller_num_layers=nested_config['controller_num_layers'],
            mode=nested_config['mode'],
            meta_update_freq=nested_config['meta_update_freq'],
            weight_decay=nested_config['weight_decay'],
            max_grad_norm=nested_config['max_grad_norm'],
            use_cms_updates=False,
            controller_update_freq=nested_config['controller_update_freq'],
        )

    # AMP setup
    use_amp = nested_config.get('use_amp', False)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    print(f"AMP: {'enabled' if use_amp else 'disabled'}")

    # Training loop profiling
    print(f"\nProfiling {num_steps} steps (warmup: {warmup_steps})...")
    print("-" * 70)

    model.train()
    step = 0
    data_iter = iter(train_loader)

    # Pre-fetch first batch
    try:
        batch = next(data_iter)
    except StopIteration:
        data_iter = iter(train_loader)
        batch = next(data_iter)

    for step in range(num_steps + warmup_steps):
        is_warmup = step < warmup_steps
        prefix = "warmup_" if is_warmup else ""

        # DataLoader timing
        with Timer(f"{prefix}dataloader", timings):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(train_loader)
                batch = next(data_iter)

        # Data transfer timing
        with Timer(f"{prefix}data_transfer", timings):
            if isinstance(batch, dict):
                x = batch["input_ids"].to(device, non_blocking=True)
                y = batch["labels"].to(device, non_blocking=True)
            else:
                x, y = batch[0].to(device, non_blocking=True), batch[-1].to(device, non_blocking=True)

        # Zero grad timing
        with Timer(f"{prefix}zero_grad", timings):
            optimizer.zero_grad()

        # Forward pass timing
        with Timer(f"{prefix}forward", timings):
            if use_amp:
                with torch.amp.autocast('cuda'):
                    logits, aux_loss = model(x, return_aux_loss=True)
            else:
                logits, aux_loss = model(x, return_aux_loss=True)

        # Loss computation timing
        with Timer(f"{prefix}loss_compute", timings):
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = y[:, 1:].contiguous()
            if use_amp:
                with torch.amp.autocast('cuda'):
                    ce_loss = F.cross_entropy(
                        shift_logits.view(-1, config.vocab_size),
                        shift_labels.view(-1)
                    )
                    total_loss = ce_loss + (aux_loss if aux_loss is not None else 0)
            else:
                ce_loss = F.cross_entropy(
                    shift_logits.view(-1, config.vocab_size),
                    shift_labels.view(-1)
                )
                total_loss = ce_loss + (aux_loss if aux_loss is not None else 0)

        # Backward pass timing
        with Timer(f"{prefix}backward", timings):
            if use_amp:
                scaler.scale(total_loss).backward()
            else:
                total_loss.backward()

        # Optimizer step timing (detailed)
        if use_amp:
            with Timer(f"{prefix}scaler_unscale", timings):
                scaler.unscale_(optimizer.base_optimizer)

        with Timer(f"{prefix}optimizer_step", timings):
            optimizer.step(loss_value=total_loss.item())

        if use_amp:
            with Timer(f"{prefix}scaler_update", timings):
                scaler.update()

        # Progress
        if (step + 1) % 20 == 0:
            print(f"Step {step + 1}/{num_steps + warmup_steps}")

    # Compute and print statistics
    print("\n" + "=" * 70)
    print("PROFILING RESULTS (excluding warmup)")
    print("=" * 70)

    # Filter out warmup timings
    profile_timings = {k: v for k, v in timings.items() if not k.startswith("warmup_")}

    # Compute totals and percentages
    total_time = 0
    for name, times in profile_timings.items():
        if name not in ['model_creation', 'torch_compile', 'optimizer_creation']:
            total_time += sum(times)

    print(f"\nTotal training time: {total_time:.2f} ms ({num_steps} steps)")
    print(f"Average step time: {total_time / num_steps:.2f} ms")
    print(f"Throughput: {num_steps / (total_time / 1000):.2f} steps/sec")

    print("\n--- Time Breakdown ---")
    breakdown = []
    for name, times in sorted(profile_timings.items()):
        if name in ['model_creation', 'torch_compile', 'optimizer_creation']:
            continue  # One-time costs
        avg_time = sum(times) / len(times)
        total = sum(times)
        pct = (total / total_time) * 100 if total_time > 0 else 0
        breakdown.append((pct, name, avg_time, total))

    # Sort by percentage
    breakdown.sort(reverse=True)

    for pct, name, avg_time, total in breakdown:
        print(f"  {name:25s}: {avg_time:8.2f} ms avg | {total:8.2f} ms total | {pct:5.1f}%")

    print("\n--- One-time Costs ---")
    for name in ['model_creation', 'torch_compile', 'optimizer_creation']:
        if name in profile_timings:
            times = profile_timings[name]
            print(f"  {name:25s}: {sum(times):.2f} ms")

    # Memory analysis
    if torch.cuda.is_available():
        print("\n--- Memory Analysis ---")
        print(f"  Peak VRAM allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
        print(f"  Peak VRAM reserved: {torch.cuda.max_memory_reserved() / 1e9:.2f} GB")

    # Identify bottlenecks
    print("\n--- TOP BOTTLENECKS ---")
    for i, (pct, name, avg_time, total) in enumerate(breakdown[:5]):
        print(f"{i+1}. {name}: {pct:.1f}% of time ({avg_time:.2f} ms/step)")

    return timings, breakdown


def profile_optimizer_internals(num_steps: int = 50):
    """Profile internal optimizer operations in detail."""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(42)
    config = DebugTitanMACConfig()

    # Timings storage
    timings = defaultdict(list)

    # Setup data
    data_cfg = DataConfig(
        seq_length=config.max_seq_len,
        num_samples=config.num_documents,
        cache_dir="./hf_cache",
    )

    from data.loader import setup_tokenizer
    tokenizer = setup_tokenizer(data_cfg)
    config.vocab_size = tokenizer.vocab_size

    from train_titanmac_nested import prepare_datasets
    train_ds, _ = prepare_datasets(data_cfg, tokenizer)

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        persistent_workers=True,
    )

    # Create model (no compile for accurate profiling)
    model = create_titanmac_model(config).to(device)

    # Create optimizer
    optimizer = DeepNestedOptimizer(
        model=model,
        base_lr=3e-4,
        meta_lr=1e-4,
        k_unroll=5,
        momentum_hidden_dim=64,
        momentum_num_layers=4,
        controller_hidden_dim=32,
        controller_num_layers=5,
        mode='explicit',
        meta_update_freq=10,
        weight_decay=0.2,
        max_grad_norm=1.0,
        use_cms_updates=False,
        controller_update_freq=1,  # Every step for detailed profiling
    )

    print(f"\nProfiling optimizer internals ({num_steps} steps)...")
    print("-" * 70)

    model.train()
    data_iter = iter(train_loader)

    for step in range(num_steps):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        if isinstance(batch, dict):
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)
        else:
            x, y = batch[0].to(device), batch[-1].to(device)

        optimizer.zero_grad()

        logits, aux_loss = model(x, return_aux_loss=True)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = y[:, 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, config.vocab_size),
            shift_labels.view(-1)
        )
        total_loss = ce_loss + (aux_loss if aux_loss is not None else 0)
        total_loss.backward()

        # Profile optimizer step components manually
        torch.cuda.synchronize()

        # EMA loss update
        start = time.perf_counter()
        optimizer.ema_loss = (1 - optimizer.beta_ema) * optimizer.ema_loss + optimizer.beta_ema * total_loss.item()
        torch.cuda.synchronize()
        timings['ema_loss_update'].append((time.perf_counter() - start) * 1000)

        # Compute group stats (fused)
        start = time.perf_counter()
        stats = optimizer._compute_group_stats()
        torch.cuda.synchronize()
        timings['compute_group_stats'].append((time.perf_counter() - start) * 1000)

        # Controller forward
        start = time.perf_counter()
        with torch.no_grad():
            lr_mults = optimizer.controller(stats)
        torch.cuda.synchronize()
        timings['controller_forward'].append((time.perf_counter() - start) * 1000)

        # Gradient clipping
        start = time.perf_counter()
        from titans_core.opt.deep_nested_optimizer import _fused_clip_grad_norm_
        _fused_clip_grad_norm_(model.parameters(), optimizer.max_grad_norm)
        torch.cuda.synchronize()
        timings['gradient_clipping'].append((time.perf_counter() - start) * 1000)

        # AdamW step
        start = time.perf_counter()
        optimizer.base_optimizer.step()
        torch.cuda.synchronize()
        timings['adamw_step'].append((time.perf_counter() - start) * 1000)

        if (step + 1) % 20 == 0:
            print(f"Step {step + 1}/{num_steps}")

    # Print results
    print("\n" + "=" * 70)
    print("OPTIMIZER INTERNALS BREAKDOWN")
    print("=" * 70)

    total = sum(sum(v) for v in timings.values())

    breakdown = []
    for name, times in timings.items():
        avg = sum(times) / len(times)
        tot = sum(times)
        pct = (tot / total) * 100
        breakdown.append((pct, name, avg, tot))

    breakdown.sort(reverse=True)

    for pct, name, avg, tot in breakdown:
        print(f"  {name:25s}: {avg:8.3f} ms avg | {tot:8.2f} ms total | {pct:5.1f}%")

    return timings


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=100, help="Number of profiling steps")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup steps")
    parser.add_argument("--no-speedy", action="store_true", help="Disable speedy mode")
    parser.add_argument("--optimizer-only", action="store_true", help="Profile optimizer internals only")
    args = parser.parse_args()

    if args.optimizer_only:
        profile_optimizer_internals(num_steps=args.steps)
    else:
        profile_training(
            num_steps=args.steps,
            warmup_steps=args.warmup,
            use_speedy=not args.no_speedy,
        )
