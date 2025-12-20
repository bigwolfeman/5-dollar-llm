#!/usr/bin/env python3
"""
Minimal profiling script for TitanMAC + DeepNestedOptimizer.

This script runs a few training steps for GPU profiling with ncu.

Usage:
    # Run with ncu profiler (kernel analysis)
    ncu --set full -o titanmac_profile python profile_titanmac_nested.py

    # Simpler - just get top kernels
    ncu --target-processes all python profile_titanmac_nested.py

    # Get memory stats
    ncu --metrics gpu__time_active.avg,l1tex__t_bytes.sum,dram__bytes.sum \
        python profile_titanmac_nested.py
"""

import sys
import os
import time
import argparse
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "111TitanMAC-Standalone"))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import TitanMAC components
from titans_core.config import TitanMACConfig
from titans_core.models.titanmac import TitanMAC
from titans_core.opt import DeepNestedOptimizer


def create_synthetic_data(
    batch_size: int = 4,
    seq_length: int = 256,
    vocab_size: int = 1000,
    num_batches: int = 10,
):
    """Create synthetic training data for profiling."""
    data = []
    for _ in range(num_batches):
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_length))
        labels = input_ids.clone()
        labels[:, :-1] = input_ids[:, 1:]  # Shift for causal LM
        labels[:, -1] = 0  # Pad last token
        data.append({"input_ids": input_ids, "labels": labels})
    return data


def profile_training_loop(
    model: nn.Module,
    optimizer,
    data: list,
    device: torch.device,
    num_steps: int = 5,
    warmup_steps: int = 2,
    use_amp: bool = True,
):
    """Run training loop with profiling annotations."""
    model.train()

    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    total_time = 0.0
    step_times = []

    print(f"Running {num_steps} training steps (+ {warmup_steps} warmup)...")

    for step, batch in enumerate(data[:num_steps + warmup_steps]):
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        # Synchronize for accurate timing
        torch.cuda.synchronize()
        start_time = time.perf_counter()

        # ==================== FORWARD PASS ====================
        optimizer.zero_grad()

        # Note: Neural memory uses torch.autograd.grad internally, which
        # conflicts with AMP's autocast. We need to handle this carefully.
        if use_amp:
            # Use autocast only for forward, but allow memory's internal grad
            with torch.amp.autocast('cuda', enabled=True):
                outputs = model(input_ids=input_ids, labels=labels)
                loss = outputs["loss"]
                # Get memory loss separately if exists
                memory_loss = outputs.get("memory_loss", None)
        else:
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs["loss"]

        # ==================== BACKWARD PASS ====================
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer.base_optimizer)
        else:
            loss.backward()

        # ==================== OPTIMIZER STEP ====================
        # This is where DeepNestedOptimizer does its work
        optimizer.set_loss(loss.item())

        if use_amp:
            # Note: DeepNestedOptimizer wraps AdamW, so we step it directly
            scaler.step(optimizer.base_optimizer)
            scaler.update()
            # Still need to run optimizer's internal logic
            optimizer.global_step += 1
            if optimizer.global_step % optimizer.meta_update_freq == 0:
                optimizer._update_meta_components(loss.item())
        else:
            optimizer.step(loss.item())

        torch.cuda.synchronize()
        step_time = time.perf_counter() - start_time

        # Skip warmup for timing statistics
        if step >= warmup_steps:
            step_times.append(step_time)
            total_time += step_time
            print(f"Step {step - warmup_steps + 1}: loss={loss.item():.4f}, time={step_time*1000:.2f}ms")
        else:
            print(f"Warmup {step + 1}: loss={loss.item():.4f}, time={step_time*1000:.2f}ms")

    if step_times:
        avg_time = sum(step_times) / len(step_times)
        print(f"\nAverage step time: {avg_time*1000:.2f}ms")
        print(f"Throughput: {1.0/avg_time:.2f} steps/sec")

    return step_times


def main():
    parser = argparse.ArgumentParser(description="Profile TitanMAC + DeepNestedOptimizer")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=256, help="Sequence length")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num-steps", type=int, default=5, help="Number of training steps")
    parser.add_argument("--warmup-steps", type=int, default=2, help="Warmup steps")
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")
    parser.add_argument("--use-cms", action="store_true", help="Enable CMS updates in optimizer")
    parser.add_argument("--momentum-layers", type=int, default=4, help="MomentumMLP layers")
    parser.add_argument("--controller-layers", type=int, default=5, help="Controller MLP layers")
    parser.add_argument("--profile-memory", action="store_true", help="Profile memory usage")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("ERROR: CUDA device required for profiling")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    # Create model config
    config = TitanMACConfig(
        vocab_size=1000,  # Small vocab for profiling
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_model * 4,
        max_seq_len=args.seq_length,
        window_size=min(64, args.seq_length // 4),
        n_persistent=8,
        use_neural_memory=True,  # Enable neural memory for full profiling
        n_memory_layers=2,
        titans_variant="MAG",  # Use MAG for simpler profiling
        dropout=0.0,
    )

    print(f"\nConfig: {config}")

    # Create model
    print("\nCreating model...")
    model = TitanMAC(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Create optimizer with specified nested architecture
    print("\nCreating DeepNestedOptimizer...")
    optimizer = DeepNestedOptimizer(
        model=model,
        base_lr=1e-4,
        meta_lr=1e-4,
        momentum_hidden_dim=64,
        momentum_num_layers=args.momentum_layers,
        controller_hidden_dim=32,
        controller_num_layers=args.controller_layers,
        mode='simple',
        meta_update_freq=50,
        weight_decay=0.01,
        max_grad_norm=1.0,
        use_cms_updates=args.use_cms,
        use_preprocessing=True,  # Enable DirectUpdateMLP
    )

    momentum_params = sum(p.numel() for p in optimizer.momentum_mlp.parameters())
    controller_params = sum(p.numel() for p in optimizer.controller.parameters())
    print(f"MomentumMLP params: {momentum_params:,}")
    print(f"Controller params: {controller_params:,}")
    print(f"Optimizer mode: {'CMS' if args.use_cms else 'AdamW'} + learned LR multipliers")

    # Memory snapshot before training
    if args.profile_memory:
        torch.cuda.reset_peak_memory_stats()
        print(f"\nInitial GPU memory: {torch.cuda.memory_allocated()/1e9:.2f} GB")

    # Create synthetic data
    print("\nCreating synthetic data...")
    data = create_synthetic_data(
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        vocab_size=1000,
        num_batches=args.num_steps + args.warmup_steps,
    )

    # CUDA profiling markers
    print("\n" + "="*60)
    print("STARTING PROFILED TRAINING LOOP")
    print("="*60)

    # Profile the training loop
    step_times = profile_training_loop(
        model=model,
        optimizer=optimizer,
        data=data,
        device=device,
        num_steps=args.num_steps,
        warmup_steps=args.warmup_steps,
        use_amp=not args.no_amp,
    )

    print("="*60)
    print("PROFILING COMPLETE")
    print("="*60)

    # Memory snapshot after training
    if args.profile_memory:
        peak_mem = torch.cuda.max_memory_allocated() / 1e9
        current_mem = torch.cuda.memory_allocated() / 1e9
        print(f"\nPeak GPU memory: {peak_mem:.2f} GB")
        print(f"Current GPU memory: {current_mem:.2f} GB")

    # Print optimizer state for debugging
    print("\n--- Optimizer State ---")
    lr_mults = optimizer.get_lr_multipliers()
    print(f"LR multipliers: {lr_mults.tolist()}")
    print(f"Global step: {optimizer.global_step}")

    momentum_stats = optimizer.get_momentum_stats()
    print(f"Momentum norm: {momentum_stats['momentum_total_norm']:.4f}")

    # Memory stats from neural memory
    if model.neural_memory is not None:
        mem_stats = model.get_neural_memory_stats()
        if mem_stats:
            print(f"\n--- Neural Memory Stats ---")
            print(f"Memory param norm: {mem_stats.get('memory_param_norm', 'N/A'):.4f}")
            print(f"Momentum norm: {mem_stats.get('momentum_norm', 'N/A'):.4f}")
            print(f"Alpha (forget): {mem_stats.get('alpha_t', 'N/A'):.4f}")
            print(f"Eta (decay): {mem_stats.get('eta_t', 'N/A'):.4f}")


if __name__ == "__main__":
    main()
