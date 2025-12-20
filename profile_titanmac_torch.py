#!/usr/bin/env python3
"""
PyTorch native profiler for TitanMAC + DeepNestedOptimizer.

Uses torch.profiler for detailed kernel analysis without needing
elevated GPU permissions.

Usage:
    python profile_titanmac_torch.py --output-dir ./profiling_results
    tensorboard --logdir ./profiling_results  # View in TensorBoard
"""

import sys
import os
import time
import argparse
import json
from pathlib import Path
from collections import defaultdict

# Add paths
sys.path.insert(0, str(Path(__file__).parent / "111TitanMAC-Standalone"))

import torch
import torch.nn as nn
from torch.profiler import profile, record_function, ProfilerActivity

# Import TitanMAC components
from titans_core.config import TitanMACConfig
from titans_core.models.titanmac import TitanMAC
from titans_core.opt import DeepNestedOptimizer
from titans_core.opt.deep_nested_optimizer import _fused_clip_grad_norm_


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
        labels[:, :-1] = input_ids[:, 1:]
        labels[:, -1] = 0
        data.append({"input_ids": input_ids, "labels": labels})
    return data


def profile_single_step(
    model: nn.Module,
    optimizer,
    batch: dict,
    device: torch.device,
    prof: torch.profiler.profile,
):
    """Profile a single training step with detailed annotations."""
    input_ids = batch["input_ids"].to(device)
    labels = batch["labels"].to(device)

    with record_function("optimizer.zero_grad"):
        optimizer.zero_grad()

    # Forward pass
    with record_function("forward"):
        with record_function("model_forward"):
            outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs["loss"]

    # Backward pass
    with record_function("backward"):
        loss.backward()

    # Optimizer step (this is where DeepNestedOptimizer does its work)
    with record_function("optimizer_step"):
        with record_function("set_loss"):
            optimizer.set_loss(loss.item())

        with record_function("compute_group_stats"):
            stats = optimizer._compute_group_stats()

        with record_function("controller_forward"):
            with torch.no_grad():
                lr_mults = optimizer.controller(stats)

        with record_function("clip_grad_norm_fused"):
            if optimizer.max_grad_norm > 0:
                _fused_clip_grad_norm_(model.parameters(), optimizer.max_grad_norm)

        with record_function("base_optimizer_step"):
            optimizer.base_optimizer.step()

        optimizer.global_step += 1

        # Meta update if needed
        if optimizer.global_step % optimizer.meta_update_freq == 0:
            with record_function("meta_update"):
                optimizer._update_meta_components(loss.item())

    prof.step()
    return loss.item()


def analyze_profile_results(prof):
    """Analyze and summarize profiling results."""
    print("\n" + "="*80)
    print("PROFILING ANALYSIS")
    print("="*80)

    # Get key averages
    key_averages = prof.key_averages()

    # Helper to get cuda time (handle API changes)
    def get_cuda_time(e):
        if hasattr(e, 'cuda_time_total'):
            return e.cuda_time_total
        elif hasattr(e, 'device_time_total'):
            return e.device_time_total
        else:
            return e.self_cuda_time_total if hasattr(e, 'self_cuda_time_total') else 0

    # Sort by CUDA time (self time)
    sorted_by_cuda = sorted(
        key_averages,
        key=lambda x: get_cuda_time(x),
        reverse=True
    )

    # Sort by CPU time for operations without GPU
    sorted_by_cpu = sorted(
        [e for e in key_averages],
        key=lambda x: x.cpu_time_total,
        reverse=True
    )

    print("\n" + "-"*80)
    print("TOP 20 OPERATIONS BY CUDA TIME")
    print("-"*80)
    print(f"{'Operation':<50} {'CUDA Time (ms)':<15} {'Calls':<10} {'CUDA Mem (MB)':<15}")
    print("-"*80)

    for event in sorted_by_cuda[:20]:
        cuda_time_ms = get_cuda_time(event) / 1000.0
        cuda_mem_mb = getattr(event, 'cuda_memory_usage', 0) / 1e6 if getattr(event, 'cuda_memory_usage', 0) else 0
        print(f"{event.key[:50]:<50} {cuda_time_ms:<15.3f} {event.count:<10} {cuda_mem_mb:<15.3f}")

    print("\n" + "-"*80)
    print("TOP 10 OPERATIONS BY CPU TIME")
    print("-"*80)
    print(f"{'Operation':<50} {'CPU Time (ms)':<15} {'Calls':<10}")
    print("-"*80)

    for event in sorted_by_cpu[:10]:
        cpu_time_ms = event.cpu_time_total / 1000.0
        print(f"{event.key[:50]:<50} {cpu_time_ms:<15.3f} {event.count:<10}")

    # Analyze by category
    print("\n" + "-"*80)
    print("TIME BREAKDOWN BY REGION")
    print("-"*80)

    regions = ["forward", "backward", "optimizer_step", "optimizer.zero_grad",
               "meta_update", "controller_forward", "base_optimizer_step"]

    for region in regions:
        events = [e for e in key_averages if e.key == region]
        if events:
            event = events[0]
            cuda_time = get_cuda_time(event) / 1000.0
            cpu_time = event.cpu_time_total / 1000.0
            print(f"{region:<30} CUDA: {cuda_time:>10.3f}ms  CPU: {cpu_time:>10.3f}ms  Calls: {event.count}")

    # Memory analysis
    print("\n" + "-"*80)
    print("TOP MEMORY-CONSUMING OPERATIONS")
    print("-"*80)

    def get_cuda_mem(e):
        return getattr(e, 'cuda_memory_usage', 0) or 0

    sorted_by_mem = sorted(
        [e for e in key_averages if get_cuda_mem(e) > 0],
        key=lambda x: get_cuda_mem(x),
        reverse=True
    )[:10]

    for event in sorted_by_mem:
        mem_mb = get_cuda_mem(event) / 1e6
        print(f"{event.key[:50]:<50} {mem_mb:>10.2f} MB")

    # Kernel-level analysis
    print("\n" + "-"*80)
    print("KERNEL-LEVEL ANALYSIS")
    print("-"*80)

    kernel_events = [e for e in key_averages if "aten::" in e.key or "cudnn" in e.key.lower()]
    kernel_events_sorted = sorted(kernel_events, key=lambda x: get_cuda_time(x), reverse=True)

    print(f"{'Kernel':<50} {'CUDA Time (ms)':<15} {'Calls':<10} {'Avg (us)':<12}")
    print("-"*80)

    for event in kernel_events_sorted[:15]:
        cuda_time_total = get_cuda_time(event)
        cuda_time_ms = cuda_time_total / 1000.0
        avg_us = cuda_time_total / max(event.count, 1)
        print(f"{event.key[:50]:<50} {cuda_time_ms:<15.3f} {event.count:<10} {avg_us:<12.1f}")

    return key_averages


def main():
    parser = argparse.ArgumentParser(description="Profile TitanMAC + DeepNestedOptimizer")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--seq-length", type=int, default=256, help="Sequence length")
    parser.add_argument("--d-model", type=int, default=256, help="Model dimension")
    parser.add_argument("--n-heads", type=int, default=8, help="Number of attention heads")
    parser.add_argument("--n-layers", type=int, default=4, help="Number of transformer layers")
    parser.add_argument("--num-steps", type=int, default=10, help="Number of training steps")
    parser.add_argument("--warmup-steps", type=int, default=3, help="Warmup steps")
    parser.add_argument("--use-cms", action="store_true", help="Enable CMS updates in optimizer")
    parser.add_argument("--momentum-layers", type=int, default=4, help="MomentumMLP layers")
    parser.add_argument("--controller-layers", type=int, default=5, help="Controller MLP layers")
    parser.add_argument("--output-dir", type=str, default="./profiling_results", help="Output directory")
    parser.add_argument("--trace", action="store_true", help="Export chrome trace")
    parser.add_argument("--controller-update-freq", type=int, default=1, help="Update controller every N steps")
    parser.add_argument("--use-compile", action="store_true", help="Enable torch.compile for optimizer")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("ERROR: CUDA device required for profiling")
        return

    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA: {torch.version.cuda}")

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Create model config
    config = TitanMACConfig(
        vocab_size=1000,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_model * 4,
        max_seq_len=args.seq_length,
        window_size=min(64, args.seq_length // 4),
        n_persistent=8,
        use_neural_memory=True,
        n_memory_layers=2,
        titans_variant="MAG",
        dropout=0.0,
    )

    print(f"\nConfig: {config}")

    # Create model
    print("\nCreating model...")
    model = TitanMAC(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,} ({n_params/1e6:.2f}M)")

    # Create optimizer
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
        use_preprocessing=True,
        controller_update_freq=args.controller_update_freq,
        use_compile=args.use_compile,
    )

    momentum_params = sum(p.numel() for p in optimizer.momentum_mlp.parameters())
    controller_params = sum(p.numel() for p in optimizer.controller.parameters())
    print(f"MomentumMLP params: {momentum_params:,} ({args.momentum_layers} layers)")
    print(f"Controller params: {controller_params:,} ({args.controller_layers} layers)")
    print(f"Optimizer mode: {'CMS' if args.use_cms else 'AdamW'} + learned LR multipliers")

    # Memory baseline
    torch.cuda.reset_peak_memory_stats()
    initial_mem = torch.cuda.memory_allocated() / 1e9

    # Create synthetic data
    data = create_synthetic_data(
        batch_size=args.batch_size,
        seq_length=args.seq_length,
        vocab_size=1000,
        num_batches=args.num_steps + args.warmup_steps,
    )

    print(f"\nWarmup for {args.warmup_steps} steps...")
    model.train()
    for i in range(args.warmup_steps):
        batch = data[i]
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, labels=labels)
        outputs["loss"].backward()
        optimizer.set_loss(outputs["loss"].item())
        optimizer.base_optimizer.step()
        optimizer.global_step += 1
        torch.cuda.synchronize()

    print(f"\nStarting profiled training for {args.num_steps} steps...")

    # Setup profiler
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        with_flops=True,
    ) as prof:
        for step in range(args.num_steps):
            batch = data[args.warmup_steps + step]
            loss = profile_single_step(model, optimizer, batch, device, prof)
            print(f"Step {step + 1}/{args.num_steps}: loss={loss:.4f}")

    # Analyze results
    key_averages = analyze_profile_results(prof)

    # Export trace if requested
    if args.trace:
        trace_path = output_dir / "trace.json"
        prof.export_chrome_trace(str(trace_path))
        print(f"\nChrome trace exported to: {trace_path}")

    # Export table
    table_path = output_dir / "profile_table.txt"
    with open(table_path, "w") as f:
        f.write(prof.key_averages().table(sort_by="cuda_time_total", row_limit=100))
    print(f"Profile table exported to: {table_path}")

    # Memory summary
    peak_mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"\n--- Memory Summary ---")
    print(f"Initial: {initial_mem:.3f} GB")
    print(f"Peak: {peak_mem:.3f} GB")
    print(f"Growth: {peak_mem - initial_mem:.3f} GB")

    # Export JSON summary
    summary = {
        "config": {
            "d_model": args.d_model,
            "n_heads": args.n_heads,
            "n_layers": args.n_layers,
            "batch_size": args.batch_size,
            "seq_length": args.seq_length,
            "momentum_layers": args.momentum_layers,
            "controller_layers": args.controller_layers,
            "use_cms": args.use_cms,
        },
        "model_params": n_params,
        "momentum_params": momentum_params,
        "controller_params": controller_params,
        "memory": {
            "initial_gb": initial_mem,
            "peak_gb": peak_mem,
            "growth_gb": peak_mem - initial_mem,
        },
        "top_kernels": [
            {
                "name": e.key,
                "cuda_time_ms": (e.self_cuda_time_total if hasattr(e, 'self_cuda_time_total') else getattr(e, 'device_time_total', 0)) / 1000.0,
                "cpu_time_ms": e.cpu_time_total / 1000.0,
                "calls": e.count,
            }
            for e in sorted(key_averages, key=lambda x: x.self_cuda_time_total if hasattr(x, 'self_cuda_time_total') else getattr(x, 'device_time_total', 0), reverse=True)[:20]
        ],
    }

    summary_path = output_dir / "profile_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary exported to: {summary_path}")


if __name__ == "__main__":
    main()
