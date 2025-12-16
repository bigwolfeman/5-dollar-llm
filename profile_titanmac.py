#!/usr/bin/env python3
"""
TitanMAC Hot Path Profiler

Measures exact execution times (ms/us) for all hot path operations
in TitanMAC training to identify optimization targets.

Usage:
    python profile_titanmac.py --steps 100 --warmup 10 --output profile_results.json
    python profile_titanmac.py --steps 10 --quick  # Quick sanity check

Output:
    - Per-operation timing breakdown (mean, std, min, max, p50, p95, p99)
    - Bottleneck ranking by cumulative time
    - JSON report for analysis
"""

import argparse
import json
import time
import os
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import statistics

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class OpTiming:
    """Timing data for a single operation invocation."""
    wall_ns: int
    cuda_ms: float


@dataclass
class OpStats:
    """Aggregated statistics for an operation."""
    name: str
    count: int = 0
    total_wall_ms: float = 0.0
    total_cuda_ms: float = 0.0
    wall_times_ms: List[float] = field(default_factory=list)
    cuda_times_ms: List[float] = field(default_factory=list)

    def add(self, wall_ns: int, cuda_ms: float):
        wall_ms = wall_ns / 1e6
        self.count += 1
        self.total_wall_ms += wall_ms
        self.total_cuda_ms += cuda_ms
        self.wall_times_ms.append(wall_ms)
        self.cuda_times_ms.append(cuda_ms)

    def compute_stats(self) -> Dict[str, Any]:
        if self.count == 0:
            return {"name": self.name, "count": 0}

        wall = self.wall_times_ms
        cuda = self.cuda_times_ms

        def percentile(data, p):
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data) - 1)]

        return {
            "name": self.name,
            "count": self.count,
            "total_wall_ms": round(self.total_wall_ms, 3),
            "total_cuda_ms": round(self.total_cuda_ms, 3),
            "wall_mean_ms": round(statistics.mean(wall), 4),
            "wall_std_ms": round(statistics.stdev(wall), 4) if len(wall) > 1 else 0,
            "wall_min_ms": round(min(wall), 4),
            "wall_max_ms": round(max(wall), 4),
            "wall_p50_ms": round(percentile(wall, 50), 4),
            "wall_p95_ms": round(percentile(wall, 95), 4),
            "wall_p99_ms": round(percentile(wall, 99), 4),
            "cuda_mean_ms": round(statistics.mean(cuda), 4),
            "cuda_std_ms": round(statistics.stdev(cuda), 4) if len(cuda) > 1 else 0,
        }


class ProfileRegistry:
    """Global registry for profiling data."""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.ops: Dict[str, OpStats] = {}
            cls._instance.enabled = True
            cls._instance.step_times_ms: List[float] = []
        return cls._instance

    @classmethod
    def reset(cls):
        instance = cls()
        instance.ops = {}
        instance.step_times_ms = []

    @classmethod
    def log(cls, name: str, wall_ns: int, cuda_ms: float):
        instance = cls()
        if not instance.enabled:
            return
        if name not in instance.ops:
            instance.ops[name] = OpStats(name=name)
        instance.ops[name].add(wall_ns, cuda_ms)

    @classmethod
    def log_step_time(cls, step_ms: float):
        instance = cls()
        instance.step_times_ms.append(step_ms)

    @classmethod
    def get_report(cls) -> Dict[str, Any]:
        instance = cls()

        ops_stats = {name: stats.compute_stats() for name, stats in instance.ops.items()}

        # Compute bottleneck ranking
        ranking = sorted(
            ops_stats.values(),
            key=lambda x: x.get("total_wall_ms", 0),
            reverse=True
        )

        total_step_time = statistics.mean(instance.step_times_ms) if instance.step_times_ms else 0

        # Add percentage of step time
        for op in ranking:
            if total_step_time > 0:
                op["pct_of_step"] = round(op.get("wall_mean_ms", 0) / total_step_time * 100, 2)
            else:
                op["pct_of_step"] = 0

        return {
            "operations": ops_stats,
            "bottleneck_ranking": [
                {"op": r["name"], "total_ms": r["total_wall_ms"], "pct": r["pct_of_step"]}
                for r in ranking[:15]  # Top 15
            ],
            "step_summary": {
                "count": len(instance.step_times_ms),
                "mean_ms": round(statistics.mean(instance.step_times_ms), 3) if instance.step_times_ms else 0,
                "std_ms": round(statistics.stdev(instance.step_times_ms), 3) if len(instance.step_times_ms) > 1 else 0,
                "throughput_it_s": round(1000 / statistics.mean(instance.step_times_ms), 3) if instance.step_times_ms else 0,
            }
        }


@contextmanager
def profile_op(name: str, sync: bool = True):
    """Context manager for profiling an operation."""
    if not ProfileRegistry().enabled:
        yield
        return

    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end_event = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None

    if start_event:
        start_event.record()
    start_ns = time.perf_counter_ns()

    yield

    if end_event:
        end_event.record()

    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()

    end_ns = time.perf_counter_ns()

    cuda_ms = start_event.elapsed_time(end_event) if start_event and end_event else 0

    ProfileRegistry.log(name, end_ns - start_ns, cuda_ms)


def patch_neural_memory_for_profiling():
    """Patch neural memory module to add profiling hooks."""
    try:
        from titans_core.memory.neural_memory import NeuralMemory

        original_retrieve = NeuralMemory.retrieve if hasattr(NeuralMemory, 'retrieve') else None
        original_update = NeuralMemory.update if hasattr(NeuralMemory, 'update') else None
        original_forward = NeuralMemory.forward

        def profiled_forward(self, *args, **kwargs):
            with profile_op("M1_neural_memory_forward"):
                return original_forward(self, *args, **kwargs)

        NeuralMemory.forward = profiled_forward

        if original_retrieve:
            def profiled_retrieve(self, *args, **kwargs):
                with profile_op("M1_memory_retrieve"):
                    return original_retrieve(self, *args, **kwargs)
            NeuralMemory.retrieve = profiled_retrieve

        if original_update:
            def profiled_update(self, *args, **kwargs):
                with profile_op("M2_memory_update"):
                    return original_update(self, *args, **kwargs)
            NeuralMemory.update = profiled_update

        print("  Patched NeuralMemory for profiling")
        return True
    except ImportError:
        print("  NeuralMemory not found, skipping patch")
        return False


def patch_titanmac_for_profiling():
    """Patch TitanMAC model for profiling."""
    try:
        from models.titanmac_llm import TitanMACMinimalLLM

        original_forward = TitanMACMinimalLLM.forward

        def profiled_forward(self, input_ids, targets=None):
            with profile_op("T1_full_forward"):
                return original_forward(self, input_ids, targets)

        TitanMACMinimalLLM.forward = profiled_forward
        print("  Patched TitanMACMinimalLLM for profiling")
        return True
    except ImportError:
        print("  TitanMACMinimalLLM not found, skipping patch")
        return False


def patch_nested_optimizer_for_profiling():
    """Patch nested optimizer for profiling."""
    try:
        from optimizers.nested_optimizer import DeepNestedOptimizer

        original_step = DeepNestedOptimizer.step
        original_meta_step = getattr(DeepNestedOptimizer, '_simplified_meta_step', None)

        def profiled_step(self, *args, **kwargs):
            with profile_op("O0_optimizer_step"):
                return original_step(self, *args, **kwargs)

        DeepNestedOptimizer.step = profiled_step

        if original_meta_step:
            def profiled_meta_step(self, *args, **kwargs):
                with profile_op("O3_meta_update"):
                    return original_meta_step(self, *args, **kwargs)
            DeepNestedOptimizer._simplified_meta_step = profiled_meta_step

        print("  Patched DeepNestedOptimizer for profiling")
        return True
    except ImportError:
        print("  DeepNestedOptimizer not found, skipping patch")
        return False


def patch_attention_for_profiling():
    """Patch attention modules for profiling."""
    patched = False

    try:
        from titans_core.attention.block_sparse_attention import BlockSparseAttention
        original_forward = BlockSparseAttention.forward

        def profiled_forward(self, *args, **kwargs):
            with profile_op("A1_block_sparse_attention"):
                return original_forward(self, *args, **kwargs)

        BlockSparseAttention.forward = profiled_forward
        print("  Patched BlockSparseAttention for profiling")
        patched = True
    except ImportError:
        pass

    try:
        from models.layers import MultiHeadAttention
        original_forward = MultiHeadAttention.forward

        def profiled_forward(self, *args, **kwargs):
            with profile_op("A2_multihead_attention"):
                return original_forward(self, *args, **kwargs)

        MultiHeadAttention.forward = profiled_forward
        print("  Patched MultiHeadAttention for profiling")
        patched = True
    except ImportError:
        pass

    return patched


def run_profiling(
    steps: int = 100,
    warmup: int = 10,
    batch_size: Optional[int] = None,
    output_file: Optional[str] = None,
):
    """Run profiling on TitanMAC training."""

    print("\n" + "=" * 60)
    print("TitanMAC Performance Profiler")
    print("=" * 60)

    # Import config and model
    print("\n[1/5] Loading configuration...")
    from configs.titanmac_config import TitanMACConfig, TitanMACGPU24GBConfig
    from configs.dataset_config import DataConfig

    config = TitanMACGPU24GBConfig()
    if batch_size:
        config.batch_size = batch_size

    print(f"  Config: {type(config).__name__}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Seq length: {config.seq_length}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    # Apply profiling patches
    print("\n[2/5] Applying profiling patches...")
    patch_titanmac_for_profiling()
    patch_neural_memory_for_profiling()
    patch_nested_optimizer_for_profiling()
    patch_attention_for_profiling()

    # Create model
    print("\n[3/5] Creating model...")
    from models.titanmac_llm import TitanMACMinimalLLM

    model = TitanMACMinimalLLM(config).to(device)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count / 1e6:.2f}M")

    # Create optimizer
    print("\n[4/5] Creating optimizer...")
    from optimizers.nested_optimizer import DeepNestedOptimizer, group_titanmac_params

    param_groups = group_titanmac_params(model, config)
    optimizer = DeepNestedOptimizer(
        param_groups,
        base_lr=config.muon_lr,
        meta_lr=1e-4,
        device=device,
        model=model,
    )

    # Create dummy data
    print("\n[5/5] Starting profiling...")
    print(f"  Warmup steps: {warmup}")
    print(f"  Profile steps: {steps}")

    ProfileRegistry.reset()

    # Warmup
    print("\n  Warming up...", end="", flush=True)
    ProfileRegistry().enabled = False

    for i in range(warmup):
        input_ids = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_length), device=device)
        targets = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_length), device=device)

        output = model(input_ids, targets)
        loss = output["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if (i + 1) % max(1, warmup // 5) == 0:
            print(".", end="", flush=True)

    print(" done")

    # Profile
    print("  Profiling", end="", flush=True)
    ProfileRegistry().enabled = True

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for i in range(steps):
        step_start = time.perf_counter_ns()

        input_ids = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_length), device=device)
        targets = torch.randint(0, config.vocab_size, (config.batch_size, config.seq_length), device=device)

        with profile_op("T1_forward_pass"):
            output = model(input_ids, targets)
            loss = output["loss"]

        with profile_op("T2_backward_pass"):
            loss.backward()

        with profile_op("T3_optimizer_step"):
            optimizer.step()

        optimizer.zero_grad()

        step_end = time.perf_counter_ns()
        step_ms = (step_end - step_start) / 1e6
        ProfileRegistry.log_step_time(step_ms)

        if (i + 1) % max(1, steps // 10) == 0:
            print(".", end="", flush=True)

    print(" done")

    # Collect results
    print("\n" + "=" * 60)
    print("PROFILING RESULTS")
    print("=" * 60)

    report = ProfileRegistry.get_report()

    # Add metadata
    report["metadata"] = {
        "model": "TitanMAC",
        "config": type(config).__name__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "parameters_M": round(param_count / 1e6, 2),
        "batch_size": config.batch_size,
        "seq_length": config.seq_length,
        "warmup_steps": warmup,
        "profile_steps": steps,
        "timestamp": datetime.now().isoformat(),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
    }

    # Print summary
    summary = report["step_summary"]
    print(f"\nStep Timing:")
    print(f"  Mean: {summary['mean_ms']:.2f} ms")
    print(f"  Std:  {summary['std_ms']:.2f} ms")
    print(f"  Throughput: {summary['throughput_it_s']:.2f} it/s")

    print(f"\nTop Bottlenecks:")
    for i, item in enumerate(report["bottleneck_ranking"][:10]):
        print(f"  {i+1}. {item['op']}: {item['total_ms']:.2f} ms total ({item['pct']:.1f}% of step)")

    print(f"\nDetailed Operation Timings:")
    for name, stats in sorted(report["operations"].items()):
        if stats["count"] > 0:
            print(f"  {name}:")
            print(f"    count={stats['count']}, mean={stats['wall_mean_ms']:.3f}ms, "
                  f"cuda={stats['cuda_mean_ms']:.3f}ms, total={stats['total_wall_ms']:.1f}ms")

    # Save report
    if output_file:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to: {output_path}")

    return report


def main():
    parser = argparse.ArgumentParser(description="Profile TitanMAC training hot paths")
    parser.add_argument("--steps", type=int, default=100, help="Number of steps to profile")
    parser.add_argument("--warmup", type=int, default=10, help="Number of warmup steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--output", "-o", type=str, default="profile_results.json", help="Output JSON file")
    parser.add_argument("--quick", action="store_true", help="Quick mode (10 steps, 2 warmup)")

    args = parser.parse_args()

    if args.quick:
        args.steps = 10
        args.warmup = 2

    run_profiling(
        steps=args.steps,
        warmup=args.warmup,
        batch_size=args.batch_size,
        output_file=args.output,
    )


if __name__ == "__main__":
    main()
