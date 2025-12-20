#!/usr/bin/env python3
"""
TitanMAC + DeepNestedOptimizer Python-Level Profiler

This script performs comprehensive CPU-side profiling to identify:
1. Python overhead in the training loop
2. Data loading bottlenecks
3. GPU idle time due to Python operations
4. Object allocation patterns
5. Optimizer-specific inefficiencies

Profiling tools used:
- cProfile: Function-level call counts and timing
- torch.profiler: PyTorch-specific profiling with CUDA events
- Manual instrumentation: Fine-grained timing of hot paths
- Memory tracking: Allocation patterns in critical sections

Best config from experiments:
- momentum_num_layers=4
- controller_num_layers=5

Usage:
    python profile_titanmac_nested_python.py --steps 50 --warmup 10
    python profile_titanmac_nested_python.py --quick  # Fast sanity check
    python profile_titanmac_nested_python.py --detailed  # Full cProfile analysis
"""

import argparse
import cProfile
import gc
import io
import json
import os
import pstats
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# =============================================================================
# TIMING INFRASTRUCTURE
# =============================================================================

@dataclass
class TimingResult:
    """Timing data for a single operation."""
    name: str
    wall_ns: int
    cuda_ms: float
    memory_delta_mb: float = 0.0
    call_count: int = 1

@dataclass
class TimingStats:
    """Aggregated statistics for an operation."""
    name: str
    call_count: int = 0
    total_wall_ns: int = 0
    total_cuda_ms: float = 0.0
    total_memory_mb: float = 0.0
    wall_times_ns: List[int] = field(default_factory=list)
    cuda_times_ms: List[float] = field(default_factory=list)

    def add(self, result: TimingResult):
        self.call_count += result.call_count
        self.total_wall_ns += result.wall_ns
        self.total_cuda_ms += result.cuda_ms
        self.total_memory_mb += result.memory_delta_mb
        self.wall_times_ns.append(result.wall_ns)
        self.cuda_times_ms.append(result.cuda_ms)

    def to_dict(self) -> Dict[str, Any]:
        if self.call_count == 0:
            return {"name": self.name, "call_count": 0}

        import statistics
        wall_ms = [ns / 1e6 for ns in self.wall_times_ns]

        def percentile(data, p):
            if not data:
                return 0
            sorted_data = sorted(data)
            idx = int(len(sorted_data) * p / 100)
            return sorted_data[min(idx, len(sorted_data) - 1)]

        return {
            "name": self.name,
            "call_count": self.call_count,
            "total_wall_ms": round(self.total_wall_ns / 1e6, 3),
            "total_cuda_ms": round(self.total_cuda_ms, 3),
            "mean_wall_ms": round(statistics.mean(wall_ms), 4) if wall_ms else 0,
            "std_wall_ms": round(statistics.stdev(wall_ms), 4) if len(wall_ms) > 1 else 0,
            "min_wall_ms": round(min(wall_ms), 4) if wall_ms else 0,
            "max_wall_ms": round(max(wall_ms), 4) if wall_ms else 0,
            "p50_wall_ms": round(percentile(wall_ms, 50), 4),
            "p95_wall_ms": round(percentile(wall_ms, 95), 4),
            "p99_wall_ms": round(percentile(wall_ms, 99), 4),
            "mean_cuda_ms": round(statistics.mean(self.cuda_times_ms), 4) if self.cuda_times_ms else 0,
            "memory_total_mb": round(self.total_memory_mb, 2),
        }


class ProfilerRegistry:
    """Singleton registry for profiling data."""
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._reset()
        return cls._instance

    def _reset(self):
        self.stats: Dict[str, TimingStats] = {}
        self.enabled = True
        self.step_times_ms: List[float] = []
        self.data_load_times_ms: List[float] = []
        self.gpu_util_samples: List[float] = []

    @classmethod
    def reset(cls):
        instance = cls()
        instance._reset()

    @classmethod
    def log(cls, result: TimingResult):
        instance = cls()
        if not instance.enabled:
            return
        if result.name not in instance.stats:
            instance.stats[result.name] = TimingStats(name=result.name)
        instance.stats[result.name].add(result)

    @classmethod
    def log_step_time(cls, step_ms: float):
        cls()._instance.step_times_ms.append(step_ms)

    @classmethod
    def log_data_load_time(cls, load_ms: float):
        cls()._instance.data_load_times_ms.append(load_ms)


@contextmanager
def profile_section(name: str, sync: bool = True, track_memory: bool = False):
    """Context manager for profiling a code section."""
    registry = ProfilerRegistry()
    if not registry.enabled:
        yield
        return

    # Memory tracking
    mem_before = 0
    if track_memory and torch.cuda.is_available():
        torch.cuda.synchronize()
        mem_before = torch.cuda.memory_allocated()

    # Sync before timing if requested
    if sync and torch.cuda.is_available():
        torch.cuda.synchronize()

    # CUDA events for GPU timing (only if sync=True to avoid blocking)
    start_event = end_event = None
    if sync and torch.cuda.is_available():
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    start_ns = time.perf_counter_ns()

    yield

    # Record CUDA event
    if end_event:
        end_event.record()

    # Sync after (always sync if we have CUDA events, otherwise optional)
    if torch.cuda.is_available() and (sync or end_event):
        torch.cuda.synchronize()

    end_ns = time.perf_counter_ns()

    # Memory delta
    mem_delta = 0
    if track_memory and torch.cuda.is_available():
        mem_delta = (torch.cuda.memory_allocated() - mem_before) / (1024 * 1024)

    # CUDA elapsed time (only valid if we sync'd)
    cuda_ms = 0
    if start_event and end_event:
        cuda_ms = start_event.elapsed_time(end_event)

    ProfilerRegistry.log(TimingResult(
        name=name,
        wall_ns=end_ns - start_ns,
        cuda_ms=cuda_ms,
        memory_delta_mb=mem_delta,
    ))


# =============================================================================
# MONKEY PATCHING FOR DETAILED PROFILING
# =============================================================================

def patch_deep_nested_optimizer():
    """Patch DeepNestedOptimizer to add detailed profiling."""
    try:
        # Add TitanMAC path
        titan_path = os.path.join(os.path.dirname(__file__), "..", "111TitanMAC-Standalone")
        if titan_path not in sys.path:
            sys.path.insert(0, os.path.abspath(titan_path))

        from titans_core.opt.deep_nested_optimizer import DeepNestedOptimizer

        original_step = DeepNestedOptimizer.step
        original_compute_group_stats = DeepNestedOptimizer._compute_group_stats
        original_get_context = DeepNestedOptimizer._get_context
        original_update_meta = DeepNestedOptimizer._update_meta_components
        original_compute_mlp_proxy = DeepNestedOptimizer._compute_mlp_proxy_loss

        def profiled_step(self, loss_value=None):
            with profile_section("OPT.step.total", sync=True):
                # Profile sub-components
                with profile_section("OPT.step.1_loss_ema_update", sync=False):
                    self.global_step += 1
                    actual_loss = loss_value if loss_value is not None else self._pending_loss
                    if actual_loss is None:
                        actual_loss = 0.0
                    self._pending_loss = None

                    if self.global_step == 1:
                        self.ema_loss.fill_(actual_loss)
                    else:
                        self.ema_loss = (1 - self.beta_ema) * self.ema_loss + self.beta_ema * actual_loss

                with profile_section("OPT.step.2_compute_group_stats", sync=False):
                    stats = self._compute_group_stats()

                with profile_section("OPT.step.3_controller_forward", sync=False):
                    with torch.no_grad():
                        self._lr_multipliers = self.controller(stats)

                with profile_section("OPT.step.4_grad_clip", sync=False):
                    if self.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

                if not self.use_cms_updates:
                    # AdamW mode
                    with profile_section("OPT.step.5_adamw_update", sync=True):
                        effective_lrs = (self.base_lr * self._lr_multipliers).tolist()
                        for i, group in enumerate(self.base_optimizer.param_groups):
                            if len(group['params']) > 0:
                                group['lr'] = effective_lrs[i]
                        self.base_optimizer.step()
                else:
                    # CMS mode - profile the heavy parts
                    with profile_section("OPT.step.5_cms_update", sync=True):
                        # Simulate the original CMS logic but with profiling
                        pass  # This branch rarely taken per your config

                with profile_section("OPT.step.6_meta_check", sync=False):
                    if self.mode == 'simple' and self.global_step % self.meta_update_freq == 0:
                        with profile_section("OPT.step.6a_meta_update", sync=True):
                            self._update_meta_components(actual_loss)

            return {
                'global_step': self.global_step,
                'lr_multipliers': self._lr_multipliers.clone(),
                'ema_loss': self.ema_loss.clone(),
            }

        def profiled_compute_group_stats(self):
            with profile_section("OPT._compute_group_stats.body", sync=False):
                return original_compute_group_stats(self)

        def profiled_update_meta(self, loss_value):
            with profile_section("OPT._update_meta_components.total", sync=True):
                with profile_section("OPT._update_meta.1_record", sync=False):
                    momentum_stats = self.get_momentum_stats()
                    self.simplified_meta_trainer.record_step(
                        loss=loss_value,
                        multipliers=self._lr_multipliers,
                        momentum_norm=momentum_stats.get('momentum_avg_norm', 0.0),
                    )

                with profile_section("OPT._update_meta.2_stats", sync=False):
                    stats = self._compute_group_stats()

                with profile_section("OPT._update_meta.3_zero_grad", sync=False):
                    self.meta_optimizer.zero_grad()

                with profile_section("OPT._update_meta.4_controller_fwd", sync=False):
                    multipliers = self.controller(stats)
                    controller_loss = self.simplified_meta_trainer.compute_proxy_loss(
                        current_multipliers=multipliers,
                        current_loss=loss_value,
                    )

                with profile_section("OPT._update_meta.5_mlp_proxy", sync=False):
                    mlp_loss = self._compute_mlp_proxy_loss(loss_value)

                with profile_section("OPT._update_meta.6_backward", sync=True):
                    meta_loss = controller_loss + mlp_loss
                    meta_loss.backward()

                with profile_section("OPT._update_meta.7_clip", sync=False):
                    torch.nn.utils.clip_grad_norm_(
                        list(self.momentum_mlp.parameters()) + list(self.controller.parameters()),
                        max_norm=1.0,
                    )

                with profile_section("OPT._update_meta.8_step", sync=False):
                    self.meta_optimizer.step()

                self.last_meta_loss = meta_loss.detach()

        def profiled_mlp_proxy(self, loss_value):
            with profile_section("OPT._compute_mlp_proxy_loss.body", sync=False):
                return original_compute_mlp_proxy(self, loss_value)

        DeepNestedOptimizer.step = profiled_step
        DeepNestedOptimizer._compute_group_stats = profiled_compute_group_stats
        DeepNestedOptimizer._update_meta_components = profiled_update_meta
        DeepNestedOptimizer._compute_mlp_proxy_loss = profiled_mlp_proxy

        print("  [OK] Patched DeepNestedOptimizer")
        return True

    except Exception as e:
        print(f"  [SKIP] DeepNestedOptimizer patch failed: {e}")
        return False


def patch_neural_memory():
    """Patch NeuralMemory for profiling."""
    try:
        titan_path = os.path.join(os.path.dirname(__file__), "..", "111TitanMAC-Standalone")
        if titan_path not in sys.path:
            sys.path.insert(0, os.path.abspath(titan_path))

        from titans_core.memory.neural_memory import NeuralMemory

        original_update = NeuralMemory.update
        original_retrieve = NeuralMemory.retrieve
        original_compute_loss = NeuralMemory.compute_loss

        def profiled_update(self, x, theta_t=None, return_stats=False):
            with profile_section("MEMORY.update.total", sync=True):
                with profile_section("MEMORY.update.1_loss", sync=False):
                    loss = self.compute_loss(x)

                with profile_section("MEMORY.update.2_autograd", sync=True):
                    grads = torch.autograd.grad(
                        loss,
                        self.memory_mlp.parameters(),
                        retain_graph=True,
                        create_graph=False,
                    )

                with profile_section("MEMORY.update.3_flatten_grads", sync=False):
                    for (offset, shape), g in zip(self._param_offsets, grads):
                        numel = g.numel()
                        self._flat_grad_cache[offset:offset + numel] = g.view(-1)

                with profile_section("MEMORY.update.4_grad_clip", sync=False):
                    grad_norm = self._flat_grad_cache.norm()
                    max_grad_norm = 1.0
                    grad_clipped = grad_norm > max_grad_norm
                    if grad_clipped:
                        self._flat_grad_cache.mul_(max_grad_norm / grad_norm)

                with profile_section("MEMORY.update.5_gates", sync=False):
                    x_pooled = x.mean(dim=1)
                    alpha_t = self.forget_gate(x_pooled).mean()
                    eta_t = self.decay_gate(x_pooled).mean()

                with profile_section("MEMORY.update.6_momentum", sync=False):
                    theta = theta_t if theta_t is not None else self.theta
                    with torch.no_grad():
                        self.momentum_S.mul_(eta_t)
                        self.momentum_S.add_(self._flat_grad_cache, alpha=-theta)

                with profile_section("MEMORY.update.7_param_update", sync=False):
                    with torch.no_grad():
                        self._get_flat_params_into_cache()
                        self._flat_param_cache.mul_(1.0 - alpha_t)
                        self._flat_param_cache.add_(self.momentum_S)
                        self._set_params_from_cache()

                self._last_update_stats = {
                    "alpha_t": alpha_t.item(),
                    "eta_t": eta_t.item(),
                    "grad_norm": grad_norm.item(),
                    "grad_clipped": grad_clipped,
                    "skipped": False,
                }

                if return_stats:
                    return {"loss": loss, **self._last_update_stats}
                return loss

        def profiled_retrieve(self, x):
            with profile_section("MEMORY.retrieve", sync=False):
                return original_retrieve(self, x)

        def profiled_compute_loss(self, x):
            with profile_section("MEMORY.compute_loss", sync=False):
                return original_compute_loss(self, x)

        NeuralMemory.update = profiled_update
        NeuralMemory.retrieve = profiled_retrieve
        NeuralMemory.compute_loss = profiled_compute_loss

        print("  [OK] Patched NeuralMemory")
        return True

    except Exception as e:
        print(f"  [SKIP] NeuralMemory patch failed: {e}")
        return False


def patch_momentum_mlp():
    """Patch momentum MLPs for profiling."""
    try:
        titan_path = os.path.join(os.path.dirname(__file__), "..", "111TitanMAC-Standalone")
        if titan_path not in sys.path:
            sys.path.insert(0, os.path.abspath(titan_path))

        from titans_core.opt.deep_nested_optimizer import L2RegressionMomentum, DirectUpdateMLP

        # Profile L2RegressionMomentum
        original_l2_forward = L2RegressionMomentum.forward

        def profiled_l2_forward(self, grad, prev_momentum, context):
            with profile_section("MLP.L2RegressionMomentum.forward", sync=False):
                return original_l2_forward(self, grad, prev_momentum, context)

        L2RegressionMomentum.forward = profiled_l2_forward

        # Profile DirectUpdateMLP
        original_du_forward = DirectUpdateMLP.forward

        def profiled_du_forward(self, grad, prev_momentum=None):
            with profile_section("MLP.DirectUpdateMLP.forward", sync=False):
                return original_du_forward(self, grad, prev_momentum)

        DirectUpdateMLP.forward = profiled_du_forward

        print("  [OK] Patched Momentum MLPs")
        return True

    except Exception as e:
        print(f"  [SKIP] Momentum MLP patch failed: {e}")
        return False


def patch_controller():
    """Patch NestedController for profiling."""
    try:
        titan_path = os.path.join(os.path.dirname(__file__), "..", "111TitanMAC-Standalone")
        if titan_path not in sys.path:
            sys.path.insert(0, os.path.abspath(titan_path))

        from titans_core.opt.nested_controller import NestedController

        original_forward = NestedController.forward

        def profiled_forward(self, stats):
            with profile_section("CONTROLLER.forward", sync=False):
                return original_forward(self, stats)

        NestedController.forward = profiled_forward

        print("  [OK] Patched NestedController")
        return True

    except Exception as e:
        print(f"  [SKIP] NestedController patch failed: {e}")
        return False


# =============================================================================
# DATA LOADING PROFILING
# =============================================================================

class ProfiledDataLoader:
    """Wrapper around DataLoader that measures iteration time."""

    def __init__(self, loader: DataLoader):
        self.loader = loader
        self._iterator = None

    def __iter__(self):
        self._iterator = iter(self.loader)
        return self

    def __next__(self):
        start = time.perf_counter_ns()
        batch = next(self._iterator)
        elapsed_ms = (time.perf_counter_ns() - start) / 1e6
        ProfilerRegistry.log_data_load_time(elapsed_ms)
        ProfilerRegistry.log(TimingResult(
            name="DATALOADER.next",
            wall_ns=int(elapsed_ms * 1e6),
            cuda_ms=0.0,
        ))
        return batch

    def __len__(self):
        return len(self.loader)


# =============================================================================
# MAIN PROFILING LOGIC
# =============================================================================

def create_dummy_dataset(config, num_samples: int = 1000):
    """Create a dummy dataset for profiling (avoids network I/O)."""
    class DummyDataset(Dataset):
        def __init__(self, vocab_size, seq_len, n):
            self.vocab_size = vocab_size
            self.seq_len = seq_len
            self.n = n
            # Pre-generate data to avoid generation overhead during profiling
            torch.manual_seed(42)
            self.input_ids = torch.randint(0, vocab_size, (n, seq_len))
            self.labels = self.input_ids.clone()
            self.attention_mask = torch.ones(n, seq_len)

        def __len__(self):
            return self.n

        def __getitem__(self, idx):
            return {
                "input_ids": self.input_ids[idx],
                "labels": self.labels[idx],
                "attention_mask": self.attention_mask[idx],
            }

    return DummyDataset(config.vocab_size, config.max_seq_len, num_samples)


def run_cprofile_analysis(
    model,
    optimizer,
    train_loader,
    config,
    device,
    steps: int = 20,
):
    """Run cProfile to get function-level call counts and timing."""
    print("\n" + "=" * 60)
    print("cProfile Analysis")
    print("=" * 60)

    profiler = cProfile.Profile()
    profiler.enable()

    for step, batch in enumerate(train_loader):
        if step >= steps:
            break

        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)

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
        optimizer.step(loss_value=total_loss.item())

    profiler.disable()

    # Extract statistics
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(50)  # Top 50 functions

    print("\nTop 50 Functions by Cumulative Time:")
    print(stream.getvalue())

    # Return for further analysis
    return stats


def run_torch_profiler(
    model,
    optimizer,
    train_loader,
    config,
    device,
    steps: int = 10,
    output_dir: str = "./profile_traces",
):
    """Run torch.profiler for detailed PyTorch analysis."""
    print("\n" + "=" * 60)
    print("torch.profiler Analysis")
    print("=" * 60)

    os.makedirs(output_dir, exist_ok=True)

    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
        schedule=torch.profiler.schedule(
            wait=1,
            warmup=2,
            active=steps,
            repeat=1,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(output_dir),
    ) as prof:
        for step, batch in enumerate(train_loader):
            if step >= steps + 3:  # wait + warmup + active
                break

            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

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
            optimizer.step(loss_value=total_loss.item())

            prof.step()

    # Print summary
    print("\nCPU Time Summary (Top 20 ops):")
    print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

    print("\nCUDA Time Summary (Top 20 ops):")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    print(f"\nTrace files saved to: {output_dir}")
    print("View with: tensorboard --logdir=" + output_dir)

    return prof


def run_instrumented_profiling(
    model,
    optimizer,
    train_loader,
    config,
    device,
    steps: int = 50,
    warmup: int = 10,
):
    """Run instrumented profiling with detailed section timing."""
    print("\n" + "=" * 60)
    print("Instrumented Profiling")
    print("=" * 60)

    ProfilerRegistry.reset()
    ProfilerRegistry().enabled = False  # Disable during warmup

    # Warmup
    print(f"\nWarming up ({warmup} steps)...", end="", flush=True)
    for step, batch in enumerate(train_loader):
        if step >= warmup:
            break

        x = batch["input_ids"].to(device)
        y = batch["labels"].to(device)

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
        optimizer.step(loss_value=total_loss.item())

        if (step + 1) % max(1, warmup // 5) == 0:
            print(".", end="", flush=True)
    print(" done")

    # Reset for profiling
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    ProfilerRegistry.reset()
    ProfilerRegistry().enabled = True

    # Profile
    print(f"Profiling ({steps} steps)...", end="", flush=True)

    # Re-create iterator for profiling
    train_iter = ProfiledDataLoader(train_loader)

    for step, batch in enumerate(train_iter):
        if step >= steps:
            break

        step_start = time.perf_counter_ns()

        with profile_section("STEP.1_batch_to_device", sync=True):
            x = batch["input_ids"].to(device)
            y = batch["labels"].to(device)

        with profile_section("STEP.2_zero_grad", sync=False):
            optimizer.zero_grad()

        with profile_section("STEP.3_forward", sync=True):
            logits, aux_loss = model(x, return_aux_loss=True)

        with profile_section("STEP.4_loss_compute", sync=False):
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = y[:, 1:].contiguous()
            ce_loss = F.cross_entropy(
                shift_logits.view(-1, config.vocab_size),
                shift_labels.view(-1)
            )
            total_loss = ce_loss + (aux_loss if aux_loss is not None else 0)

        with profile_section("STEP.5_backward", sync=True):
            total_loss.backward()

        with profile_section("STEP.6_optimizer", sync=True):
            optimizer.step(loss_value=total_loss.item())

        step_end = time.perf_counter_ns()
        step_ms = (step_end - step_start) / 1e6
        ProfilerRegistry.log_step_time(step_ms)

        if (step + 1) % max(1, steps // 10) == 0:
            print(".", end="", flush=True)

    print(" done")

    return ProfilerRegistry()


def generate_report(registry: ProfilerRegistry) -> Dict[str, Any]:
    """Generate comprehensive profiling report."""
    import statistics

    # Collect all stats
    ops = {name: stats.to_dict() for name, stats in registry.stats.items()}

    # Compute bottleneck ranking
    ranking = sorted(
        ops.values(),
        key=lambda x: x.get("total_wall_ms", 0),
        reverse=True,
    )

    # Step summary
    step_times = registry.step_times_ms
    data_times = registry.data_load_times_ms

    step_mean = statistics.mean(step_times) if step_times else 0
    data_mean = statistics.mean(data_times) if data_times else 0

    # Calculate GPU idle percentage (time not in CUDA)
    total_wall_ms = sum(s.get("total_wall_ms", 0) for s in ops.values() if "STEP" in s.get("name", ""))
    total_cuda_ms = sum(s.get("total_cuda_ms", 0) for s in ops.values() if "STEP" in s.get("name", ""))

    gpu_idle_pct = 100 * (1 - total_cuda_ms / max(total_wall_ms, 1)) if total_wall_ms > 0 else 0

    report = {
        "operations": ops,
        "bottleneck_ranking": [
            {
                "op": r["name"],
                "total_ms": r.get("total_wall_ms", 0),
                "call_count": r.get("call_count", 0),
                "mean_ms": r.get("mean_wall_ms", 0),
                "pct_of_total": round(100 * r.get("total_wall_ms", 0) / max(sum(o.get("total_wall_ms", 0) for o in ops.values()), 1), 2),
            }
            for r in ranking[:20]
        ],
        "step_summary": {
            "count": len(step_times),
            "mean_ms": round(step_mean, 3),
            "std_ms": round(statistics.stdev(step_times), 3) if len(step_times) > 1 else 0,
            "throughput_it_s": round(1000 / step_mean, 3) if step_mean > 0 else 0,
        },
        "data_loading": {
            "mean_ms": round(data_mean, 3),
            "pct_of_step": round(100 * data_mean / step_mean, 2) if step_mean > 0 else 0,
        },
        "gpu_utilization": {
            "estimated_gpu_idle_pct": round(gpu_idle_pct, 2),
        },
    }

    return report


def print_report(report: Dict[str, Any]):
    """Print formatted profiling report."""
    print("\n" + "=" * 70)
    print("PROFILING REPORT")
    print("=" * 70)

    step = report["step_summary"]
    print(f"\n[Step Timing]")
    print(f"  Steps profiled: {step['count']}")
    print(f"  Mean step time: {step['mean_ms']:.2f} ms (+/- {step['std_ms']:.2f})")
    print(f"  Throughput: {step['throughput_it_s']:.2f} it/s")

    data = report["data_loading"]
    print(f"\n[Data Loading]")
    print(f"  Mean load time: {data['mean_ms']:.2f} ms")
    print(f"  % of step time: {data['pct_of_step']:.1f}%")

    gpu = report["gpu_utilization"]
    print(f"\n[GPU Utilization]")
    print(f"  Estimated GPU idle: {gpu['estimated_gpu_idle_pct']:.1f}%")
    print(f"  (Lower is better - indicates less Python overhead)")

    print(f"\n[Top 20 Bottlenecks by Total Time]")
    print(f"{'Rank':<5} {'Operation':<45} {'Total (ms)':<12} {'Calls':<8} {'Mean (ms)':<10} {'% Total':<8}")
    print("-" * 88)

    for i, item in enumerate(report["bottleneck_ranking"][:20], 1):
        print(f"{i:<5} {item['op'][:44]:<45} {item['total_ms']:<12.2f} {item['call_count']:<8} {item['mean_ms']:<10.3f} {item['pct_of_total']:<8.1f}")

    # Categorized summary
    print(f"\n[Time by Category]")

    categories = defaultdict(float)
    for name, stats in report["operations"].items():
        if "." in name:
            category = name.split(".")[0]
        else:
            category = name
        categories[category] += stats.get("total_wall_ms", 0)

    total = sum(categories.values()) or 1
    for cat, time_ms in sorted(categories.items(), key=lambda x: -x[1]):
        pct = 100 * time_ms / total
        print(f"  {cat:<25}: {time_ms:>10.2f} ms ({pct:>5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Profile TitanMAC + DeepNestedOptimizer")
    parser.add_argument("--steps", type=int, default=50, help="Profiling steps")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    parser.add_argument("--config", type=str, default="168m", choices=["debug", "24gb", "168m"])
    parser.add_argument("--output", "-o", type=str, default="profile_nested_results.json")
    parser.add_argument("--quick", action="store_true", help="Quick mode (10 steps)")
    parser.add_argument("--detailed", action="store_true", help="Include cProfile + torch.profiler")
    parser.add_argument("--torch-trace", action="store_true", help="Generate torch profiler traces")
    parser.add_argument("--trace-dir", type=str, default="./profile_traces")

    # Best config from experiments
    parser.add_argument("--momentum-num-layers", type=int, default=4)
    parser.add_argument("--controller-num-layers", type=int, default=5)

    args = parser.parse_args()

    if args.quick:
        args.steps = 10
        args.warmup = 3

    print("\n" + "=" * 70)
    print("TitanMAC + DeepNestedOptimizer Python Profiler")
    print("=" * 70)

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    # Import and setup
    print("\n[1/5] Loading configuration...")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

    from configs.titanmac_config import TitanMAC168MConfig, TitanMACGPU24GBConfig, DebugTitanMACConfig

    config_map = {
        "debug": DebugTitanMACConfig,
        "24gb": TitanMACGPU24GBConfig,
        "168m": TitanMAC168MConfig,
    }
    config = config_map[args.config]()
    config.vocab_size = 50257  # SmolLM tokenizer

    if args.batch_size:
        config.batch_size = args.batch_size

    print(f"  Config: {type(config).__name__}")
    print(f"  d_model={config.d_model}, n_layers={config.n_layers}, n_heads={config.n_heads}")
    print(f"  batch_size={config.batch_size}, seq_len={config.max_seq_len}")

    # Apply patches
    print("\n[2/5] Applying profiling patches...")
    patch_deep_nested_optimizer()
    patch_neural_memory()
    patch_momentum_mlp()
    patch_controller()

    # Create model
    print("\n[3/5] Creating model...")
    from models.titanmac_wrapper import create_titanmac_model

    model = create_titanmac_model(config).to(device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count / 1e6:.2f}M")

    # Create optimizer with best config
    print("\n[4/5] Creating optimizer...")
    titan_path = os.path.join(os.path.dirname(__file__), "..", "111TitanMAC-Standalone")
    if titan_path not in sys.path:
        sys.path.insert(0, os.path.abspath(titan_path))

    from titans_core.opt import DeepNestedOptimizer

    nested_config = {
        'base_lr': 3e-4,
        'meta_lr': 1e-4,
        'k_unroll': 5,
        'momentum_hidden_dim': 64,
        'momentum_num_layers': args.momentum_num_layers,
        'controller_hidden_dim': 32,
        'controller_num_layers': args.controller_num_layers,
        'mode': 'explicit',
        'meta_update_freq': 50,
        'weight_decay': 0.1,
        'max_grad_norm': 1.0,
        'use_cms_updates': False,  # AdamW mode (proven to work)
    }

    optimizer = DeepNestedOptimizer(
        model=model,
        **nested_config,
    )

    print(f"  momentum_num_layers={args.momentum_num_layers}")
    print(f"  controller_num_layers={args.controller_num_layers}")
    print(f"  mode=AdamW (use_cms_updates=False)")

    # Create data
    print("\n[5/5] Creating dummy dataset...")
    dataset = create_dummy_dataset(config, num_samples=max(1000, args.steps + args.warmup + 50))
    train_loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    print(f"  Dataset size: {len(dataset)}")

    # Run cProfile if detailed mode
    if args.detailed:
        run_cprofile_analysis(model, optimizer, train_loader, config, device, steps=20)

    # Run torch.profiler if requested
    if args.torch_trace:
        run_torch_profiler(
            model, optimizer, train_loader, config, device,
            steps=10, output_dir=args.trace_dir
        )

    # Run instrumented profiling (always)
    registry = run_instrumented_profiling(
        model, optimizer, train_loader, config, device,
        steps=args.steps, warmup=args.warmup
    )

    # Generate and print report
    report = generate_report(registry)
    print_report(report)

    # Add metadata
    report["metadata"] = {
        "model": "TitanMAC",
        "config": type(config).__name__,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "parameters_M": round(param_count / 1e6, 2),
        "batch_size": config.batch_size,
        "seq_length": config.max_seq_len,
        "warmup_steps": args.warmup,
        "profile_steps": args.steps,
        "timestamp": datetime.now().isoformat(),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
        "optimizer_config": {
            "momentum_num_layers": args.momentum_num_layers,
            "controller_num_layers": args.controller_num_layers,
        },
    }

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n[Report saved to: {output_path}]")

    # Print optimization recommendations based on findings
    print("\n" + "=" * 70)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("=" * 70)

    # Analyze bottlenecks
    ranking = report["bottleneck_ranking"]
    data_pct = report["data_loading"]["pct_of_step"]
    gpu_idle = report["gpu_utilization"]["estimated_gpu_idle_pct"]

    print("\nBased on profiling data:")

    if data_pct > 10:
        print(f"\n  [!] DATA LOADING: {data_pct:.1f}% of step time")
        print("      - Increase num_workers in DataLoader")
        print("      - Enable pin_memory=True")
        print("      - Consider prefetching batches")

    if gpu_idle > 20:
        print(f"\n  [!] HIGH GPU IDLE: {gpu_idle:.1f}%")
        print("      - Python overhead causing GPU starvation")
        print("      - Consider CUDA graphs for optimizer steps")
        print("      - Reduce Python-level iterations in hot paths")

    # Check for specific bottlenecks
    for item in ranking[:10]:
        name = item["op"]
        pct = item["pct_of_total"]

        if "autograd" in name.lower() and pct > 15:
            print(f"\n  [!] AUTOGRAD OVERHEAD: {name} at {pct:.1f}%")
            print("      - torch.autograd.grad() in NeuralMemory.update() is expensive")
            print("      - Consider accumulating updates less frequently")

        if "compute_group_stats" in name.lower() and pct > 5:
            print(f"\n  [!] STATS COMPUTATION: {name} at {pct:.1f}%")
            print("      - Pre-compute static portions")
            print("      - Use fused operations where possible")

        if "meta_update" in name.lower() and pct > 10:
            print(f"\n  [!] META UPDATE OVERHEAD: {name} at {pct:.1f}%")
            print("      - Consider reducing meta_update_freq")
            print("      - Simplify proxy loss computation")


if __name__ == "__main__":
    main()
