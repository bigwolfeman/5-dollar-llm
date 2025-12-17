#!/usr/bin/env python3
"""
Depth-focused hyperparameter fuzzing for Nested Optimizer.

Research indicates depth > width for learned optimizer components.
This script focuses on fuzzing:
- momentum_num_layers: Depth of learned momentum network
- controller_num_layers: Depth of LR controller network

Uses best LR params from previous fuzzing runs as baseline.

Usage:
    # Fuzz MoE + Nested depth
    python fuzz_depth.py --experiment moe_nested --n_trials 25 --steps 1200

    # Fuzz TitanMAC + Nested depth
    python fuzz_depth.py --experiment titanmac_nested --n_trials 25 --steps 1200

    # Run both
    python fuzz_depth.py --experiment both --n_trials 25 --steps 1200

    # Analyze existing results and generate charts
    python fuzz_depth.py --analyze fuzz_results_depth/moe_nested_depth
"""

import argparse
import gc
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import optuna
from optuna.trial import Trial
from optuna.samplers import GridSampler, TPESampler

# Best params from previous fuzzing (LR sweep)
BEST_MOE_NESTED_PARAMS = {
    "base_lr": 0.0004603431029542576,
    "meta_lr": 9.56441656770269e-05,
    "k_unroll": 1,
    "momentum_hidden_dim": 128,
    "controller_hidden_dim": 32,
}

BEST_TITANMAC_NESTED_PARAMS = {
    "base_lr": 0.0004072398672148361,
    "meta_lr": 9.089757036214503e-05,
    "k_unroll": 5,
    "momentum_hidden_dim": 64,
    "controller_hidden_dim": 16,
}


@dataclass
class ExperimentConfig:
    name: str
    script: str
    description: str
    base_params: Dict[str, Any]  # Fixed params from previous fuzzing
    depth_params: Dict[str, Any]  # Depth params to fuzz


# Grid definitions for different search intensities
FULL_GRID = {
    "momentum_num_layers": {"type": "int", "low": 1, "high": 12},
    "controller_num_layers": {"type": "int", "low": 1, "high": 12},
}  # 12x12 = 144 configs

PRUNED_GRID = {
    "momentum_num_layers": {"type": "categorical", "choices": [1, 2, 4, 6, 8, 10]},
    "controller_num_layers": {"type": "categorical", "choices": [1, 2, 4, 6, 8, 10]},
}  # 6x6 = 36 configs

QUICK_GRID = {
    "momentum_num_layers": {"type": "categorical", "choices": [1, 4, 8, 12]},
    "controller_num_layers": {"type": "categorical", "choices": [1, 4, 8, 12]},
}  # 4x4 = 16 configs

EXPERIMENTS = {
    # Full 12x12 grid (144 configs) - overnight runs
    "moe_nested": ExperimentConfig(
        name="moe_nested_depth",
        script="train_moe_nested.py",
        description="MoE + Nested Optimizer (12x12 full grid)",
        base_params=BEST_MOE_NESTED_PARAMS,
        depth_params=FULL_GRID,
    ),
    "titanmac_nested": ExperimentConfig(
        name="titanmac_nested_depth",
        script="train_titanmac_nested.py",
        description="TitanMAC + Nested Optimizer (12x12 full grid)",
        base_params=BEST_TITANMAC_NESTED_PARAMS,
        depth_params=FULL_GRID,
    ),
    # Pruned 6x6 grid (36 configs) - ~3 hours
    "moe_pruned": ExperimentConfig(
        name="moe_nested_pruned",
        script="train_moe_nested.py",
        description="MoE + Nested Optimizer (6x6 pruned grid)",
        base_params=BEST_MOE_NESTED_PARAMS,
        depth_params=PRUNED_GRID,
    ),
    "titanmac_pruned": ExperimentConfig(
        name="titanmac_nested_pruned",
        script="train_titanmac_nested.py",
        description="TitanMAC + Nested Optimizer (6x6 pruned grid)",
        base_params=BEST_TITANMAC_NESTED_PARAMS,
        depth_params=PRUNED_GRID,
    ),
    # Quick 4x4 grid (16 configs) - ~40 min
    "moe_quick": ExperimentConfig(
        name="moe_nested_quick",
        script="train_moe_nested.py",
        description="MoE + Nested Optimizer (4x4 quick grid)",
        base_params=BEST_MOE_NESTED_PARAMS,
        depth_params=QUICK_GRID,
    ),
    "titanmac_quick": ExperimentConfig(
        name="titanmac_nested_quick",
        script="train_titanmac_nested.py",
        description="TitanMAC + Nested Optimizer (4x4 quick grid)",
        base_params=BEST_TITANMAC_NESTED_PARAMS,
        depth_params=QUICK_GRID,
    ),
}


def clear_gpu_memory(wait_time: float = 2.0):
    """Aggressively clear GPU memory between trials."""
    try:
        import torch
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
            time.sleep(wait_time)
            gc.collect()
            torch.cuda.empty_cache()
    except Exception as e:
        print(f"Warning: Could not clear GPU memory: {e}")


def sample_params(trial: Trial, config: ExperimentConfig) -> Dict[str, Any]:
    """Sample depth parameters for a trial."""
    params = {}

    for name, spec in config.depth_params.items():
        if spec["type"] == "int":
            params[name] = trial.suggest_int(name, spec["low"], spec["high"])
        elif spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])

    return params


def run_training(
    script: str,
    base_params: Dict[str, Any],
    depth_params: Dict[str, Any],
    max_steps: int,
    experiment_name: str,
    trial_number: int,
    output_base_dir: str = "./checkpoints",
) -> Dict[str, Any]:
    """Run a single training trial and return metrics including full trajectory."""

    trial_name = f"{experiment_name}_trial_{trial_number}"
    metrics_file = Path(output_base_dir) / trial_name / "metrics.json"

    # Build command
    cmd = [sys.executable, script]

    # Add base params (fixed from previous fuzzing)
    for key, value in base_params.items():
        cmd.extend([f"--{key}", str(value)])

    # Add depth params (being fuzzed)
    for key, value in depth_params.items():
        cmd.extend([f"--{key}", str(value)])

    # Add training config
    cmd.extend([
        "--steps", str(max_steps),
        "--experiment_name", trial_name,
        "--output_dir", output_base_dir,
    ])

    print(f"\n{'='*60}")
    print(f"Trial {trial_number}: {depth_params}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'='*60}\n")

    start_time = time.time()

    try:
        # Use Popen for live output streaming
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output and collect it
        output_lines = []
        for line in process.stdout:
            print(line, end='')  # Live output
            output_lines.append(line)

        process.wait(timeout=3600)
        elapsed = time.time() - start_time
        full_output = ''.join(output_lines)

        # Parse final metrics from output
        metrics = parse_metrics(full_output)
        metrics["elapsed_seconds"] = elapsed
        metrics["return_code"] = process.returncode

        # Load full trajectory from metrics.json if it exists
        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    saved_metrics = json.load(f)

                # Extract trajectory data
                history = saved_metrics.get("history", {})
                metrics["trajectory"] = {
                    "steps": history.get("steps", []),
                    "val_losses": history.get("val_losses", []),
                    "val_accuracies": history.get("val_accuracies", []),
                    "val_perplexities": history.get("val_perplexities", []),
                    "meta_losses": history.get("meta_losses", []),
                    "lr_multipliers_core": history.get("lr_multipliers_core", []),
                    "lr_multipliers_embed": history.get("lr_multipliers_embed", []),
                }
                print(f"  Loaded trajectory with {len(metrics['trajectory']['steps'])} datapoints")
            except Exception as e:
                print(f"  Warning: Could not load trajectory from {metrics_file}: {e}")
                metrics["trajectory"] = None
        else:
            print(f"  Warning: No metrics file found at {metrics_file}")
            metrics["trajectory"] = None

        if process.returncode != 0:
            print(f"Trial failed with return code {process.returncode}")
            metrics["val_loss"] = float("inf")

        return metrics

    except subprocess.TimeoutExpired:
        print("Trial timed out!")
        process.kill()
        return {"val_loss": float("inf"), "timeout": True, "trajectory": None}
    except Exception as e:
        print(f"Trial error: {e}")
        return {"val_loss": float("inf"), "error": str(e), "trajectory": None}


def parse_metrics(output: str) -> Dict[str, Any]:
    """Parse training output for metrics."""
    metrics = {}

    for line in output.split('\n'):
        line_lower = line.lower()

        # Look for val loss
        if 'val loss:' in line_lower or 'val_loss:' in line_lower:
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    val = parts[-1].strip()
                    # Handle "nan" string
                    if val.lower() == 'nan':
                        metrics['val_loss'] = float('inf')
                    else:
                        metrics['val_loss'] = float(val)
            except (ValueError, IndexError):
                pass

        # Look for val accuracy
        if 'val accuracy:' in line_lower or 'val_accuracy:' in line_lower:
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    metrics['val_accuracy'] = float(parts[-1].strip())
            except (ValueError, IndexError):
                pass

        # Look for val perplexity
        if 'val perplexity:' in line_lower or 'val_perplexity:' in line_lower:
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    metrics['val_perplexity'] = float(parts[-1].strip())
            except (ValueError, IndexError):
                pass

        # Look for peak VRAM
        if 'peak vram' in line_lower or 'peak_vram' in line_lower:
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    val_str = parts[-1].strip().replace('GB', '').strip()
                    metrics['peak_vram_gb'] = float(val_str)
            except (ValueError, IndexError):
                pass

    return metrics


def create_objective(
    config: ExperimentConfig,
    max_steps: int,
    results_dir: Path,
    skip_configs: set = None,
    existing_data: dict = None,
) -> callable:
    """Create Optuna objective function."""

    all_results = []
    skip_configs = skip_configs or set()
    existing_data = existing_data or {}

    def objective(trial: Trial) -> float:
        try:
            return _objective_impl(trial)
        except Exception as e:
            print(f"\n❌ Trial {trial.number} failed with exception: {e}")
            print("   Continuing with next trial...")
            return float("inf")

    def _objective_impl(trial: Trial) -> float:
        # Sample depth params
        depth_params = sample_params(trial, config)

        # Check if this config should be skipped
        config_key = (depth_params["momentum_num_layers"], depth_params["controller_num_layers"])
        if config_key in skip_configs:
            # Return the existing result without running
            existing = existing_data.get(config_key, {})
            loss = existing.get("loss", float("inf"))
            print(f"\nSkipping existing config M={config_key[0]}, C={config_key[1]} -> loss={loss:.4f}")

            # Store in results
            result = {
                "trial": trial.number,
                "depth_params": depth_params,
                "base_params": config.base_params,
                "metrics": {
                    "val_loss": loss,
                    "val_accuracy": existing.get("accuracy"),
                    "source": "existing",
                },
                "timestamp": datetime.now().isoformat(),
            }
            all_results.append(result)

            # Save intermediate results
            with open(results_dir / "all_trials.json", "w") as f:
                json.dump(all_results, f, indent=2, default=str)

            return loss

        # Clear GPU memory before each trial
        clear_gpu_memory(wait_time=3.0)

        # Run training
        metrics = run_training(
            script=config.script,
            base_params=config.base_params,
            depth_params=depth_params,
            max_steps=max_steps,
            experiment_name=config.name,
            trial_number=trial.number,
        )

        # Store result
        result = {
            "trial": trial.number,
            "depth_params": depth_params,
            "base_params": config.base_params,
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        all_results.append(result)

        # Save intermediate results
        with open(results_dir / "all_trials.json", "w") as f:
            json.dump(all_results, f, indent=2, default=str)

        val_loss = metrics.get("val_loss", float("inf"))

        # Check for NaN/Inf
        if val_loss != val_loss or val_loss == float("inf"):  # NaN check
            print(f"Trial {trial.number}: Invalid loss, returning inf")
            return float("inf")

        print(f"\nTrial {trial.number} Result: val_loss={val_loss:.4f}")
        if "val_accuracy" in metrics:
            print(f"  val_accuracy={metrics['val_accuracy']:.4f}")
        if "peak_vram_gb" in metrics:
            print(f"  peak_vram={metrics['peak_vram_gb']:.2f} GB")
        print(f"  depth_params={depth_params}")

        # Clear memory after trial
        clear_gpu_memory(wait_time=2.0)

        return val_loss

    return objective, all_results


def load_existing_configs(results_dirs: List[str], target_step: int = 600) -> Dict[Tuple[int, int], Dict]:
    """
    Load existing trial data and extract loss at target_step.

    Returns dict mapping (momentum_layers, controller_layers) -> {loss, accuracy, source}
    """
    existing = {}

    for results_dir in results_dirs:
        trials_file = Path(results_dir) / "all_trials.json"
        if not trials_file.exists():
            continue

        with open(trials_file) as f:
            trials = json.load(f)

        for trial in trials:
            m = trial["depth_params"]["momentum_num_layers"]
            c = trial["depth_params"]["controller_num_layers"]
            key = (m, c)

            # Try to get loss at target_step from trajectory
            traj = trial.get("metrics", {}).get("trajectory", {})
            steps = traj.get("steps", [])
            losses = traj.get("val_losses", [])
            accuracies = traj.get("val_accuracies", [])

            loss_at_step = None
            acc_at_step = None

            if steps and losses:
                # Find closest step to target
                for i, s in enumerate(steps):
                    if s >= target_step:
                        loss_at_step = losses[i]
                        if accuracies and i < len(accuracies):
                            acc_at_step = accuracies[i]
                        break
                # If we didn't find it, use final value
                if loss_at_step is None:
                    loss_at_step = losses[-1]
                    if accuracies:
                        acc_at_step = accuracies[-1]
            else:
                # Fall back to final loss
                loss_at_step = trial.get("metrics", {}).get("val_loss")
                acc_at_step = trial.get("metrics", {}).get("val_accuracy")

            if loss_at_step is not None:
                # Only keep best result for each config
                if key not in existing or loss_at_step < existing[key]["loss"]:
                    existing[key] = {
                        "loss": loss_at_step,
                        "accuracy": acc_at_step,
                        "source": str(results_dir),
                        "original_trial": trial.get("trial"),
                    }

    return existing


def run_study(
    experiment: str,
    n_trials: int,
    max_steps: int,
    results_base_dir: str = "fuzz_results_depth",
    resume: bool = False,
    use_grid: bool = False,
    skip_existing_dirs: List[str] = None,
):
    """Run depth fuzzing study for an experiment."""

    if experiment not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {experiment}. Choose from: {list(EXPERIMENTS.keys())}")

    config = EXPERIMENTS[experiment]

    # Create results directory
    results_dir = Path(results_base_dir) / config.name
    results_dir.mkdir(parents=True, exist_ok=True)

    storage = f"sqlite:///{results_dir}/optuna_study.db"

    # Load existing configs to skip
    skip_configs = set()
    existing_data = {}
    if skip_existing_dirs:
        existing_data = load_existing_configs(skip_existing_dirs, target_step=max_steps)
        skip_configs = set(existing_data.keys())
        print(f"\nLoaded {len(skip_configs)} existing configs to skip")
        print(f"  Sources: {skip_existing_dirs}")

    # Choose sampler
    if use_grid:
        # Build grid from depth params
        search_space = {}
        for name, spec in config.depth_params.items():
            if spec["type"] == "int":
                search_space[name] = list(range(spec["low"], spec["high"] + 1))
            elif spec["type"] == "categorical":
                search_space[name] = spec["choices"]

        total_configs = 1
        for v in search_space.values():
            total_configs *= len(v)

        # Filter out existing configs from search space for reporting
        new_configs = total_configs - len(skip_configs)

        sampler = GridSampler(search_space)
        print(f"\nUsing GRID search: {total_configs} total configurations")
        print(f"  Search space: {search_space}")
        print(f"  Skipping {len(skip_configs)} existing configs")
        print(f"  New configs to run: {new_configs}")
        # Override n_trials to cover full grid
        n_trials = total_configs
    else:
        sampler = TPESampler(seed=42)
        print(f"\nUsing BAYESIAN (TPE) search")

    # Handle resume vs new study
    if resume:
        # Find existing study in the database
        try:
            study_summaries = optuna.study.get_all_study_summaries(storage=storage)
            if not study_summaries:
                print(f"No existing studies found in {storage}. Starting fresh.")
                resume = False
            else:
                # Get the most recent study
                latest_study = max(study_summaries, key=lambda s: s.datetime_start or datetime.min)
                study_name = latest_study.study_name
                completed_trials = latest_study.n_trials
                print(f"\n{'#'*60}")
                print(f"# RESUMING Depth Fuzzing: {config.name}")
                print(f"# Existing study: {study_name}")
                print(f"# Completed trials: {completed_trials}/{n_trials}")
                print(f"# Remaining trials: {max(0, n_trials - completed_trials)}")
                print(f"{'#'*60}\n")
        except Exception as e:
            print(f"Could not load existing study: {e}. Starting fresh.")
            resume = False

    if not resume:
        study_name = f"{config.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        print(f"\n{'#'*60}")
        print(f"# Depth Fuzzing: {config.name}")
        print(f"# Description: {config.description}")
        print(f"# Trials: {n_trials}")
        print(f"# Steps per trial: {max_steps}")
        print(f"# Results: {results_dir}")
        print(f"{'#'*60}\n")

    print("Base params (fixed from LR sweep):")
    for k, v in config.base_params.items():
        print(f"  {k}: {v}")
    print("\nDepth params (being fuzzed):")
    for k, v in config.depth_params.items():
        print(f"  {k}: {v}")

    # Create/load study
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        direction="minimize",
        sampler=sampler,
        load_if_exists=True,
    )

    # Load existing all_trials.json if resuming
    all_results = []
    trials_file = results_dir / "all_trials.json"
    if resume and trials_file.exists():
        with open(trials_file) as f:
            all_results = json.load(f)
        print(f"Loaded {len(all_results)} existing trial results")

    # Create objective with existing results and skip configs
    objective_fn, all_results_ref = create_objective(
        config, max_steps, results_dir,
        skip_configs=skip_configs,
        existing_data=existing_data,
    )
    # Replace the empty list with loaded results from resume
    all_results_ref.extend(all_results)

    # Calculate remaining trials
    completed = len(study.trials)
    remaining = max(0, n_trials - completed)

    if remaining == 0:
        print(f"\nAll {n_trials} trials already completed!")
    else:
        print(f"\nRunning {remaining} remaining trials...")
        # Run optimization for remaining trials only
        study.optimize(objective_fn, n_trials=remaining, show_progress_bar=True)

    # Save final summary
    summary = {
        "experiment": config.name,
        "description": config.description,
        "base_params": config.base_params,
        "best_trial": study.best_trial.number,
        "best_value": study.best_value,
        "best_depth_params": study.best_params,
        "n_trials": n_trials,
        "max_steps": max_steps,
        "timestamp": datetime.now().isoformat(),
    }

    with open(results_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("DEPTH FUZZING COMPLETE")
    print(f"{'='*60}")
    print(f"Best trial: #{study.best_trial.number}")
    print(f"Best val_loss: {study.best_value:.4f}")
    print(f"Best depth params:")
    for k, v in study.best_params.items():
        print(f"  {k}: {v}")
    print(f"\nResults saved to: {results_dir}")

    # Generate analysis
    analyze_results(results_dir)

    return study


def reconstruct_trajectories(results_dir: Path, checkpoint_base: str = "./checkpoints"):
    """Reconstruct trajectory data from checkpoint metrics.json files."""
    results_dir = Path(results_dir)
    trials_file = results_dir / "all_trials.json"

    if not trials_file.exists():
        print(f"No trials file found at {trials_file}")
        return

    with open(trials_file) as f:
        trials = json.load(f)

    print(f"Reconstructing trajectories for {len(trials)} trials...")
    checkpoint_base = Path(checkpoint_base)

    reconstructed = 0
    for trial in trials:
        trial_num = trial.get("trial", 0)
        # Try to find the metrics.json for this trial
        # Pattern: moe_nested_depth_trial_0, moe_nested_depth_trial_1, etc.
        experiment_name = results_dir.name  # e.g., "moe_nested_depth"
        trial_dir = checkpoint_base / f"{experiment_name}_trial_{trial_num}"
        metrics_file = trial_dir / "metrics.json"

        if metrics_file.exists():
            try:
                with open(metrics_file) as f:
                    saved_metrics = json.load(f)

                history = saved_metrics.get("history", {})
                trial["metrics"]["trajectory"] = {
                    "steps": history.get("steps", []),
                    "val_losses": history.get("val_losses", []),
                    "val_accuracies": history.get("val_accuracies", []),
                    "val_perplexities": history.get("val_perplexities", []),
                    "meta_losses": history.get("meta_losses", []),
                    "lr_multipliers_core": history.get("lr_multipliers_core", []),
                    "lr_multipliers_embed": history.get("lr_multipliers_embed", []),
                }
                reconstructed += 1
                print(f"  Trial {trial_num}: Loaded {len(history.get('steps', []))} datapoints")
            except Exception as e:
                print(f"  Trial {trial_num}: Failed to load - {e}")
        else:
            print(f"  Trial {trial_num}: No metrics file at {metrics_file}")

    # Save updated trials
    with open(trials_file, "w") as f:
        json.dump(trials, f, indent=2, default=str)

    print(f"\nReconstructed {reconstructed}/{len(trials)} trajectories")
    print(f"Updated {trials_file}")
    return trials


# Colorblind-friendly palette (IBM Design Library)
COLORBLIND_COLORS = [
    '#648FFF',  # Blue
    '#785EF0',  # Purple
    '#DC267F',  # Magenta
    '#FE6100',  # Orange
    '#FFB000',  # Gold
    '#009E73',  # Teal
]

# Distinct markers for each depth level
MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h']

# Line styles for additional differentiation
LINE_STYLES = ['-', '--', '-.', ':']


def get_style_for_depth(depth: int, max_depth: int = 6) -> Tuple[str, str, str]:
    """Get colorblind-friendly color, marker, and linestyle for a depth value."""
    idx = (depth - 1) % len(COLORBLIND_COLORS)
    color = COLORBLIND_COLORS[idx]
    marker = MARKERS[idx % len(MARKERS)]
    linestyle = LINE_STYLES[idx % len(LINE_STYLES)]
    return color, marker, linestyle


def plot_trajectory_overlays(trials: List[Dict], results_dir: Path):
    """Generate trajectory overlay charts with colorblind-friendly styling."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        from matplotlib.colors import Normalize
        from matplotlib.cm import ScalarMappable
    except ImportError:
        print("matplotlib not available - skipping trajectory charts")
        return

    # Extract trials with valid trajectories
    valid_trials = []
    for trial in trials:
        traj = trial.get("metrics", {}).get("trajectory")
        if traj and traj.get("steps") and traj.get("val_losses"):
            valid_trials.append(trial)

    if not valid_trials:
        print("No valid trajectories found for overlay charts")
        return

    print(f"\nGenerating trajectory overlays for {len(valid_trials)} trials...")

    # Create figure with multiple panels
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # Get unique depths for legend
    momentum_depths = [t["depth_params"]["momentum_num_layers"] for t in valid_trials]
    controller_depths = [t["depth_params"]["controller_num_layers"] for t in valid_trials]
    total_depths = [m + c for m, c in zip(momentum_depths, controller_depths)]

    unique_m = sorted(set(momentum_depths))
    unique_c = sorted(set(controller_depths))
    unique_t = sorted(set(total_depths))

    # Panel 1: Loss trajectories by momentum depth (colorblind-friendly)
    ax1 = axes[0, 0]
    legend_handles_m = {}
    for trial in valid_trials:
        traj = trial["metrics"]["trajectory"]
        m_depth = trial["depth_params"]["momentum_num_layers"]
        color, marker, ls = get_style_for_depth(m_depth)
        steps = traj["steps"]
        losses = traj["val_losses"]
        # Add markers every N points to avoid clutter
        markevery = max(1, len(steps) // 8)
        line, = ax1.plot(steps, losses, alpha=0.7, color=color, linewidth=1.5,
                        linestyle=ls, marker=marker, markersize=5, markevery=markevery)
        if m_depth not in legend_handles_m:
            legend_handles_m[m_depth] = line
    ax1.set_xlabel("Step")
    ax1.set_ylabel("Val Loss")
    ax1.set_title("Loss Trajectories by Momentum Depth")
    ax1.grid(True, alpha=0.3)
    ax1.legend([legend_handles_m[d] for d in sorted(legend_handles_m.keys())],
               [f"depth={d}" for d in sorted(legend_handles_m.keys())],
               loc='upper right', fontsize=8)

    # Panel 2: Loss trajectories by controller depth
    ax2 = axes[0, 1]
    legend_handles_c = {}
    for trial in valid_trials:
        traj = trial["metrics"]["trajectory"]
        c_depth = trial["depth_params"]["controller_num_layers"]
        color, marker, ls = get_style_for_depth(c_depth)
        steps = traj["steps"]
        losses = traj["val_losses"]
        markevery = max(1, len(steps) // 8)
        line, = ax2.plot(steps, losses, alpha=0.7, color=color, linewidth=1.5,
                        linestyle=ls, marker=marker, markersize=5, markevery=markevery)
        if c_depth not in legend_handles_c:
            legend_handles_c[c_depth] = line
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Val Loss")
    ax2.set_title("Loss Trajectories by Controller Depth")
    ax2.grid(True, alpha=0.3)
    ax2.legend([legend_handles_c[d] for d in sorted(legend_handles_c.keys())],
               [f"depth={d}" for d in sorted(legend_handles_c.keys())],
               loc='upper right', fontsize=8)

    # Panel 3: Loss trajectories by total depth
    ax3 = axes[0, 2]
    legend_handles_t = {}
    for trial in valid_trials:
        traj = trial["metrics"]["trajectory"]
        t_depth = trial["depth_params"]["momentum_num_layers"] + trial["depth_params"]["controller_num_layers"]
        color, marker, ls = get_style_for_depth(t_depth)
        steps = traj["steps"]
        losses = traj["val_losses"]
        markevery = max(1, len(steps) // 8)
        line, = ax3.plot(steps, losses, alpha=0.7, color=color, linewidth=1.5,
                        linestyle=ls, marker=marker, markersize=5, markevery=markevery)
        if t_depth not in legend_handles_t:
            legend_handles_t[t_depth] = line
    ax3.set_xlabel("Step")
    ax3.set_ylabel("Val Loss")
    ax3.set_title("Loss Trajectories by Total Depth")
    ax3.grid(True, alpha=0.3)
    ax3.legend([legend_handles_t[d] for d in sorted(legend_handles_t.keys())],
               [f"total={d}" for d in sorted(legend_handles_t.keys())],
               loc='upper right', fontsize=8)

    # Panel 4: Convergence rate by momentum depth (slope of loss curve)
    ax4 = axes[1, 0]
    convergence_data = []
    for trial in valid_trials:
        traj = trial["metrics"]["trajectory"]
        steps = np.array(traj["steps"])
        losses = np.array(traj["val_losses"])
        # Filter out nan/inf
        valid_mask = np.isfinite(losses)
        if np.sum(valid_mask) >= 2:
            steps_clean = steps[valid_mask]
            losses_clean = losses[valid_mask]
            # Compute slope (negative = improving)
            if len(steps_clean) >= 2:
                slope = (losses_clean[-1] - losses_clean[0]) / (steps_clean[-1] - steps_clean[0])
                convergence_data.append({
                    "momentum_depth": trial["depth_params"]["momentum_num_layers"],
                    "controller_depth": trial["depth_params"]["controller_num_layers"],
                    "slope": slope,
                    "final_loss": losses_clean[-1],
                    "stability": np.std(losses_clean) if len(losses_clean) > 1 else 0,
                })

    if convergence_data:
        # Scatter with markers by depth
        for d in convergence_data:
            m_depth = d["momentum_depth"]
            color, marker, _ = get_style_for_depth(m_depth)
            ax4.scatter(m_depth, d["slope"], alpha=0.7, c=color, marker=marker, s=80, edgecolors='black', linewidth=0.5)
        ax4.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='No improvement')
        ax4.set_xlabel("Momentum Num Layers")
        ax4.set_ylabel("Convergence Rate (slope)")
        ax4.set_title("Convergence Rate by Momentum Depth\n(more negative = faster convergence)")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    # Panel 5: Stability (std of loss) by controller depth
    ax5 = axes[1, 1]
    if convergence_data:
        for d in convergence_data:
            c_depth = d["controller_depth"]
            color, marker, _ = get_style_for_depth(c_depth)
            ax5.scatter(c_depth, d["stability"], alpha=0.7, c=color, marker=marker, s=80, edgecolors='black', linewidth=0.5)
        ax5.set_xlabel("Controller Num Layers")
        ax5.set_ylabel("Loss Variance (std)")
        ax5.set_title("Training Stability by Controller Depth\n(lower = more stable)")
        ax5.grid(True, alpha=0.3)

    # Panel 6: Meta-loss trajectories (if available)
    ax6 = axes[1, 2]
    has_meta = False
    legend_handles_meta = {}
    for trial in valid_trials:
        traj = trial["metrics"]["trajectory"]
        meta_losses = traj.get("meta_losses", [])
        if meta_losses and len(meta_losses) > 0 and any(m != 0 for m in meta_losses):
            has_meta = True
            t_depth = trial["depth_params"]["momentum_num_layers"] + trial["depth_params"]["controller_num_layers"]
            color, marker, ls = get_style_for_depth(t_depth)
            steps = traj["steps"][:len(meta_losses)]
            markevery = max(1, len(steps) // 8)
            line, = ax6.plot(steps, meta_losses, alpha=0.7, color=color, linewidth=1.5,
                            linestyle=ls, marker=marker, markersize=5, markevery=markevery)
            if t_depth not in legend_handles_meta:
                legend_handles_meta[t_depth] = line

    if has_meta:
        ax6.set_xlabel("Step")
        ax6.set_ylabel("Meta Loss")
        ax6.set_title("Meta-Loss Trajectories by Total Depth")
        ax6.grid(True, alpha=0.3)
        if legend_handles_meta:
            ax6.legend([legend_handles_meta[d] for d in sorted(legend_handles_meta.keys())],
                       [f"total={d}" for d in sorted(legend_handles_meta.keys())],
                       loc='upper right', fontsize=8)
    else:
        ax6.text(0.5, 0.5, "No meta-loss data available",
                 ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title("Meta-Loss (not recorded)")

    plt.tight_layout()
    chart_path = results_dir / "trajectory_overlays.png"
    plt.savefig(chart_path, dpi=150)
    print(f"Trajectory overlay chart saved to: {chart_path}")
    plt.close()

    # Also generate per-depth aggregated curves (mean ± std)
    plot_aggregated_trajectories(valid_trials, results_dir)

    return convergence_data


def plot_aggregated_trajectories(trials: List[Dict], results_dir: Path):
    """Generate aggregated trajectory curves showing mean ± std by depth (colorblind-friendly)."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Group by momentum depth
    momentum_groups = {}
    for trial in trials:
        m_depth = trial["depth_params"]["momentum_num_layers"]
        if m_depth not in momentum_groups:
            momentum_groups[m_depth] = []
        momentum_groups[m_depth].append(trial)

    # Find common step range
    all_steps = set()
    for trial in trials:
        all_steps.update(trial["metrics"]["trajectory"]["steps"])
    common_steps = sorted(all_steps)

    # Plot aggregated by momentum depth with colorblind-friendly styling
    ax1 = axes[0]
    m_depths = sorted(momentum_groups.keys())

    for m_depth in m_depths:
        group = momentum_groups[m_depth]
        # Interpolate all trajectories to common steps
        interpolated = []
        for trial in group:
            traj = trial["metrics"]["trajectory"]
            steps = np.array(traj["steps"])
            losses = np.array(traj["val_losses"])
            # Simple interpolation
            interp_losses = np.interp(common_steps, steps, losses)
            interpolated.append(interp_losses)

        if interpolated:
            interpolated = np.array(interpolated)
            mean_loss = np.nanmean(interpolated, axis=0)
            std_loss = np.nanstd(interpolated, axis=0)

            color, marker, ls = get_style_for_depth(m_depth)
            markevery = max(1, len(common_steps) // 10)
            ax1.plot(common_steps, mean_loss, color=color, linewidth=2.5, linestyle=ls,
                    marker=marker, markersize=6, markevery=markevery, label=f"depth={m_depth}")
            ax1.fill_between(common_steps, mean_loss - std_loss, mean_loss + std_loss,
                            alpha=0.2, color=color)

    ax1.set_xlabel("Step")
    ax1.set_ylabel("Val Loss")
    ax1.set_title("Aggregated Loss by Momentum Depth (mean ± std)")
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Group by controller depth
    controller_groups = {}
    for trial in trials:
        c_depth = trial["depth_params"]["controller_num_layers"]
        if c_depth not in controller_groups:
            controller_groups[c_depth] = []
        controller_groups[c_depth].append(trial)

    ax2 = axes[1]
    c_depths = sorted(controller_groups.keys())

    for c_depth in c_depths:
        group = controller_groups[c_depth]
        interpolated = []
        for trial in group:
            traj = trial["metrics"]["trajectory"]
            steps = np.array(traj["steps"])
            losses = np.array(traj["val_losses"])
            interp_losses = np.interp(common_steps, steps, losses)
            interpolated.append(interp_losses)

        if interpolated:
            interpolated = np.array(interpolated)
            mean_loss = np.nanmean(interpolated, axis=0)
            std_loss = np.nanstd(interpolated, axis=0)

            color, marker, ls = get_style_for_depth(c_depth)
            markevery = max(1, len(common_steps) // 10)
            ax2.plot(common_steps, mean_loss, color=color, linewidth=2.5, linestyle=ls,
                    marker=marker, markersize=6, markevery=markevery, label=f"depth={c_depth}")
            ax2.fill_between(common_steps, mean_loss - std_loss, mean_loss + std_loss,
                            alpha=0.2, color=color)

    ax2.set_xlabel("Step")
    ax2.set_ylabel("Val Loss")
    ax2.set_title("Aggregated Loss by Controller Depth (mean ± std)")
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = results_dir / "aggregated_trajectories.png"
    plt.savefig(chart_path, dpi=150)
    print(f"Aggregated trajectory chart saved to: {chart_path}")
    plt.close()


def analyze_results(results_dir: Path):
    """Analyze results and generate charts with trend analysis."""
    results_dir = Path(results_dir)
    trials_file = results_dir / "all_trials.json"

    if not trials_file.exists():
        print(f"No trials file found at {trials_file}")
        return

    with open(trials_file) as f:
        trials = json.load(f)

    if not trials:
        print("No trial data found")
        return

    # Generate trajectory overlay charts first
    convergence_data = plot_trajectory_overlays(trials, results_dir)

    # Extract data for final-value analysis
    momentum_depths = []
    controller_depths = []
    val_losses = []

    for trial in trials:
        depth_params = trial.get("depth_params", {})
        metrics = trial.get("metrics", {})

        m_depth = depth_params.get("momentum_num_layers")
        c_depth = depth_params.get("controller_num_layers")
        loss = metrics.get("val_loss")

        if m_depth is not None and c_depth is not None and loss is not None:
            if loss != float("inf") and loss == loss:  # Filter inf and NaN
                momentum_depths.append(m_depth)
                controller_depths.append(c_depth)
                val_losses.append(loss)

    if not val_losses:
        print("No valid trial results to analyze")
        return

    # Try to import matplotlib
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        has_matplotlib = True
    except ImportError:
        print("matplotlib not available - generating text analysis only")
        has_matplotlib = False

    # Text analysis (always generated)
    print(f"\n{'='*60}")
    print("DEPTH TREND ANALYSIS")
    print(f"{'='*60}")
    print(f"Valid trials analyzed: {len(val_losses)}")

    # Analyze momentum depth trend
    print("\n--- Momentum Depth Analysis ---")
    momentum_stats = {}
    for m, loss in zip(momentum_depths, val_losses):
        if m not in momentum_stats:
            momentum_stats[m] = []
        momentum_stats[m].append(loss)

    print(f"{'Depth':<8} {'Avg Loss':<12} {'Min Loss':<12} {'Count':<8}")
    print("-" * 40)
    for depth in sorted(momentum_stats.keys()):
        losses = momentum_stats[depth]
        avg = sum(losses) / len(losses)
        min_loss = min(losses)
        print(f"{depth:<8} {avg:<12.4f} {min_loss:<12.4f} {len(losses):<8}")

    # Check trend
    depths_sorted = sorted(momentum_stats.keys())
    if len(depths_sorted) >= 2:
        min_losses = [min(momentum_stats[d]) for d in depths_sorted]
        if min_losses[-1] < min_losses[0]:
            trend = "IMPROVING"
            if depths_sorted[-1] == max(EXPERIMENTS["moe_nested"].depth_params["momentum_num_layers"]["high"],
                                        EXPERIMENTS["titanmac_nested"].depth_params["momentum_num_layers"]["high"]):
                recommendation = "⚠️  Best at max depth - EXPAND SEARCH SPACE"
            else:
                recommendation = "Trend positive but not at boundary"
        else:
            trend = "NOT IMPROVING"
            recommendation = "Deeper may not help for momentum"
        print(f"\nMomentum depth trend: {trend}")
        print(f"Recommendation: {recommendation}")

    # Analyze controller depth trend
    print("\n--- Controller Depth Analysis ---")
    controller_stats = {}
    for c, loss in zip(controller_depths, val_losses):
        if c not in controller_stats:
            controller_stats[c] = []
        controller_stats[c].append(loss)

    print(f"{'Depth':<8} {'Avg Loss':<12} {'Min Loss':<12} {'Count':<8}")
    print("-" * 40)
    for depth in sorted(controller_stats.keys()):
        losses = controller_stats[depth]
        avg = sum(losses) / len(losses)
        min_loss = min(losses)
        print(f"{depth:<8} {avg:<12.4f} {min_loss:<12.4f} {len(losses):<8}")

    # Check trend
    depths_sorted = sorted(controller_stats.keys())
    if len(depths_sorted) >= 2:
        min_losses = [min(controller_stats[d]) for d in depths_sorted]
        if min_losses[-1] < min_losses[0]:
            trend = "IMPROVING"
            if depths_sorted[-1] == max(EXPERIMENTS["moe_nested"].depth_params["controller_num_layers"]["high"],
                                        EXPERIMENTS["titanmac_nested"].depth_params["controller_num_layers"]["high"]):
                recommendation = "⚠️  Best at max depth - EXPAND SEARCH SPACE"
            else:
                recommendation = "Trend positive but not at boundary"
        else:
            trend = "NOT IMPROVING"
            recommendation = "Deeper may not help for controller"
        print(f"\nController depth trend: {trend}")
        print(f"Recommendation: {recommendation}")

    # Best overall
    best_idx = val_losses.index(min(val_losses))
    print(f"\n--- Best Configuration ---")
    print(f"momentum_num_layers: {momentum_depths[best_idx]}")
    print(f"controller_num_layers: {controller_depths[best_idx]}")
    print(f"val_loss: {val_losses[best_idx]:.4f}")

    # Generate charts if matplotlib available
    if has_matplotlib:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Momentum depth vs loss
        ax1 = axes[0, 0]
        ax1.scatter(momentum_depths, val_losses, alpha=0.6, c='blue')
        # Add trend line
        if len(set(momentum_depths)) > 1:
            z = np.polyfit(momentum_depths, val_losses, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(momentum_depths), max(momentum_depths), 100)
            ax1.plot(x_line, p(x_line), "r--", alpha=0.8, label=f"trend (slope={z[0]:.4f})")
            ax1.legend()
        ax1.set_xlabel("Momentum Num Layers")
        ax1.set_ylabel("Val Loss")
        ax1.set_title("Momentum Depth vs Loss")
        ax1.grid(True, alpha=0.3)

        # 2. Controller depth vs loss
        ax2 = axes[0, 1]
        ax2.scatter(controller_depths, val_losses, alpha=0.6, c='green')
        if len(set(controller_depths)) > 1:
            z = np.polyfit(controller_depths, val_losses, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(controller_depths), max(controller_depths), 100)
            ax2.plot(x_line, p(x_line), "r--", alpha=0.8, label=f"trend (slope={z[0]:.4f})")
            ax2.legend()
        ax2.set_xlabel("Controller Num Layers")
        ax2.set_ylabel("Val Loss")
        ax2.set_title("Controller Depth vs Loss")
        ax2.grid(True, alpha=0.3)

        # 3. Heatmap of depth combinations
        ax3 = axes[1, 0]
        # Create grid
        unique_m = sorted(set(momentum_depths))
        unique_c = sorted(set(controller_depths))
        grid = np.full((len(unique_c), len(unique_m)), np.nan)
        for m, c, loss in zip(momentum_depths, controller_depths, val_losses):
            mi = unique_m.index(m)
            ci = unique_c.index(c)
            if np.isnan(grid[ci, mi]) or loss < grid[ci, mi]:
                grid[ci, mi] = loss

        im = ax3.imshow(grid, cmap='viridis_r', aspect='auto')
        ax3.set_xticks(range(len(unique_m)))
        ax3.set_xticklabels(unique_m)
        ax3.set_yticks(range(len(unique_c)))
        ax3.set_yticklabels(unique_c)
        ax3.set_xlabel("Momentum Num Layers")
        ax3.set_ylabel("Controller Num Layers")
        ax3.set_title("Best Loss by Depth Combination")
        plt.colorbar(im, ax=ax3, label="Val Loss")

        # 4. Total depth (sum) vs loss
        ax4 = axes[1, 1]
        total_depths = [m + c for m, c in zip(momentum_depths, controller_depths)]
        ax4.scatter(total_depths, val_losses, alpha=0.6, c='purple')
        if len(set(total_depths)) > 1:
            z = np.polyfit(total_depths, val_losses, 1)
            p = np.poly1d(z)
            x_line = np.linspace(min(total_depths), max(total_depths), 100)
            ax4.plot(x_line, p(x_line), "r--", alpha=0.8, label=f"trend (slope={z[0]:.4f})")
            ax4.legend()
        ax4.set_xlabel("Total Depth (Momentum + Controller)")
        ax4.set_ylabel("Val Loss")
        ax4.set_title("Total Depth vs Loss")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        chart_path = results_dir / "depth_analysis.png"
        plt.savefig(chart_path, dpi=150)
        print(f"\nChart saved to: {chart_path}")
        plt.close()

    # Save analysis to text file
    analysis_path = results_dir / "depth_analysis.txt"
    with open(analysis_path, "w") as f:
        f.write("DEPTH TREND ANALYSIS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Valid trials: {len(val_losses)}\n\n")

        f.write("Momentum Depth Stats (Final Loss):\n")
        for depth in sorted(momentum_stats.keys()):
            losses = momentum_stats[depth]
            f.write(f"  Depth {depth}: avg={sum(losses)/len(losses):.4f}, min={min(losses):.4f}, n={len(losses)}\n")

        f.write("\nController Depth Stats (Final Loss):\n")
        for depth in sorted(controller_stats.keys()):
            losses = controller_stats[depth]
            f.write(f"  Depth {depth}: avg={sum(losses)/len(losses):.4f}, min={min(losses):.4f}, n={len(losses)}\n")

        f.write(f"\nBest: momentum={momentum_depths[best_idx]}, controller={controller_depths[best_idx]}, loss={val_losses[best_idx]:.4f}\n")

        # Add convergence analysis if available
        if convergence_data:
            f.write("\n" + "=" * 60 + "\n")
            f.write("CONVERGENCE ANALYSIS (from trajectories)\n")
            f.write("=" * 60 + "\n\n")

            # Group convergence by depth
            m_conv = {}
            c_conv = {}
            for d in convergence_data:
                m = d["momentum_depth"]
                c = d["controller_depth"]
                if m not in m_conv:
                    m_conv[m] = {"slopes": [], "stabilities": []}
                if c not in c_conv:
                    c_conv[c] = {"slopes": [], "stabilities": []}
                m_conv[m]["slopes"].append(d["slope"])
                m_conv[m]["stabilities"].append(d["stability"])
                c_conv[c]["slopes"].append(d["slope"])
                c_conv[c]["stabilities"].append(d["stability"])

            f.write("Momentum Depth Convergence:\n")
            f.write(f"{'Depth':<8} {'Avg Slope':<12} {'Avg Stability':<14} {'n':<6}\n")
            f.write("-" * 44 + "\n")
            for depth in sorted(m_conv.keys()):
                data = m_conv[depth]
                avg_slope = sum(data["slopes"]) / len(data["slopes"])
                avg_stab = sum(data["stabilities"]) / len(data["stabilities"])
                f.write(f"{depth:<8} {avg_slope:<12.6f} {avg_stab:<14.4f} {len(data['slopes']):<6}\n")

            f.write("\nController Depth Convergence:\n")
            f.write(f"{'Depth':<8} {'Avg Slope':<12} {'Avg Stability':<14} {'n':<6}\n")
            f.write("-" * 44 + "\n")
            for depth in sorted(c_conv.keys()):
                data = c_conv[depth]
                avg_slope = sum(data["slopes"]) / len(data["slopes"])
                avg_stab = sum(data["stabilities"]) / len(data["stabilities"])
                f.write(f"{depth:<8} {avg_slope:<12.6f} {avg_stab:<14.4f} {len(data['slopes']):<6}\n")

            f.write("\nNote: Slope = (final_loss - initial_loss) / steps")
            f.write("\n      More negative slope = faster convergence")
            f.write("\n      Lower stability (std) = more stable training\n")

    print(f"Analysis saved to: {analysis_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Depth-focused nested optimizer fuzzing",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run depth fuzzing
  python fuzz_depth.py --experiment moe_nested --n_trials 25 --steps 1200

  # Reconstruct trajectories from existing checkpoint metrics.json files
  python fuzz_depth.py --reconstruct fuzz_results_depth/moe_nested_depth

  # Analyze results and generate charts
  python fuzz_depth.py --analyze fuzz_results_depth/moe_nested_depth
        """
    )
    parser.add_argument(
        "--experiment",
        type=str,
        choices=["moe_nested", "titanmac_nested", "both"],
        help="Which experiment to fuzz",
    )
    parser.add_argument("--n_trials", type=int, default=25, help="Number of trials")
    parser.add_argument("--steps", type=int, default=1200, help="Training steps per trial")
    parser.add_argument("--results_dir", type=str, default="fuzz_results_depth", help="Results directory")
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory for reconstruction")
    parser.add_argument("--reconstruct", type=str, help="Reconstruct trajectories from checkpoint metrics.json files")
    parser.add_argument("--analyze", type=str, help="Analyze existing results from this directory (skip training)")
    parser.add_argument("--resume", action="store_true", help="Resume an interrupted fuzzing run from where it left off")
    parser.add_argument("--grid", action="store_true", help="Use grid search instead of Bayesian (TPE) - tests ALL combinations")
    parser.add_argument("--skip_existing", type=str, nargs="+", help="Directories with existing results to skip (reuse data)")
    args = parser.parse_args()

    # Reconstruct mode - load trajectories from checkpoints
    if args.reconstruct:
        trials = reconstruct_trajectories(Path(args.reconstruct), args.checkpoint_dir)
        if trials:
            print("\nNow running analysis with reconstructed trajectories...")
            analyze_results(Path(args.reconstruct))
        return

    # Analysis-only mode
    if args.analyze:
        analyze_results(Path(args.analyze))
        return

    # Training mode requires experiment
    if not args.experiment:
        parser.error("--experiment is required unless using --analyze or --reconstruct")

    if args.experiment == "both":
        # Run both experiments
        for exp in ["moe_nested", "titanmac_nested"]:
            run_study(exp, args.n_trials, args.steps, args.results_dir,
                     resume=args.resume, use_grid=args.grid,
                     skip_existing_dirs=args.skip_existing)
    else:
        run_study(args.experiment, args.n_trials, args.steps, args.results_dir,
                 resume=args.resume, use_grid=args.grid,
                 skip_existing_dirs=args.skip_existing)


if __name__ == "__main__":
    main()
