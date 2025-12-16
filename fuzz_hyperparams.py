#!/usr/bin/env python3
"""
Hyperparameter Fuzzing/Optimization Harness for 5-dollar-llm experiments.

Uses Optuna for Bayesian optimization to find optimal hyperparameters for:
- Baseline (train_moe.py): MoE + Muon+AdamW
- Exp 1 (train_titanmac.py): TitanMAC + Muon+AdamW
- Exp 2 (train_moe_nested.py): MoE + DeepNestedOptimizer
- Exp 3 (train_titanmac_nested.py): TitanMAC + DeepNestedOptimizer

Usage:
    # Fuzz MoE baseline
    python fuzz_hyperparams.py --experiment baseline --n_trials 50 --steps 8000

    # Fuzz TitanMAC with Muon+AdamW
    python fuzz_hyperparams.py --experiment titanmac --n_trials 50 --steps 8000

    # Fuzz MoE with nested optimizer
    python fuzz_hyperparams.py --experiment moe_nested --n_trials 50 --steps 8000

    # Fuzz TitanMAC with nested optimizer
    python fuzz_hyperparams.py --experiment titanmac_nested --n_trials 50 --steps 8000

    # Dry run (test without actually training)
    python fuzz_hyperparams.py --experiment baseline --n_trials 3 --dry_run

    # Resume a previous study
    python fuzz_hyperparams.py --experiment baseline --n_trials 50 --resume
"""

import argparse
import gc
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler


def clear_gpu_memory(wait_time: float = 2.0):
    """Clear GPU memory between trials to prevent OOM accumulation.

    Args:
        wait_time: Seconds to wait for GPU memory to be released by killed processes
    """
    if torch.cuda.is_available():
        # Force garbage collection first
        gc.collect()

        # Empty CUDA cache
        torch.cuda.empty_cache()

        # Collect IPC handles from dead processes
        torch.cuda.ipc_collect()

        # Reset memory stats
        torch.cuda.reset_peak_memory_stats()

        # Wait for GPU memory to be released by killed subprocesses
        # This is crucial when a subprocess OOMs and gets killed
        time.sleep(wait_time)

        # Second pass after waiting
        gc.collect()
        torch.cuda.empty_cache()
    else:
        gc.collect()

# Directory for this script (project root)
PROJECT_ROOT = Path(__file__).parent.absolute()


@dataclass
class ExperimentConfig:
    """Configuration for a hyperparameter fuzzing experiment."""
    name: str
    script: str
    hyperparams: Dict[str, Dict[str, Any]]
    description: str = ""
    fixed_args: Dict[str, Any] = field(default_factory=dict)


# Define hyperparameter search spaces for each experiment
EXPERIMENT_CONFIGS = {
    "baseline": ExperimentConfig(
        name="baseline",
        script="train_moe.py",
        description="MoE + Muon+AdamW (already tuned, for reference)",
        hyperparams={
            "muon_lr": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
            "adamw_lr": {"type": "float", "low": 0.001, "high": 0.01, "log": True},
        },
    ),
    "titanmac": ExperimentConfig(
        name="titanmac",
        script="train_titanmac.py",
        description="TitanMAC + Muon+AdamW",
        hyperparams={
            "muon_lr": {"type": "float", "low": 0.01, "high": 0.1, "log": True},
            "adamw_lr": {"type": "float", "low": 0.001, "high": 0.01, "log": True},
        },
    ),
    "moe_nested": ExperimentConfig(
        name="moe_nested",
        script="train_moe_nested.py",
        description="MoE + DeepNestedOptimizer",
        hyperparams={
            # Tighter bounds: high LRs (>3e-3) cause NaN losses
            "base_lr": {"type": "float", "low": 1e-4, "high": 3e-3, "log": True},
            "meta_lr": {"type": "float", "low": 1e-5, "high": 5e-4, "log": True},
            "k_unroll": {"type": "categorical", "choices": [1, 3, 5]},
            "momentum_hidden_dim": {"type": "categorical", "choices": [32, 64, 128]},
            "controller_hidden_dim": {"type": "categorical", "choices": [16, 32, 64]},
        },
    ),
    "titanmac_nested": ExperimentConfig(
        name="titanmac_nested",
        script="train_titanmac_nested.py",
        description="TitanMAC + DeepNestedOptimizer",
        hyperparams={
            # Tighter bounds: high LRs (>3e-3) cause NaN losses
            "base_lr": {"type": "float", "low": 1e-4, "high": 3e-3, "log": True},
            "meta_lr": {"type": "float", "low": 1e-5, "high": 5e-4, "log": True},
            "k_unroll": {"type": "categorical", "choices": [1, 3, 5]},
            "momentum_hidden_dim": {"type": "categorical", "choices": [32, 64, 128]},
            "controller_hidden_dim": {"type": "categorical", "choices": [16, 32, 64]},
        },
    ),
}


def suggest_hyperparams(trial: optuna.Trial, config: ExperimentConfig) -> Dict[str, Any]:
    """
    Use Optuna trial to suggest hyperparameters based on experiment config.

    Args:
        trial: Optuna trial object
        config: Experiment configuration with hyperparameter definitions

    Returns:
        Dictionary of suggested hyperparameter values
    """
    params = {}

    for name, spec in config.hyperparams.items():
        if spec["type"] == "float":
            params[name] = trial.suggest_float(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False)
            )
        elif spec["type"] == "int":
            params[name] = trial.suggest_int(
                name,
                spec["low"],
                spec["high"],
                log=spec.get("log", False)
            )
        elif spec["type"] == "categorical":
            params[name] = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"Unknown hyperparameter type: {spec['type']}")

    return params


def build_training_command(
    config: ExperimentConfig,
    params: Dict[str, Any],
    steps: int,
    output_dir: Path,
    experiment_name: str,
) -> List[str]:
    """
    Build the command line arguments for the training script.

    Args:
        config: Experiment configuration
        params: Suggested hyperparameters
        steps: Number of training steps
        output_dir: Output directory for checkpoints/metrics
        experiment_name: Name for this trial's experiment

    Returns:
        List of command line arguments
    """
    script_path = PROJECT_ROOT / config.script

    cmd = [
        sys.executable,
        str(script_path),
        "--steps", str(steps),
        "--output_dir", str(output_dir),
        "--experiment_name", experiment_name,
    ]

    # Add hyperparameters
    for name, value in params.items():
        cmd.extend([f"--{name}", str(value)])

    # Add fixed arguments
    for name, value in config.fixed_args.items():
        if isinstance(value, bool):
            if value:
                cmd.append(f"--{name}")
        else:
            cmd.extend([f"--{name}", str(value)])

    return cmd


def parse_val_loss_from_output(output: str) -> Optional[float]:
    """
    Parse validation loss from training script output.

    The training scripts print:
    - "Val loss:       X.XXXX"
    - Or "Val Loss: X.XXXX"

    Args:
        output: Stdout from training script

    Returns:
        Validation loss if found, None otherwise
    """
    # Try different patterns the scripts use
    patterns = [
        r"Val loss:\s+([\d.]+)",
        r"Val Loss:\s+([\d.]+)",
        r"Final Val Loss:\s+([\d.]+)",
        r"'val_loss':\s*([\d.]+)",
    ]

    for pattern in patterns:
        matches = re.findall(pattern, output, re.IGNORECASE)
        if matches:
            # Return the last match (final validation loss)
            return float(matches[-1])

    return None


def parse_metrics_from_json(output_dir: Path) -> Optional[float]:
    """
    Try to parse validation loss from metrics.json file.

    Args:
        output_dir: Directory containing metrics.json

    Returns:
        Validation loss if found, None otherwise
    """
    # Try different possible metric file locations
    possible_paths = [
        output_dir / "metrics.json",
        output_dir / "metrics_history.json",
    ]

    for metrics_path in possible_paths:
        if metrics_path.exists():
            try:
                with open(metrics_path) as f:
                    metrics = json.load(f)

                # Different scripts structure the JSON differently
                if "final_metrics" in metrics:
                    return metrics["final_metrics"].get("val_loss")
                elif "val_loss" in metrics:
                    return metrics["val_loss"]
                elif "val_losses" in metrics and metrics["val_losses"]:
                    return metrics["val_losses"][-1]

            except (json.JSONDecodeError, KeyError, IndexError):
                continue

    return None


def run_training(
    config: ExperimentConfig,
    params: Dict[str, Any],
    steps: int,
    output_dir: Path,
    trial_number: int,
    dry_run: bool = False,
    timeout: int = 7200,  # 2 hour timeout
) -> Tuple[float, Dict[str, Any]]:
    """
    Run a single training trial with the given hyperparameters.

    Args:
        config: Experiment configuration
        params: Suggested hyperparameters
        steps: Number of training steps
        output_dir: Base output directory
        trial_number: Trial number for naming
        dry_run: If True, simulate training without running
        timeout: Timeout in seconds

    Returns:
        Tuple of (validation_loss, result_dict)
    """
    experiment_name = f"trial_{trial_number:04d}"
    trial_output_dir = output_dir / experiment_name
    trial_output_dir.mkdir(parents=True, exist_ok=True)

    cmd = build_training_command(
        config, params, steps, trial_output_dir, experiment_name
    )

    result = {
        "trial_number": trial_number,
        "params": params,
        "command": " ".join(cmd),
        "output_dir": str(trial_output_dir),
        "status": "pending",
    }

    if dry_run:
        # Simulate training for dry run
        print(f"\n[DRY RUN] Would run: {' '.join(cmd)}")
        print(f"[DRY RUN] Simulating random val_loss between 3.0 and 6.0")
        import random
        val_loss = random.uniform(3.0, 6.0)
        result["status"] = "completed"
        result["val_loss"] = val_loss
        result["simulated"] = True
        return val_loss, result

    print(f"\n{'='*70}")
    print(f"Trial {trial_number}: Running with params:")
    for k, v in params.items():
        print(f"  {k}: {v}")
    print(f"Command: {' '.join(cmd[:4])} ...")
    print(f"{'='*70}")

    start_time = time.time()

    try:
        # Clear GPU memory before starting
        clear_gpu_memory()

        # Run the training script
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(PROJECT_ROOT),
        )

        output_lines = []
        nan_detected = False
        oom_detected = False

        # Stream output and collect it, with NaN/OOM detection
        for line in process.stdout:
            print(line, end="")
            output_lines.append(line)

            # Early detection of NaN losses - abort trial
            if "loss=nan" in line.lower() or "val loss: nan" in line.lower():
                nan_detected = True
                print(f"\n[ABORT] NaN loss detected - killing trial early")
                process.kill()
                break

            # Detect OOM errors
            if "cuda out of memory" in line.lower() or "outofmemoryerror" in line.lower():
                oom_detected = True
                print(f"\n[ABORT] OOM detected - killing trial")
                process.kill()
                break

        if not nan_detected and not oom_detected:
            process.wait(timeout=timeout)

        elapsed_time = time.time() - start_time
        output = "".join(output_lines)

        # Clear GPU memory after trial
        clear_gpu_memory()

        if nan_detected:
            result["status"] = "failed"
            result["error"] = "NaN loss detected"
            result["elapsed_time"] = elapsed_time
            return float("inf"), result

        if oom_detected:
            # Extra long wait after OOM to ensure GPU memory is released
            print("[OOM] Waiting 5s for GPU memory to be released...")
            clear_gpu_memory(wait_time=5.0)
            result["status"] = "failed"
            result["error"] = "CUDA OOM"
            result["elapsed_time"] = elapsed_time
            return float("inf"), result

        if process.returncode != 0:
            print(f"\n[ERROR] Training failed with return code {process.returncode}")
            result["status"] = "failed"
            result["error"] = f"Return code: {process.returncode}"
            result["elapsed_time"] = elapsed_time
            return float("inf"), result

        # Parse validation loss from output
        val_loss = parse_val_loss_from_output(output)

        # If not found in output, try metrics file
        if val_loss is None:
            val_loss = parse_metrics_from_json(trial_output_dir)

        if val_loss is None:
            print("\n[ERROR] Could not parse validation loss from output")
            result["status"] = "failed"
            result["error"] = "Could not parse validation loss"
            result["elapsed_time"] = elapsed_time
            return float("inf"), result

        # Check for NaN val_loss (training diverged)
        import math
        if math.isnan(val_loss) or math.isinf(val_loss):
            print(f"\n[ERROR] Invalid val_loss: {val_loss} (training diverged)")
            result["status"] = "failed"
            result["error"] = f"Invalid val_loss: {val_loss}"
            result["elapsed_time"] = elapsed_time
            return float("inf"), result

        result["status"] = "completed"
        result["val_loss"] = val_loss
        result["elapsed_time"] = elapsed_time

        print(f"\n[SUCCESS] Trial {trial_number} completed in {elapsed_time/60:.1f} min")
        print(f"          Val Loss: {val_loss:.4f}")

        return val_loss, result

    except subprocess.TimeoutExpired:
        process.kill()
        clear_gpu_memory()
        result["status"] = "timeout"
        result["error"] = f"Timeout after {timeout} seconds"
        print(f"\n[TIMEOUT] Trial {trial_number} timed out after {timeout}s")
        return float("inf"), result

    except Exception as e:
        clear_gpu_memory()
        result["status"] = "error"
        result["error"] = str(e)
        print(f"\n[ERROR] Trial {trial_number} failed: {e}")
        return float("inf"), result


def create_objective(
    config: ExperimentConfig,
    steps: int,
    output_dir: Path,
    dry_run: bool = False,
    trial_results: List[Dict[str, Any]] = None,
) -> Callable[[optuna.Trial], float]:
    """
    Create an Optuna objective function for the given experiment.

    Args:
        config: Experiment configuration
        steps: Number of training steps per trial
        output_dir: Base output directory
        dry_run: If True, simulate training
        trial_results: List to append trial results to

    Returns:
        Objective function for Optuna
    """
    if trial_results is None:
        trial_results = []

    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters
        params = suggest_hyperparams(trial, config)

        # Run training
        val_loss, result = run_training(
            config=config,
            params=params,
            steps=steps,
            output_dir=output_dir,
            trial_number=trial.number,
            dry_run=dry_run,
        )

        # Store result
        trial_results.append(result)

        # Report for pruning (if we had intermediate results)
        # For now, we just return the final loss

        return val_loss

    return objective


def save_results(
    study: optuna.Study,
    config: ExperimentConfig,
    trial_results: List[Dict[str, Any]],
    output_dir: Path,
):
    """
    Save optimization results to files.

    Args:
        study: Optuna study object
        config: Experiment configuration
        trial_results: List of trial result dictionaries
        output_dir: Output directory
    """
    # Save best parameters
    best_params_path = output_dir / "best_params.json"
    best_params = {
        "experiment": config.name,
        "description": config.description,
        "best_params": study.best_params,
        "best_value": study.best_value,
        "best_trial_number": study.best_trial.number,
        "n_trials": len(study.trials),
        "timestamp": datetime.now().isoformat(),
    }

    with open(best_params_path, "w") as f:
        json.dump(best_params, f, indent=2)

    print(f"\nBest parameters saved to: {best_params_path}")

    # Save all trial results
    all_results_path = output_dir / "all_trials.json"
    all_results = {
        "experiment": config.name,
        "description": config.description,
        "trials": trial_results,
        "summary": {
            "total_trials": len(study.trials),
            "completed_trials": len([r for r in trial_results if r.get("status") == "completed"]),
            "best_trial": study.best_trial.number,
            "best_value": study.best_value,
        },
        "timestamp": datetime.now().isoformat(),
    }

    with open(all_results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"All trial results saved to: {all_results_path}")

    # Generate summary report
    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"Hyperparameter Optimization Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Experiment: {config.name}\n")
        f.write(f"Description: {config.description}\n")
        f.write(f"Script: {config.script}\n")
        f.write(f"Total trials: {len(study.trials)}\n\n")

        f.write(f"Best Trial (#{study.best_trial.number})\n")
        f.write(f"{'-'*30}\n")
        f.write(f"Val Loss: {study.best_value:.4f}\n")
        f.write(f"Parameters:\n")
        for k, v in study.best_params.items():
            f.write(f"  {k}: {v}\n")
        f.write(f"\n")

        f.write(f"Top 5 Trials\n")
        f.write(f"{'-'*30}\n")

        # Sort trials by value
        sorted_trials = sorted(
            [t for t in study.trials if t.value is not None],
            key=lambda t: t.value
        )

        for i, trial in enumerate(sorted_trials[:5]):
            f.write(f"\n{i+1}. Trial #{trial.number}: Val Loss = {trial.value:.4f}\n")
            for k, v in trial.params.items():
                f.write(f"   {k}: {v}\n")

        f.write(f"\n{'='*50}\n")
        f.write(f"Generated at: {datetime.now().isoformat()}\n")

    print(f"Summary saved to: {summary_path}")


def print_banner(config: ExperimentConfig, n_trials: int, steps: int):
    """Print a banner with experiment information."""
    print("\n" + "="*70)
    print("  HYPERPARAMETER FUZZING HARNESS")
    print("="*70)
    print(f"  Experiment: {config.name}")
    print(f"  Description: {config.description}")
    print(f"  Script: {config.script}")
    print(f"  Trials: {n_trials}")
    print(f"  Steps per trial: {steps}")
    print(f"\n  Hyperparameters to optimize:")
    for name, spec in config.hyperparams.items():
        if spec["type"] == "float":
            print(f"    - {name}: [{spec['low']}, {spec['high']}] (log={spec.get('log', False)})")
        elif spec["type"] == "categorical":
            print(f"    - {name}: {spec['choices']}")
    print("="*70 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter fuzzing harness for 5-dollar-llm experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=list(EXPERIMENT_CONFIGS.keys()),
        help="Which experiment to fuzz"
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        default=50,
        help="Number of optimization trials (default: 50)"
    )

    parser.add_argument(
        "--steps",
        type=int,
        default=8000,
        help="Training steps per trial (default: 8000)"
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./fuzz_results",
        help="Base output directory for results (default: ./fuzz_results)"
    )

    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Test without running actual training (simulates random losses)"
    )

    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a previous study from the database (default behavior now)"
    )

    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Start fresh by deleting any existing study database"
    )

    parser.add_argument(
        "--timeout",
        type=int,
        default=7200,
        help="Timeout per trial in seconds (default: 7200 = 2 hours)"
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )

    parser.add_argument(
        "--n_startup_trials",
        type=int,
        default=10,
        help="Number of random trials before TPE kicks in (default: 10)"
    )

    parser.add_argument(
        "--pruning_warmup_steps",
        type=int,
        default=5,
        help="Minimum trials before pruning starts (default: 5)"
    )

    args = parser.parse_args()

    # Get experiment config
    config = EXPERIMENT_CONFIGS[args.experiment]

    # Setup output directory
    output_dir = Path(args.output_dir) / config.name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print banner
    print_banner(config, args.n_trials, args.steps)

    if args.dry_run:
        print("[DRY RUN MODE] No actual training will be performed\n")

    # Setup study storage (SQLite database)
    db_path = output_dir / "study.db"
    storage = f"sqlite:///{db_path}"

    study_name = f"{config.name}_optimization"

    # Handle --fresh flag: delete existing database
    if args.fresh and db_path.exists():
        print(f"[FRESH] Deleting existing study database: {db_path}")
        db_path.unlink()
        # Also clean up trial directories
        for trial_dir in output_dir.glob("trial_*"):
            if trial_dir.is_dir():
                import shutil
                shutil.rmtree(trial_dir)
        print("[FRESH] Cleaned up previous trial results\n")

    # Create sampler with seed
    sampler = TPESampler(
        seed=args.seed,
        n_startup_trials=args.n_startup_trials,
    )

    # Create pruner
    pruner = MedianPruner(
        n_startup_trials=args.pruning_warmup_steps,
        n_warmup_steps=0,
    )

    # Create or load study
    # Always use load_if_exists=True to handle interrupted runs gracefully
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        sampler=sampler,
        pruner=pruner,
        direction="minimize",
        load_if_exists=True,  # Continue from existing study if present
    )

    existing_trials = len(study.trials)
    if existing_trials > 0:
        print(f"Resuming study from {db_path}")
        print(f"Found {existing_trials} previous trials")
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        if completed > 0:
            print(f"  Completed: {completed}, Best so far: {study.best_value:.4f}\n")
        else:
            print(f"  Completed: {completed} (no successful trials yet)\n")
    else:
        print(f"Created new study: {study_name}")
        print(f"Database: {db_path}\n")

    # Track trial results
    trial_results = []

    # Create objective function
    objective = create_objective(
        config=config,
        steps=args.steps,
        output_dir=output_dir,
        dry_run=args.dry_run,
        trial_results=trial_results,
    )

    # Run optimization
    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            show_progress_bar=True,
            gc_after_trial=True,  # Help with GPU memory cleanup
        )
    except KeyboardInterrupt:
        print("\n\n[INTERRUPTED] Saving partial results...")

    # Save results
    save_results(study, config, trial_results, output_dir)

    # Print final summary
    print("\n" + "="*70)
    print("  OPTIMIZATION COMPLETE")
    print("="*70)
    print(f"\n  Best trial: #{study.best_trial.number}")
    print(f"  Best val_loss: {study.best_value:.4f}")
    print(f"\n  Best hyperparameters:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    print(f"\n  Results saved to: {output_dir}")
    print("="*70 + "\n")

    return study


if __name__ == "__main__":
    main()
