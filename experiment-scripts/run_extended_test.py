#!/usr/bin/env python3
"""
Extended Training Test: 3000 Steps

Tests whether the nested optimizer needs more training time to converge.
Runs best and median configs from grid search + baselines for comparison.

Configs from 600-step grid search:
  - MoE Best:     M=2, C=2,   loss=8.53
  - MoE Median:   M=6, C=10,  loss=8.83
  - TitanMAC Best:   M=6, C=6,   loss=8.42
  - TitanMAC Median: M=10, C=10, loss=8.76

Baselines at 600 steps:
  - MoE:      3.94 loss
  - TitanMAC: 4.53 loss

Hypothesis: Nested optimizer needs more steps to warm up CMS + learned momentum.

Usage:
    python run_extended_test.py

Estimated time: ~2.5 hours (6 runs @ ~25 min each)
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# Configuration
STEPS = 3000
RESULTS_DIR = "extended_test_results"
CHECKPOINT_DIR = "checkpoints_extended"

# Best configs from grid search (with their base params)
CONFIGS = {
    "moe_best": {
        "script": "train_moe_nested.py",
        "momentum_num_layers": 2,
        "controller_num_layers": 2,
        "base_lr": 0.0004603431029542576,
        "meta_lr": 9.56441656770269e-05,
        "momentum_hidden_dim": 128,
        "controller_hidden_dim": 32,
    },
    "moe_median": {
        "script": "train_moe_nested.py",
        "momentum_num_layers": 6,
        "controller_num_layers": 10,
        "base_lr": 0.0004603431029542576,
        "meta_lr": 9.56441656770269e-05,
        "momentum_hidden_dim": 128,
        "controller_hidden_dim": 32,
    },
    "titanmac_best": {
        "script": "train_titanmac_nested.py",
        "momentum_num_layers": 6,
        "controller_num_layers": 6,
        "base_lr": 0.0004072398672148361,
        "meta_lr": 9.089757036214503e-05,
        "momentum_hidden_dim": 64,
        "controller_hidden_dim": 16,
    },
    "titanmac_median": {
        "script": "train_titanmac_nested.py",
        "momentum_num_layers": 10,
        "controller_num_layers": 10,
        "base_lr": 0.0004072398672148361,
        "meta_lr": 9.089757036214503e-05,
        "momentum_hidden_dim": 64,
        "controller_hidden_dim": 16,
    },
}


def run_command(cmd: list, description: str) -> bool:
    """Run a command and stream output."""
    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}\n")

    start = time.time()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        for line in process.stdout:
            print(line, end='')

        process.wait()
        elapsed = time.time() - start

        if process.returncode == 0:
            print(f"\n[SUCCESS] {description} completed in {elapsed/60:.1f} minutes")
            return True
        else:
            print(f"\n[FAILED] {description} failed with code {process.returncode}")
            return False

    except Exception as e:
        print(f"\n[ERROR] {description} failed: {e}")
        return False


def run_baseline(model: str, steps: int) -> bool:
    """Run baseline Muon+AdamW training."""
    if model == "moe":
        script = "train_moe.py"
        name = f"moe_baseline_{steps}"
    else:
        script = "train_titanmac.py"
        name = f"titanmac_baseline_{steps}"

    cmd = [
        sys.executable, script,
        "--max_steps", str(steps),
        "--experiment_name", name,
        "--output_dir", CHECKPOINT_DIR,
    ]

    return run_command(cmd, f"Baseline: {model.upper()} with Muon+AdamW ({steps} steps)")


def run_nested(name: str, config: dict, steps: int) -> bool:
    """Run nested optimizer training with specific config."""
    cmd = [
        sys.executable, config["script"],
        "--max_steps", str(steps),
        "--experiment_name", f"{name}_{steps}",
        "--output_dir", CHECKPOINT_DIR,
        "--base_lr", str(config["base_lr"]),
        "--meta_lr", str(config["meta_lr"]),
        "--momentum_num_layers", str(config["momentum_num_layers"]),
        "--controller_num_layers", str(config["controller_num_layers"]),
        "--momentum_hidden_dim", str(config["momentum_hidden_dim"]),
        "--controller_hidden_dim", str(config["controller_hidden_dim"]),
    ]

    desc = (f"Nested: {name} M={config['momentum_num_layers']}, "
            f"C={config['controller_num_layers']} ({steps} steps)")
    return run_command(cmd, desc)


def main():
    print(f"""
{'#'*70}
#  EXTENDED TRAINING TEST (3000 Steps)
#  Testing if Nested Optimizer needs more warmup time
#  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'#'*70}

Configuration:
  - Steps: {STEPS}
  - Results: {RESULTS_DIR}
  - Checkpoints: {CHECKPOINT_DIR}

Tests:
  1. MoE Baseline (Muon+AdamW)
  2. TitanMAC Baseline (Muon+AdamW)
  3. MoE Nested Best (M=2, C=2)
  4. MoE Nested Median (M=6, C=10)
  5. TitanMAC Nested Best (M=6, C=6)
  6. TitanMAC Nested Median (M=10, C=10)

Estimated time: ~2.5 hours
""")

    # Create directories
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    results = {}

    # Phase 1: Baselines
    print("\n" + "="*70)
    print("PHASE 1: BASELINES (3000 steps)")
    print("="*70)

    results["moe_baseline"] = run_baseline("moe", STEPS)
    results["titanmac_baseline"] = run_baseline("titanmac", STEPS)

    # Phase 2: Nested configs
    print("\n" + "="*70)
    print("PHASE 2: NESTED OPTIMIZER CONFIGS (3000 steps)")
    print("="*70)

    for name, config in CONFIGS.items():
        results[name] = run_nested(name, config, STEPS)

    # Summary
    total_time = (time.time() - start_time) / 3600
    print(f"""
{'#'*70}
#  EXTENDED TEST COMPLETE
#  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
#  Total time: {total_time:.1f} hours
{'#'*70}

Results:
""")

    for name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"  {name}: {status}")

    print(f"\nCheckpoints saved to: {CHECKPOINT_DIR}/")
    print("\nTo compare results, check the final val_loss in each run's output above.")
    print("Look for 'val_loss' in the final evaluation step.")

    # Report failures
    failed = [name for name, success in results.items() if not success]
    if failed:
        print(f"\nWarning: {len(failed)} run(s) had issues: {', '.join(failed)}")
        sys.exit(2)


if __name__ == "__main__":
    main()
