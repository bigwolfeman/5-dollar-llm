#!/usr/bin/env python3
"""
Overnight Grid Search Runner

Runs 6x6 pruned depth grid search for both MoE and TitanMAC,
plus baseline comparisons with Muon+AdamW.

Uses pruned grid (M,C in [1,2,4,6,8,10]) for efficient exploration.

Usage:
    python run_overnight_grid.py

Estimated time: ~3-4 hours for pruned grid (72 trials + 2 baselines @ ~2.5 min each)
"""

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


# Configuration
STEPS = 600
RESULTS_DIR = "fuzz_results_cms"
CHECKPOINT_DIR = "checkpoints_cms"

# No existing results - CMS fix invalidates previous data
EXISTING_RESULTS = []


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
        name = "moe_baseline_600"
    else:
        script = "train_titanmac.py"
        name = "titanmac_baseline_600"

    cmd = [
        sys.executable, script,
        "--max_steps", str(steps),
        "--experiment_name", name,
        "--output_dir", CHECKPOINT_DIR,
    ]

    return run_command(cmd, f"Baseline: {model.upper()} with Muon+AdamW ({steps} steps)")


def run_grid_search(experiment: str, steps: int, skip_dirs: list) -> bool:
    """Run grid search for an experiment."""
    cmd = [
        sys.executable, "fuzz_depth.py",
        "--experiment", experiment,
        "--steps", str(steps),
        "--grid",
        "--results_dir", RESULTS_DIR,
    ]

    if skip_dirs:
        cmd.extend(["--skip_existing"] + skip_dirs)

    return run_command(cmd, f"Grid Search: {experiment} (12x12, {steps} steps)")


def run_analysis() -> bool:
    """Run analysis and generate charts."""
    cmd = [
        sys.executable, "analyze_grid.py",
        "--results_dir", RESULTS_DIR,
        "--baseline_dir", CHECKPOINT_DIR,
        "--output", f"{RESULTS_DIR}/analysis",
    ]

    return run_command(cmd, "Generating analysis and charts")


def main():
    print(f"""
{'#'*70}
#  CMS GRID SEARCH (with proper Continuum Memory System)
#  6x6 Pruned Depth Grid + Baselines
#  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'#'*70}

Configuration:
  - Grid: 6x6 = 36 configs per architecture (M,C in [1,2,4,6,8,10])
  - Steps: {STEPS}
  - Results: {RESULTS_DIR}
  - Checkpoints: {CHECKPOINT_DIR}
  - Fresh run (CMS fix invalidates previous data)

Estimated time: ~3-4 hours
""")

    # Create directories
    Path(RESULTS_DIR).mkdir(parents=True, exist_ok=True)
    Path(CHECKPOINT_DIR).mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    results = {}

    # Step 1: Run baselines
    print("\n" + "="*70)
    print("PHASE 1: BASELINES")
    print("="*70)

    results["moe_baseline"] = run_baseline("moe", STEPS)
    results["titanmac_baseline"] = run_baseline("titanmac", STEPS)

    # Step 2: Run MoE grid (6x6 pruned)
    print("\n" + "="*70)
    print("PHASE 2: MOE GRID SEARCH (6x6 pruned)")
    print("="*70)

    # Use pruned experiment for 6x6 grid
    moe_existing = [d for d in EXISTING_RESULTS if "moe" in d.lower()]
    results["moe_grid"] = run_grid_search("moe_pruned", STEPS, moe_existing)

    # Step 3: Run TitanMAC grid (6x6 pruned)
    print("\n" + "="*70)
    print("PHASE 3: TITANMAC GRID SEARCH (6x6 pruned)")
    print("="*70)

    titan_existing = [d for d in EXISTING_RESULTS if "titanmac" in d.lower()]
    results["titanmac_grid"] = run_grid_search("titanmac_pruned", STEPS, titan_existing)

    # Step 4: Analysis
    print("\n" + "="*70)
    print("PHASE 4: ANALYSIS")
    print("="*70)

    # Check if analyze_grid.py exists
    if Path("analyze_grid.py").exists():
        results["analysis"] = run_analysis()
    else:
        print("analyze_grid.py not found - skipping analysis")
        results["analysis"] = False

    # Summary
    total_time = (time.time() - start_time) / 3600
    print(f"""
{'#'*70}
#  OVERNIGHT RUN COMPLETE
#  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
#  Total time: {total_time:.1f} hours
{'#'*70}

Results:
""")

    for name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED"
        print(f"  {name}: {status}")

    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"Checkpoints saved to: {CHECKPOINT_DIR}/")

    # Report failures but don't exit - partial results are still valuable
    failed = [name for name, success in results.items() if not success]
    if failed:
        print(f"\n⚠️  Warning: {len(failed)} phase(s) had issues: {', '.join(failed)}")
        print("   Check logs above for details. Partial results may still be usable.")
        # Exit with warning code (not error) so caller knows there were issues
        sys.exit(2)  # 2 = completed with warnings, 1 = fatal error


if __name__ == "__main__":
    main()
