#!/usr/bin/env python3
"""
Comprehensive Grid Search: 168M Parameter Comparison

Compares MoE and TitanMAC architectures with multiple optimizers:
- Muon+AdamW (baseline)
- DeepNestedOptimizer (with 6x6 grid search)

All models scaled to ~168M parameters for fair comparison.

Usage:
    # Baseline mode: Run all non-grid configurations
    python run_comprehensive_grid.py --mode baseline --steps 600

    # Grid mode: Run 6x6 nested optimizer grid search
    python run_comprehensive_grid.py --mode grid --steps 600

    # Winners mode: Run top nested configs + Muon baselines for extended training
    python run_comprehensive_grid.py --mode winners --steps 5000

    # Dry run: Validate configurations without GPU
    python run_comprehensive_grid.py --mode baseline --dry-run
    python run_comprehensive_grid.py --mode grid --dry-run

Estimated times (600 steps each):
    - Baseline mode: ~4 configs × 2-4 min = 8-16 min
    - Grid mode: 72 configs × 2-4 min = 2.5-5 hours
    - Winners mode: 4 configs × N min (depends on --steps)
"""

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ============================================================================
# Configuration
# ============================================================================

STEPS = 600
RESULTS_DIR = "grid_results_168m"
CHECKPOINT_DIR = "checkpoints_168m"

# Nested optimizer best params from previous fuzzing
BEST_NESTED_PARAMS = {
    "base_lr": 3e-4,
    "meta_lr": 1e-4,
    "k_unroll": 5,
    "momentum_hidden_dim": 64,
    "controller_hidden_dim": 32,
    "momentum_num_layers": 2,
    "controller_num_layers": 2,
}

# Grid search space (6x6 = 36 configs)
DEPTH_GRID = [1, 2, 3, 4, 5, 6]


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    name: str
    model: str  # "moe", "titanmac", "hope"
    optimizer: str  # "muon", "nested", "adamw"
    description: str
    # Nested-specific params (None for non-nested)
    momentum_layers: Optional[int] = None
    controller_layers: Optional[int] = None


# ============================================================================
# Baseline Experiments
# ============================================================================

BASELINE_EXPERIMENTS = [
    ExperimentConfig(
        name="moe_muon",
        model="moe",
        optimizer="muon",
        description="MoE 168M + Muon+AdamW (baseline)",
    ),
    ExperimentConfig(
        name="moe_nested_best",
        model="moe",
        optimizer="nested",
        description="MoE 168M + DeepNested (best params)",
        momentum_layers=2,
        controller_layers=2,
    ),
    ExperimentConfig(
        name="titanmac_muon",
        model="titanmac",
        optimizer="muon",
        description="TitanMAC 168M + Muon+AdamW",
    ),
    ExperimentConfig(
        name="titanmac_nested_best",
        model="titanmac",
        optimizer="nested",
        description="TitanMAC 168M + DeepNested (best params)",
        momentum_layers=2,
        controller_layers=2,
    ),
    # HOPE removed - delta-rule memory too VRAM-intensive for 168M scale
]


# ============================================================================
# Grid Experiments
# ============================================================================

def generate_grid_experiments() -> List[ExperimentConfig]:
    """Generate 6x6 grid experiments for each model."""
    experiments = []

    for model in ["moe", "titanmac"]:  # HOPE removed - delta-rule memory too VRAM-intensive
        for m in DEPTH_GRID:
            for c in DEPTH_GRID:
                experiments.append(ExperimentConfig(
                    name=f"{model}_nested_M{m}_C{c}",
                    model=model,
                    optimizer="nested",
                    description=f"{model.upper()} 168M + DeepNested (M={m}, C={c})",
                    momentum_layers=m,
                    controller_layers=c,
                ))

    return experiments


def generate_winners_experiments(results_dir: str = RESULTS_DIR) -> List[ExperimentConfig]:
    """
    Generate experiments for winners mode.

    Reads grid_results.json to find:
    - Top 1 MoE nested config (lowest val_loss)
    - Top 1 TitanMAC nested config (lowest val_loss)

    Plus Muon baselines:
    - moe_muon
    - titanmac_muon

    Returns 4 experiments total.
    """
    import re

    experiments = []

    # Add Muon baselines
    experiments.append(ExperimentConfig(
        name="moe_muon",
        model="moe",
        optimizer="muon",
        description="MoE 168M + Muon+AdamW (baseline)",
    ))
    experiments.append(ExperimentConfig(
        name="titanmac_muon",
        model="titanmac",
        optimizer="muon",
        description="TitanMAC 168M + Muon+AdamW (baseline)",
    ))

    # Load grid results
    grid_file = Path(results_dir) / "grid_results.json"
    if not grid_file.exists():
        print(f"WARNING: {grid_file} not found. Run --mode grid first.")
        print("Falling back to default best params (M=2, C=2)")
        # Fallback to default best params
        experiments.append(ExperimentConfig(
            name="moe_nested_winner",
            model="moe",
            optimizer="nested",
            description="MoE 168M + DeepNested (M=2, C=2 fallback)",
            momentum_layers=2,
            controller_layers=2,
        ))
        experiments.append(ExperimentConfig(
            name="titanmac_nested_winner",
            model="titanmac",
            optimizer="nested",
            description="TitanMAC 168M + DeepNested (M=2, C=2 fallback)",
            momentum_layers=2,
            controller_layers=2,
        ))
        return experiments

    with open(grid_file) as f:
        grid_results = json.load(f)

    # Find top 1 for each model
    for model in ["moe", "titanmac"]:
        model_results = [
            r for r in grid_results
            if r.get("status") == "success" and model in r["name"].lower()
        ]

        if not model_results:
            print(f"WARNING: No successful {model} results found. Using default M=2, C=2.")
            m_layers, c_layers = 2, 2
        else:
            # Sort by val_loss and get best
            best = min(model_results, key=lambda x: x.get("metrics", {}).get("val_loss", float("inf")))

            # Parse M and C from name like "moe_nested_M3_C2"
            match = re.search(r"_M(\d+)_C(\d+)", best["name"])
            if match:
                m_layers = int(match.group(1))
                c_layers = int(match.group(2))
            else:
                m_layers, c_layers = 2, 2

            best_loss = best.get("metrics", {}).get("val_loss", "N/A")
            print(f"  {model.upper()} winner: M={m_layers}, C={c_layers} (val_loss={best_loss:.4f})")

        model_upper = "TitanMAC" if model == "titanmac" else "MoE"
        experiments.append(ExperimentConfig(
            name=f"{model}_nested_winner",
            model=model,
            optimizer="nested",
            description=f"{model_upper} 168M + DeepNested (M={m_layers}, C={c_layers} - WINNER)",
            momentum_layers=m_layers,
            controller_layers=c_layers,
        ))

    return experiments


# ============================================================================
# Training Scripts
# ============================================================================

def get_train_script(model: str, optimizer: str) -> str:
    """Get the training script for a model+optimizer combination."""
    if optimizer == "nested":
        if model == "moe":
            return "train_moe_nested.py"
        elif model == "titanmac":
            return "train_titanmac_nested.py"
        elif model == "hope":
            return "train_hope_nested.py"
    else:
        # Muon or AdamW uses base training scripts
        if model == "moe":
            return "train_moe.py"
        elif model == "titanmac":
            return "train_titanmac.py"
        elif model == "hope":
            return "train_hope.py"

    raise ValueError(f"Unknown model+optimizer: {model}+{optimizer}")


def find_checkpoint(experiment_dir: Path) -> Optional[Path]:
    """Find an existing checkpoint in the experiment directory."""
    if not experiment_dir.exists():
        return None

    # Check for checkpoint files in order of preference
    for ckpt_name in ["model.pt", "final_model.pt", "checkpoint.pt"]:
        ckpt_path = experiment_dir / ckpt_name
        if ckpt_path.exists():
            return ckpt_path

    return None


def build_command(
    experiment: ExperimentConfig,
    steps: int,
    output_dir: str,
    resume: bool = False,
) -> List[str]:
    """Build the command to run an experiment."""
    script = get_train_script(experiment.model, experiment.optimizer)

    cmd = [
        sys.executable, script,
        "--max_steps", str(steps),
        "--experiment_name", experiment.name,
        "--output_dir", output_dir,
        "--config", "168m",  # Use 168M config variant
    ]

    # Add nested-specific params
    if experiment.optimizer == "nested":
        cmd.extend([
            "--base_lr", str(BEST_NESTED_PARAMS["base_lr"]),
            "--meta_lr", str(BEST_NESTED_PARAMS["meta_lr"]),
            "--k_unroll", str(BEST_NESTED_PARAMS["k_unroll"]),
            "--momentum_hidden_dim", str(BEST_NESTED_PARAMS["momentum_hidden_dim"]),
            "--controller_hidden_dim", str(BEST_NESTED_PARAMS["controller_hidden_dim"]),
        ])

        if experiment.momentum_layers is not None:
            cmd.extend(["--momentum_num_layers", str(experiment.momentum_layers)])
        if experiment.controller_layers is not None:
            cmd.extend(["--controller_num_layers", str(experiment.controller_layers)])

    # Check for existing checkpoint and add --resume if found
    if resume:
        experiment_path = Path(output_dir) / experiment.name
        checkpoint = find_checkpoint(experiment_path)
        if checkpoint:
            cmd.extend(["--resume", str(checkpoint)])
            print(f"  [Resume] Found checkpoint: {checkpoint}")

    return cmd


# ============================================================================
# Execution
# ============================================================================

def run_experiment(
    experiment: ExperimentConfig,
    steps: int,
    output_dir: str,
    dry_run: bool = False,
    resume: bool = False,
) -> Dict[str, Any]:
    """Run a single experiment."""
    cmd = build_command(experiment, steps, output_dir, resume=resume)

    print(f"\n{'='*70}")
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {experiment.description}")
    print(f"{'='*70}")
    print(f"Command: {' '.join(cmd)}")

    if dry_run:
        print("[DRY RUN] Would execute above command")
        return {
            "name": experiment.name,
            "status": "dry_run",
            "command": cmd,
        }

    start_time = time.time()

    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        output_lines = []
        for line in process.stdout:
            print(line, end='')
            output_lines.append(line)

        process.wait(timeout=7200)  # 2 hour timeout
        elapsed = time.time() - start_time

        # Parse metrics from output
        metrics = parse_output_metrics(''.join(output_lines))

        result = {
            "name": experiment.name,
            "status": "success" if process.returncode == 0 else "failed",
            "return_code": process.returncode,
            "elapsed_seconds": elapsed,
            "metrics": metrics,
        }

        print(f"\n[{'SUCCESS' if process.returncode == 0 else 'FAILED'}] "
              f"{experiment.name} in {elapsed/60:.1f} min")

        return result

    except subprocess.TimeoutExpired:
        print(f"\n[TIMEOUT] {experiment.name}")
        process.kill()
        return {"name": experiment.name, "status": "timeout"}
    except Exception as e:
        print(f"\n[ERROR] {experiment.name}: {e}")
        return {"name": experiment.name, "status": "error", "error": str(e)}


def parse_output_metrics(output: str) -> Dict[str, Any]:
    """Parse training output for metrics."""
    metrics = {}

    for line in output.split('\n'):
        line_lower = line.lower()

        # Val loss
        if 'val loss:' in line_lower or 'val_loss:' in line_lower:
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    val = parts[-1].strip().split()[0]
                    metrics['val_loss'] = float(val) if val.lower() != 'nan' else float('inf')
            except (ValueError, IndexError):
                pass

        # Val accuracy
        if 'val accuracy:' in line_lower or 'val_accuracy:' in line_lower:
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    metrics['val_accuracy'] = float(parts[-1].strip().split()[0])
            except (ValueError, IndexError):
                pass

        # Params
        if 'params:' in line_lower or 'parameters:' in line_lower:
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    val = parts[-1].strip().replace('M', '').replace(',', '')
                    metrics['params_m'] = float(val)
            except (ValueError, IndexError):
                pass

        # Peak VRAM
        if 'peak vram' in line_lower or 'peak_vram' in line_lower:
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    val = parts[-1].strip().replace('GB', '').replace('gb', '').strip()
                    metrics['peak_vram_gb'] = float(val)
            except (ValueError, IndexError):
                pass

    return metrics


# ============================================================================
# Dry Run Validation
# ============================================================================

def validate_scripts_exist(experiments: List[ExperimentConfig]) -> bool:
    """Check that all required training scripts exist."""
    scripts_needed = set()
    for exp in experiments:
        scripts_needed.add(get_train_script(exp.model, exp.optimizer))

    all_exist = True
    for script in sorted(scripts_needed):
        exists = Path(script).exists()
        status = "✓" if exists else "✗ MISSING"
        print(f"  {script}: {status}")
        if not exists:
            all_exist = False

    return all_exist


def validate_configs_exist() -> bool:
    """Check that 168M configs can be imported."""
    print("\nValidating configs...")

    try:
        from configs import GPU24GBMoEModelConfig, TitanMAC168MConfig, HOPE168MConfig
        print("  GPU24GBMoEModelConfig: ✓")
        print("  TitanMAC168MConfig: ✓")
        print("  HOPE168MConfig: ✓")
        return True
    except ImportError as e:
        print(f"  Config import failed: {e}")
        return False


def dry_run_summary(experiments: List[ExperimentConfig], steps: int):
    """Print summary for dry run."""
    print(f"\n{'='*70}")
    print("DRY RUN SUMMARY")
    print(f"{'='*70}")
    print(f"\nExperiments: {len(experiments)}")
    print(f"Steps per experiment: {steps}")
    print(f"Estimated time per experiment: 2-4 minutes")
    print(f"Estimated total time: {len(experiments) * 3 / 60:.1f} hours")

    print("\nExperiments to run:")
    for i, exp in enumerate(experiments, 1):
        grid_info = ""
        if exp.momentum_layers is not None:
            grid_info = f" (M={exp.momentum_layers}, C={exp.controller_layers})"
        print(f"  {i:3d}. {exp.name}{grid_info}")

    print(f"\n{'='*70}")
    print("Checking dependencies...")
    print(f"{'='*70}")

    print("\nRequired scripts:")
    scripts_ok = validate_scripts_exist(experiments)

    configs_ok = validate_configs_exist()

    print("\n" + "="*70)
    if scripts_ok and configs_ok:
        print("✓ All dependencies satisfied. Ready to run.")
    else:
        print("✗ Missing dependencies. See above for details.")
        print("\nMissing training scripts may need to be created:")
        print("  - train_hope.py (HOPE + AdamW/Muon)")
        print("  - train_hope_nested.py (HOPE + DeepNestedOptimizer)")
    print("="*70)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive 168M parameter grid search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["baseline", "grid", "winners"],
        required=True,
        help="baseline: Run 4 non-grid experiments. grid: Run 72 grid experiments. winners: Run top nested + Muon baselines.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=STEPS,
        help=f"Training steps per experiment (default: {STEPS})",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration without running experiments",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=RESULTS_DIR,
        help=f"Results directory (default: {RESULTS_DIR})",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=CHECKPOINT_DIR,
        help=f"Checkpoint directory (default: {CHECKPOINT_DIR})",
    )
    args = parser.parse_args()

    # Select experiments based on mode
    if args.mode == "baseline":
        experiments = BASELINE_EXPERIMENTS
    elif args.mode == "grid":
        experiments = generate_grid_experiments()
    elif args.mode == "winners":
        print("\nLoading winners from grid search results...")
        experiments = generate_winners_experiments(args.results_dir)

    print(f"""
{'#'*70}
#  COMPREHENSIVE 168M GRID SEARCH
#  Mode: {args.mode.upper()}
#  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'#'*70}

Configuration:
  - Mode: {args.mode}
  - Experiments: {len(experiments)}
  - Steps: {args.steps}
  - Results: {args.results_dir}
  - Checkpoints: {args.checkpoint_dir}
  - Dry run: {args.dry_run}
""")

    if args.dry_run:
        dry_run_summary(experiments, args.steps)
        return

    # Create directories
    Path(args.results_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Run experiments
    all_results = []
    start_time = time.time()

    # Enable resume for winners mode
    use_resume = (args.mode == "winners")

    for i, experiment in enumerate(experiments, 1):
        print(f"\n[{i}/{len(experiments)}] Starting {experiment.name}...")
        result = run_experiment(
            experiment,
            args.steps,
            args.checkpoint_dir,
            dry_run=args.dry_run,
            resume=use_resume,
        )
        all_results.append(result)

        # Save intermediate results
        results_file = Path(args.results_dir) / f"{args.mode}_results.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2, default=str)

    # Final summary
    total_time = time.time() - start_time
    successful = sum(1 for r in all_results if r.get("status") == "success")
    failed = sum(1 for r in all_results if r.get("status") == "failed")

    print(f"""
{'#'*70}
#  GRID SEARCH COMPLETE
#  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
#  Total time: {total_time/3600:.1f} hours
{'#'*70}

Results:
  - Successful: {successful}/{len(experiments)}
  - Failed: {failed}/{len(experiments)}
  - Results saved to: {args.results_dir}/

""")

    # Print best results
    if successful > 0:
        print("Top results by val_loss:")
        sorted_results = sorted(
            [r for r in all_results if r.get("status") == "success" and r.get("metrics", {}).get("val_loss")],
            key=lambda x: x["metrics"].get("val_loss", float("inf")),
        )
        for i, r in enumerate(sorted_results[:10], 1):
            loss = r["metrics"].get("val_loss", "N/A")
            time_s = r.get("elapsed_seconds", 0)
            print(f"  {i}. {r['name']}: loss={loss:.4f}, time={time_s/60:.1f}min")


if __name__ == "__main__":
    main()
