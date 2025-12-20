#!/usr/bin/env python3
"""
Run all experiments to 5000 steps with best fuzzed hyperparameters.

Experiments:
1. MoE + Muon (baseline) - default config LRs
2. MoE + DeepNestedOptimizer - fuzzed params
3. TitanMAC + Muon - fuzzed params
4. TitanMAC + DeepNestedOptimizer - fuzzed params

Results are written to experiment_results/ with JSON summaries.
"""

import subprocess
import os
import sys
import json
import time
from datetime import datetime
from pathlib import Path

# Configuration
MAX_STEPS = 5000
RESULTS_DIR = Path("experiment_results")
CHECKPOINTS_DIR = Path("checkpoints")

# Best hyperparameters from fuzzing
EXPERIMENTS = {
    "moe_baseline": {
        "script": "train_moe.py",
        "description": "MoE + Muon (baseline)",
        "args": {
            "max_steps": MAX_STEPS,
            "experiment_name": "moe_baseline_5000",
        }
    },
    "moe_nested": {
        "script": "train_moe_nested.py",
        "description": "MoE + DeepNestedOptimizer",
        "args": {
            "max_steps": MAX_STEPS,
            "experiment_name": "moe_nested_5000",
            "base_lr": 0.0004603431029542576,
            "meta_lr": 9.56441656770269e-05,
            "k_unroll": 1,
            "momentum_hidden_dim": 128,
            "controller_hidden_dim": 32,
        }
    },
    "titanmac": {
        "script": "train_titanmac.py",
        "description": "TitanMAC + Muon",
        "args": {
            "max_steps": MAX_STEPS,
            "experiment_name": "titanmac_5000",
            "muon_lr": 0.015533097160640025,
            "adamw_lr": 0.007022787930614363,
        }
    },
    "titanmac_nested": {
        "script": "train_titanmac_nested.py",
        "description": "TitanMAC + DeepNestedOptimizer",
        "args": {
            "max_steps": MAX_STEPS,
            "experiment_name": "titanmac_nested_5000",
            "base_lr": 0.0004072398672148361,
            "meta_lr": 9.089757036214503e-05,
            "k_unroll": 5,
            "momentum_hidden_dim": 64,
            "controller_hidden_dim": 16,
        }
    },
}


def build_command(script: str, args: dict) -> list:
    """Build command line arguments for a training script."""
    cmd = [sys.executable, script]
    for key, value in args.items():
        cmd.append(f"--{key}")
        cmd.append(str(value))
    return cmd


def run_experiment(name: str, config: dict, results_path: Path) -> dict:
    """Run a single experiment with live output streaming."""
    print(f"\n{'='*70}")
    print(f"STARTING EXPERIMENT: {name}")
    print(f"Description: {config['description']}")
    print(f"Script: {config['script']}")
    print(f"Args: {config['args']}")
    print(f"{'='*70}\n")

    cmd = build_command(config["script"], config["args"])
    start_time = time.time()

    result = {
        "name": name,
        "description": config["description"],
        "script": config["script"],
        "args": config["args"],
        "start_time": datetime.now().isoformat(),
        "status": "running",
    }

    # Log file for this experiment
    log_file = results_path / f"{name}.log"

    try:
        # Run with live output streaming AND capture to log file
        with open(log_file, 'w') as f:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # Line buffered
            )

            output_lines = []
            for line in process.stdout:
                print(line, end='')  # Stream to terminal
                f.write(line)  # Write to log
                output_lines.append(line)

            process.wait()
            full_output = ''.join(output_lines)

        elapsed_time = time.time() - start_time
        result["elapsed_seconds"] = elapsed_time
        result["elapsed_minutes"] = elapsed_time / 60
        result["return_code"] = process.returncode
        result["log_file"] = str(log_file)

        if process.returncode == 0:
            result["status"] = "completed"
            # Try to extract metrics from output
            metrics = extract_metrics(full_output)
            result["metrics"] = metrics
            print(f"\nEXPERIMENT {name} COMPLETED in {elapsed_time/60:.2f} minutes")
            if metrics:
                print(f"  Val Loss: {metrics.get('val_loss', 'N/A')}")
                print(f"  Val Accuracy: {metrics.get('val_accuracy', 'N/A')}")
                print(f"  Val Perplexity: {metrics.get('val_perplexity', 'N/A')}")
                print(f"  Peak VRAM: {metrics.get('peak_vram_gb', 'N/A')} GB")
        else:
            result["status"] = "failed"
            print(f"\nEXPERIMENT {name} FAILED (return code {process.returncode})")
            print(f"See log: {log_file}")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        print(f"\nEXPERIMENT {name} ERROR: {e}")
        import traceback
        traceback.print_exc()

    result["end_time"] = datetime.now().isoformat()
    return result


def extract_metrics(stdout: str) -> dict:
    """Extract metrics from training script output."""
    metrics = {}
    lines = stdout.split('\n')

    for line in lines:
        line_lower = line.lower()
        # Look for final results section
        if 'val loss:' in line_lower or 'val_loss:' in line_lower:
            try:
                # Extract number after colon
                parts = line.split(':')
                if len(parts) >= 2:
                    val = float(parts[-1].strip())
                    metrics['val_loss'] = val
            except (ValueError, IndexError):
                pass

        if 'val accuracy:' in line_lower or 'val_accuracy:' in line_lower:
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    val = float(parts[-1].strip())
                    metrics['val_accuracy'] = val
            except (ValueError, IndexError):
                pass

        if 'val perplexity:' in line_lower or 'val_perplexity:' in line_lower:
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    val = float(parts[-1].strip())
                    metrics['val_perplexity'] = val
            except (ValueError, IndexError):
                pass

        if 'peak vram:' in line_lower or 'peak_vram' in line_lower:
            try:
                parts = line.split(':')
                if len(parts) >= 2:
                    val_str = parts[-1].strip().replace('GB', '').strip()
                    val = float(val_str)
                    metrics['peak_vram_gb'] = val
            except (ValueError, IndexError):
                pass

    return metrics


def save_results(all_results: list, summary_path: Path):
    """Save all results to JSON and create summary."""
    # Save full results
    results_json = summary_path / "full_results.json"
    with open(results_json, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    # Create human-readable summary
    summary_txt = summary_path / "summary.txt"
    with open(summary_txt, 'w') as f:
        f.write("="*70 + "\n")
        f.write("EXPERIMENT RESULTS SUMMARY\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        # Get max_steps from first result's args
        steps_used = all_results[0]['args'].get('max_steps', 'N/A') if all_results else 'N/A'
        f.write(f"Max Steps: {steps_used}\n")
        f.write("="*70 + "\n\n")

        # Results table
        f.write(f"{'Experiment':<25} {'Val Loss':<12} {'Val Acc':<12} {'Perplexity':<12} {'VRAM (GB)':<12} {'Status':<10}\n")
        f.write("-"*85 + "\n")

        for r in all_results:
            metrics = r.get('metrics', {})
            val_loss = metrics.get('val_loss', 'N/A')
            val_acc = metrics.get('val_accuracy', 'N/A')
            ppl = metrics.get('val_perplexity', 'N/A')
            vram = metrics.get('peak_vram_gb', 'N/A')
            status = r.get('status', 'unknown')

            # Format values
            val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else str(val_loss)
            val_acc_str = f"{val_acc:.4f}" if isinstance(val_acc, float) else str(val_acc)
            ppl_str = f"{ppl:.2f}" if isinstance(ppl, float) else str(ppl)
            vram_str = f"{vram:.2f}" if isinstance(vram, float) else str(vram)

            f.write(f"{r['name']:<25} {val_loss_str:<12} {val_acc_str:<12} {ppl_str:<12} {vram_str:<12} {status:<10}\n")

        f.write("\n" + "="*70 + "\n")
        f.write("DETAILED RESULTS\n")
        f.write("="*70 + "\n\n")

        for r in all_results:
            f.write(f"\n{'-'*50}\n")
            f.write(f"Experiment: {r['name']}\n")
            f.write(f"Description: {r['description']}\n")
            f.write(f"Status: {r['status']}\n")
            f.write(f"Script: {r['script']}\n")
            f.write(f"Arguments:\n")
            for k, v in r.get('args', {}).items():
                f.write(f"  {k}: {v}\n")

            if r.get('elapsed_minutes'):
                f.write(f"Runtime: {r['elapsed_minutes']:.2f} minutes\n")

            metrics = r.get('metrics', {})
            if metrics:
                f.write(f"Metrics:\n")
                for k, v in metrics.items():
                    if isinstance(v, float):
                        f.write(f"  {k}: {v:.6f}\n")
                    else:
                        f.write(f"  {k}: {v}\n")

            if r.get('status') == 'failed' and r.get('stderr'):
                f.write(f"Error (last 500 chars):\n{r['stderr'][-500:]}\n")

    print(f"\nResults saved to:")
    print(f"  {results_json}")
    print(f"  {summary_txt}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run all experiments")
    parser.add_argument("--only", type=str, help="Run only specific experiment (comma-separated)")
    parser.add_argument("--skip", type=str, help="Skip specific experiments (comma-separated)")
    parser.add_argument("--steps", type=int, default=5000, help="Max steps (default: 5000)")
    args = parser.parse_args()

    # Update max steps for all experiments
    max_steps = args.steps
    for exp in EXPERIMENTS.values():
        exp["args"]["max_steps"] = max_steps

    # Filter experiments
    experiments_to_run = list(EXPERIMENTS.keys())

    if args.only:
        only_list = [x.strip() for x in args.only.split(",")]
        experiments_to_run = [e for e in experiments_to_run if e in only_list]

    if args.skip:
        skip_list = [x.strip() for x in args.skip.split(",")]
        experiments_to_run = [e for e in experiments_to_run if e not in skip_list]

    if not experiments_to_run:
        print("No experiments to run!")
        return

    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = RESULTS_DIR / f"run_{timestamp}"
    results_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#'*70}")
    print(f"# RUNNING {len(experiments_to_run)} EXPERIMENTS")
    print(f"# Max Steps: {max_steps}")
    print(f"# Results will be saved to: {results_path}")
    print(f"{'#'*70}\n")

    print("Experiments to run:")
    for i, name in enumerate(experiments_to_run, 1):
        print(f"  {i}. {name}: {EXPERIMENTS[name]['description']}")
    print()

    # Run experiments
    all_results = []
    total_start = time.time()

    for i, name in enumerate(experiments_to_run, 1):
        print(f"\n[{i}/{len(experiments_to_run)}] Running {name}...")
        result = run_experiment(name, EXPERIMENTS[name], results_path)
        all_results.append(result)

        # Save intermediate results after each experiment
        save_results(all_results, results_path)

    total_elapsed = time.time() - total_start

    # Final summary
    print(f"\n{'#'*70}")
    print(f"# ALL EXPERIMENTS COMPLETE")
    print(f"# Total runtime: {total_elapsed/60:.2f} minutes ({total_elapsed/3600:.2f} hours)")
    print(f"{'#'*70}\n")

    # Print summary table
    print(f"\n{'='*85}")
    print("SUMMARY")
    print(f"{'='*85}")
    print(f"{'Experiment':<25} {'Val Loss':<12} {'Val Acc':<12} {'Perplexity':<12} {'Status':<10}")
    print("-"*85)

    for r in all_results:
        metrics = r.get('metrics', {})
        val_loss = metrics.get('val_loss', 'N/A')
        val_acc = metrics.get('val_accuracy', 'N/A')
        ppl = metrics.get('val_perplexity', 'N/A')
        status = r.get('status', 'unknown')

        val_loss_str = f"{val_loss:.4f}" if isinstance(val_loss, float) else str(val_loss)
        val_acc_str = f"{val_acc:.4f}" if isinstance(val_acc, float) else str(val_acc)
        ppl_str = f"{ppl:.2f}" if isinstance(ppl, float) else str(ppl)

        print(f"{r['name']:<25} {val_loss_str:<12} {val_acc_str:<12} {ppl_str:<12} {status:<10}")

    print(f"{'='*85}\n")
    print(f"Full results: {results_path}/summary.txt")


if __name__ == "__main__":
    main()
