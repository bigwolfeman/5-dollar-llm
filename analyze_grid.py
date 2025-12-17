#!/usr/bin/env python3
"""
Grid Search Analysis with Baseline Comparison

Generates comprehensive charts comparing nested optimizer configurations
against Muon+AdamW baseline, with best values annotated on each chart.

Usage:
    python analyze_grid.py --results_dir fuzz_results_grid12 --baseline_dir checkpoints_grid12
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

# Colorblind-friendly palette
COLORS = {
    'moe_nested': '#648FFF',      # Blue
    'titanmac_nested': '#DC267F',  # Magenta
    'moe_baseline': '#009E73',     # Teal
    'titanmac_baseline': '#FE6100', # Orange
    'best': '#FFB000',             # Gold (for highlighting best)
}

MARKERS = ['o', 's', '^', 'D', 'v', 'P', 'X', 'h', '*', '+']


def load_grid_results(results_dir: Path) -> Dict:
    """Load all grid search results."""
    results = {"moe": [], "titanmac": []}

    # Load MoE results
    moe_file = results_dir / "moe_nested_depth" / "all_trials.json"
    if moe_file.exists():
        with open(moe_file) as f:
            results["moe"] = json.load(f)
        print(f"Loaded {len(results['moe'])} MoE trials")

    # Load TitanMAC results
    titan_file = results_dir / "titanmac_nested_depth" / "all_trials.json"
    if titan_file.exists():
        with open(titan_file) as f:
            results["titanmac"] = json.load(f)
        print(f"Loaded {len(results['titanmac'])} TitanMAC trials")

    return results


def load_baseline_results(baseline_dir: Path) -> Dict:
    """Load baseline Muon+AdamW results."""
    baselines = {}

    for name in ["moe_baseline_600", "titanmac_baseline_600"]:
        metrics_file = baseline_dir / name / "metrics.json"
        if metrics_file.exists():
            with open(metrics_file) as f:
                data = json.load(f)
            baselines[name] = {
                "loss": data.get("final_metrics", {}).get("val_loss"),
                "accuracy": data.get("final_metrics", {}).get("val_accuracy"),
                "history": data.get("history", {}),
            }
            print(f"Loaded baseline: {name} -> loss={baselines[name]['loss']:.4f}")

    return baselines


def extract_grid_data(trials: List[Dict]) -> Tuple[np.ndarray, Dict]:
    """Extract grid data from trials."""
    # Find grid dimensions
    max_m = max(t["depth_params"]["momentum_num_layers"] for t in trials) if trials else 0
    max_c = max(t["depth_params"]["controller_num_layers"] for t in trials) if trials else 0

    # Create grid (NaN for missing values)
    grid = np.full((max_m, max_c), np.nan)

    # Track best
    best = {"loss": float("inf"), "m": 0, "c": 0, "trial": None}

    for trial in trials:
        m = trial["depth_params"]["momentum_num_layers"]
        c = trial["depth_params"]["controller_num_layers"]
        loss = trial.get("metrics", {}).get("val_loss")

        if loss is not None and loss != float("inf"):
            grid[m-1, c-1] = loss

            if loss < best["loss"]:
                best["loss"] = loss
                best["m"] = m
                best["c"] = c
                best["trial"] = trial

    return grid, best


def annotate_best(ax, best: Dict, baseline_loss: Optional[float] = None):
    """Add best value annotation to a chart."""
    text_parts = [f"Best: {best['loss']:.4f}"]
    text_parts.append(f"(M={best['m']}, C={best['c']})")

    if baseline_loss is not None:
        diff = best['loss'] - baseline_loss
        if diff < 0:
            text_parts.append(f"vs baseline: {diff:.4f} better")
        else:
            text_parts.append(f"vs baseline: +{diff:.4f} worse")

    # Add text box in upper right
    text = "\n".join(text_parts)
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.98, text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)


def plot_heatmap(ax, grid: np.ndarray, title: str, best: Dict, baseline_loss: Optional[float] = None):
    """Plot a heatmap with best value annotation."""
    # Mask NaN values
    masked_grid = np.ma.masked_invalid(grid)

    # Plot heatmap
    im = ax.imshow(masked_grid, cmap='viridis_r', aspect='auto', origin='lower')

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Val Loss')

    # Mark best with a star
    if best["m"] > 0 and best["c"] > 0:
        ax.plot(best["c"]-1, best["m"]-1, marker='*', color=COLORS['best'],
                markersize=20, markeredgecolor='black', markeredgewidth=1)

    # Labels
    ax.set_xlabel('Controller Layers')
    ax.set_ylabel('Momentum Layers')
    ax.set_title(title)

    # Tick labels
    ax.set_xticks(range(grid.shape[1]))
    ax.set_xticklabels(range(1, grid.shape[1]+1))
    ax.set_yticks(range(grid.shape[0]))
    ax.set_yticklabels(range(1, grid.shape[0]+1))

    # Add best annotation
    annotate_best(ax, best, baseline_loss)


def plot_trajectory_overlay(ax, trials: List[Dict], baseline: Optional[Dict],
                           title: str, best: Dict):
    """Plot trajectory overlay with all configs and baseline."""
    # Plot each trial's trajectory
    for trial in trials:
        traj = trial.get("metrics", {}).get("trajectory", {})
        steps = traj.get("steps", [])
        losses = traj.get("val_losses", [])

        if steps and losses:
            m = trial["depth_params"]["momentum_num_layers"]
            c = trial["depth_params"]["controller_num_layers"]
            total_depth = m + c

            # Color by total depth
            alpha = 0.3 + 0.05 * min(total_depth, 10)
            color = plt.cm.viridis(total_depth / 24)  # Normalize to 12+12 max

            # Highlight best config
            if m == best["m"] and c == best["c"]:
                ax.plot(steps, losses, color=COLORS['best'], linewidth=3,
                       alpha=1.0, label=f"Best: M={m},C={c}", zorder=10)
            else:
                ax.plot(steps, losses, color=color, linewidth=0.8, alpha=alpha)

    # Plot baseline
    if baseline and baseline.get("history"):
        hist = baseline["history"]
        steps = hist.get("steps", [])
        losses = hist.get("val_losses", [])
        if steps and losses:
            ax.plot(steps, losses, color='red', linewidth=2.5, linestyle='--',
                   label=f"Muon+AdamW: {baseline['loss']:.4f}", zorder=11)

    ax.set_xlabel('Step')
    ax.set_ylabel('Val Loss')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add best annotation
    annotate_best(ax, best, baseline.get("loss") if baseline else None)


def plot_depth_vs_loss(ax, trials: List[Dict], title: str, best: Dict,
                      baseline_loss: Optional[float] = None):
    """Scatter plot of total depth vs loss."""
    depths = []
    losses = []
    is_best = []

    for trial in trials:
        m = trial["depth_params"]["momentum_num_layers"]
        c = trial["depth_params"]["controller_num_layers"]
        loss = trial.get("metrics", {}).get("val_loss")

        if loss is not None and loss != float("inf") and loss < 10:
            depths.append(m + c)
            losses.append(loss)
            is_best.append(m == best["m"] and c == best["c"])

    # Plot all points
    ax.scatter(depths, losses, alpha=0.6, c='steelblue', s=50, edgecolors='white', linewidth=0.5)

    # Highlight best
    best_idx = [i for i, b in enumerate(is_best) if b]
    if best_idx:
        ax.scatter([depths[i] for i in best_idx], [losses[i] for i in best_idx],
                  c=COLORS['best'], s=150, marker='*', edgecolors='black',
                  linewidth=1, zorder=10, label=f"Best: {best['loss']:.4f}")

    # Add baseline line
    if baseline_loss is not None:
        ax.axhline(y=baseline_loss, color='red', linestyle='--', linewidth=2,
                  label=f"Muon+AdamW: {baseline_loss:.4f}")

    ax.set_xlabel('Total Depth (M + C)')
    ax.set_ylabel('Val Loss')
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    # Add best annotation
    annotate_best(ax, best, baseline_loss)


def plot_comparison_bar(ax, moe_best: Dict, titan_best: Dict,
                       moe_baseline: Optional[float], titan_baseline: Optional[float]):
    """Bar chart comparing best configs vs baselines."""
    labels = ['MoE\nNested', 'MoE\nBaseline', 'TitanMAC\nNested', 'TitanMAC\nBaseline']
    values = [
        moe_best["loss"] if moe_best["loss"] < float("inf") else 0,
        moe_baseline if moe_baseline else 0,
        titan_best["loss"] if titan_best["loss"] < float("inf") else 0,
        titan_baseline if titan_baseline else 0,
    ]
    colors = [COLORS['moe_nested'], COLORS['moe_baseline'],
              COLORS['titanmac_nested'], COLORS['titanmac_baseline']]

    bars = ax.bar(labels, values, color=colors, edgecolor='black', linewidth=1)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                   f'{val:.4f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    ax.set_ylabel('Val Loss')
    ax.set_title('Best Configurations Comparison')
    ax.grid(True, alpha=0.3, axis='y')

    # Find overall best
    valid_values = [(v, l) for v, l in zip(values, labels) if v > 0]
    if valid_values:
        best_val, best_label = min(valid_values, key=lambda x: x[0])
        ax.text(0.02, 0.98, f"Overall Best: {best_label.replace(chr(10), ' ')} = {best_val:.4f}",
               transform=ax.transAxes, fontsize=11, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))


def plot_placeholder(ax, title: str, message: str = "Not applicable"):
    """Create a placeholder chart for consistent layout."""
    ax.text(0.5, 0.5, message, transform=ax.transAxes, fontsize=14,
            verticalalignment='center', horizontalalignment='center',
            color='gray', style='italic')
    ax.set_title(title, color='gray')
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color('lightgray')


def generate_per_experiment_charts(name: str, trials: List[Dict], baseline: Optional[Dict],
                                   output_dir: Path, is_nested: bool = True):
    """
    Generate consistent 4-panel chart for each experiment.

    Layout (same for all experiments):
    - Panel 1: Heatmap (12x12) - only for nested, placeholder for baseline
    - Panel 2: Trajectory overlay
    - Panel 3: Depth vs Loss scatter - only for nested, placeholder for baseline
    - Panel 4: Summary stats box
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle(f'{name} Analysis', fontsize=16, fontweight='bold')

    baseline_loss = baseline.get("loss") if baseline else None

    if is_nested and trials:
        grid, best = extract_grid_data(trials)

        # Panel 1: Heatmap
        plot_heatmap(axes[0, 0], grid, f"Val Loss by Depth Configuration", best, baseline_loss)

        # Panel 2: Trajectory overlay
        plot_trajectory_overlay(axes[0, 1], trials, baseline,
                               "Training Trajectories", best)

        # Panel 3: Depth vs Loss
        plot_depth_vs_loss(axes[1, 0], trials, "Total Depth vs Loss", best, baseline_loss)

        # Panel 4: Summary
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = f"""
SUMMARY

Best Configuration:
  Momentum Layers: {best['m']}
  Controller Layers: {best['c']}
  Total Depth: {best['m'] + best['c']}

Best Val Loss: {best['loss']:.4f}
"""
        if baseline_loss:
            diff = best['loss'] - baseline_loss
            summary_text += f"""
Baseline (Muon+AdamW): {baseline_loss:.4f}
Difference: {diff:+.4f} ({'better' if diff < 0 else 'worse'})
Improvement: {-diff/baseline_loss*100:.1f}%
"""
        summary_text += f"""
Total Configs Tested: {len(trials)}
Grid Coverage: {len(set((t['depth_params']['momentum_num_layers'],
                          t['depth_params']['controller_num_layers'])
                         for t in trials))}/144
"""
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax.set_title("Summary Statistics")

    else:
        # Baseline - only trajectory applies
        # Panel 1: Placeholder
        plot_placeholder(axes[0, 0], "Depth Heatmap", "N/A for baseline\n(single configuration)")

        # Panel 2: Trajectory (if available)
        if baseline and baseline.get("history"):
            hist = baseline["history"]
            steps = hist.get("steps", [])
            losses = hist.get("val_losses", [])
            if steps and losses:
                axes[0, 1].plot(steps, losses, color='red', linewidth=2.5,
                               marker='o', markersize=4, markevery=max(1, len(steps)//10))
                axes[0, 1].set_xlabel('Step')
                axes[0, 1].set_ylabel('Val Loss')
                axes[0, 1].set_title('Training Trajectory')
                axes[0, 1].grid(True, alpha=0.3)
                # Annotate final value
                final_loss = losses[-1] if losses else baseline_loss
                axes[0, 1].text(0.98, 0.98, f"Final: {final_loss:.4f}",
                               transform=axes[0, 1].transAxes, fontsize=11,
                               verticalalignment='top', horizontalalignment='right',
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            else:
                plot_placeholder(axes[0, 1], "Training Trajectory", "No trajectory data")
        else:
            plot_placeholder(axes[0, 1], "Training Trajectory", "No data available")

        # Panel 3: Placeholder
        plot_placeholder(axes[1, 0], "Depth vs Loss", "N/A for baseline\n(single configuration)")

        # Panel 4: Summary
        ax = axes[1, 1]
        ax.axis('off')
        summary_text = f"""
SUMMARY

Configuration: Muon + AdamW
(Default hyperparameters)

Val Loss: {baseline_loss:.4f if baseline_loss else 'N/A'}
"""
        ax.text(0.1, 0.9, summary_text, transform=ax.transAxes, fontsize=12,
               verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        ax.set_title("Summary Statistics")

    plt.tight_layout()
    filename = f"{name.lower().replace(' ', '_').replace('+', '_')}.png"
    plt.savefig(output_dir / filename, dpi=150)
    plt.close()
    print(f"Saved: {output_dir / filename}")


def generate_report(results: Dict, baselines: Dict, output_dir: Path):
    """Generate full analysis report with charts."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)

    # Extract grid data
    moe_grid, moe_best = extract_grid_data(results.get("moe", []))
    titan_grid, titan_best = extract_grid_data(results.get("titanmac", []))

    moe_baseline_loss = baselines.get("moe_baseline_600", {}).get("loss")
    titan_baseline_loss = baselines.get("titanmac_baseline_600", {}).get("loss")

    print(f"\nMoE Best: M={moe_best['m']}, C={moe_best['c']} -> {moe_best['loss']:.4f}")
    print(f"TitanMAC Best: M={titan_best['m']}, C={titan_best['c']} -> {titan_best['loss']:.4f}")
    if moe_baseline_loss:
        print(f"MoE Baseline: {moe_baseline_loss:.4f}")
    if titan_baseline_loss:
        print(f"TitanMAC Baseline: {titan_baseline_loss:.4f}")

    # Generate consistent per-experiment charts (same layout for all)
    print("\nGenerating per-experiment charts (consistent layout)...")

    # 1. MoE Nested
    generate_per_experiment_charts(
        "MoE Nested Optimizer",
        results.get("moe", []),
        baselines.get("moe_baseline_600"),
        output_dir,
        is_nested=True
    )

    # 2. TitanMAC Nested
    generate_per_experiment_charts(
        "TitanMAC Nested Optimizer",
        results.get("titanmac", []),
        baselines.get("titanmac_baseline_600"),
        output_dir,
        is_nested=True
    )

    # 3. MoE Baseline (Muon+AdamW)
    generate_per_experiment_charts(
        "MoE Muon+AdamW Baseline",
        [],  # No nested trials
        baselines.get("moe_baseline_600"),
        output_dir,
        is_nested=False
    )

    # 4. TitanMAC Baseline (Muon+AdamW)
    generate_per_experiment_charts(
        "TitanMAC Muon+AdamW Baseline",
        [],  # No nested trials
        baselines.get("titanmac_baseline_600"),
        output_dir,
        is_nested=False
    )

    # Combined comparison charts
    print("\nGenerating combined comparison charts...")

    # Chart 1: Side-by-side heatmaps
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plot_heatmap(axes[0], moe_grid, "MoE Nested: Val Loss by Depth", moe_best, moe_baseline_loss)
    plot_heatmap(axes[1], titan_grid, "TitanMAC Nested: Val Loss by Depth", titan_best, titan_baseline_loss)
    plt.tight_layout()
    plt.savefig(output_dir / "heatmaps_combined.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'heatmaps_combined.png'}")

    # Chart 2: Trajectory overlays combined
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    plot_trajectory_overlay(axes[0], results.get("moe", []),
                           baselines.get("moe_baseline_600"),
                           "MoE: All Configs vs Baseline", moe_best)
    plot_trajectory_overlay(axes[1], results.get("titanmac", []),
                           baselines.get("titanmac_baseline_600"),
                           "TitanMAC: All Configs vs Baseline", titan_best)
    plt.tight_layout()
    plt.savefig(output_dir / "trajectory_overlays_combined.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'trajectory_overlays_combined.png'}")

    # Chart 3: Comparison bar chart
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_comparison_bar(ax, moe_best, titan_best, moe_baseline_loss, titan_baseline_loss)
    plt.tight_layout()
    plt.savefig(output_dir / "comparison.png", dpi=150)
    plt.close()
    print(f"Saved: {output_dir / 'comparison.png'}")

    # Save summary JSON
    summary = {
        "moe": {
            "best_config": {"momentum_layers": moe_best["m"], "controller_layers": moe_best["c"]},
            "best_loss": moe_best["loss"],
            "baseline_loss": moe_baseline_loss,
            "improvement_vs_baseline": (moe_baseline_loss - moe_best["loss"]) if moe_baseline_loss else None,
            "total_configs_tested": len(results.get("moe", [])),
        },
        "titanmac": {
            "best_config": {"momentum_layers": titan_best["m"], "controller_layers": titan_best["c"]},
            "best_loss": titan_best["loss"],
            "baseline_loss": titan_baseline_loss,
            "improvement_vs_baseline": (titan_baseline_loss - titan_best["loss"]) if titan_baseline_loss else None,
            "total_configs_tested": len(results.get("titanmac", [])),
        },
    }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {output_dir / 'summary.json'}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Analyze grid search results")
    parser.add_argument("--results_dir", type=str, required=True, help="Grid results directory")
    parser.add_argument("--baseline_dir", type=str, default="checkpoints_grid12", help="Baseline checkpoints directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory for charts")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    baseline_dir = Path(args.baseline_dir)
    output_dir = Path(args.output) if args.output else results_dir / "analysis"

    print(f"Loading results from: {results_dir}")
    print(f"Loading baselines from: {baseline_dir}")

    # Load data
    results = load_grid_results(results_dir)
    baselines = load_baseline_results(baseline_dir)

    # Generate report
    summary = generate_report(results, baselines, output_dir)

    print(f"\nAnalysis complete! Charts saved to: {output_dir}")


# Need matplotlib for plotting
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:
    print("Warning: matplotlib not available. Install with: pip install matplotlib")


if __name__ == "__main__":
    main()
