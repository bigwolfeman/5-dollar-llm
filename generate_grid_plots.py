#!/usr/bin/env python3
"""
Generate comprehensive plots from grid search results.

Usage:
    python generate_grid_plots.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# Import plotting utilities
from utils.plotting import (
    MODEL_STYLES,
    OPTIMIZER_STYLES,
    FIGURE_DEFAULTS,
    METRIC_LABELS,
    load_grid_results,
    plot_grid_heatmap,
    _save_figure,
    _setup_axis,
)

RESULTS_DIR = Path("grid_results_168m")
OUTPUT_DIR = Path("plots_168m")


def load_all_results():
    """Load baseline and grid results."""
    baseline_file = RESULTS_DIR / "baseline_results.json"
    grid_file = RESULTS_DIR / "grid_results.json"

    with open(baseline_file) as f:
        baseline = json.load(f)

    with open(grid_file) as f:
        grid = json.load(f)

    return baseline, grid


def plot_baseline_comparison(baseline_results, output_dir):
    """
    Create bar charts comparing baseline results.
    """
    # Filter successful experiments
    successful = [r for r in baseline_results if r.get('status') == 'success']

    if not successful:
        print("No successful baseline experiments found")
        return

    # Extract data
    names = []
    val_losses = []
    val_accuracies = []
    times = []
    vrams = []
    colors = []

    for r in successful:
        name = r['name']
        metrics = r.get('metrics', {})

        # Determine model type for coloring
        if 'moe' in name.lower():
            model = 'MoE'
        elif 'titan' in name.lower():
            model = 'TitanMAC'
        else:
            model = 'HOPE'

        # Determine optimizer for display name
        if 'muon' in name.lower():
            opt = 'Muon'
        elif 'nested' in name.lower():
            opt = 'Nested'
        else:
            opt = 'AdamW'

        display_name = f"{model}\n{opt}"
        names.append(display_name)

        val_losses.append(metrics.get('val_loss', float('nan')))
        val_accuracies.append(metrics.get('val_accuracy', 0) * 100)  # Convert to %
        times.append(r.get('elapsed_seconds', 0) / 60)  # Convert to minutes
        vrams.append(metrics.get('peak_vram_gb', 0))
        colors.append(MODEL_STYLES[model]['color'])

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Baseline Comparison: MoE vs TitanMAC (600 steps)',
                 fontsize=18, fontweight='bold', y=0.98)

    x_pos = np.arange(len(names))
    bar_width = 0.6

    # Plot 1: Validation Loss
    ax = axes[0, 0]
    bars = ax.bar(x_pos, val_losses, bar_width, color=colors,
                  edgecolor='black', linewidth=1.5)
    best_idx = np.nanargmin(val_losses)
    bars[best_idx].set_edgecolor('#FFB000')
    bars[best_idx].set_linewidth(3)
    for i, (bar, val) in enumerate(zip(bars, val_losses)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
               f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Validation Loss', fontsize=13)
    ax.set_title('Validation Loss (lower is better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 2: Validation Accuracy
    ax = axes[0, 1]
    bars = ax.bar(x_pos, val_accuracies, bar_width, color=colors,
                  edgecolor='black', linewidth=1.5)
    best_idx = np.nanargmax(val_accuracies)
    bars[best_idx].set_edgecolor('#FFB000')
    bars[best_idx].set_linewidth(3)
    for bar, val in zip(bars, val_accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Accuracy (%)', fontsize=13)
    ax.set_title('Validation Accuracy (higher is better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 3: Training Time
    ax = axes[1, 0]
    bars = ax.bar(x_pos, times, bar_width, color=colors,
                  edgecolor='black', linewidth=1.5)
    best_idx = np.nanargmin(times)
    bars[best_idx].set_edgecolor('#FFB000')
    bars[best_idx].set_linewidth(3)
    for bar, val in zip(bars, times):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
               f'{val:.1f}m', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('Time (minutes)', fontsize=13)
    ax.set_title('Training Time (lower is better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    # Plot 4: Peak VRAM
    ax = axes[1, 1]
    bars = ax.bar(x_pos, vrams, bar_width, color=colors,
                  edgecolor='black', linewidth=1.5)
    best_idx = np.nanargmin(vrams)
    bars[best_idx].set_edgecolor('#FFB000')
    bars[best_idx].set_linewidth(3)
    for bar, val in zip(bars, vrams):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
               f'{val:.1f}GB', ha='center', va='bottom', fontsize=11, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, fontsize=11)
    ax.set_ylabel('VRAM (GB)', fontsize=13)
    ax.set_title('Peak VRAM Usage (lower is better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    _save_figure(fig, output_dir, 'baseline_comparison')
    plt.close(fig)


def plot_grid_heatmaps(grid_results, output_dir):
    """
    Create heatmaps for MoE and TitanMAC grid search.
    """
    for model in ['moe', 'titanmac']:
        grid_data = load_grid_results(RESULTS_DIR / "grid_results.json", model_filter=model)

        if not grid_data['trials']:
            print(f"No trials found for {model}")
            continue

        print(f"\nGenerating heatmaps for {model.upper()} ({len(grid_data['trials'])} trials)...")

        # Val loss heatmap
        model_name = 'TitanMAC' if model == 'titanmac' else 'MoE'
        plot_loss_heatmap(grid_data, output_dir, model_name)

        # Val accuracy heatmap
        plot_accuracy_heatmap(grid_data, output_dir, model_name)


def plot_loss_heatmap(grid_data, output_dir, model_type):
    """Plot loss heatmap (lower is better)."""
    trials = grid_data.get('trials', [])
    if not trials:
        return

    # Extract grid dimensions
    momentum_layers = sorted(set(t['depth_params']['momentum_num_layers'] for t in trials))
    controller_layers = sorted(set(t['depth_params']['controller_num_layers'] for t in trials))

    # Create grid matrix
    grid = np.full((len(momentum_layers), len(controller_layers)), np.nan)

    best_value = float('inf')
    best_m, best_c = 0, 0

    for trial in trials:
        m = trial['depth_params']['momentum_num_layers']
        c = trial['depth_params']['controller_num_layers']
        value = trial.get('metrics', {}).get('val_loss', float('nan'))

        if value is not None and not np.isnan(value):
            m_idx = momentum_layers.index(m)
            c_idx = controller_layers.index(c)
            grid[m_idx, c_idx] = value

            if value < best_value:
                best_value = value
                best_m, best_c = m, c

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(grid, cmap='RdYlGn_r', aspect='auto', interpolation='nearest')

    ax.set_xticks(np.arange(len(controller_layers)))
    ax.set_yticks(np.arange(len(momentum_layers)))
    ax.set_xticklabels(controller_layers)
    ax.set_yticklabels(momentum_layers)

    ax.set_xlabel('Controller Layers', fontsize=14)
    ax.set_ylabel('Momentum Layers', fontsize=14)
    ax.set_title(f'{model_type} Grid Search: Validation Loss',
                fontsize=16, fontweight='bold', pad=15)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Validation Loss', fontsize=14)

    # Annotate cells
    for i in range(len(momentum_layers)):
        for j in range(len(controller_layers)):
            value = grid[i, j]
            if not np.isnan(value):
                text_color = 'white' if value > np.nanmean(grid) else 'black'
                ax.text(j, i, f'{value:.2f}', ha='center', va='center',
                       color=text_color, fontsize=9, fontweight='bold')

    # Highlight best
    if best_m > 0 and best_c > 0:
        best_m_idx = momentum_layers.index(best_m)
        best_c_idx = controller_layers.index(best_c)
        rect = mpatches.Rectangle((best_c_idx - 0.5, best_m_idx - 0.5), 1, 1,
                                  linewidth=3, edgecolor='#FFB000',
                                  facecolor='none')
        ax.add_patch(rect)
        ax.text(len(controller_layers) / 2, -1.5,
               f'Best: M={best_m}, C={best_c} (loss={best_value:.4f})',
               ha='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='#FFB000', alpha=0.3))

    _save_figure(fig, output_dir, f'grid_heatmap_loss_{model_type.lower()}')
    plt.close(fig)


def plot_accuracy_heatmap(grid_data, output_dir, model_type):
    """Plot accuracy heatmap (separate because higher is better)."""
    trials = grid_data.get('trials', [])
    if not trials:
        return

    # Extract grid dimensions
    momentum_layers = sorted(set(t['depth_params']['momentum_num_layers'] for t in trials))
    controller_layers = sorted(set(t['depth_params']['controller_num_layers'] for t in trials))

    # Create grid matrix
    grid = np.full((len(momentum_layers), len(controller_layers)), np.nan)

    best_value = float('-inf')
    best_m, best_c = 0, 0

    for trial in trials:
        m = trial['depth_params']['momentum_num_layers']
        c = trial['depth_params']['controller_num_layers']
        value = trial.get('metrics', {}).get('val_accuracy', 0) * 100  # Convert to %

        if value is not None and not np.isnan(value):
            m_idx = momentum_layers.index(m)
            c_idx = controller_layers.index(c)
            grid[m_idx, c_idx] = value

            if value > best_value:
                best_value = value
                best_m, best_c = m, c

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(grid, cmap='RdYlGn', aspect='auto', interpolation='nearest')

    ax.set_xticks(np.arange(len(controller_layers)))
    ax.set_yticks(np.arange(len(momentum_layers)))
    ax.set_xticklabels(controller_layers)
    ax.set_yticklabels(momentum_layers)

    ax.set_xlabel('Controller Layers', fontsize=14)
    ax.set_ylabel('Momentum Layers', fontsize=14)
    ax.set_title(f'{model_type} Grid Search: Validation Accuracy (%)',
                fontsize=16, fontweight='bold', pad=15)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Accuracy (%)', fontsize=14)

    # Annotate cells
    for i in range(len(momentum_layers)):
        for j in range(len(controller_layers)):
            value = grid[i, j]
            if not np.isnan(value):
                text_color = 'white' if value < np.nanmean(grid) else 'black'
                ax.text(j, i, f'{value:.1f}', ha='center', va='center',
                       color=text_color, fontsize=9, fontweight='bold')

    # Highlight best
    if best_m > 0 and best_c > 0:
        best_m_idx = momentum_layers.index(best_m)
        best_c_idx = controller_layers.index(best_c)
        rect = mpatches.Rectangle((best_c_idx - 0.5, best_m_idx - 0.5), 1, 1,
                                  linewidth=3, edgecolor='#FFB000',
                                  facecolor='none')
        ax.add_patch(rect)
        ax.text(len(controller_layers) / 2, -1.5,
               f'Best: M={best_m}, C={best_c} (acc={best_value:.1f}%)',
               ha='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='#FFB000', alpha=0.3))

    _save_figure(fig, output_dir, f'grid_heatmap_accuracy_{model_type.lower()}')
    plt.close(fig)


def plot_efficiency_scatter(baseline_results, grid_results, output_dir):
    """
    Create efficiency scatter plots (time vs loss, VRAM vs loss).
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Efficiency Analysis: Resource vs Performance',
                 fontsize=16, fontweight='bold', y=0.98)

    # Combine all results
    all_results = []

    for r in baseline_results:
        if r.get('status') != 'success':
            continue
        name = r['name']
        metrics = r.get('metrics', {})

        if 'moe' in name.lower():
            model = 'MoE'
        elif 'titan' in name.lower():
            model = 'TitanMAC'
        else:
            continue

        all_results.append({
            'name': name,
            'model': model,
            'optimizer': 'Nested' if 'nested' in name else 'Muon',
            'time': r.get('elapsed_seconds', 0) / 60,
            'vram': metrics.get('peak_vram_gb', 0),
            'val_loss': metrics.get('val_loss', float('nan')),
            'is_grid': False,
        })

    for r in grid_results:
        if r.get('status') != 'success':
            continue
        name = r['name']
        metrics = r.get('metrics', {})

        if 'moe' in name.lower():
            model = 'MoE'
        elif 'titan' in name.lower():
            model = 'TitanMAC'
        else:
            continue

        all_results.append({
            'name': name,
            'model': model,
            'optimizer': 'Nested',
            'time': r.get('elapsed_seconds', 0) / 60,
            'vram': metrics.get('peak_vram_gb', 0),
            'val_loss': metrics.get('val_loss', float('nan')),
            'is_grid': True,
        })

    # Plot 1: Time vs Val Loss
    ax = axes[0]
    for model in ['MoE', 'TitanMAC']:
        for opt in ['Muon', 'Nested']:
            subset = [r for r in all_results if r['model'] == model and r['optimizer'] == opt]
            if not subset:
                continue

            times = [r['time'] for r in subset]
            losses = [r['val_loss'] for r in subset]

            style = MODEL_STYLES[model]
            marker = 'o' if opt == 'Muon' else 's'
            alpha = 1.0 if opt == 'Muon' else 0.5
            size = 150 if opt == 'Muon' else 50

            ax.scatter(times, losses, c=style['color'], marker=marker,
                      s=size, alpha=alpha, edgecolors='black', linewidth=1,
                      label=f'{model} + {opt}')

    ax.set_xlabel('Training Time (minutes)', fontsize=13)
    ax.set_ylabel('Validation Loss', fontsize=13)
    ax.set_title('Time vs Performance', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    # Plot 2: VRAM vs Val Loss
    ax = axes[1]
    for model in ['MoE', 'TitanMAC']:
        for opt in ['Muon', 'Nested']:
            subset = [r for r in all_results if r['model'] == model and r['optimizer'] == opt]
            if not subset:
                continue

            vrams = [r['vram'] for r in subset]
            losses = [r['val_loss'] for r in subset]

            style = MODEL_STYLES[model]
            marker = 'o' if opt == 'Muon' else 's'
            alpha = 1.0 if opt == 'Muon' else 0.5
            size = 150 if opt == 'Muon' else 50

            ax.scatter(vrams, losses, c=style['color'], marker=marker,
                      s=size, alpha=alpha, edgecolors='black', linewidth=1,
                      label=f'{model} + {opt}')

    ax.set_xlabel('Peak VRAM (GB)', fontsize=13)
    ax.set_ylabel('Validation Loss', fontsize=13)
    ax.set_title('VRAM vs Performance', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_figure(fig, output_dir, 'efficiency_scatter')
    plt.close(fig)


def plot_grid_best_configs(grid_results, output_dir):
    """
    Find and highlight the best configurations from grid search.
    Shows only val_loss with accuracy annotated on bars.
    """
    import re

    # Find best configs for each model
    best_configs = {}

    for model in ['moe', 'titanmac']:
        model_results = [r for r in grid_results
                        if r.get('status') == 'success' and model in r['name'].lower()]

        if not model_results:
            continue

        # Sort by val_loss
        sorted_results = sorted(model_results,
                               key=lambda x: x.get('metrics', {}).get('val_loss', float('inf')))

        best_configs[model] = sorted_results[:5]  # Top 5

    # Create summary figure - just loss bars with accuracy annotations
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Top 5 Nested Optimizer Configurations (Ranked by Val Loss)',
                 fontsize=16, fontweight='bold', y=0.98)

    for idx, (model, results) in enumerate(best_configs.items()):
        ax = axes[idx]

        names = []
        losses = []
        accs = []

        for r in results:
            match = re.search(r'_M(\d+)_C(\d+)', r['name'])
            if match:
                m, c = match.groups()
                names.append(f'M{m} C{c}')
            else:
                names.append(r['name'].split('_')[-1])

            losses.append(r['metrics'].get('val_loss', 0))
            accs.append(r['metrics'].get('val_accuracy', 0) * 100)

        x = np.arange(len(names))
        width = 0.6

        model_name = 'TitanMAC' if model == 'titanmac' else 'MoE'
        color = MODEL_STYLES[model_name]['color']

        # Single bar for loss
        bars = ax.bar(x, losses, width, color=color, alpha=0.85,
                     edgecolor='black', linewidth=1.5)

        # Highlight #1 config
        bars[0].set_edgecolor('#FFB000')
        bars[0].set_linewidth(3)

        ax.set_xlabel('Configuration (M=Momentum Layers, C=Controller Layers)', fontsize=12)
        ax.set_ylabel('Validation Loss', fontsize=13)
        ax.set_title(f'{model_name}: Top 5 Configurations', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')

        # Add loss value on top, accuracy inside bar
        for bar, loss, acc in zip(bars, losses, accs):
            # Loss on top
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                   f'{loss:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
            # Accuracy inside bar
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() / 2,
                   f'{acc:.1f}%', ha='center', va='center', fontsize=10,
                   color='white', fontweight='bold')

    plt.tight_layout()
    _save_figure(fig, output_dir, 'grid_best_configs')
    plt.close(fig)


def print_summary_statistics(baseline_results, grid_results):
    """Print summary statistics to console."""
    print("\n" + "="*70)
    print("GRID SEARCH SUMMARY")
    print("="*70)

    # Baseline summary
    print("\nBASELINE RESULTS:")
    print("-"*50)
    successful_baseline = [r for r in baseline_results if r.get('status') == 'success']
    for r in sorted(successful_baseline, key=lambda x: x['metrics'].get('val_loss', float('inf'))):
        metrics = r['metrics']
        print(f"  {r['name']:25s} | Loss: {metrics.get('val_loss', 0):.4f} | "
              f"Acc: {metrics.get('val_accuracy', 0)*100:.2f}% | "
              f"VRAM: {metrics.get('peak_vram_gb', 0):.1f}GB")

    # Grid summary - best per model
    print("\nGRID SEARCH BEST (per model):")
    print("-"*50)
    for model in ['moe', 'titanmac']:
        model_results = [r for r in grid_results
                        if r.get('status') == 'success' and model in r['name'].lower()]
        if not model_results:
            continue

        best = min(model_results, key=lambda x: x.get('metrics', {}).get('val_loss', float('inf')))
        metrics = best['metrics']
        print(f"  {best['name']:25s} | Loss: {metrics.get('val_loss', 0):.4f} | "
              f"Acc: {metrics.get('val_accuracy', 0)*100:.2f}%")

    # Key finding
    print("\n" + "="*70)
    print("KEY FINDINGS:")
    print("="*70)

    # Best overall
    all_successful = [r for r in baseline_results + grid_results if r.get('status') == 'success']
    best_overall = min(all_successful, key=lambda x: x.get('metrics', {}).get('val_loss', float('inf')))

    print(f"\n  Best overall: {best_overall['name']}")
    print(f"    - Val Loss: {best_overall['metrics'].get('val_loss', 0):.4f}")
    print(f"    - Accuracy: {best_overall['metrics'].get('val_accuracy', 0)*100:.2f}%")

    # Muon baseline comparison
    moe_muon = next((r for r in baseline_results if r['name'] == 'moe_muon'), None)
    if moe_muon:
        moe_nested_best = min(
            [r for r in grid_results if 'moe' in r['name'].lower() and r.get('status') == 'success'],
            key=lambda x: x.get('metrics', {}).get('val_loss', float('inf')),
            default=None
        )
        if moe_nested_best:
            muon_loss = moe_muon['metrics']['val_loss']
            nested_loss = moe_nested_best['metrics']['val_loss']
            diff = nested_loss - muon_loss
            print(f"\n  MoE: Muon ({muon_loss:.4f}) vs Best Nested ({nested_loss:.4f})")
            print(f"    Difference: {'+' if diff > 0 else ''}{diff:.4f} (Muon {'better' if diff > 0 else 'worse'})")


def main():
    print("Generating plots from grid search results...")

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load results
    baseline, grid = load_all_results()
    print(f"Loaded {len(baseline)} baseline and {len(grid)} grid experiments")

    # Generate plots
    print("\n1. Generating baseline comparison...")
    plot_baseline_comparison(baseline, OUTPUT_DIR)

    print("\n2. Generating grid heatmaps...")
    plot_grid_heatmaps(grid, OUTPUT_DIR)

    print("\n3. Generating efficiency scatter plots...")
    plot_efficiency_scatter(baseline, grid, OUTPUT_DIR)

    print("\n4. Generating best configs summary...")
    plot_grid_best_configs(grid, OUTPUT_DIR)

    # Print summary
    print_summary_statistics(baseline, grid)

    print(f"\n{'='*70}")
    print(f"All plots saved to: {OUTPUT_DIR.absolute()}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
