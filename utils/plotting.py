"""
Production-quality plotting utilities for ML research paper.

This module provides standardized plotting functions for comparing model architectures
(MoE, TitanMAC, HOPE) with different optimizers (Muon+AdamW, DeepNestedOptimizer, AdamW).

Key Features:
- Colorblind-friendly color schemes
- Print-ready (grayscale-compatible)
- Publication-quality fonts and DPI
- Comprehensive metric support
- Grid search visualization
- Efficiency frontier analysis

Author: 5-Dollar LLM Research Team
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Union
import json
import warnings


# ============================================================================
# STYLE CONFIGURATION - HIGH VISIBILITY
# ============================================================================

# High-contrast colors - maximally distinct, print-safe
MODEL_STYLES = {
    'MoE': {
        'color': '#0072B2',      # Deep blue (distinct)
        'marker': 'o',           # Circle
        'markersize': 12,
        'markeredgecolor': 'black',
        'markeredgewidth': 1.5,
        'label': 'MoE',
    },
    'TitanMAC': {
        'color': '#D55E00',      # Vermillion/Orange (very distinct from blue)
        'marker': 's',           # Square
        'markersize': 11,
        'markeredgecolor': 'black',
        'markeredgewidth': 1.5,
        'label': 'TitanMAC',
    },
    'HOPE': {
        'color': '#009E73',      # Bluish green (distinct from both)
        'marker': '^',           # Triangle
        'markersize': 12,
        'markeredgecolor': 'black',
        'markeredgewidth': 1.5,
        'label': 'HOPE',
    },
}

# Line styles - very distinct patterns
OPTIMIZER_STYLES = {
    'Muon+AdamW': {
        'linestyle': '-',        # Solid
        'linewidth': 3.0,
        'alpha': 1.0,
        'label': 'Muon+AdamW',
        'dashes': [],            # Solid line
    },
    'DeepNestedOptimizer': {
        'linestyle': '--',       # Dashed
        'linewidth': 3.0,
        'alpha': 1.0,
        'label': 'Nested',
        'dashes': [8, 4],        # Long dash
    },
    'AdamW': {
        'linestyle': ':',        # Dotted
        'linewidth': 3.5,        # Thicker for visibility
        'alpha': 1.0,
        'label': 'AdamW',
        'dashes': [2, 2],        # Dots
    },
}

# Metric labels with proper units
METRIC_LABELS = {
    'val_loss': 'Validation Loss',
    'val_accuracy': 'Accuracy (%)',
    'val_perplexity': 'Perplexity',
    'val_aux_loss': 'Aux Loss',
    'elapsed_times': 'Time (min)',
    'steps': 'Steps',
    'learning_rates': 'Learning Rate',
    'lr_multipliers_core': 'LR Mult (Core)',
    'lr_multipliers_embed': 'LR Mult (Embed)',
    'meta_losses': 'Meta Loss',
    'memory_alpha_t': 'Forget Gate (α)',
    'memory_eta_t': 'Decay Gate (η)',
    'peak_vram_gb': 'VRAM (GB)',
}

# Figure defaults - larger fonts for readability
FIGURE_DEFAULTS = {
    'dpi': 300,
    'single_col_width': 4.0,      # inches
    'double_col_width': 8.0,      # inches
    'full_page_width': 12.0,      # inches
    'aspect_ratio': 0.75,
    'title_fontsize': 16,         # Bigger
    'label_fontsize': 14,         # Bigger
    'tick_fontsize': 12,          # Bigger
    'legend_fontsize': 12,        # Bigger
    'grid_alpha': 0.4,
    'grid_color': '#AAAAAA',
}

# Set matplotlib defaults
plt.rcParams.update({
    'font.size': FIGURE_DEFAULTS['label_fontsize'],
    'axes.labelsize': FIGURE_DEFAULTS['label_fontsize'],
    'axes.titlesize': FIGURE_DEFAULTS['title_fontsize'],
    'xtick.labelsize': FIGURE_DEFAULTS['tick_fontsize'],
    'ytick.labelsize': FIGURE_DEFAULTS['tick_fontsize'],
    'legend.fontsize': FIGURE_DEFAULTS['legend_fontsize'],
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.grid': True,
    'grid.alpha': FIGURE_DEFAULTS['grid_alpha'],
    'grid.color': FIGURE_DEFAULTS['grid_color'],
    'axes.axisbelow': True,  # Grid behind data
    'figure.constrained_layout.use': True,
})


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _get_combined_style(model: str, optimizer: str) -> Dict[str, Any]:
    """
    Combine model and optimizer styles into a single style dict.

    Args:
        model: Model name (e.g., 'MoE', 'TitanMAC', 'HOPE')
        optimizer: Optimizer name (e.g., 'Muon+AdamW', 'DeepNestedOptimizer')

    Returns:
        Combined style dictionary
    """
    style = {}

    # Get base styles with defaults
    model_style = MODEL_STYLES.get(model, MODEL_STYLES['MoE'])
    opt_style = OPTIMIZER_STYLES.get(optimizer, OPTIMIZER_STYLES['Muon+AdamW'])

    # Combine
    style.update(model_style)
    style.update(opt_style)

    # Create combined label
    style['label'] = f"{model} + {optimizer}"

    return style


def _convert_accuracy_to_percent(values: Union[List, np.ndarray]) -> np.ndarray:
    """
    Convert accuracy values to percentage if needed.

    Args:
        values: Accuracy values (either 0-1 or 0-100 range)

    Returns:
        Accuracy values in percentage (0-100)
    """
    values = np.array(values)
    if np.max(values) <= 1.0:
        return values * 100
    return values


def _add_best_marker(ax, x_data, y_data, mode='min', label_prefix='Best'):
    """
    Add a star marker at the best point on a curve.

    Args:
        ax: Matplotlib axis
        x_data: X coordinates
        y_data: Y coordinates
        mode: 'min' or 'max' - whether to find minimum or maximum
        label_prefix: Prefix for the legend label
    """
    if len(y_data) == 0:
        return

    y_array = np.array(y_data)
    x_array = np.array(x_data)

    if mode == 'min':
        best_idx = np.nanargmin(y_array)
    else:
        best_idx = np.nanargmax(y_array)

    best_x = x_array[best_idx]
    best_y = y_array[best_idx]

    ax.plot(best_x, best_y, marker='*', color='#FFB000', markersize=15,
            label=f'{label_prefix}: {best_y:.4f}', zorder=100,
            markeredgecolor='black', markeredgewidth=0.5)


def _setup_axis(ax, title: str, xlabel: str, ylabel: str,
                use_log_y: bool = False, use_log_x: bool = False):
    """
    Standard axis setup with consistent styling.

    Args:
        ax: Matplotlib axis
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        use_log_y: Use logarithmic Y-axis
        use_log_x: Use logarithmic X-axis
    """
    ax.set_title(title, fontsize=FIGURE_DEFAULTS['title_fontsize'],
                 fontweight='bold', pad=10)
    ax.set_xlabel(xlabel, fontsize=FIGURE_DEFAULTS['label_fontsize'])
    ax.set_ylabel(ylabel, fontsize=FIGURE_DEFAULTS['label_fontsize'])

    if use_log_y:
        ax.set_yscale('log')
    if use_log_x:
        ax.set_xscale('log')

    ax.grid(True, alpha=FIGURE_DEFAULTS['grid_alpha'],
            color=FIGURE_DEFAULTS['grid_color'], linestyle='-', linewidth=0.5)
    ax.tick_params(labelsize=FIGURE_DEFAULTS['tick_fontsize'])


def _save_figure(fig, output_path: Path, filename: str):
    """
    Save figure with publication-quality settings.

    Args:
        fig: Matplotlib figure
        output_path: Output directory path
        filename: Filename (without extension)
    """
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save as PNG and PDF
    png_path = output_path / f"{filename}.png"
    pdf_path = output_path / f"{filename}.pdf"

    fig.savefig(png_path, dpi=FIGURE_DEFAULTS['dpi'], bbox_inches='tight')
    fig.savefig(pdf_path, bbox_inches='tight')

    print(f"   Saved: {png_path}")
    print(f"   Saved: {pdf_path}")


# ============================================================================
# CORE PLOTTING FUNCTIONS
# ============================================================================

def plot_experiment_summary(
    metrics_history: Dict[str, List],
    output_path: Union[str, Path],
    experiment_name: str = "Training",
    model_type: str = "MoE",
    optimizer_type: str = "Muon+AdamW",
) -> None:
    """
    Plot comprehensive 2x2 summary for a single experiment.

    Creates a 2x2 grid showing:
    - Top-left: Validation loss vs time (with best point marked)
    - Top-right: Validation loss vs steps
    - Bottom-left: Validation accuracy vs steps
    - Bottom-right: Learning rate schedule

    Args:
        metrics_history: Dictionary containing metric histories
        output_path: Directory to save plots
        experiment_name: Name of the experiment for title
        model_type: Model architecture ('MoE', 'TitanMAC', 'HOPE')
        optimizer_type: Optimizer used ('Muon+AdamW', 'DeepNestedOptimizer', 'AdamW')
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{experiment_name} - Training Metrics',
                 fontsize=16, fontweight='bold', y=0.995)

    style = _get_combined_style(model_type, optimizer_type)

    # Plot 1: Val Loss vs Time
    ax = axes[0, 0]
    if 'elapsed_times' in metrics_history and 'val_losses' in metrics_history:
        ax.plot(metrics_history['elapsed_times'], metrics_history['val_losses'],
                color=style['color'], linestyle=style['linestyle'],
                linewidth=style['linewidth'], marker=style['marker'],
                markersize=4, alpha=style['alpha'])
        _add_best_marker(ax, metrics_history['elapsed_times'],
                        metrics_history['val_losses'], mode='min')
        _setup_axis(ax, 'Validation Loss vs Time',
                   METRIC_LABELS['elapsed_times'], METRIC_LABELS['val_loss'])
        ax.legend(loc='best')

    # Plot 2: Val Loss vs Steps
    ax = axes[0, 1]
    if 'steps' in metrics_history and 'val_losses' in metrics_history:
        ax.plot(metrics_history['steps'], metrics_history['val_losses'],
                color=style['color'], linestyle=style['linestyle'],
                linewidth=style['linewidth'], marker=style['marker'],
                markersize=4, alpha=style['alpha'])
        _add_best_marker(ax, metrics_history['steps'],
                        metrics_history['val_losses'], mode='min')
        _setup_axis(ax, 'Validation Loss vs Steps',
                   METRIC_LABELS['steps'], METRIC_LABELS['val_loss'])

    # Plot 3: Val Accuracy vs Steps
    ax = axes[1, 0]
    if 'steps' in metrics_history and 'val_accuracies' in metrics_history:
        acc_pct = _convert_accuracy_to_percent(metrics_history['val_accuracies'])
        ax.plot(metrics_history['steps'], acc_pct,
                color=style['color'], linestyle=style['linestyle'],
                linewidth=style['linewidth'], marker=style['marker'],
                markersize=4, alpha=style['alpha'])
        _add_best_marker(ax, metrics_history['steps'], acc_pct, mode='max')
        _setup_axis(ax, 'Validation Accuracy vs Steps',
                   METRIC_LABELS['steps'], METRIC_LABELS['val_accuracy'])

    # Plot 4: Learning Rate Schedule
    ax = axes[1, 1]
    if 'steps' in metrics_history and 'learning_rates' in metrics_history:
        ax.plot(metrics_history['steps'], metrics_history['learning_rates'],
                color='#EE7733', linewidth=2.5)
        _setup_axis(ax, 'Learning Rate Schedule',
                   METRIC_LABELS['steps'], METRIC_LABELS['learning_rates'])

    _save_figure(fig, output_path, 'experiment_summary')
    plt.close(fig)


def plot_nested_optimizer_details(
    metrics_history: Dict[str, List],
    output_path: Union[str, Path],
    experiment_name: str = "Nested Optimizer Training",
    model_type: str = "MoE",
) -> None:
    """
    Plot detailed metrics specific to nested optimizer experiments.

    Creates a 2x3 grid showing:
    - Standard metrics (loss, accuracy)
    - LR multipliers (learned per parameter group)
    - Meta-learning loss
    - Memory gates (if TitanMAC)

    Args:
        metrics_history: Dictionary containing metric histories
        output_path: Directory to save plots
        experiment_name: Name of the experiment
        model_type: Model architecture ('MoE', 'TitanMAC', 'HOPE')
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{experiment_name} - Nested Optimizer Details',
                 fontsize=16, fontweight='bold', y=0.995)

    style = MODEL_STYLES.get(model_type, MODEL_STYLES['MoE'])

    # Plot 1: Val Loss vs Time
    ax = axes[0, 0]
    if 'elapsed_times' in metrics_history and 'val_losses' in metrics_history:
        ax.plot(metrics_history['elapsed_times'], metrics_history['val_losses'],
                color=style['color'], linewidth=2.5, marker='o', markersize=4)
        _add_best_marker(ax, metrics_history['elapsed_times'],
                        metrics_history['val_losses'], mode='min')
        _setup_axis(ax, 'Validation Loss vs Time',
                   METRIC_LABELS['elapsed_times'], METRIC_LABELS['val_loss'])
        ax.legend(loc='best')

    # Plot 2: Val Loss vs Steps
    ax = axes[0, 1]
    if 'steps' in metrics_history and 'val_losses' in metrics_history:
        ax.plot(metrics_history['steps'], metrics_history['val_losses'],
                color=style['color'], linewidth=2.5, marker='o', markersize=4)
        _setup_axis(ax, 'Validation Loss vs Steps',
                   METRIC_LABELS['steps'], METRIC_LABELS['val_loss'])

    # Plot 3: Val Accuracy vs Steps
    ax = axes[0, 2]
    if 'steps' in metrics_history and 'val_accuracies' in metrics_history:
        acc_pct = _convert_accuracy_to_percent(metrics_history['val_accuracies'])
        ax.plot(metrics_history['steps'], acc_pct,
                color=style['color'], linewidth=2.5, marker='o', markersize=4)
        _add_best_marker(ax, metrics_history['steps'], acc_pct, mode='max')
        _setup_axis(ax, 'Validation Accuracy vs Steps',
                   METRIC_LABELS['steps'], METRIC_LABELS['val_accuracy'])

    # Plot 4: LR Multipliers
    ax = axes[1, 0]
    if ('steps' in metrics_history and
        'lr_multipliers_core' in metrics_history and
        'lr_multipliers_embed' in metrics_history):
        ax.plot(metrics_history['steps'], metrics_history['lr_multipliers_core'],
                'b-', linewidth=2.5, label='Core')
        ax.plot(metrics_history['steps'], metrics_history['lr_multipliers_embed'],
                'r-', linewidth=2.5, label='Embed')
        ax.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Baseline')
        _setup_axis(ax, 'Learned LR Multipliers',
                   METRIC_LABELS['steps'], 'LR Multiplier')
        ax.legend(loc='best')

    # Plot 5: Effective Learning Rate
    ax = axes[1, 1]
    if 'steps' in metrics_history and 'learning_rates' in metrics_history:
        ax.plot(metrics_history['steps'], metrics_history['learning_rates'],
                color='#EE7733', linewidth=2.5)
        _setup_axis(ax, 'Effective Learning Rate',
                   METRIC_LABELS['steps'], METRIC_LABELS['learning_rates'])

    # Plot 6: Meta Loss or Memory Gates
    ax = axes[1, 2]
    if 'steps' in metrics_history and 'meta_losses' in metrics_history:
        # Show meta-learning loss
        ax.plot(metrics_history['steps'], metrics_history['meta_losses'],
                color='#33BBEE', linewidth=2.5)
        _setup_axis(ax, 'Meta-Learning Loss',
                   METRIC_LABELS['steps'], METRIC_LABELS['meta_losses'])
    elif (model_type == 'TitanMAC' and
          'memory_alpha_t' in metrics_history and
          'memory_eta_t' in metrics_history):
        # Show TitanMAC memory gates
        ax.plot(metrics_history['steps'], metrics_history['memory_alpha_t'],
                'r-', linewidth=2.5, label='α (forget)')
        ax.plot(metrics_history['steps'], metrics_history['memory_eta_t'],
                'b-', linewidth=2.5, label='η (decay)')
        _setup_axis(ax, 'TitanMAC Memory Gates',
                   METRIC_LABELS['steps'], 'Gate Value')
        ax.legend(loc='best')

    _save_figure(fig, output_path, 'nested_optimizer_details')
    plt.close(fig)


def plot_comparison_overlay(
    experiments_data: Dict[str, Dict],
    output_path: Union[str, Path],
    metrics: List[str] = ['val_loss', 'val_accuracy'],
) -> None:
    """
    Overlay multiple experiments on the same axes for direct comparison.

    Creates a grid of plots (one per metric) where all experiments are
    overlaid on the same axes. This makes it easy to see which approach
    performs best at each point in training.

    Args:
        experiments_data: Dictionary mapping experiment names to their data.
            Each experiment dict should contain:
            - 'history': metrics_history dictionary
            - 'model': model type ('MoE', 'TitanMAC', 'HOPE')
            - 'optimizer': optimizer type ('Muon+AdamW', etc.)
        output_path: Directory to save plots
        metrics: List of metrics to plot (default: val_loss and val_accuracy)

    Example:
        experiments_data = {
            'MoE-Baseline': {
                'history': {...},
                'model': 'MoE',
                'optimizer': 'Muon+AdamW',
            },
            'MoE-Nested': {
                'history': {...},
                'model': 'MoE',
                'optimizer': 'DeepNestedOptimizer',
            },
        }
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(7 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    fig.suptitle('Cross-Experiment Comparison',
                 fontsize=16, fontweight='bold', y=0.995)

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        # Plot each experiment
        for exp_name, exp_data in experiments_data.items():
            history = exp_data['history']
            model = exp_data.get('model', 'MoE')
            optimizer = exp_data.get('optimizer', 'Muon+AdamW')

            style = _get_combined_style(model, optimizer)

            # Determine metric key (handle different naming conventions)
            metric_key = metric
            if metric == 'val_losses':
                metric_key = 'val_losses'
            elif metric == 'val_loss' and 'val_losses' in history:
                metric_key = 'val_losses'
            elif metric == 'val_accuracies':
                metric_key = 'val_accuracies'
            elif metric == 'val_accuracy' and 'val_accuracies' in history:
                metric_key = 'val_accuracies'

            if 'steps' in history and metric_key in history:
                y_data = history[metric_key]

                # Convert accuracy to percent if needed
                if 'accuracy' in metric.lower():
                    y_data = _convert_accuracy_to_percent(y_data)

                ax.plot(history['steps'], y_data,
                       color=style['color'], linestyle=style['linestyle'],
                       linewidth=style['linewidth'], marker=style['marker'],
                       markersize=5, alpha=style['alpha'],
                       label=f"{model} + {optimizer}")

        # Setup axis
        y_label = METRIC_LABELS.get(metric, metric)
        _setup_axis(ax, f'{y_label} Comparison',
                   METRIC_LABELS['steps'], y_label)
        ax.legend(loc='best', framealpha=0.9)

    _save_figure(fig, output_path, 'comparison_overlay')
    plt.close(fig)


def plot_grid_heatmap(
    grid_results: Dict[str, Any],
    output_path: Union[str, Path],
    model_type: str = "MoE",
    metric: str = 'val_loss',
    annotate_best: bool = True,
) -> None:
    """
    Plot heatmap for M x C grid search (momentum layers x controller layers).

    Visualizes the grid search results as a 2D heatmap where:
    - X-axis: Number of controller layers
    - Y-axis: Number of momentum layers
    - Color: Performance metric (lower is better for loss)

    Args:
        grid_results: Dictionary containing grid search results.
            Should have structure: {
                'trials': [
                    {
                        'depth_params': {'momentum_num_layers': int, 'controller_num_layers': int},
                        'metrics': {'val_loss': float, ...},
                    },
                    ...
                ]
            }
        output_path: Directory to save plots
        model_type: Model architecture ('MoE', 'TitanMAC', 'HOPE')
        metric: Metric to visualize (default: 'val_loss')
        annotate_best: Whether to annotate the best configuration
    """
    trials = grid_results.get('trials', [])
    if not trials:
        warnings.warn("No trials found in grid_results")
        return

    # Extract grid dimensions
    momentum_layers = sorted(set(t['depth_params']['momentum_num_layers'] for t in trials))
    controller_layers = sorted(set(t['depth_params']['controller_num_layers'] for t in trials))

    # Create grid matrix
    grid = np.full((len(momentum_layers), len(controller_layers)), np.nan)

    best_value = float('inf') if 'loss' in metric.lower() else float('-inf')
    best_m, best_c = 0, 0

    for trial in trials:
        m = trial['depth_params']['momentum_num_layers']
        c = trial['depth_params']['controller_num_layers']
        value = trial.get('metrics', {}).get(metric)

        if value is not None and not np.isnan(value):
            m_idx = momentum_layers.index(m)
            c_idx = controller_layers.index(c)
            grid[m_idx, c_idx] = value

            # Track best
            if 'loss' in metric.lower():
                if value < best_value:
                    best_value = value
                    best_m, best_c = m, c
            else:
                if value > best_value:
                    best_value = value
                    best_m, best_c = m, c

    # Create heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    # Use colormap (lower is better for losses, so reverse for intuitive coloring)
    cmap = 'RdYlGn_r' if 'loss' in metric.lower() else 'RdYlGn'

    im = ax.imshow(grid, cmap=cmap, aspect='auto', interpolation='nearest')

    # Set ticks
    ax.set_xticks(np.arange(len(controller_layers)))
    ax.set_yticks(np.arange(len(momentum_layers)))
    ax.set_xticklabels(controller_layers)
    ax.set_yticklabels(momentum_layers)

    # Labels
    ax.set_xlabel('Controller Layers', fontsize=FIGURE_DEFAULTS['label_fontsize'])
    ax.set_ylabel('Momentum Layers', fontsize=FIGURE_DEFAULTS['label_fontsize'])
    ax.set_title(f'{model_type} Grid Search: {METRIC_LABELS.get(metric, metric)}',
                fontsize=FIGURE_DEFAULTS['title_fontsize'], fontweight='bold', pad=15)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(METRIC_LABELS.get(metric, metric),
                   fontsize=FIGURE_DEFAULTS['label_fontsize'])

    # Annotate cells with values
    for i in range(len(momentum_layers)):
        for j in range(len(controller_layers)):
            value = grid[i, j]
            if not np.isnan(value):
                text_color = 'white' if value < np.nanmean(grid) else 'black'
                ax.text(j, i, f'{value:.3f}', ha='center', va='center',
                       color=text_color, fontsize=9)

    # Highlight best configuration
    if annotate_best and best_m > 0 and best_c > 0:
        best_m_idx = momentum_layers.index(best_m)
        best_c_idx = controller_layers.index(best_c)

        rect = mpatches.Rectangle((best_c_idx - 0.5, best_m_idx - 0.5), 1, 1,
                                  linewidth=3, edgecolor='#FFB000',
                                  facecolor='none', linestyle='-')
        ax.add_patch(rect)

        # Add annotation
        ax.text(len(controller_layers) / 2, -1.5,
               f'Best: M={best_m}, C={best_c} ({metric}={best_value:.4f})',
               ha='center', fontsize=12, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='#FFB000', alpha=0.3))

    _save_figure(fig, output_path, f'grid_heatmap_{metric}')
    plt.close(fig)


def plot_efficiency_frontier(
    experiments_data: Dict[str, Dict],
    output_path: Union[str, Path],
    x_metrics: List[str] = ['elapsed_times', 'peak_vram_gb'],
    y_metric: str = 'val_loss',
) -> None:
    """
    Plot efficiency frontiers showing trade-offs.

    Creates plots showing the Pareto frontier for different trade-offs:
    - Time vs Performance
    - VRAM vs Performance
    - FLOPs vs Performance (if available)

    This helps identify which approach offers the best trade-off for
    different constraints (e.g., limited time, limited memory).

    Args:
        experiments_data: Dictionary mapping experiment names to their data.
            Each experiment dict should contain:
            - 'final_metrics': final evaluation metrics
            - 'model': model type
            - 'optimizer': optimizer type
        output_path: Directory to save plots
        x_metrics: List of resource metrics to plot (x-axis)
        y_metric: Performance metric (y-axis, lower is better)
    """
    n_plots = len(x_metrics)
    fig, axes = plt.subplots(1, n_plots, figsize=(7 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    fig.suptitle('Efficiency Frontier Analysis',
                 fontsize=16, fontweight='bold', y=0.995)

    for idx, x_metric in enumerate(x_metrics):
        ax = axes[idx]

        # Collect data points
        points = []
        labels = []
        styles = []

        for exp_name, exp_data in experiments_data.items():
            final_metrics = exp_data.get('final_metrics', {})
            model = exp_data.get('model', 'MoE')
            optimizer = exp_data.get('optimizer', 'Muon+AdamW')

            # Get metric values
            if x_metric == 'elapsed_times':
                x_val = exp_data.get('total_time_minutes')
            else:
                x_val = final_metrics.get(x_metric)

            y_val = final_metrics.get(y_metric)

            if x_val is not None and y_val is not None:
                points.append((x_val, y_val))
                labels.append(exp_name)
                styles.append(_get_combined_style(model, optimizer))

        if not points:
            continue

        # Plot points
        for (x, y), label, style in zip(points, labels, styles):
            ax.scatter(x, y, color=style['color'], marker=style['marker'],
                      s=150, alpha=style['alpha'], edgecolors='black',
                      linewidth=1.5, label=label, zorder=10)

        # Identify Pareto frontier (lower-left is better)
        points_array = np.array(points)
        pareto_indices = []

        for i, (x_i, y_i) in enumerate(points):
            # A point is on the Pareto frontier if no other point dominates it
            # (i.e., no point has both lower x AND lower y)
            dominated = False
            for x_j, y_j in points:
                if x_j < x_i and y_j < y_i:
                    dominated = True
                    break
            if not dominated:
                pareto_indices.append(i)

        # Draw Pareto frontier
        if len(pareto_indices) > 1:
            pareto_points = points_array[pareto_indices]
            # Sort by x for proper line drawing
            pareto_points = pareto_points[pareto_points[:, 0].argsort()]
            ax.plot(pareto_points[:, 0], pareto_points[:, 1],
                   'k--', linewidth=2, alpha=0.5, zorder=5,
                   label='Pareto Frontier')

        # Setup axis
        x_label = METRIC_LABELS.get(x_metric, x_metric)
        y_label = METRIC_LABELS.get(y_metric, y_metric)

        _setup_axis(ax, f'{y_label} vs {x_label}', x_label, y_label)
        ax.legend(loc='best', framealpha=0.9, fontsize=9)

    _save_figure(fig, output_path, 'efficiency_frontier')
    plt.close(fig)


def plot_final_metrics_bar(
    experiments_data: Dict[str, Dict],
    output_path: Union[str, Path],
    metrics: List[str] = ['val_loss', 'val_accuracy', 'peak_vram_gb'],
) -> None:
    """
    Plot bar chart comparing final metrics across experiments.

    Creates a grouped bar chart showing final performance metrics for
    all experiments side-by-side. This provides a clear snapshot of
    which approach performs best overall.

    Args:
        experiments_data: Dictionary mapping experiment names to their data.
            Each experiment dict should contain:
            - 'final_metrics': final evaluation metrics
            - 'model': model type
            - 'optimizer': optimizer type
        output_path: Directory to save plots
        metrics: List of metrics to compare
    """
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(6 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    fig.suptitle('Final Metrics Comparison',
                 fontsize=16, fontweight='bold', y=0.995)

    exp_names = list(experiments_data.keys())
    x_pos = np.arange(len(exp_names))

    for idx, metric in enumerate(metrics):
        ax = axes[idx]

        values = []
        colors = []

        for exp_name in exp_names:
            exp_data = experiments_data[exp_name]
            final_metrics = exp_data.get('final_metrics', {})
            model = exp_data.get('model', 'MoE')

            value = final_metrics.get(metric, 0)

            # Convert accuracy to percent if needed
            if 'accuracy' in metric.lower():
                value = _convert_accuracy_to_percent([value])[0]

            values.append(value)

            # Get color from model style
            style = MODEL_STYLES.get(model, MODEL_STYLES['MoE'])
            colors.append(style['color'])

        # Create bars
        bars = ax.bar(x_pos, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

        # Highlight best
        if 'loss' in metric.lower() or 'vram' in metric.lower():
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)

        bars[best_idx].set_edgecolor('#FFB000')
        bars[best_idx].set_linewidth(3)

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                   f'{value:.3f}', ha='center', va='bottom', fontsize=9)

        # Setup axis
        ax.set_xticks(x_pos)
        ax.set_xticklabels(exp_names, rotation=45, ha='right')
        ax.set_ylabel(METRIC_LABELS.get(metric, metric),
                     fontsize=FIGURE_DEFAULTS['label_fontsize'])
        ax.set_title(METRIC_LABELS.get(metric, metric),
                    fontsize=FIGURE_DEFAULTS['title_fontsize'], fontweight='bold')
        ax.grid(True, alpha=FIGURE_DEFAULTS['grid_alpha'], axis='y')

    _save_figure(fig, output_path, 'final_metrics_bar')
    plt.close(fig)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def load_experiment_from_json(metrics_file: Union[str, Path]) -> Dict[str, Any]:
    """
    Load experiment data from a metrics.json file.

    Args:
        metrics_file: Path to metrics.json file

    Returns:
        Dictionary with experiment data suitable for plotting functions
    """
    metrics_file = Path(metrics_file)

    with open(metrics_file, 'r') as f:
        data = json.load(f)

    # Extract relevant fields
    experiment = {
        'history': data.get('history', {}),
        'final_metrics': data.get('final_metrics', {}),
        'total_time_minutes': data.get('total_time_minutes', 0),
        'peak_vram_gb': data.get('peak_vram_gb', 0),
        'model': 'MoE',  # Default, should be overridden
        'optimizer': 'Muon+AdamW',  # Default
    }

    # Try to infer model and optimizer from config
    if 'experiment_config' in data:
        config = data['experiment_config']
        if 'model_type' in config:
            experiment['model'] = config['model_type']

    if 'optimizer_type' in data:
        experiment['optimizer'] = data['optimizer_type']
    elif 'nested_config' in data:
        experiment['optimizer'] = 'DeepNestedOptimizer'

    return experiment


def load_grid_results(results_file: Union[str, Path], model_filter: Optional[str] = None) -> Dict[str, Any]:
    """
    Load and convert grid search results from run_comprehensive_grid.py format.

    Converts from:
        [{"name": "moe_nested_M2_C2", "metrics": {...}}, ...]
    To:
        {'trials': [{'depth_params': {...}, 'metrics': {...}}, ...]}

    Args:
        results_file: Path to grid_results.json or baseline_results.json
        model_filter: Optional filter for model type ('moe', 'titanmac', 'hope')

    Returns:
        Dictionary formatted for plot_grid_heatmap()
    """
    import re

    results_file = Path(results_file)
    with open(results_file, 'r') as f:
        raw_results = json.load(f)

    trials = []
    for result in raw_results:
        name = result.get('name', '')

        # Skip failed experiments
        if result.get('status') != 'success':
            continue

        # Filter by model if specified
        if model_filter:
            if not name.startswith(model_filter.lower()):
                continue

        # Parse M and C from name like "moe_nested_M2_C2"
        match = re.search(r'_M(\d+)_C(\d+)', name)
        if match:
            m_layers = int(match.group(1))
            c_layers = int(match.group(2))
        elif '_nested_best' in name:
            # Best params use M=2, C=2
            m_layers = 2
            c_layers = 2
        else:
            # Not a grid experiment
            continue

        trial = {
            'name': name,
            'depth_params': {
                'momentum_num_layers': m_layers,
                'controller_num_layers': c_layers,
            },
            'metrics': result.get('metrics', {}),
            'elapsed_seconds': result.get('elapsed_seconds', 0),
        }
        trials.append(trial)

    return {'trials': trials}


def plot_all_grid_heatmaps(
    results_file: Union[str, Path],
    output_dir: Union[str, Path],
) -> None:
    """
    Generate heatmaps for all model types in grid search results.

    Args:
        results_file: Path to grid_results.json
        output_dir: Directory to save plots
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model in ['moe', 'titanmac', 'hope']:
        grid_data = load_grid_results(results_file, model_filter=model)
        if grid_data['trials']:
            print(f"\nGenerating heatmap for {model.upper()} ({len(grid_data['trials'])} trials)...")
            plot_grid_heatmap(
                grid_data,
                output_dir,
                model_type=model.upper() if model != 'titanmac' else 'TitanMAC',
                metric='val_loss',
            )


def plot_all_experiment_summaries(
    experiments_dir: Union[str, Path],
    output_dir: Union[str, Path],
) -> None:
    """
    Automatically find and plot all experiments in a directory.

    Searches for metrics.json files and creates summary plots for each.

    Args:
        experiments_dir: Directory containing experiment subdirectories
        output_dir: Directory to save plots
    """
    experiments_dir = Path(experiments_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all metrics.json files
    metrics_files = list(experiments_dir.rglob('metrics.json'))

    print(f"Found {len(metrics_files)} experiments in {experiments_dir}")

    for metrics_file in metrics_files:
        exp_name = metrics_file.parent.name
        print(f"\nPlotting {exp_name}...")

        try:
            exp_data = load_experiment_from_json(metrics_file)

            exp_output = output_dir / exp_name
            exp_output.mkdir(parents=True, exist_ok=True)

            plot_experiment_summary(
                exp_data['history'],
                exp_output,
                experiment_name=exp_name,
                model_type=exp_data['model'],
                optimizer_type=exp_data['optimizer'],
            )

            # If nested optimizer, plot detailed metrics
            if exp_data['optimizer'] == 'DeepNestedOptimizer':
                plot_nested_optimizer_details(
                    exp_data['history'],
                    exp_output,
                    experiment_name=exp_name,
                    model_type=exp_data['model'],
                )

        except Exception as e:
            print(f"   Error plotting {exp_name}: {e}")

    print(f"\nAll plots saved to {output_dir}")


# ============================================================================
# MAIN (Example usage)
# ============================================================================

if __name__ == '__main__':
    """
    Example usage demonstrating the plotting utilities.
    """
    import argparse

    parser = argparse.ArgumentParser(description="Plot ML experiment results")
    parser.add_argument('--experiments_dir', type=str, required=True,
                       help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='./plots',
                       help='Directory to save plots')
    args = parser.parse_args()

    plot_all_experiment_summaries(args.experiments_dir, args.output_dir)
