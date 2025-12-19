"""
Usage examples for utils/plotting.py

This file demonstrates how to use the plotting utilities for analyzing
your ML experiments.
"""

from pathlib import Path
from utils.plotting import (
    plot_experiment_summary,
    plot_nested_optimizer_details,
    plot_comparison_overlay,
    plot_grid_heatmap,
    plot_efficiency_frontier,
    plot_final_metrics_bar,
    load_experiment_from_json,
    plot_all_experiment_summaries,
)


# ============================================================================
# Example 1: Plot a single experiment
# ============================================================================

def example_single_experiment():
    """
    Plot summary for a single training run.

    This is the most basic usage - just point it at your metrics.json file.
    """
    # Load experiment data
    exp_data = load_experiment_from_json('checkpoints/moe_baseline_600/metrics.json')

    # Create summary plot
    plot_experiment_summary(
        metrics_history=exp_data['history'],
        output_path='plots/moe_baseline',
        experiment_name='MoE Baseline (Muon+AdamW)',
        model_type='MoE',
        optimizer_type='Muon+AdamW',
    )

    print("Created: plots/moe_baseline/experiment_summary.png")


# ============================================================================
# Example 2: Plot nested optimizer details
# ============================================================================

def example_nested_optimizer():
    """
    Plot detailed metrics for nested optimizer experiments.

    This shows LR multipliers, meta-learning loss, and other nested-specific metrics.
    """
    exp_data = load_experiment_from_json('checkpoints/moe_nested_m2c2/metrics.json')

    plot_nested_optimizer_details(
        metrics_history=exp_data['history'],
        output_path='plots/moe_nested',
        experiment_name='MoE with Nested Optimizer',
        model_type='MoE',
    )

    print("Created: plots/moe_nested/nested_optimizer_details.png")


# ============================================================================
# Example 3: Compare multiple experiments
# ============================================================================

def example_comparison():
    """
    Overlay multiple experiments on the same axes.

    This makes it easy to see which approach converges faster and to what level.
    """
    # Load multiple experiments
    experiments = {
        'MoE-Baseline': {
            **load_experiment_from_json('checkpoints/moe_baseline_600/metrics.json'),
            'model': 'MoE',
            'optimizer': 'Muon+AdamW',
        },
        'MoE-Nested-2x2': {
            **load_experiment_from_json('checkpoints/moe_nested_m2c2/metrics.json'),
            'model': 'MoE',
            'optimizer': 'DeepNestedOptimizer',
        },
        'TitanMAC-Baseline': {
            **load_experiment_from_json('checkpoints/titanmac_baseline_600/metrics.json'),
            'model': 'TitanMAC',
            'optimizer': 'Muon+AdamW',
        },
    }

    # Create comparison overlay
    plot_comparison_overlay(
        experiments_data=experiments,
        output_path='plots/comparisons',
        metrics=['val_loss', 'val_accuracy'],
    )

    print("Created: plots/comparisons/comparison_overlay.png")


# ============================================================================
# Example 4: Visualize grid search results
# ============================================================================

def example_grid_search():
    """
    Create heatmap for M x C grid search.

    This helps identify the optimal depth configuration.
    """
    import json

    # Load grid search results
    with open('grid_results_168m/moe_nested/all_trials.json', 'r') as f:
        trials = json.load(f)

    grid_results = {'trials': trials}

    # Create heatmap for validation loss
    plot_grid_heatmap(
        grid_results=grid_results,
        output_path='plots/grid_search',
        model_type='MoE',
        metric='val_loss',
        annotate_best=True,
    )

    # Also create heatmap for validation accuracy
    plot_grid_heatmap(
        grid_results=grid_results,
        output_path='plots/grid_search',
        model_type='MoE',
        metric='val_accuracy',
        annotate_best=True,
    )

    print("Created: plots/grid_search/grid_heatmap_val_loss.png")
    print("Created: plots/grid_search/grid_heatmap_val_accuracy.png")


# ============================================================================
# Example 5: Efficiency frontier analysis
# ============================================================================

def example_efficiency_frontier():
    """
    Plot efficiency frontiers showing trade-offs.

    This helps answer questions like:
    - Which approach gives the best performance for a given time budget?
    - Which approach is most memory-efficient?
    """
    # Load all experiments with their final metrics
    experiments = {
        'MoE-Baseline': {
            **load_experiment_from_json('checkpoints/moe_baseline_600/metrics.json'),
            'model': 'MoE',
            'optimizer': 'Muon+AdamW',
        },
        'MoE-Nested-2x2': {
            **load_experiment_from_json('checkpoints/moe_nested_m2c2/metrics.json'),
            'model': 'MoE',
            'optimizer': 'DeepNestedOptimizer',
        },
        'TitanMAC-Baseline': {
            **load_experiment_from_json('checkpoints/titanmac_baseline_600/metrics.json'),
            'model': 'TitanMAC',
            'optimizer': 'Muon+AdamW',
        },
        'HOPE-Baseline': {
            **load_experiment_from_json('checkpoints/hope_baseline_600/metrics.json'),
            'model': 'HOPE',
            'optimizer': 'Muon+AdamW',
        },
    }

    # Plot time vs performance and VRAM vs performance
    plot_efficiency_frontier(
        experiments_data=experiments,
        output_path='plots/efficiency',
        x_metrics=['elapsed_times', 'peak_vram_gb'],
        y_metric='val_loss',
    )

    print("Created: plots/efficiency/efficiency_frontier.png")


# ============================================================================
# Example 6: Final metrics bar chart
# ============================================================================

def example_final_metrics_bar():
    """
    Create bar chart comparing final metrics.

    This gives a clear at-a-glance comparison of which approach won.
    """
    experiments = {
        'MoE-Baseline': {
            **load_experiment_from_json('checkpoints/moe_baseline_600/metrics.json'),
            'model': 'MoE',
            'optimizer': 'Muon+AdamW',
        },
        'MoE-Nested-2x2': {
            **load_experiment_from_json('checkpoints/moe_nested_m2c2/metrics.json'),
            'model': 'MoE',
            'optimizer': 'DeepNestedOptimizer',
        },
        'TitanMAC-Baseline': {
            **load_experiment_from_json('checkpoints/titanmac_baseline_600/metrics.json'),
            'model': 'TitanMAC',
            'optimizer': 'Muon+AdamW',
        },
    }

    plot_final_metrics_bar(
        experiments_data=experiments,
        output_path='plots/final_comparison',
        metrics=['val_loss', 'val_accuracy', 'peak_vram_gb'],
    )

    print("Created: plots/final_comparison/final_metrics_bar.png")


# ============================================================================
# Example 7: Automatically plot all experiments
# ============================================================================

def example_plot_all():
    """
    Automatically find and plot all experiments in a directory.

    This is the easiest way to generate plots for a large number of experiments.
    """
    plot_all_experiment_summaries(
        experiments_dir='checkpoints_168m',
        output_dir='plots/all_experiments',
    )

    print("Created plots for all experiments in: plots/all_experiments/")


# ============================================================================
# Example 8: Custom styling for presentations
# ============================================================================

def example_custom_style():
    """
    Customize plot styles for presentations or specific requirements.
    """
    import matplotlib.pyplot as plt
    from utils import plotting

    # Temporarily modify defaults for larger fonts (presentations)
    original_defaults = plotting.FIGURE_DEFAULTS.copy()

    plotting.FIGURE_DEFAULTS['title_fontsize'] = 18
    plotting.FIGURE_DEFAULTS['label_fontsize'] = 16
    plotting.FIGURE_DEFAULTS['tick_fontsize'] = 14
    plotting.FIGURE_DEFAULTS['legend_fontsize'] = 14

    # Update matplotlib settings
    plt.rcParams.update({
        'font.size': plotting.FIGURE_DEFAULTS['label_fontsize'],
        'axes.labelsize': plotting.FIGURE_DEFAULTS['label_fontsize'],
        'axes.titlesize': plotting.FIGURE_DEFAULTS['title_fontsize'],
    })

    # Create plots with larger fonts
    exp_data = load_experiment_from_json('checkpoints/moe_baseline_600/metrics.json')

    plot_experiment_summary(
        metrics_history=exp_data['history'],
        output_path='plots/presentation',
        experiment_name='MoE Baseline',
        model_type='MoE',
        optimizer_type='Muon+AdamW',
    )

    # Restore original settings
    plotting.FIGURE_DEFAULTS = original_defaults

    print("Created: plots/presentation/experiment_summary.png (with larger fonts)")


# ============================================================================
# Example 9: Generate all plots for paper
# ============================================================================

def generate_paper_plots():
    """
    Complete workflow to generate all plots needed for a research paper.

    This is what you'd run to create all publication-ready figures.
    """
    print("Generating all plots for research paper...")
    print("=" * 70)

    # 1. Individual experiment summaries
    print("\n1. Creating individual experiment summaries...")
    plot_all_experiment_summaries(
        experiments_dir='checkpoints_168m',
        output_dir='paper_plots/experiments',
    )

    # 2. Cross-experiment comparisons
    print("\n2. Creating cross-experiment comparisons...")
    baseline_experiments = {
        'MoE-Baseline': {
            **load_experiment_from_json('checkpoints_168m/moe_baseline_600/metrics.json'),
            'model': 'MoE',
            'optimizer': 'Muon+AdamW',
        },
        'TitanMAC-Baseline': {
            **load_experiment_from_json('checkpoints_168m/titanmac_baseline_600/metrics.json'),
            'model': 'TitanMAC',
            'optimizer': 'Muon+AdamW',
        },
        'HOPE-Baseline': {
            **load_experiment_from_json('checkpoints_168m/hope_baseline_600/metrics.json'),
            'model': 'HOPE',
            'optimizer': 'Muon+AdamW',
        },
    }

    plot_comparison_overlay(
        experiments_data=baseline_experiments,
        output_path='paper_plots/comparisons',
        metrics=['val_loss', 'val_accuracy'],
    )

    # 3. Grid search heatmaps
    print("\n3. Creating grid search heatmaps...")
    import json

    for model in ['moe', 'titanmac', 'hope']:
        try:
            with open(f'grid_results_168m/{model}_nested/all_trials.json', 'r') as f:
                trials = json.load(f)

            plot_grid_heatmap(
                grid_results={'trials': trials},
                output_path='paper_plots/grid_search',
                model_type=model.title() if model != 'hope' else 'HOPE',
                metric='val_loss',
                annotate_best=True,
            )
        except FileNotFoundError:
            print(f"   Warning: No grid results found for {model}")

    # 4. Efficiency frontiers
    print("\n4. Creating efficiency frontier analysis...")
    all_experiments = baseline_experiments.copy()
    # Add best nested optimizer configurations
    # (You would load these from your grid search results)

    plot_efficiency_frontier(
        experiments_data=all_experiments,
        output_path='paper_plots/efficiency',
        x_metrics=['elapsed_times', 'peak_vram_gb'],
        y_metric='val_loss',
    )

    # 5. Final metrics comparison
    print("\n5. Creating final metrics bar chart...")
    plot_final_metrics_bar(
        experiments_data=all_experiments,
        output_path='paper_plots/final',
        metrics=['val_loss', 'val_accuracy', 'peak_vram_gb'],
    )

    print("\n" + "=" * 70)
    print("All paper plots generated successfully!")
    print("Output directory: paper_plots/")
    print("\nPlots ready for inclusion in LaTeX with:")
    print("  \\includegraphics[width=0.45\\textwidth]{paper_plots/comparisons/comparison_overlay.pdf}")


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    """
    Run this file to generate example plots.

    Usage:
        python utils/plotting_examples.py

    Or run specific examples:
        python -c "from utils.plotting_examples import example_single_experiment; example_single_experiment()"
    """
    import argparse

    parser = argparse.ArgumentParser(description="Generate example plots")
    parser.add_argument('--example', type=str, default='all',
                       help='Which example to run (single, nested, comparison, grid, efficiency, bar, all, paper)')
    args = parser.parse_args()

    examples = {
        'single': example_single_experiment,
        'nested': example_nested_optimizer,
        'comparison': example_comparison,
        'grid': example_grid_search,
        'efficiency': example_efficiency_frontier,
        'bar': example_final_metrics_bar,
        'all': example_plot_all,
        'paper': generate_paper_plots,
    }

    if args.example in examples:
        print(f"Running example: {args.example}")
        try:
            examples[args.example]()
        except FileNotFoundError as e:
            print(f"\nNote: Example requires experiment results that don't exist yet.")
            print(f"Error: {e}")
            print("\nRun your experiments first, then come back to generate plots.")
    else:
        print(f"Unknown example: {args.example}")
        print(f"Available examples: {', '.join(examples.keys())}")
