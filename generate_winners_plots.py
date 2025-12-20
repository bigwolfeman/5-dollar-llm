"""
Generate comparison plots for Winners Mode training (4800 steps).

Handles the TitanMAC nested model that ran to 6400 steps by truncating to 4800.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Paths
CHECKPOINT_DIR = Path("checkpoints_168m")
OUTPUT_DIR = Path("experiment_results_final/charts/winners_comparison")

# Model configs
MODELS = {
    "moe_muon": {
        "path": CHECKPOINT_DIR / "moe_muon/metrics.json",
        "label": "MoE + Muon",
        "color": "#2ecc71",  # Green
        "linestyle": "-",
    },
    "titanmac_muon": {
        "path": CHECKPOINT_DIR / "titanmac_muon/metrics.json",
        "label": "TitanMAC + Muon",
        "color": "#3498db",  # Blue
        "linestyle": "-",
    },
    "moe_nested_winner": {
        "path": CHECKPOINT_DIR / "moe_nested_winner/metrics.json",
        "label": "MoE + Nested (M3,C2)",
        "color": "#e74c3c",  # Red
        "linestyle": "--",
    },
    "titanmac_nested_winner": {
        "path": CHECKPOINT_DIR / "titanmac_nested_winner/metrics.json",
        "label": "TitanMAC + Nested (M4,C5)",
        "color": "#9b59b6",  # Purple
        "linestyle": "--",
    },
}

TARGET_STEP = 4800


def load_and_normalize_data():
    """Load all model data and normalize to TARGET_STEP."""
    data = {}

    for name, config in MODELS.items():
        with open(config["path"]) as f:
            raw = json.load(f)

        history = raw["history"]
        steps = history["steps"]

        # Find index for TARGET_STEP
        idx_target = None
        for i, s in enumerate(steps):
            if s <= TARGET_STEP:
                idx_target = i

        # Truncate to target step
        truncated = {
            "steps": steps[:idx_target + 1],
            "val_losses": history["val_losses"][:idx_target + 1],
            "val_accuracies": history["val_accuracies"][:idx_target + 1],
            "val_perplexities": history["val_perplexities"][:idx_target + 1],
        }

        # Add optional fields if present
        for field in ["lr_multipliers_core", "lr_multipliers_embed", "meta_losses"]:
            if field in history and history[field]:
                truncated[field] = history[field][:idx_target + 1]

        data[name] = {
            "history": truncated,
            "final_metrics": raw["final_metrics"],
            "config": config,
            "actual_final_step": steps[-1],
            "truncated_to": steps[idx_target],
        }

        print(f"{name}: {steps[0]}-{steps[-1]} -> truncated to {steps[idx_target]}")

    return data


def plot_training_curves(data, output_dir):
    """Plot training curves comparison (loss, accuracy, perplexity)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Val Loss
    ax = axes[0]
    for name, d in data.items():
        h = d["history"]
        cfg = d["config"]
        # Add marker at start if data doesn't start at beginning
        marker = 'o' if h["steps"][0] > 100 else None
        ax.plot(h["steps"], h["val_losses"],
                color=cfg["color"], linestyle=cfg["linestyle"],
                label=cfg["label"], linewidth=2, marker=marker,
                markevery=[0] if marker else [], markersize=8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, TARGET_STEP)
    # Add note about MoE nested
    ax.annotate('* MoE Nested: data starts at step 1600\n  (early history lost during resume)',
               xy=(0.02, 0.02), xycoords='axes fraction',
               fontsize=7, color='gray', style='italic')

    # Add final values annotation
    y_offset = 0
    for name, d in data.items():
        h = d["history"]
        final_loss = h["val_losses"][-1]
        ax.annotate(f'{final_loss:.3f}',
                   xy=(h["steps"][-1], final_loss),
                   xytext=(5, y_offset), textcoords='offset points',
                   fontsize=8, color=d["config"]["color"])
        y_offset += 12

    # Val Accuracy
    ax = axes[1]
    for name, d in data.items():
        h = d["history"]
        cfg = d["config"]
        marker = 'o' if h["steps"][0] > 100 else None
        ax.plot(h["steps"], [a * 100 for a in h["val_accuracies"]],
                color=cfg["color"], linestyle=cfg["linestyle"],
                label=cfg["label"], linewidth=2, marker=marker,
                markevery=[0] if marker else [], markersize=8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Validation Accuracy")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, TARGET_STEP)

    # Val Perplexity
    ax = axes[2]
    for name, d in data.items():
        h = d["history"]
        cfg = d["config"]
        marker = 'o' if h["steps"][0] > 100 else None
        ax.plot(h["steps"], h["val_perplexities"],
                color=cfg["color"], linestyle=cfg["linestyle"],
                label=cfg["label"], linewidth=2, marker=marker,
                markevery=[0] if marker else [], markersize=8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity")
    ax.set_title("Validation Perplexity")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, TARGET_STEP)
    ax.set_ylim(0, 200)  # Cap for readability

    plt.suptitle(f"Winners Comparison @ {TARGET_STEP} Steps", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: training_curves_comparison.png")


def plot_final_metrics_bar(data, output_dir):
    """Bar chart of final metrics at TARGET_STEP."""
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    names = list(data.keys())
    labels = [data[n]["config"]["label"] for n in names]
    colors = [data[n]["config"]["color"] for n in names]

    # Get metrics at truncation point (step 4800)
    losses = [data[n]["history"]["val_losses"][-1] for n in names]
    accuracies = [data[n]["history"]["val_accuracies"][-1] * 100 for n in names]
    perplexities = [data[n]["history"]["val_perplexities"][-1] for n in names]

    x = np.arange(len(names))

    # Val Loss (lower is better)
    ax = axes[0]
    bars = ax.bar(x, losses, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_ylabel("Validation Loss")
    ax.set_title("Validation Loss (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0, max(losses) * 1.15)
    # Add value labels
    for bar, val in zip(bars, losses):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Val Accuracy (higher is better)
    ax = axes[1]
    bars = ax.bar(x, accuracies, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Validation Accuracy (higher is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0, max(accuracies) * 1.15)
    for bar, val in zip(bars, accuracies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Val Perplexity (lower is better)
    ax = axes[2]
    bars = ax.bar(x, perplexities, color=colors, edgecolor='black', linewidth=1.2)
    ax.set_ylabel("Perplexity")
    ax.set_title("Perplexity (lower is better)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha='right', fontsize=9)
    ax.set_ylim(0, max(perplexities) * 1.15)
    for bar, val in zip(bars, perplexities):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.suptitle(f"Final Metrics @ {TARGET_STEP} Steps", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "final_metrics_bar.png", dpi=150)
    plt.close()
    print(f"Saved: final_metrics_bar.png")


def plot_muon_vs_nested(data, output_dir):
    """Side-by-side comparison: Muon vs Nested for each architecture."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # MoE comparison
    moe_muon = data["moe_muon"]
    moe_nested = data["moe_nested_winner"]

    # MoE Loss
    ax = axes[0, 0]
    ax.plot(moe_muon["history"]["steps"], moe_muon["history"]["val_losses"],
            color=moe_muon["config"]["color"], label="Muon+AdamW", linewidth=2)
    ax.plot(moe_nested["history"]["steps"], moe_nested["history"]["val_losses"],
            color=moe_nested["config"]["color"], label="Nested (M3,C2)*", linewidth=2, linestyle="--",
            marker='o', markevery=[0], markersize=8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("MoE: Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, TARGET_STEP)
    ax.annotate('* Nested data starts at step 1600', xy=(0.02, 0.02),
               xycoords='axes fraction', fontsize=7, color='gray', style='italic')

    # MoE Accuracy
    ax = axes[0, 1]
    ax.plot(moe_muon["history"]["steps"],
            [a * 100 for a in moe_muon["history"]["val_accuracies"]],
            color=moe_muon["config"]["color"], label="Muon+AdamW", linewidth=2)
    ax.plot(moe_nested["history"]["steps"],
            [a * 100 for a in moe_nested["history"]["val_accuracies"]],
            color=moe_nested["config"]["color"], label="Nested (M3,C2)*", linewidth=2, linestyle="--",
            marker='o', markevery=[0], markersize=8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("MoE: Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, TARGET_STEP)

    # TitanMAC comparison
    titan_muon = data["titanmac_muon"]
    titan_nested = data["titanmac_nested_winner"]

    # TitanMAC Loss
    ax = axes[1, 0]
    ax.plot(titan_muon["history"]["steps"], titan_muon["history"]["val_losses"],
            color=titan_muon["config"]["color"], label="Muon+AdamW", linewidth=2)
    ax.plot(titan_nested["history"]["steps"], titan_nested["history"]["val_losses"],
            color=titan_nested["config"]["color"], label="Nested (M4,C5)", linewidth=2, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("TitanMAC: Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, TARGET_STEP)

    # TitanMAC Accuracy
    ax = axes[1, 1]
    ax.plot(titan_muon["history"]["steps"],
            [a * 100 for a in titan_muon["history"]["val_accuracies"]],
            color=titan_muon["config"]["color"], label="Muon+AdamW", linewidth=2)
    ax.plot(titan_nested["history"]["steps"],
            [a * 100 for a in titan_nested["history"]["val_accuracies"]],
            color=titan_nested["config"]["color"], label="Nested (M4,C5)", linewidth=2, linestyle="--")
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("TitanMAC: Validation Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, TARGET_STEP)

    plt.suptitle(f"Muon vs Nested Optimizer @ {TARGET_STEP} Steps", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "muon_vs_nested_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: muon_vs_nested_comparison.png")


def plot_architecture_comparison(data, output_dir):
    """Compare MoE vs TitanMAC for each optimizer."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Muon optimizer comparison
    moe_muon = data["moe_muon"]
    titan_muon = data["titanmac_muon"]

    ax = axes[0, 0]
    ax.plot(moe_muon["history"]["steps"], moe_muon["history"]["val_losses"],
            color=moe_muon["config"]["color"], label="MoE", linewidth=2)
    ax.plot(titan_muon["history"]["steps"], titan_muon["history"]["val_losses"],
            color=titan_muon["config"]["color"], label="TitanMAC", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Muon+AdamW: Loss by Architecture")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, TARGET_STEP)

    ax = axes[0, 1]
    ax.plot(moe_muon["history"]["steps"],
            [a * 100 for a in moe_muon["history"]["val_accuracies"]],
            color=moe_muon["config"]["color"], label="MoE", linewidth=2)
    ax.plot(titan_muon["history"]["steps"],
            [a * 100 for a in titan_muon["history"]["val_accuracies"]],
            color=titan_muon["config"]["color"], label="TitanMAC", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Muon+AdamW: Accuracy by Architecture")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, TARGET_STEP)

    # Nested optimizer comparison
    moe_nested = data["moe_nested_winner"]
    titan_nested = data["titanmac_nested_winner"]

    ax = axes[1, 0]
    ax.plot(moe_nested["history"]["steps"], moe_nested["history"]["val_losses"],
            color=moe_nested["config"]["color"], label="MoE (M3,C2)*", linewidth=2,
            marker='o', markevery=[0], markersize=8)
    ax.plot(titan_nested["history"]["steps"], titan_nested["history"]["val_losses"],
            color=titan_nested["config"]["color"], label="TitanMAC (M4,C5)", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Loss")
    ax.set_title("Nested Optimizer: Loss by Architecture")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, TARGET_STEP)
    ax.annotate('* MoE data starts at step 1600', xy=(0.02, 0.02),
               xycoords='axes fraction', fontsize=7, color='gray', style='italic')

    ax = axes[1, 1]
    ax.plot(moe_nested["history"]["steps"],
            [a * 100 for a in moe_nested["history"]["val_accuracies"]],
            color=moe_nested["config"]["color"], label="MoE (M3,C2)*", linewidth=2,
            marker='o', markevery=[0], markersize=8)
    ax.plot(titan_nested["history"]["steps"],
            [a * 100 for a in titan_nested["history"]["val_accuracies"]],
            color=titan_nested["config"]["color"], label="TitanMAC (M4,C5)", linewidth=2)
    ax.set_xlabel("Step")
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Nested Optimizer: Accuracy by Architecture")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, TARGET_STEP)

    plt.suptitle(f"Architecture Comparison @ {TARGET_STEP} Steps", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_dir / "architecture_comparison.png", dpi=150)
    plt.close()
    print(f"Saved: architecture_comparison.png")


def plot_summary_table(data, output_dir):
    """Create a summary table as an image."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis('off')

    # Prepare table data
    headers = ["Model", "Optimizer", "Val Loss", "Val Acc", "Perplexity", "Steps"]
    rows = []

    for name, d in data.items():
        h = d["history"]
        cfg = d["config"]

        # Determine model and optimizer from name
        if "moe" in name:
            model = "MoE"
        else:
            model = "TitanMAC"

        if "muon" in name:
            optimizer = "Muon+AdamW"
        else:
            optimizer = "Nested"

        rows.append([
            model,
            optimizer,
            f"{h['val_losses'][-1]:.4f}",
            f"{h['val_accuracies'][-1]*100:.2f}%",
            f"{h['val_perplexities'][-1]:.2f}",
            str(d["truncated_to"]),
        ])

    # Sort by val_loss
    rows.sort(key=lambda x: float(x[2]))

    # Create table
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)

    # Style header
    for j, header in enumerate(headers):
        table[(0, j)].set_facecolor('#4a90d9')
        table[(0, j)].set_text_props(color='white', fontweight='bold')

    # Highlight best row (first after sorting)
    for j in range(len(headers)):
        table[(1, j)].set_facecolor('#d4edda')

    plt.title(f"Winners Comparison Summary @ {TARGET_STEP} Steps\n(sorted by validation loss)",
              fontsize=12, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(output_dir / "summary_table.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: summary_table.png")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading and normalizing data...\n")
    data = load_and_normalize_data()

    print("\nGenerating plots...")
    plot_training_curves(data, OUTPUT_DIR)
    plot_final_metrics_bar(data, OUTPUT_DIR)
    plot_muon_vs_nested(data, OUTPUT_DIR)
    plot_architecture_comparison(data, OUTPUT_DIR)
    plot_summary_table(data, OUTPUT_DIR)

    print(f"\nAll plots saved to: {OUTPUT_DIR}")

    # Print summary
    print("\n" + "=" * 60)
    print(f"SUMMARY @ {TARGET_STEP} STEPS")
    print("=" * 60)

    results = []
    for name, d in data.items():
        h = d["history"]
        results.append({
            "name": d["config"]["label"],
            "loss": h["val_losses"][-1],
            "acc": h["val_accuracies"][-1] * 100,
            "ppl": h["val_perplexities"][-1],
        })

    results.sort(key=lambda x: x["loss"])

    for i, r in enumerate(results, 1):
        print(f"{i}. {r['name']}")
        print(f"   Loss: {r['loss']:.4f} | Acc: {r['acc']:.2f}% | PPL: {r['ppl']:.2f}")


if __name__ == "__main__":
    main()
