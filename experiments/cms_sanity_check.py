#!/usr/bin/env python3
"""
CMS Sanity Check: Verify DeepNestedOptimizer fixes

Tests the hypothesis: cms_frequencies=[1] (single level) should approximately
match Muon+AdamW baseline since it eliminates multi-level complexity.

This validates that the core momentum MLP and optimizer mechanics work before
testing multi-level CMS.

Usage:
    python experiments/cms_sanity_check.py

Expected: Single-level CMS should reach ~4.0 loss (similar to baseline)
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add parent dir to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

# Suppress tokenizer warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def run_sanity_check():
    """Run a 600-step comparison: baseline vs single-level CMS."""

    print(f"""
{'='*70}
CMS SANITY CHECK
Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
{'='*70}

This test validates the CMS bug fixes by comparing:
1. Muon+AdamW baseline (600 steps)
2. Single-level CMS (cms_frequencies=[1]) (600 steps)

If single-level CMS matches baseline (~4.0 loss), the core mechanics work.
Then we can test multi-level CMS with confidence.
""")

    from configs.moe_config import GPU24GBMoEModelConfig
    from models.moe_llm import MoEMinimalLLM
    from data.loader import get_dataloaders
    from training.trainer import train_model
    from optimizers.muon import Muon
    from titans_core.opt import DeepNestedOptimizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    config = GPU24GBMoEModelConfig()
    config.vocab_size = 49152  # SmolLM

    max_steps = 600
    results = {}

    # Test 1: Baseline (Muon+AdamW)
    print(f"\n{'='*70}")
    print("TEST 1: Muon+AdamW Baseline")
    print(f"{'='*70}\n")

    model = MoEMinimalLLM(config).to(device)
    train_loader, val_loader = get_dataloaders(
        batch_size=config.batch_size,
        seq_length=config.seq_length,
        cache_dir=".cache/sanity_check",
    )

    # Muon for 2D weights, AdamW for rest
    muon_params = []
    adamw_params = []
    for name, p in model.named_parameters():
        if p.ndim >= 2 and 'embed' not in name.lower() and 'norm' not in name.lower():
            muon_params.append(p)
        else:
            adamw_params.append(p)

    optimizer = Muon(muon_params, lr=0.02, momentum=0.95)
    adamw = torch.optim.AdamW(adamw_params, lr=3e-4, weight_decay=0.01)

    start = time.time()

    # Simple training loop
    model.train()
    step = 0
    for batch in train_loader:
        if step >= max_steps:
            break

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids, return_aux_loss=True)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs

        # Cross-entropy loss
        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        adamw.step()
        optimizer.zero_grad()
        adamw.zero_grad()

        if step % 100 == 0:
            print(f"Step {step}: loss={loss.item():.4f}")

        step += 1

    baseline_time = time.time() - start
    baseline_loss = loss.item()
    results["baseline"] = {"loss": baseline_loss, "time": baseline_time}
    print(f"\nBaseline final loss: {baseline_loss:.4f} ({baseline_time/60:.1f} min)")

    del model, optimizer, adamw
    torch.cuda.empty_cache()

    # Test 2: Single-level CMS
    print(f"\n{'='*70}")
    print("TEST 2: Single-level CMS (cms_frequencies=[1])")
    print(f"{'='*70}\n")

    model = MoEMinimalLLM(config).to(device)

    optimizer = DeepNestedOptimizer(
        model,
        base_lr=3e-4,
        meta_lr=1e-4,
        cms_frequencies=[1],  # Single level - should match baseline
        momentum_hidden_dim=64,
        momentum_num_layers=2,
        controller_hidden_dim=32,
        controller_num_layers=2,
        mode='simple',
        meta_update_freq=100,
    )

    start = time.time()

    model.train()
    step = 0
    for batch in train_loader:
        if step >= max_steps:
            break

        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)

        optimizer.zero_grad()

        outputs = model(input_ids, return_aux_loss=True)
        logits = outputs[0] if isinstance(outputs, tuple) else outputs

        loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
        )

        loss.backward()
        optimizer.step(loss.item())

        if step % 100 == 0:
            print(f"Step {step}: loss={loss.item():.4f}")

        step += 1

    cms1_time = time.time() - start
    cms1_loss = loss.item()
    results["single_cms"] = {"loss": cms1_loss, "time": cms1_time}
    print(f"\nSingle-level CMS final loss: {cms1_loss:.4f} ({cms1_time/60:.1f} min)")

    del model, optimizer
    torch.cuda.empty_cache()

    # Summary
    print(f"""
{'='*70}
SANITY CHECK RESULTS
{'='*70}

| Configuration      | Final Loss | Time      |
|--------------------|------------|-----------|
| Muon+AdamW         | {results['baseline']['loss']:.4f}     | {results['baseline']['time']/60:.1f} min  |
| Single-level CMS   | {results['single_cms']['loss']:.4f}     | {results['single_cms']['time']/60:.1f} min  |

Analysis:
""")

    ratio = cms1_loss / baseline_loss
    if ratio < 1.5:
        print(f"  PASS: Single-level CMS is within 50% of baseline (ratio: {ratio:.2f})")
        print("  The core optimizer mechanics are working correctly.")
        print("  Ready to test multi-level CMS (cms_frequencies=[1,10,100]).")
    elif ratio < 2.0:
        print(f"  WARN: Single-level CMS is 50-100% worse than baseline (ratio: {ratio:.2f})")
        print("  May indicate remaining issues with momentum MLP or controller.")
    else:
        print(f"  FAIL: Single-level CMS is 2x+ worse than baseline (ratio: {ratio:.2f})")
        print("  Critical bugs remain. Check momentum MLP initialization/training.")

    print(f"\n{'='*70}")


if __name__ == "__main__":
    run_sanity_check()
