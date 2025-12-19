#!/usr/bin/env python3
"""
Test: Paper-Aligned CMS vs AdamW Baseline

This tests the paper-aligned CMS optimizer changes:
1. Second-moment tracking (Adam-style adaptive LR)
2. L2 regression loss for MomentumMLP
3. Frequency-based level weighting
4. Direct loss-delta meta-training signal

Compares:
- Plain AdamW (baseline)
- CMS (paper-aligned, new default)
- CMS with AdamW fallback (for reference)
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler

from configs.moe_config import GPU24GBMoEModelConfig
from configs.dataset_config import DataConfig
from models.moe_llm import MoEMinimalLLM
from data.loader import setup_tokenizer, load_smollm_corpus, tokenize_and_chunk, finalize_dataset
from datasets import Dataset as HFDataset
from utils.helpers import set_seed


def train_with_adamw(model, train_loader, max_steps, lr=3e-4):
    """Train with plain AdamW (baseline)."""
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.1)
    scaler = GradScaler()

    losses = []
    model.train()
    step = 0

    for batch in train_loader:
        if step >= max_steps:
            break

        x = batch['input_ids'].to(device)
        y = batch['labels'].to(device)

        optimizer.zero_grad()

        with autocast('cuda', dtype=torch.float16):
            logits, aux_loss = model(x, return_aux_loss=True)
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                y[:, 1:].contiguous().view(-1)
            )
            if aux_loss is not None:
                loss = loss + aux_loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        if step % 50 == 0:
            losses.append((step, loss.item()))
            print(f"  [AdamW] Step {step}: loss={loss.item():.4f}")

        step += 1

    return losses


def train_with_cms(model, train_loader, max_steps, use_cms_updates=True, name="CMS"):
    """Train with CMS optimizer."""
    from titans_core.opt import DeepNestedOptimizer

    device = next(model.parameters()).device
    optimizer = DeepNestedOptimizer(
        model,
        base_lr=3e-4,
        meta_lr=1e-4,
        cms_frequencies=[1, 10, 100],
        momentum_hidden_dim=64,
        momentum_num_layers=2,
        controller_hidden_dim=32,
        controller_num_layers=2,
        mode='simple',
        meta_update_freq=50,
        weight_decay=0.0,
        max_grad_norm=1.0,
        use_cms_updates=use_cms_updates,
    )

    scaler = GradScaler()
    losses = []
    model.train()
    step = 0

    for batch in train_loader:
        if step >= max_steps:
            break

        x = batch['input_ids'].to(device)
        y = batch['labels'].to(device)

        optimizer.zero_grad()

        with autocast('cuda', dtype=torch.float16):
            logits, aux_loss = model(x, return_aux_loss=True)
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, logits.size(-1)),
                y[:, 1:].contiguous().view(-1)
            )
            if aux_loss is not None:
                loss = loss + aux_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer.base_optimizer)
        optimizer.step(loss_value=loss.item())
        scaler.update()

        if step % 50 == 0:
            losses.append((step, loss.item()))
            lr_mults = optimizer.get_lr_multipliers()
            print(f"  [{name}] Step {step}: loss={loss.item():.4f}, "
                  f"lr_mult=[{lr_mults[0].item():.3f}, {lr_mults[1].item():.3f}]")

        step += 1

    return losses


def main():
    print("=" * 70)
    print("PAPER-ALIGNED CMS VS BASELINE TEST")
    print("=" * 70)

    device = torch.device('cuda')
    set_seed(42)

    # Load data
    config = GPU24GBMoEModelConfig()
    data_cfg = DataConfig(seq_length=config.max_seq_len, num_samples=20000)
    tokenizer = setup_tokenizer(data_cfg)
    config.vocab_size = tokenizer.vocab_size

    print("Loading data...")
    raw_dataset = load_smollm_corpus(data_cfg)
    raw_samples = list(raw_dataset.take(data_cfg.num_samples))
    train_ds = finalize_dataset(
        tokenize_and_chunk(HFDataset.from_list(raw_samples), tokenizer, data_cfg),
        data_cfg
    )
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=2)

    max_steps = 300
    results = {}

    # Test 1: Plain AdamW (baseline reference)
    print("\n" + "=" * 70)
    print("TEST 1: Plain AdamW (baseline)")
    print("=" * 70)

    set_seed(42)
    model = MoEMinimalLLM(config).to(device)
    results['adamw'] = train_with_adamw(model, train_loader, max_steps)
    del model
    torch.cuda.empty_cache()

    # Test 2: Paper-Aligned CMS (new default)
    print("\n" + "=" * 70)
    print("TEST 2: Paper-Aligned CMS (new default)")
    print("=" * 70)

    set_seed(42)
    model = MoEMinimalLLM(config).to(device)
    results['cms_paper'] = train_with_cms(model, train_loader, max_steps, use_cms_updates=True, name="CMS-Paper")
    del model
    torch.cuda.empty_cache()

    # Test 3: CMS with AdamW fallback (for comparison)
    print("\n" + "=" * 70)
    print("TEST 3: CMS with AdamW fallback (old mode)")
    print("=" * 70)

    set_seed(42)
    model = MoEMinimalLLM(config).to(device)
    results['cms_adamw'] = train_with_cms(model, train_loader, max_steps, use_cms_updates=False, name="CMS-AdamW")
    del model
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Step':<10} {'AdamW':<15} {'CMS-Paper':<15} {'CMS-AdamW':<15}")
    print("-" * 55)

    for i in range(len(results['adamw'])):
        step = results['adamw'][i][0]
        adamw_loss = results['adamw'][i][1]
        cms_paper_loss = results['cms_paper'][i][1] if i < len(results['cms_paper']) else float('nan')
        cms_adamw_loss = results['cms_adamw'][i][1] if i < len(results['cms_adamw']) else float('nan')
        print(f"{step:<10} {adamw_loss:<15.4f} {cms_paper_loss:<15.4f} {cms_adamw_loss:<15.4f}")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    final_adamw = results['adamw'][-1][1]
    final_cms_paper = results['cms_paper'][-1][1]
    final_cms_adamw = results['cms_adamw'][-1][1]

    print(f"\nFinal losses at step {max_steps}:")
    print(f"  AdamW (baseline):  {final_adamw:.4f}")
    print(f"  CMS-Paper:         {final_cms_paper:.4f} ({final_cms_paper/final_adamw:.2f}x AdamW)")
    print(f"  CMS-AdamW:         {final_cms_adamw:.4f} ({final_cms_adamw/final_adamw:.2f}x AdamW)")

    if final_cms_paper < final_adamw * 1.2:
        print("\n✓ Paper-aligned CMS is competitive with AdamW!")
        if final_cms_paper < final_cms_adamw:
            print("  → Paper-aligned CMS beats AdamW fallback mode!")
    else:
        print("\n✗ Paper-aligned CMS is not yet competitive.")
        print("  → May need more tuning or longer training.")


if __name__ == "__main__":
    main()
