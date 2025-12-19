#!/usr/bin/env python3
"""
Test: Single-level CMS vs Multi-level CMS

Hypothesis: The multi-level averaging is crippling updates.
Single-level CMS (just momentum) should perform better.

This also compares to what plain AdamW would do.
"""

import os
import sys
import time
import math
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


def train_with_optimizer(model, train_loader, optimizer, max_steps, name, use_cms=False):
    """Train and return loss curve."""
    device = next(model.parameters()).device
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

        if use_cms:
            scaler.unscale_(optimizer.base_optimizer)
            optimizer.step(loss_value=loss.item())
        else:
            scaler.step(optimizer)

        scaler.update()

        if step % 50 == 0:
            losses.append((step, loss.item()))
            print(f"  [{name}] Step {step}: loss={loss.item():.4f}")

        step += 1

    return losses


def main():
    print("="*70)
    print("CMS SINGLE-LEVEL vs MULTI-LEVEL TEST")
    print("="*70)

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
    print("\n" + "="*70)
    print("TEST 1: Plain AdamW")
    print("="*70)

    set_seed(42)
    model = MoEMinimalLLM(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    results['adamw'] = train_with_optimizer(model, train_loader, optimizer, max_steps, "AdamW")
    del model, optimizer
    torch.cuda.empty_cache()

    # Test 2: Single-level CMS (cms_frequencies=[1])
    print("\n" + "="*70)
    print("TEST 2: Single-level CMS (no multi-level)")
    print("="*70)

    from titans_core.opt import DeepNestedOptimizer

    set_seed(42)
    model = MoEMinimalLLM(config).to(device)
    optimizer = DeepNestedOptimizer(
        model,
        base_lr=3e-4,
        meta_lr=1e-4,
        cms_frequencies=[1],  # SINGLE LEVEL ONLY
        momentum_hidden_dim=64,
        momentum_num_layers=2,
        controller_hidden_dim=32,
        controller_num_layers=2,
        mode='simple',
        meta_update_freq=50,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )

    results['cms_single'] = train_with_optimizer(
        model, train_loader, optimizer, max_steps, "CMS-Single", use_cms=True
    )
    del model, optimizer
    torch.cuda.empty_cache()

    # Test 3: Multi-level CMS (cms_frequencies=[1,10,100])
    print("\n" + "="*70)
    print("TEST 3: Multi-level CMS (with averaging bug)")
    print("="*70)

    set_seed(42)
    model = MoEMinimalLLM(config).to(device)
    optimizer = DeepNestedOptimizer(
        model,
        base_lr=3e-4,
        meta_lr=1e-4,
        cms_frequencies=[1, 10, 100],  # MULTI-LEVEL
        momentum_hidden_dim=64,
        momentum_num_layers=2,
        controller_hidden_dim=32,
        controller_num_layers=2,
        mode='simple',
        meta_update_freq=50,
        weight_decay=0.0,
        max_grad_norm=1.0,
    )

    results['cms_multi'] = train_with_optimizer(
        model, train_loader, optimizer, max_steps, "CMS-Multi", use_cms=True
    )
    del model, optimizer
    torch.cuda.empty_cache()

    # Summary
    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)

    print(f"\n{'Step':<10} {'AdamW':<15} {'CMS-Single':<15} {'CMS-Multi':<15}")
    print("-"*55)

    for i in range(len(results['adamw'])):
        step = results['adamw'][i][0]
        adamw_loss = results['adamw'][i][1]
        single_loss = results['cms_single'][i][1] if i < len(results['cms_single']) else float('nan')
        multi_loss = results['cms_multi'][i][1] if i < len(results['cms_multi']) else float('nan')
        print(f"{step:<10} {adamw_loss:<15.4f} {single_loss:<15.4f} {multi_loss:<15.4f}")

    print("\n" + "="*70)
    print("ANALYSIS")
    print("="*70)

    final_adamw = results['adamw'][-1][1]
    final_single = results['cms_single'][-1][1]
    final_multi = results['cms_multi'][-1][1]

    print(f"\nFinal losses at step {max_steps}:")
    print(f"  AdamW:      {final_adamw:.4f}")
    print(f"  CMS-Single: {final_single:.4f} ({final_single/final_adamw:.2f}x AdamW)")
    print(f"  CMS-Multi:  {final_multi:.4f} ({final_multi/final_adamw:.2f}x AdamW)")

    if final_single < final_multi:
        print("\n✓ Single-level CMS is BETTER than multi-level.")
        print("  → The multi-level averaging is indeed hurting performance.")
    else:
        print("\n✗ Multi-level CMS is better or equal.")
        print("  → The problem is elsewhere.")

    if final_single > final_adamw * 1.5:
        print("\n⚠ CMS is significantly worse than AdamW.")
        print("  → The momentum MLP design may be fundamentally limited.")


if __name__ == "__main__":
    main()
