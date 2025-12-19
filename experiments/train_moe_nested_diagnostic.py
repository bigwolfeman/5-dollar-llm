#!/usr/bin/env python3
"""
DIAGNOSTIC VERSION: MoE training with DeepNestedOptimizer

This script adds extensive logging to understand what CMS is doing at runtime.
Use this to debug why the nested optimizer is performing poorly.

Logs:
- Per-step: scale/shift/damping from MLP, momentum norms, which levels fire
- Every 10 steps: detailed CMS state dump
- Every 50 steps: comparison of update magnitudes to baseline SGD

Usage:
    python experiments/train_moe_nested_diagnostic.py --max_steps 200
"""

import argparse
import time
import os
import math
import json
import torch
import torch.nn.functional as F
import logging
from collections import deque, defaultdict
from pathlib import Path
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Optional, Dict, Any, List
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))
# Add TitanMAC for nested optimizer
sys.path.insert(0, str(Path(__file__).parent.parent / "111TitanMAC-Standalone"))

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from configs.moe_config import GPU24GBMoEModelConfig
from models.moe_llm import MoEMinimalLLM
from titans_core.opt import DeepNestedOptimizer, group_moe_params
from utils.helpers import set_seed


class CMSDiagnostics:
    """Collector for CMS diagnostic data."""

    def __init__(self):
        self.step_data = []
        self.current_step = {}

    def start_step(self, step: int, loss: float):
        self.current_step = {
            'step': step,
            'loss': loss,
            'levels_active': [],
            'mlp_outputs': [],  # (scale, shift, damping) per param
            'momentum_norms': defaultdict(list),  # level -> [norms]
            'grad_norms': [],
            'update_norms': [],
            'total_update_norms': [],
        }

    def log_level_activation(self, level: int, param_idx: int,
                             scale: float, shift: float, damping: float,
                             grad_norm: float, momentum_norm: float,
                             update_norm: float):
        self.current_step['levels_active'].append(level)
        self.current_step['mlp_outputs'].append({
            'level': level,
            'param_idx': param_idx,
            'scale': scale,
            'shift': shift,
            'damping': damping,
        })
        self.current_step['momentum_norms'][level].append(momentum_norm)
        self.current_step['grad_norms'].append(grad_norm)
        self.current_step['update_norms'].append(update_norm)

    def log_total_update(self, param_idx: int, total_update_norm: float, num_active: int):
        self.current_step['total_update_norms'].append({
            'param_idx': param_idx,
            'norm': total_update_norm,
            'num_active_levels': num_active,
        })

    def end_step(self):
        self.step_data.append(self.current_step)
        self.current_step = {}

    def get_summary(self, last_n: int = 10) -> str:
        """Get summary of last N steps."""
        if not self.step_data:
            return "No data collected yet"

        recent = self.step_data[-last_n:]

        lines = [f"\n{'='*70}", "CMS DIAGNOSTICS SUMMARY", f"{'='*70}"]

        for step_data in recent:
            step = step_data['step']
            loss = step_data['loss']

            # Count level activations
            level_counts = defaultdict(int)
            for l in step_data['levels_active']:
                level_counts[l] += 1

            # Get MLP output stats
            if step_data['mlp_outputs']:
                scales = [m['scale'] for m in step_data['mlp_outputs']]
                shifts = [m['shift'] for m in step_data['mlp_outputs']]
                dampings = [m['damping'] for m in step_data['mlp_outputs']]

                avg_scale = sum(scales) / len(scales)
                avg_shift = sum(shifts) / len(shifts)
                avg_damping = sum(dampings) / len(dampings)
            else:
                avg_scale = avg_shift = avg_damping = 0

            # Get momentum stats per level
            momentum_by_level = {}
            for level, norms in step_data['momentum_norms'].items():
                if norms:
                    momentum_by_level[level] = sum(norms) / len(norms)

            # Get gradient vs update ratio
            if step_data['grad_norms'] and step_data['update_norms']:
                avg_grad = sum(step_data['grad_norms']) / len(step_data['grad_norms'])
                avg_update = sum(step_data['update_norms']) / len(step_data['update_norms'])
                ratio = avg_update / max(avg_grad, 1e-8)
            else:
                avg_grad = avg_update = ratio = 0

            lines.append(f"\nStep {step}: loss={loss:.4f}")
            lines.append(f"  Levels fired: {dict(level_counts)}")
            lines.append(f"  MLP outputs: scale={avg_scale:.4f}, shift={avg_shift:.4f}, damping={avg_damping:.4f}")
            lines.append(f"  Momentum norms by level: {momentum_by_level}")
            lines.append(f"  Avg grad norm: {avg_grad:.6f}, Avg update norm: {avg_update:.6f}")
            lines.append(f"  Update/Grad ratio: {ratio:.4f}")

        return '\n'.join(lines)


def create_instrumented_optimizer(model, base_lr, diagnostics: CMSDiagnostics, **kwargs):
    """Create a DeepNestedOptimizer with instrumented step() for diagnostics."""

    optimizer = DeepNestedOptimizer(model, base_lr=base_lr, **kwargs)

    # Store original step
    original_step = optimizer.step

    def instrumented_step(loss_value=None):
        """Step with diagnostic logging."""

        optimizer.global_step += 1

        actual_loss = loss_value if loss_value is not None else optimizer._pending_loss
        if actual_loss is None:
            actual_loss = 0.0
        optimizer._pending_loss = None

        diagnostics.start_step(optimizer.global_step, actual_loss)

        # Update EMA loss
        if optimizer.global_step == 1:
            optimizer.ema_loss.fill_(actual_loss)
        else:
            optimizer.ema_loss = (1 - optimizer.beta_ema) * optimizer.ema_loss + optimizer.beta_ema * actual_loss

        # Get controller LR multipliers
        stats = optimizer._compute_group_stats()
        with torch.no_grad():
            optimizer._lr_multipliers = optimizer.controller(stats)

        # Clip gradients
        if optimizer.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(optimizer.model.parameters(), optimizer.max_grad_norm)

        context = optimizer._get_context(actual_loss)
        effective_lrs = (optimizer.base_lr * optimizer._lr_multipliers).tolist()

        param_idx = 0
        with torch.no_grad():
            for group_idx, group in enumerate(optimizer.base_optimizer.param_groups):
                lr = effective_lrs[group_idx]

                for param in group['params']:
                    if param.grad is None:
                        param_idx += 1
                        continue

                    grad = param.grad
                    cms = optimizer.state[param]

                    # Accumulate gradient
                    cms.accumulate_grad(grad)

                    total_update = torch.zeros_like(param)
                    num_active_levels = 0

                    for level in range(len(optimizer.cms_frequencies)):
                        if cms.should_update(level, optimizer.global_step):
                            level_grad = cms.get_update(level)
                            prev_momentum = cms.get_momentum(level)

                            freq = optimizer.cms_frequencies[level] if level < len(optimizer.cms_frequencies) else 1
                            level_grad = level_grad / max(freq, 1)

                            # Get MLP outputs
                            scale, shift, damping = optimizer.momentum_mlp(
                                level_grad, prev_momentum, context
                            )

                            new_momentum = scale * prev_momentum + shift * level_grad

                            # Soft clamp
                            grad_norm = level_grad.norm().item()
                            momentum_norm = new_momentum.norm().item()

                            if momentum_norm > 100.0 * max(grad_norm, 1e-8):
                                new_momentum = new_momentum * (100.0 * grad_norm / momentum_norm)
                                momentum_norm = 100.0 * grad_norm

                            cms.update_momentum(level, new_momentum)

                            level_weight = 1.0
                            level_update = new_momentum * (1 - damping) * level_weight

                            update_norm = level_update.norm().item()

                            # Log diagnostics
                            diagnostics.log_level_activation(
                                level=level,
                                param_idx=param_idx,
                                scale=scale.item(),
                                shift=shift.item(),
                                damping=damping.item(),
                                grad_norm=grad_norm,
                                momentum_norm=momentum_norm,
                                update_norm=update_norm,
                            )

                            if not (torch.isnan(level_update).any() or torch.isinf(level_update).any()):
                                total_update += level_update
                                num_active_levels += 1

                    if num_active_levels > 0:
                        total_update = total_update / num_active_levels

                        total_norm = total_update.norm().item()
                        diagnostics.log_total_update(param_idx, total_norm, num_active_levels)

                        if not (torch.isnan(total_update).any() or torch.isinf(total_update).any()):
                            if optimizer.weight_decay > 0:
                                total_update = total_update + optimizer.weight_decay * param
                            param.add_(total_update, alpha=-lr)

                    param_idx += 1

        diagnostics.end_step()

        # Meta update in simple mode
        if optimizer.mode == 'simple' and optimizer.global_step % optimizer.meta_update_freq == 0:
            optimizer._update_meta_components(actual_loss)

        return {
            'global_step': optimizer.global_step,
            'lr_multipliers': optimizer._lr_multipliers.clone(),
            'ema_loss': optimizer.ema_loss.clone(),
        }

    optimizer.step = instrumented_step
    return optimizer


def main():
    print(f"""
{'='*70}
CMS DIAGNOSTIC TRAINING
{'='*70}
This will log detailed CMS state every step to identify the problem.
""")

    parser = argparse.ArgumentParser()
    parser.add_argument("--max_steps", type=int, default=200)
    parser.add_argument("--log_every", type=int, default=20, help="Print diagnostics every N steps")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    set_seed(42)

    # Load data
    from configs.dataset_config import DataConfig
    from data.loader import setup_tokenizer, load_smollm_corpus, tokenize_and_chunk, finalize_dataset
    from datasets import Dataset as HFDataset

    config = GPU24GBMoEModelConfig()
    data_cfg = DataConfig(seq_length=config.max_seq_len, num_samples=10000)  # Smaller for diagnostics
    tokenizer = setup_tokenizer(data_cfg)
    config.vocab_size = tokenizer.vocab_size

    # Load and prepare dataset
    print("Loading dataset...")
    raw_dataset = load_smollm_corpus(data_cfg)
    raw_samples = list(raw_dataset.take(data_cfg.num_samples))

    num_val = int(len(raw_samples) * 0.1)
    raw_train = HFDataset.from_list(raw_samples[:-num_val])
    raw_val = HFDataset.from_list(raw_samples[-num_val:])

    print("Tokenizing...")
    train_ds = finalize_dataset(tokenize_and_chunk(raw_train, tokenizer, data_cfg), data_cfg)
    val_ds = finalize_dataset(tokenize_and_chunk(raw_val, tokenizer, data_cfg), data_cfg)

    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False, num_workers=2)

    # Create model
    model = MoEMinimalLLM(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Create diagnostics collector
    diagnostics = CMSDiagnostics()

    # Create instrumented optimizer
    optimizer = create_instrumented_optimizer(
        model,
        base_lr=3e-4,
        diagnostics=diagnostics,
        meta_lr=1e-4,
        cms_frequencies=[1, 10, 100],
        momentum_hidden_dim=64,
        momentum_num_layers=2,
        controller_hidden_dim=32,
        controller_num_layers=2,
        mode='simple',
        meta_update_freq=50,
        weight_decay=0.0,  # Disable for cleaner diagnostics
        max_grad_norm=1.0,
    )

    scaler = GradScaler()

    print(f"\nStarting training for {args.max_steps} steps...")
    print(f"CMS frequencies: {optimizer.cms_frequencies}")
    print(f"Will log diagnostics every {args.log_every} steps\n")

    # Also track what standard SGD would do
    sgd_update_norms = []
    cms_update_norms = []

    model.train()
    step = 0

    for batch in train_loader:
        if step >= args.max_steps:
            break

        x = batch['input_ids'].to(device)
        y = batch['labels'].to(device)

        optimizer.zero_grad()

        with autocast('cuda', dtype=torch.float16):
            logits, aux_loss = model(x, return_aux_loss=True)
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = y[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, config.vocab_size),
                shift_labels.view(-1)
            )
            if aux_loss is not None:
                loss = loss + aux_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer.base_optimizer)

        # Compute what plain SGD update would be (for comparison)
        sgd_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                sgd_norm += (p.grad * 3e-4).pow(2).sum().item()  # -lr * grad
        sgd_norm = math.sqrt(sgd_norm)
        sgd_update_norms.append(sgd_norm)

        # Step optimizer (this logs to diagnostics)
        optimizer.step(loss_value=loss.item())

        # Get CMS update norm from diagnostics
        if diagnostics.step_data:
            last_step = diagnostics.step_data[-1]
            if last_step['total_update_norms']:
                cms_norm = sum(u['norm'] for u in last_step['total_update_norms'])
                cms_update_norms.append(cms_norm)
            else:
                cms_update_norms.append(0.0)

        scaler.update()

        # Print diagnostics
        if step % args.log_every == 0:
            print(f"\n--- Step {step} ---")
            print(f"Loss: {loss.item():.4f}")
            print(f"LR multipliers: {optimizer.get_lr_multipliers().tolist()}")

            if diagnostics.step_data:
                last = diagnostics.step_data[-1]

                # Level activation summary
                level_counts = defaultdict(int)
                for l in last['levels_active']:
                    level_counts[l] += 1
                print(f"Levels fired this step: {dict(level_counts)}")

                # MLP output summary
                if last['mlp_outputs']:
                    scales = [m['scale'] for m in last['mlp_outputs']]
                    shifts = [m['shift'] for m in last['mlp_outputs']]
                    dampings = [m['damping'] for m in last['mlp_outputs']]
                    print(f"MLP scale: min={min(scales):.4f}, max={max(scales):.4f}, mean={sum(scales)/len(scales):.4f}")
                    print(f"MLP shift: min={min(shifts):.4f}, max={max(shifts):.4f}, mean={sum(shifts)/len(shifts):.4f}")
                    print(f"MLP damping: min={min(dampings):.4f}, max={max(dampings):.4f}, mean={sum(dampings)/len(dampings):.4f}")

                # Update magnitude comparison
                if sgd_update_norms and cms_update_norms:
                    sgd_avg = sum(sgd_update_norms[-10:]) / min(10, len(sgd_update_norms))
                    cms_avg = sum(cms_update_norms[-10:]) / min(10, len(cms_update_norms))
                    ratio = cms_avg / max(sgd_avg, 1e-10)
                    print(f"Update magnitude: SGD={sgd_avg:.6f}, CMS={cms_avg:.6f}, ratio={ratio:.4f}")

        step += 1

    # Final summary
    print("\n" + "="*70)
    print("FINAL DIAGNOSTICS SUMMARY")
    print("="*70)

    if diagnostics.step_data:
        # Analyze MLP outputs over training
        all_scales = []
        all_shifts = []
        all_dampings = []

        for step_data in diagnostics.step_data:
            for m in step_data['mlp_outputs']:
                all_scales.append(m['scale'])
                all_shifts.append(m['shift'])
                all_dampings.append(m['damping'])

        if all_scales:
            print(f"\nMLP Output Distribution (all steps):")
            print(f"  Scale:   min={min(all_scales):.4f}, max={max(all_scales):.4f}, mean={sum(all_scales)/len(all_scales):.4f}")
            print(f"  Shift:   min={min(all_shifts):.4f}, max={max(all_shifts):.4f}, mean={sum(all_shifts)/len(all_shifts):.4f}")
            print(f"  Damping: min={min(all_dampings):.4f}, max={max(all_dampings):.4f}, mean={sum(all_dampings)/len(all_dampings):.4f}")

        # Check if damping is too high
        avg_damping = sum(all_dampings) / len(all_dampings) if all_dampings else 0
        if avg_damping > 0.5:
            print(f"\n⚠️  WARNING: Average damping is {avg_damping:.2f} - this reduces updates by {avg_damping*100:.0f}%!")

        # Check if scale is too low (momentum decaying too fast)
        avg_scale = sum(all_scales) / len(all_scales) if all_scales else 0
        if avg_scale < 0.7:
            print(f"\n⚠️  WARNING: Average scale is {avg_scale:.2f} - momentum decays quickly (1/e in {-1/math.log(avg_scale+1e-10):.1f} steps)")

        # Check update magnitude
        if sgd_update_norms and cms_update_norms:
            sgd_total = sum(sgd_update_norms)
            cms_total = sum(cms_update_norms)
            ratio = cms_total / max(sgd_total, 1e-10)
            print(f"\nUpdate Magnitude Comparison:")
            print(f"  Total SGD updates (if we used SGD): {sgd_total:.6f}")
            print(f"  Total CMS updates (what we actually used): {cms_total:.6f}")
            print(f"  Ratio (CMS/SGD): {ratio:.4f}")

            if ratio < 0.1:
                print(f"\n⚠️  CRITICAL: CMS updates are {ratio*100:.1f}% of what SGD would be!")
                print("    This explains the slow learning. The optimizer is barely updating parameters.")
            elif ratio > 10:
                print(f"\n⚠️  WARNING: CMS updates are {ratio:.1f}x larger than SGD!")

    print("\n" + "="*70)


if __name__ == "__main__":
    main()
