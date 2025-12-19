"""
Comprehensive Test: Optimizers and Models Comparison

Tests:
1. MoE + AdamW (baseline)
2. MoE + DeepMomentumGD (ported from erikl2)
3. HOPE-nano + AdamW (reference architecture)

Each model trains for a fixed number of steps and we compare loss curves.
"""

import os
import sys
import time
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset
from dataclasses import dataclass
from typing import Optional, List, Tuple
import tiktoken

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# =============================================================================
# Streaming Dataset (shared across all models)
# =============================================================================

class StreamingTextDataset(IterableDataset):
    """Memory-efficient streaming dataset using TinyStories."""

    def __init__(self, split="train", block_size=256, max_samples=None):
        from datasets import load_dataset
        self.dataset = load_dataset("roneneldan/TinyStories", split=split, streaming=True)
        self.tokenizer = tiktoken.get_encoding("gpt2")
        self.block_size = block_size
        self.max_samples = max_samples

    def __iter__(self):
        buffer = []
        sample_count = 0
        for item in self.dataset:
            if self.max_samples and sample_count >= self.max_samples:
                break
            tokens = self.tokenizer.encode(item['text'])
            buffer.extend(tokens)
            while len(buffer) >= self.block_size + 1:
                if self.max_samples and sample_count >= self.max_samples:
                    break
                chunk = buffer[:self.block_size + 1]
                buffer = buffer[self.block_size:]
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                yield x, y
                sample_count += 1


# =============================================================================
# HOPE-nano Model (from hope_nano/model.py - simplified)
# =============================================================================

@dataclass
class HOPEConfig:
    vocab_size: int = 50257
    n_embd: int = 256
    n_head: int = 4
    n_layer: int = 4
    block_size: int = 256
    dropout: float = 0.1
    bias: bool = False


class TitansL2(nn.Module):
    """Titans Memory Module with L2/Delta Rule Update."""

    def __init__(self, config: HOPEConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head

        self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.alpha_raw = nn.Parameter(torch.zeros(1, self.n_head, 1, 1))
        self.beta_raw = nn.Parameter(torch.zeros(1, self.n_head, 1, 1))

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw) * 0.5

    @property
    def beta(self):
        return torch.sigmoid(self.beta_raw) * 0.5

    def forward(self, x: torch.Tensor, state: Optional[torch.Tensor] = None):
        B, T, C = x.size()

        q = self.c_q(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        k = F.normalize(k, dim=-1)

        # Simplified forward (no chunking for testing)
        D = self.head_dim
        if state is None:
            state = torch.zeros(B, self.n_head, D, D, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(T):
            qt = q[:, :, t:t+1, :]
            kt = k[:, :, t:t+1, :]
            vt = v[:, :, t:t+1, :]

            y = torch.matmul(qt, state.transpose(-1, -2))

            k_t = kt.transpose(-1, -2)
            v_t = vt.transpose(-1, -2)

            Mk = torch.matmul(state, k_t)
            forget_term = torch.matmul(Mk, kt)
            write_term = torch.matmul(v_t, kt)

            state = state - self.alpha * forget_term + self.beta * write_term
            outputs.append(y)

        y = torch.cat(outputs, dim=2)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        return self.c_proj(y), state


class CMSBlock(nn.Module):
    """Continuum Memory System Block (just MLP)."""
    def __init__(self, config: HOPEConfig):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias),
            nn.Dropout(config.dropout),
        )

    def forward(self, x):
        return self.net(x)


class HOPEBlock(nn.Module):
    def __init__(self, config: HOPEConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.titans = TitansL2(config)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.cms = CMSBlock(config)

    def forward(self, x, state=None):
        res, new_state = self.titans(self.ln1(x), state)
        x = x + res
        x = x + self.cms(self.ln2(x))
        return x, new_state


class HOPE(nn.Module):
    def __init__(self, config: HOPEConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([HOPEBlock(config) for _ in range(config.n_layer)]),
            ln_f=nn.LayerNorm(config.n_embd),
        ))

        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.transformer.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None, states=None):
        device = idx.device
        b, t = idx.size()

        pos = torch.arange(0, t, dtype=torch.long, device=device)

        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        new_states = []
        for i, block in enumerate(self.transformer.h):
            block_state = states[i] if states is not None else None
            x, new_block_state = block(x, state=block_state)
            new_states.append(new_block_state)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        else:
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss, new_states

    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


# =============================================================================
# Training Functions
# =============================================================================

def train_model(
    model: nn.Module,
    optimizer,
    train_loader,
    max_steps: int,
    device: str,
    model_name: str,
    log_interval: int = 50,
    use_states: bool = False,
):
    """Generic training loop."""
    model.train()
    model.to(device)

    if hasattr(optimizer, 'shared_memory'):
        optimizer.shared_memory = optimizer.shared_memory.to(device)

    train_iter = iter(train_loader)
    losses = []
    persistent_states = None

    print(f"\nTraining {model_name}...")
    print("-" * 60)

    start_time = time.time()

    for step in range(max_steps):
        try:
            X, Y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            X, Y = next(train_iter)
            if use_states:
                persistent_states = None  # Reset states on epoch boundary

        X, Y = X.to(device), Y.to(device)

        optimizer.zero_grad()

        if use_states:
            # HOPE-style with states
            if persistent_states is not None:
                persistent_states = [s.detach() if s is not None else None for s in persistent_states]
            logits, loss, new_states = model(X, Y, states=persistent_states)
            persistent_states = new_states
        else:
            # Standard MoE forward - returns (logits, aux_loss) tuple
            output = model(X, return_aux_loss=True)
            if isinstance(output, tuple):
                logits, aux_loss = output
            elif isinstance(output, dict):
                logits = output['logits']
                aux_loss = output.get('aux_loss', 0.0)
            else:
                logits = output
                aux_loss = 0.0

            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                Y.view(-1),
                ignore_index=-1,
            )
            # Add aux loss if available
            if aux_loss is not None and aux_loss != 0.0:
                loss = loss + 0.01 * aux_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        losses.append(loss.item())

        if step % log_interval == 0:
            elapsed = time.time() - start_time
            avg_loss = sum(losses[-log_interval:]) / min(len(losses), log_interval)
            print(f"  Step {step:4d}: loss={avg_loss:.4f} ({elapsed:.1f}s)")

    elapsed = time.time() - start_time
    final_loss = sum(losses[-100:]) / min(len(losses), 100)
    print(f"\n{model_name} complete: final_loss={final_loss:.4f}, time={elapsed:.1f}s")

    return losses


def run_comparison(
    max_steps: int = 500,
    batch_size: int = 8,
    block_size: int = 256,
    device: str = None,
):
    """Run comparison across models and optimizers."""

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("=" * 70)
    print("OPTIMIZER AND MODEL COMPARISON TEST")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Max steps: {max_steps}")
    print(f"Batch size: {batch_size}")
    print(f"Block size: {block_size}")
    print("=" * 70)

    # Create dataset
    print("\nLoading dataset...")
    train_dataset = StreamingTextDataset(
        split="train",
        block_size=block_size,
        max_samples=max_steps * batch_size * 2,  # Enough for full training
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    results = {}

    # ==========================================================================
    # Test 1: HOPE-nano + AdamW
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 1: HOPE-nano + AdamW")
    print("=" * 70)

    hope_config = HOPEConfig(
        n_embd=256,
        n_head=4,
        n_layer=4,
        block_size=block_size,
    )
    hope_model = HOPE(hope_config)
    print(f"HOPE params: {hope_model.get_num_params() / 1e6:.2f}M")

    hope_optimizer = torch.optim.AdamW(hope_model.parameters(), lr=1e-3, weight_decay=0.1)

    hope_losses = train_model(
        model=hope_model,
        optimizer=hope_optimizer,
        train_loader=train_loader,
        max_steps=max_steps,
        device=device,
        model_name="HOPE-nano + AdamW",
        use_states=True,
    )
    results['hope_adamw'] = hope_losses

    # Clear memory
    del hope_model, hope_optimizer
    torch.cuda.empty_cache() if device == 'cuda' else None

    # ==========================================================================
    # Test 2: MoE + AdamW
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 2: MoE + AdamW")
    print("=" * 70)

    from configs.moe_config import DebugMoEConfig
    from models.moe_llm import MoEMinimalLLM

    moe_config = DebugMoEConfig()
    moe_config.max_seq_len = block_size
    moe_config.vocab_size = 50257  # GPT-2 tokenizer vocab size
    moe_model = MoEMinimalLLM(moe_config)
    print(f"MoE params: {sum(p.numel() for p in moe_model.parameters()) / 1e6:.2f}M")

    moe_adamw_opt = torch.optim.AdamW(moe_model.parameters(), lr=1e-3, weight_decay=0.01)

    moe_adamw_losses = train_model(
        model=moe_model,
        optimizer=moe_adamw_opt,
        train_loader=train_loader,
        max_steps=max_steps,
        device=device,
        model_name="MoE + AdamW",
        use_states=False,
    )
    results['moe_adamw'] = moe_adamw_losses

    del moe_model, moe_adamw_opt
    torch.cuda.empty_cache() if device == 'cuda' else None

    # ==========================================================================
    # Test 3: MoE + DeepMomentumGD
    # ==========================================================================
    print("\n" + "=" * 70)
    print("TEST 3: MoE + DeepMomentumGD (ported from erikl2)")
    print("=" * 70)

    from optimizers.deep_momentum_gd import DeepMomentumGD

    moe_model2 = MoEMinimalLLM(moe_config)

    dmgd_optimizer = DeepMomentumGD(
        moe_model2.parameters(),
        lr=1e-3,
        momentum=0.9,
        memory_lr=1e-4,
        memory_hidden_dim=64,
        memory_depth=2,
        weight_decay=0.01,
        internal_loss_mode='surrogate',
    )
    dmgd_optimizer.to(device)

    dmgd_losses = train_model(
        model=moe_model2,
        optimizer=dmgd_optimizer,
        train_loader=train_loader,
        max_steps=max_steps,
        device=device,
        model_name="MoE + DeepMomentumGD",
        use_states=False,
    )
    results['moe_dmgd'] = dmgd_losses

    # ==========================================================================
    # Summary
    # ==========================================================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    def avg_loss(losses, last_n=100):
        return sum(losses[-last_n:]) / min(len(losses), last_n)

    print(f"\nFinal losses (avg of last 100 steps):")
    print(f"  HOPE-nano + AdamW:    {avg_loss(results['hope_adamw']):.4f}")
    print(f"  MoE + AdamW:          {avg_loss(results['moe_adamw']):.4f}")
    print(f"  MoE + DeepMomentumGD: {avg_loss(results['moe_dmgd']):.4f}")

    # Compute ratios
    hope_loss = avg_loss(results['hope_adamw'])
    moe_adamw_loss = avg_loss(results['moe_adamw'])
    moe_dmgd_loss = avg_loss(results['moe_dmgd'])

    print(f"\nRatios (lower is better):")
    print(f"  HOPE vs MoE-AdamW:    {hope_loss / moe_adamw_loss:.2f}x")
    print(f"  DeepMomentumGD vs AdamW: {moe_dmgd_loss / moe_adamw_loss:.2f}x")

    # Get memory stats if available
    if hasattr(dmgd_optimizer, 'get_memory_stats'):
        stats = dmgd_optimizer.get_memory_stats()
        print(f"\nDeepMomentumGD stats:")
        print(f"  Steps: {stats['step_count']}")
        print(f"  Last internal loss: {stats['last_internal_loss']:.4f}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare optimizers and models")
    parser.add_argument("--steps", type=int, default=500, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--block-size", type=int, default=256, help="Sequence length")
    parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")

    args = parser.parse_args()

    results = run_comparison(
        max_steps=args.steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
        device=args.device,
    )
