"""
Comprehensive 6-Way Model + Optimizer Comparison Test

Tests the following configurations for 600 steps each:
1. MoE + Muon (baseline)
2. TitanMAC + Muon
3. TitanMAC + DeepNestedOptimizer
4. MoE + DeepNestedOptimizer
5. HOPE-nano + AdamW (reference)
6. HOPE-nano + DeepNestedOptimizer

Reference: Nested Learning paper (Behrouz et al., NeurIPS 2025)
"""

import os
import sys
import time
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass
from datetime import datetime

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Add TitanMAC to path
TITAN_PATH = os.path.join(PROJECT_ROOT, "111TitanMAC-Standalone")
if TITAN_PATH not in sys.path:
    sys.path.insert(0, TITAN_PATH)


# ============================================================================
# HOPE-nano Model Implementation (from hope_nano reference)
# ============================================================================

class TitansL2(nn.Module):
    """Delta-rule based memory from Titans paper (Section 3.2)."""

    def __init__(self, d_model: int, n_head: int, chunk_size: int = 64):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.chunk_size = chunk_size

        self.Wq = nn.Linear(d_model, d_model, bias=False)
        self.Wk = nn.Linear(d_model, d_model, bias=False)
        self.Wv = nn.Linear(d_model, d_model, bias=False)
        self.Wo = nn.Linear(d_model, d_model, bias=False)

        # Learnable alpha and beta for delta rule
        self.alpha_raw = nn.Parameter(torch.zeros(1, n_head, 1, 1))
        self.beta_raw = nn.Parameter(torch.zeros(1, n_head, 1, 1))

    @property
    def alpha(self):
        return torch.sigmoid(self.alpha_raw) * 0.5

    @property
    def beta(self):
        return torch.sigmoid(self.beta_raw) * 2.0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, DH = self.n_head, self.head_dim

        q = self.Wq(x).view(B, T, H, DH).transpose(1, 2)
        k = self.Wk(x).view(B, T, H, DH).transpose(1, 2)
        v = self.Wv(x).view(B, T, H, DH).transpose(1, 2)

        # Normalize keys
        k = F.normalize(k, dim=-1)

        # Process in chunks with delta-rule memory
        num_chunks = (T + self.chunk_size - 1) // self.chunk_size
        outputs = []

        # Initialize memory state
        M = torch.zeros(B, H, DH, DH, device=x.device, dtype=x.dtype)

        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, T)

            q_chunk = q[:, :, start:end, :]
            k_chunk = k[:, :, start:end, :]
            v_chunk = v[:, :, start:end, :]

            # Attention within chunk
            attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / math.sqrt(DH)

            # Causal mask
            chunk_len = end - start
            mask = torch.triu(torch.ones(chunk_len, chunk_len, device=x.device), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_out = torch.matmul(attn_weights, v_chunk)

            # Memory retrieval
            mem_out = torch.matmul(q_chunk, M)

            # Combine attention and memory
            chunk_out = attn_out + 0.1 * mem_out
            outputs.append(chunk_out)

            # Update memory with delta rule
            for t in range(chunk_len):
                k_t = k_chunk[:, :, t:t+1, :].transpose(-2, -1)
                v_t = v_chunk[:, :, t:t+1, :].transpose(-2, -1)
                k_row = k_chunk[:, :, t:t+1, :]

                Mk = torch.matmul(M, k_t)
                forget_term = torch.matmul(Mk, k_row)
                write_term = torch.matmul(v_t, k_row)
                M = M - self.alpha * forget_term + self.beta * write_term

        out = torch.cat(outputs, dim=2)
        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.Wo(out)


class CMSBlock(nn.Module):
    """Simple MLP block (represents 'slow memory' in HOPE)."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class HOPEBlock(nn.Module):
    """Single HOPE block: TitansL2 + CMS with residuals."""

    def __init__(self, d_model: int, n_head: int, d_ff: int, chunk_size: int = 64):
        super().__init__()
        self.titans = TitansL2(d_model, n_head, chunk_size)
        self.cms = CMSBlock(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.titans(self.norm1(x))
        x = x + self.cms(self.norm2(x))
        return x


class HOPE(nn.Module):
    """HOPE-nano model."""

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_head: int = 8,
        n_layers: int = 6,
        d_ff: int = 2048,
        max_seq_len: int = 1024,
        chunk_size: int = 64,
    ):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList([
            HOPEBlock(d_model, n_head, d_ff, chunk_size)
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # Tie weights
        self.lm_head.weight = self.tok_emb.weight

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T = x.shape

        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)

        for block in self.blocks:
            h = block(h)

        h = self.norm(h)
        return self.lm_head(h)


# ============================================================================
# Test Configuration
# ============================================================================

@dataclass
class TestConfig:
    max_steps: int = 600
    batch_size: int = 8
    block_size: int = 256
    eval_interval: int = 100
    log_interval: int = 50
    vocab_size: int = 50257  # GPT-2 tokenizer


# ============================================================================
# Data Loading
# ============================================================================

def get_streaming_batch(dataset_iter, tokenizer, batch_size: int, block_size: int, device: str):
    """Get a batch from streaming dataset."""
    texts = []
    for _ in range(batch_size):
        try:
            sample = next(dataset_iter)
            texts.append(sample.get('text', ''))
        except StopIteration:
            texts.append('')

    tokens = tokenizer(
        texts,
        truncation=True,
        max_length=block_size + 1,
        padding='max_length',
        return_tensors='pt'
    )['input_ids']

    X = tokens[:, :-1].to(device)
    Y = tokens[:, 1:].to(device)
    return X, Y


def create_dataset_iterator():
    """Create streaming dataset iterator."""
    from datasets import load_dataset

    dataset = load_dataset(
        "roneneldan/TinyStories",
        split="train",
        streaming=True,
    )
    return iter(dataset)


# ============================================================================
# Training Functions
# ============================================================================

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train_model(
    model: nn.Module,
    optimizer: Any,
    config: TestConfig,
    device: str,
    model_name: str,
    is_moe: bool = False,
    is_nested: bool = False,
) -> Dict[str, Any]:
    """Generic training loop."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.train()
    dataset_iter = create_dataset_iterator()

    losses = []
    start_time = time.time()

    print(f"\nTraining {model_name}...")
    print("-" * 60)

    for step in range(config.max_steps):
        X, Y = get_streaming_batch(dataset_iter, tokenizer, config.batch_size, config.block_size, device)

        optimizer.zero_grad()

        # Handle different model output formats
        if is_moe:
            output = model(X, return_aux_loss=True)
            if isinstance(output, tuple):
                logits, aux_loss = output
            else:
                logits = output
                aux_loss = 0.0
        else:
            logits = model(X)
            aux_loss = 0.0

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            Y.view(-1),
            ignore_index=tokenizer.pad_token_id
        )

        if aux_loss is not None and aux_loss != 0.0:
            loss = loss + 0.01 * aux_loss

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Handle nested optimizer step
        if is_nested:
            optimizer.step(loss.item())
        else:
            optimizer.step()

        losses.append(loss.item())

        if step % config.log_interval == 0:
            elapsed = time.time() - start_time
            print(f"  Step {step:4d}: loss={loss.item():.4f} ({elapsed:.1f}s)")

    elapsed = time.time() - start_time
    final_loss = sum(losses[-10:]) / 10  # Average of last 10

    print(f"\n{model_name} complete: final_loss={final_loss:.4f}, time={elapsed:.1f}s")

    return {
        'model_name': model_name,
        'final_loss': final_loss,
        'all_losses': losses,
        'time': elapsed,
        'params': count_parameters(model),
    }


# ============================================================================
# Test Configurations
# ============================================================================

def create_moe_model(config: TestConfig, device: str):
    """Create MoE model."""
    from configs.moe_config import GPU24GBMoEModelConfig
    from models.moe_llm import MoEMinimalLLM

    moe_config = GPU24GBMoEModelConfig()
    moe_config.vocab_size = config.vocab_size
    moe_config.max_seq_len = config.block_size

    model = MoEMinimalLLM(moe_config).to(device)
    return model, moe_config


def create_titanmac_model(config: TestConfig, device: str):
    """Create TitanMAC model."""
    from configs.titanmac_config import TitanMACGPU24GBConfig
    from models.titanmac_wrapper import TitanMACWrapper

    titan_config = TitanMACGPU24GBConfig()
    titan_config.vocab_size = config.vocab_size
    titan_config.max_seq_len = config.block_size

    model = TitanMACWrapper(titan_config).to(device)
    return model, titan_config


def create_hope_model(config: TestConfig, device: str):
    """Create HOPE-nano model."""
    model = HOPE(
        vocab_size=config.vocab_size,
        d_model=512,
        n_head=8,
        n_layers=6,
        d_ff=2048,
        max_seq_len=config.block_size,
    ).to(device)
    return model


def create_muon_optimizer(model: nn.Module, lr: float = 0.02):
    """Create Muon + AdamW hybrid optimizer."""
    from optimizers import Muon

    # Separate 2D weights (Muon) from embeddings/norms (AdamW)
    muon_params = []
    adamw_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and 'embed' not in name.lower() and 'norm' not in name.lower():
            muon_params.append(param)
        else:
            adamw_params.append(param)

    muon_opt = Muon(muon_params, lr=lr, momentum=0.95)
    adamw_opt = optim.AdamW(adamw_params, lr=lr * 0.1, weight_decay=0.01)

    class HybridOptimizer:
        def __init__(self, muon, adamw):
            self.muon = muon
            self.adamw = adamw

        def zero_grad(self):
            self.muon.zero_grad()
            self.adamw.zero_grad()

        def step(self, *args, **kwargs):
            self.muon.step()
            self.adamw.step()

    return HybridOptimizer(muon_opt, adamw_opt)


def create_nested_optimizer(model: nn.Module, base_lr: float = 3e-4):
    """Create DeepNestedOptimizer."""
    from titans_core.opt import DeepNestedOptimizer

    return DeepNestedOptimizer(
        model=model,
        base_lr=base_lr,
        meta_lr=1e-4,
        k_unroll=5,
        mode='simple',
        meta_update_freq=100,
        weight_decay=0.01,
        low_memory=True,
        use_preprocessing=True,
    )


# ============================================================================
# Main Comparison
# ============================================================================

def run_comparison(config: TestConfig):
    """Run all 6 comparisons."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("=" * 70)
    print("COMPREHENSIVE MODEL + OPTIMIZER COMPARISON")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Max steps: {config.max_steps}")
    print(f"Batch size: {config.batch_size}")
    print(f"Block size: {config.block_size}")
    print("=" * 70)

    results = []

    # Test 1: MoE + Muon
    print("\n" + "=" * 70)
    print("TEST 1: MoE + Muon")
    print("=" * 70)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    model, _ = create_moe_model(config, device)
    print(f"MoE params: {count_parameters(model) / 1e6:.2f}M")
    optimizer = create_muon_optimizer(model)
    result = train_model(model, optimizer, config, device, "MoE + Muon", is_moe=True)
    results.append(result)
    del model, optimizer

    # Test 2: TitanMAC + Muon
    print("\n" + "=" * 70)
    print("TEST 2: TitanMAC + Muon")
    print("=" * 70)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    try:
        model, _ = create_titanmac_model(config, device)
        print(f"TitanMAC params: {count_parameters(model) / 1e6:.2f}M")
        optimizer = create_muon_optimizer(model)
        result = train_model(model, optimizer, config, device, "TitanMAC + Muon", is_moe=True)
        results.append(result)
        del model, optimizer
    except Exception as e:
        print(f"TitanMAC + Muon failed: {e}")
        results.append({'model_name': 'TitanMAC + Muon', 'error': str(e)})

    # Test 3: TitanMAC + DeepNestedOptimizer
    print("\n" + "=" * 70)
    print("TEST 3: TitanMAC + DeepNestedOptimizer")
    print("=" * 70)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    try:
        model, _ = create_titanmac_model(config, device)
        print(f"TitanMAC params: {count_parameters(model) / 1e6:.2f}M")
        optimizer = create_nested_optimizer(model)
        result = train_model(model, optimizer, config, device, "TitanMAC + DeepNested", is_moe=True, is_nested=True)
        results.append(result)
        del model, optimizer
    except Exception as e:
        print(f"TitanMAC + DeepNested failed: {e}")
        results.append({'model_name': 'TitanMAC + DeepNested', 'error': str(e)})

    # Test 4: MoE + DeepNestedOptimizer
    print("\n" + "=" * 70)
    print("TEST 4: MoE + DeepNestedOptimizer")
    print("=" * 70)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    model, _ = create_moe_model(config, device)
    print(f"MoE params: {count_parameters(model) / 1e6:.2f}M")
    optimizer = create_nested_optimizer(model)
    result = train_model(model, optimizer, config, device, "MoE + DeepNested", is_moe=True, is_nested=True)
    results.append(result)
    del model, optimizer

    # Test 5: HOPE-nano + AdamW
    print("\n" + "=" * 70)
    print("TEST 5: HOPE-nano + AdamW")
    print("=" * 70)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    model = create_hope_model(config, device)
    print(f"HOPE-nano params: {count_parameters(model) / 1e6:.2f}M")
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    result = train_model(model, optimizer, config, device, "HOPE-nano + AdamW")
    results.append(result)
    del model, optimizer

    # Test 6: HOPE-nano + DeepNestedOptimizer
    print("\n" + "=" * 70)
    print("TEST 6: HOPE-nano + DeepNestedOptimizer")
    print("=" * 70)
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    model = create_hope_model(config, device)
    print(f"HOPE-nano params: {count_parameters(model) / 1e6:.2f}M")
    optimizer = create_nested_optimizer(model)
    result = train_model(model, optimizer, config, device, "HOPE-nano + DeepNested", is_nested=True)
    results.append(result)
    del model, optimizer

    return results


def print_results(results: List[Dict[str, Any]]):
    """Print results summary."""
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Find baseline (MoE + Muon)
    baseline_loss = None
    for r in results:
        if r.get('model_name') == 'MoE + Muon' and 'final_loss' in r:
            baseline_loss = r['final_loss']
            break

    print("\n| Model + Optimizer          | Params | Final Loss | Time   | vs Baseline |")
    print("|---------------------------|--------|------------|--------|-------------|")

    for r in results:
        name = r.get('model_name', 'Unknown')
        if 'error' in r:
            print(f"| {name:25s} | ERROR  | {r['error'][:30]:30s} |")
        else:
            params = r.get('params', 0) / 1e6
            loss = r.get('final_loss', 0)
            time_s = r.get('time', 0)

            if baseline_loss and baseline_loss > 0:
                ratio = loss / baseline_loss
                ratio_str = f"{ratio:.2f}x"
            else:
                ratio_str = "baseline"

            print(f"| {name:25s} | {params:5.2f}M | {loss:10.4f} | {time_s:5.1f}s | {ratio_str:11s} |")


def save_results(results: List[Dict[str, Any]], output_dir: str):
    """Save results to JSON."""
    os.makedirs(output_dir, exist_ok=True)

    # Create serializable version
    serializable_results = []
    for r in results:
        sr = {k: v for k, v in r.items() if k != 'all_losses'}
        if 'all_losses' in r:
            sr['loss_samples'] = r['all_losses'][::10]  # Every 10th loss
        serializable_results.append(sr)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"comparison_results_{timestamp}.json")

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    print(f"\nResults saved to: {output_path}")


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive Model + Optimizer Comparison")
    parser.add_argument("--max_steps", type=int, default=600, help="Training steps per test")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--block_size", type=int, default=256, help="Sequence length")
    parser.add_argument("--output_dir", type=str, default="Ai-notes/12-18-2025/Optimizer-Cleanup", help="Output directory")

    args = parser.parse_args()

    config = TestConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        block_size=args.block_size,
    )

    results = run_comparison(config)
    print_results(results)
    save_results(results, args.output_dir)
