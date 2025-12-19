"""
Training script for HOPE model with AdamW or Muon optimizer.

HOPE (Hybrid Optimized Persistent Encoder) uses delta-rule memory
from the Titans paper combined with chunked attention.

Usage:
    # AdamW optimizer (default)
    python train_hope.py --experiment_name hope_adamw

    # Muon optimizer
    python train_hope.py --optimizer muon --experiment_name hope_muon

    # 168M config
    python train_hope.py --config 168m --experiment_name hope_168m
"""

import argparse
import time
import os
import math
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from typing import Optional, Dict, Any

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from configs.hope_config import HOPEModelConfig, HOPEGPU24GBConfig, HOPE168MConfig, DebugHOPEConfig
from configs.dataset_config import DataConfig
from utils.helpers import set_seed
from utils.logger import setup_logging


# ============================================================================
# HOPE Model Implementation
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

        k = F.normalize(k, dim=-1)

        num_chunks = (T + self.chunk_size - 1) // self.chunk_size
        outputs = []
        M = torch.zeros(B, H, DH, DH, device=x.device, dtype=x.dtype)

        for chunk_idx in range(num_chunks):
            start = chunk_idx * self.chunk_size
            end = min(start + self.chunk_size, T)

            q_chunk = q[:, :, start:end, :]
            k_chunk = k[:, :, start:end, :]
            v_chunk = v[:, :, start:end, :]

            attn_scores = torch.matmul(q_chunk, k_chunk.transpose(-2, -1)) / math.sqrt(DH)

            chunk_len = end - start
            mask = torch.triu(torch.ones(chunk_len, chunk_len, device=x.device), diagonal=1).bool()
            attn_scores = attn_scores.masked_fill(mask, float('-inf'))

            attn_weights = F.softmax(attn_scores, dim=-1)
            attn_out = torch.matmul(attn_weights, v_chunk)

            mem_out = torch.matmul(q_chunk, M)
            chunk_out = attn_out + 0.1 * mem_out
            outputs.append(chunk_out)

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
    """Simple MLP block."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class HOPEBlock(nn.Module):
    """Single HOPE block: TitansL2 + CMS with residuals."""

    def __init__(self, d_model: int, n_head: int, d_ff: int, chunk_size: int = 64, dropout: float = 0.1):
        super().__init__()
        self.titans = TitansL2(d_model, n_head, chunk_size)
        self.cms = CMSBlock(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.titans(self.norm1(x))
        x = x + self.cms(self.norm2(x))
        return x


class HOPE(nn.Module):
    """HOPE model."""

    def __init__(self, config: HOPEModelConfig):
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.vocab_size = config.vocab_size

        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_emb = nn.Embedding(config.max_seq_len, config.d_model)

        self.blocks = nn.ModuleList([
            HOPEBlock(config.d_model, config.n_heads, config.d_ff, config.chunk_size, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.norm = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

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

    def forward(self, x: torch.Tensor, return_aux_loss: bool = False):
        B, T = x.shape

        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok_emb(x) + self.pos_emb(pos)

        for block in self.blocks:
            h = block(h)

        h = self.norm(h)
        logits = self.lm_head(h)

        if return_aux_loss:
            return logits, torch.tensor(0.0, device=x.device)
        return logits


# ============================================================================
# Training Functions
# ============================================================================

def print_system_info():
    device = "CUDA" if torch.cuda.is_available() else "CPU"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name} ({props.total_memory / 1e9:.1f} GB)")
    print(f"PyTorch: {torch.__version__}\n")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_config(config_name: str) -> HOPEModelConfig:
    """Get config by name."""
    configs = {
        "default": HOPEGPU24GBConfig,
        "24gb": HOPEGPU24GBConfig,
        "168m": HOPE168MConfig,
        "debug": DebugHOPEConfig,
    }
    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Choose from: {list(configs.keys())}")
    return configs[config_name]()


def create_optimizer(model: nn.Module, config: HOPEModelConfig, optimizer_type: str):
    """Create optimizer based on type."""
    if optimizer_type == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=config.adamw_lr,
            weight_decay=config.weight_decay,
        )
    elif optimizer_type == "muon":
        from optimizers import Muon

        muon_params = []
        adamw_params = []
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if param.ndim >= 2 and 'embed' not in name.lower() and 'norm' not in name.lower():
                muon_params.append(param)
            else:
                adamw_params.append(param)

        muon_opt = Muon(muon_params, lr=0.02, momentum=0.95)
        adamw_opt = torch.optim.AdamW(adamw_params, lr=config.adamw_lr, weight_decay=config.weight_decay)

        class HybridOptimizer:
            def __init__(self, muon, adamw):
                self.muon = muon
                self.adamw = adamw

            def zero_grad(self):
                self.muon.zero_grad()
                self.adamw.zero_grad()

            def step(self):
                self.muon.step()
                self.adamw.step()

            def state_dict(self):
                return {"muon": self.muon.state_dict(), "adamw": self.adamw.state_dict()}

            def load_state_dict(self, state):
                self.muon.load_state_dict(state["muon"])
                self.adamw.load_state_dict(state["adamw"])

        return HybridOptimizer(muon_opt, adamw_opt)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")


def prepare_datasets(data_cfg, tokenizer, cache_dir="./processed_data"):
    """Prepare train and validation datasets."""
    import json
    import shutil
    from datasets import load_from_disk, load_dataset, Dataset
    from data.loader import tokenize_and_chunk, finalize_dataset

    train_cache = os.path.join(cache_dir, "train")
    val_cache = os.path.join(cache_dir, "val")
    info_path = os.path.join(cache_dir, "dataset_info.json")

    config_state = {
        "dataset_path": data_cfg.dataset_path,
        "dataset_name": data_cfg.dataset_name,
        "tokenizer_name": data_cfg.tokenizer_name,
        "seq_length": data_cfg.seq_length,
        "num_samples": data_cfg.num_samples,
    }

    if os.path.exists(train_cache) and os.path.exists(val_cache) and os.path.exists(info_path):
        try:
            with open(info_path, "r") as f:
                if json.load(f) == config_state:
                    print(f"Loading cached datasets from {cache_dir}...")
                    return load_from_disk(train_cache), load_from_disk(val_cache)
            print("Cache configuration mismatch. Rebuilding...")
        except Exception as e:
            print(f"Cache check failed ({e}). Rebuilding...")

    if os.path.exists(cache_dir):
        print(f"Cleaning old cache at {cache_dir}...")
        shutil.rmtree(cache_dir)

    os.makedirs(cache_dir, exist_ok=True)

    print("Loading raw dataset...")
    raw_dataset = load_dataset(
        data_cfg.dataset_path,
        data_cfg.dataset_name,
        split=data_cfg.split,
        cache_dir=data_cfg.cache_dir,
        streaming=True,
    )

    raw_samples = list(raw_dataset.take(data_cfg.num_samples))
    num_val = int(len(raw_samples) * 0.1)
    num_train = len(raw_samples) - num_val

    raw_train = Dataset.from_list(raw_samples[:num_train])
    raw_val = Dataset.from_list(raw_samples[num_train:])

    print(f"Tokenizing {num_train} train samples...")
    train_tokenized = tokenize_and_chunk(raw_train, tokenizer, data_cfg.seq_length)
    train_final = finalize_dataset(train_tokenized)

    print(f"Tokenizing {num_val} val samples...")
    val_tokenized = tokenize_and_chunk(raw_val, tokenizer, data_cfg.seq_length)
    val_final = finalize_dataset(val_tokenized)

    print("Saving cached datasets...")
    train_final.save_to_disk(train_cache)
    val_final.save_to_disk(val_cache)
    with open(info_path, "w") as f:
        json.dump(config_state, f)

    return train_final, val_final


def train_hope(
    config: HOPEModelConfig,
    data_cfg: DataConfig,
    optimizer_type: str = "adamw",
    experiment_name: str = "hope",
    output_dir: str = "./checkpoints",
    max_steps: Optional[int] = None,
):
    """Train HOPE model."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nTraining {experiment_name} on {device}")

    # Override max_steps if provided
    if max_steps is not None:
        config.max_steps = max_steps

    # Setup tokenizer
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(data_cfg.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config.vocab_size = len(tokenizer)

    # Prepare datasets
    train_dataset, val_dataset = prepare_datasets(data_cfg, tokenizer, cache_dir=f"./processed_data/{experiment_name}")

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=2)

    # Create model
    model = HOPE(config).to(device)
    num_params = count_parameters(model)
    print(f"HOPE Parameters: {num_params / 1e6:.2f}M")

    # Create optimizer
    optimizer = create_optimizer(model, config, optimizer_type)

    # Training loop
    model.train()
    step = 0
    start_time = time.time()
    metrics_history = {
        'steps': [],
        'val_losses': [],
        'val_accuracies': [],
        'elapsed_times': [],
    }

    print(f"\nStarting training for {config.max_steps} steps...")
    pbar = tqdm(total=config.max_steps, desc="Training")

    for epoch in range(1000):  # High epoch count, will break on max_steps
        for batch in train_loader:
            if step >= config.max_steps:
                break

            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()

            logits = model(input_ids)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                ignore_index=tokenizer.pad_token_id,
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

            # Logging
            if step % 100 == 0:
                with torch.no_grad():
                    predictions = logits.argmax(dim=-1)
                    accuracy = (predictions == labels).float().mean().item()
                    pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{accuracy:.3f}',
                    })

            # Evaluation
            if step % config.eval_every == 0 and step > 0:
                model.eval()
                val_losses = []
                val_accs = []
                with torch.no_grad():
                    for val_batch in val_loader:
                        val_input = val_batch['input_ids'].to(device)
                        val_labels = val_batch['labels'].to(device)
                        val_logits = model(val_input)
                        val_loss = F.cross_entropy(
                            val_logits.view(-1, val_logits.size(-1)),
                            val_labels.view(-1),
                            ignore_index=tokenizer.pad_token_id,
                        )
                        val_losses.append(val_loss.item())
                        val_preds = val_logits.argmax(dim=-1)
                        val_accs.append((val_preds == val_labels).float().mean().item())

                avg_val_loss = sum(val_losses) / len(val_losses)
                avg_val_acc = sum(val_accs) / len(val_accs)
                elapsed = (time.time() - start_time) / 60

                metrics_history['steps'].append(step)
                metrics_history['val_losses'].append(avg_val_loss)
                metrics_history['val_accuracies'].append(avg_val_acc)
                metrics_history['elapsed_times'].append(elapsed)

                print(f"\nStep {step}: Val Loss: {avg_val_loss:.4f}, Val Acc: {avg_val_acc:.4f}")
                model.train()

            step += 1
            pbar.update(1)

        if step >= config.max_steps:
            break

    pbar.close()

    # Final evaluation
    model.eval()
    final_val_losses = []
    with torch.no_grad():
        for val_batch in val_loader:
            val_input = val_batch['input_ids'].to(device)
            val_labels = val_batch['labels'].to(device)
            val_logits = model(val_input)
            val_loss = F.cross_entropy(
                val_logits.view(-1, val_logits.size(-1)),
                val_labels.view(-1),
                ignore_index=tokenizer.pad_token_id,
            )
            final_val_losses.append(val_loss.item())

    final_val_loss = sum(final_val_losses) / len(final_val_losses)
    total_time = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Final Val Loss: {final_val_loss:.4f}")
    print(f"Total Time: {total_time/60:.1f} minutes")
    print(f"Parameters: {num_params/1e6:.2f}M")
    print(f"{'='*60}")

    # Save
    output_path = Path(output_dir) / experiment_name
    output_path.mkdir(parents=True, exist_ok=True)

    torch.save(model.state_dict(), output_path / "model.pt")
    with open(output_path / "metrics.json", "w") as f:
        json.dump({
            "final_metrics": {"val_loss": final_val_loss},
            "history": metrics_history,
            "config": {
                "d_model": config.d_model,
                "n_layers": config.n_layers,
                "n_heads": config.n_heads,
                "params_m": num_params / 1e6,
            },
            "total_time_s": total_time,
        }, f, indent=2)

    print(f"Saved to {output_path}")
    return final_val_loss


def main():
    parser = argparse.ArgumentParser(description="Train HOPE model")
    parser.add_argument("--config", type=str, default="24gb", help="Config name: 24gb, 168m, debug")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "muon"], help="Optimizer type")
    parser.add_argument("--experiment_name", type=str, default="hope", help="Experiment name")
    parser.add_argument("--output_dir", type=str, default="./checkpoints", help="Output directory")
    parser.add_argument("--max_steps", "--steps", type=int, default=None, dest="max_steps", help="Max training steps")
    args = parser.parse_args()

    print_system_info()
    set_seed(42)

    config = get_config(args.config)
    data_cfg = DataConfig()

    train_hope(
        config=config,
        data_cfg=data_cfg,
        optimizer_type=args.optimizer,
        experiment_name=args.experiment_name,
        output_dir=args.output_dir,
        max_steps=args.max_steps,
    )


if __name__ == "__main__":
    main()
