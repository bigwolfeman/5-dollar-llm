#!/usr/bin/env python3
"""
Inference Sample Generator

Runs 3 evaluation prompts through each trained checkpoint to compare
qualitative output between baseline and nested optimizer models.

Usage:
    python run_inference_samples.py

Expects checkpoints in checkpoints_extended/ from run_extended_test.py
"""

import os
import sys
import torch
from pathlib import Path
from transformers import AutoTokenizer

# Suppress tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Sample prompts for evaluation
PROMPTS = [
    "The meaning of life is",
    "In the year 2050, artificial intelligence will",
    "Once upon a time, in a kingdom far away,",
]

MAX_NEW_TOKENS = 50
TEMPERATURE = 0.7
TOP_K = 50


def load_tokenizer():
    """Load the SmolLM tokenizer."""
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "HuggingFaceTB/SmolLM-135M",
        use_fast=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_moe_model(checkpoint_path: Path, device: torch.device):
    """Load MoE model from checkpoint."""
    from models.moe_llm import MoEMinimalLLM
    from configs.moe_config import GPU24GBMoEModelConfig

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint or use default
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = GPU24GBMoEModelConfig()
        config.vocab_size = 49152  # SmolLM vocab size

    model = MoEMinimalLLM(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config


def load_titanmac_model(checkpoint_path: Path, device: torch.device):
    """Load TitanMAC model from checkpoint."""
    from models.titanmac_wrapper import TitanMACWrapper
    from configs.titanmac_config import TitanMACGPU24GBConfig

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get config from checkpoint or use default
    if 'config' in checkpoint:
        config = checkpoint['config']
    else:
        config = TitanMACGPU24GBConfig()
        config.vocab_size = 49152

    model = TitanMACWrapper(config).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    return model, config


@torch.no_grad()
def generate_text(model, tokenizer, prompt: str, device: torch.device,
                  max_new_tokens: int = 50, temperature: float = 0.7,
                  top_k: int = 50, is_titanmac: bool = False) -> str:
    """Generate text using greedy/sampling decoding."""

    # Tokenize input
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    generated = input_ids.clone()

    for _ in range(max_new_tokens):
        # Get model output
        if is_titanmac:
            outputs = model(generated)
            if isinstance(outputs, dict):
                logits = outputs.get('logits', outputs.get('output'))
            else:
                logits = outputs[0] if isinstance(outputs, tuple) else outputs
        else:
            outputs = model(generated, return_aux_loss=False)
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

        # Get next token logits
        next_token_logits = logits[:, -1, :] / temperature

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
            next_token_logits[indices_to_remove] = float('-inf')

        # Sample
        probs = torch.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        # Append
        generated = torch.cat([generated, next_token], dim=1)

        # Stop at EOS
        if next_token.item() == tokenizer.eos_token_id:
            break

    # Decode
    output_text = tokenizer.decode(generated[0], skip_special_tokens=True)
    return output_text


def find_checkpoints(checkpoint_dir: Path) -> dict:
    """Find all model checkpoints in the directory."""
    checkpoints = {}

    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        return checkpoints

    for subdir in checkpoint_dir.iterdir():
        if subdir.is_dir():
            model_path = subdir / "model.pt"
            if model_path.exists():
                checkpoints[subdir.name] = model_path

    return checkpoints


def main():
    print("=" * 70)
    print("INFERENCE SAMPLE GENERATOR")
    print("=" * 70)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load tokenizer
    tokenizer = load_tokenizer()

    # Find checkpoints
    checkpoint_dir = Path("checkpoints_extended")
    checkpoints = find_checkpoints(checkpoint_dir)

    if not checkpoints:
        print(f"\nNo checkpoints found in {checkpoint_dir}/")
        print("Run run_extended_test.py first to generate checkpoints.")
        sys.exit(1)

    print(f"\nFound {len(checkpoints)} checkpoints:")
    for name in sorted(checkpoints.keys()):
        print(f"  - {name}")

    # Generate samples for each checkpoint
    results = {}

    for name, ckpt_path in sorted(checkpoints.items()):
        print(f"\n{'='*70}")
        print(f"MODEL: {name}")
        print(f"{'='*70}")

        try:
            # Determine model type
            is_titanmac = "titanmac" in name.lower()

            # Load model
            print(f"Loading {ckpt_path}...")
            if is_titanmac:
                model, config = load_titanmac_model(ckpt_path, device)
            else:
                model, config = load_moe_model(ckpt_path, device)

            print(f"Model loaded successfully")

            # Generate for each prompt
            results[name] = []
            for i, prompt in enumerate(PROMPTS, 1):
                print(f"\n--- Prompt {i} ---")
                print(f"Input: {prompt}")

                try:
                    output = generate_text(
                        model, tokenizer, prompt, device,
                        max_new_tokens=MAX_NEW_TOKENS,
                        temperature=TEMPERATURE,
                        top_k=TOP_K,
                        is_titanmac=is_titanmac,
                    )
                    print(f"Output: {output}")
                    results[name].append({"prompt": prompt, "output": output})
                except Exception as e:
                    print(f"Generation failed: {e}")
                    results[name].append({"prompt": prompt, "output": f"ERROR: {e}"})

            # Free memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        except Exception as e:
            print(f"Failed to load model: {e}")
            results[name] = [{"error": str(e)}]

    # Summary comparison
    print(f"\n{'='*70}")
    print("SUMMARY COMPARISON")
    print(f"{'='*70}")

    for i, prompt in enumerate(PROMPTS):
        print(f"\n### Prompt {i+1}: \"{prompt}\"")
        print("-" * 50)

        for name in sorted(results.keys()):
            if results[name] and "output" in results[name][i]:
                output = results[name][i]["output"]
                # Truncate for display
                if len(output) > 200:
                    output = output[:200] + "..."
                print(f"\n[{name}]")
                print(f"{output}")

    print(f"\n{'='*70}")
    print("INFERENCE COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
