"""
Autoregressive text generation from the trained LLM.
Supports temperature, top-k, and top-p (nucleus) sampling.
Includes interactive chat mode with optional feedback collection for learning.
"""

import json
import argparse
from pathlib import Path
from typing import Optional

import torch

from config import DEFAULT_GENERATE_CONFIG, FEEDBACK_FILE
from model import TransformerLM
from tokenizer import get_tokenizer
from train import load_checkpoint


def top_k_filter(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Zero out logits for tokens not in the top-k."""
    if k <= 0:
        return logits
    v, _ = torch.topk(logits, min(k, logits.size(-1)))
    threshold = v[..., -1, None]
    return torch.where(logits < threshold, torch.full_like(logits, float("-inf")), logits)


def top_p_filter(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus (top-p) sampling: keep smallest set of tokens whose cumulative prob >= p."""
    if p >= 1.0:
        return logits
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    mask = cumsum - sorted_probs > p
    sorted_probs = torch.where(mask, torch.zeros_like(sorted_probs), sorted_probs)
    probs = torch.zeros_like(probs).scatter_(-1, sorted_indices, sorted_probs)
    return torch.log(probs.clamp(min=1e-10))


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 1.0,
    device: torch.device = None,
    seed: Optional[int] = None,
) -> str:
    """
    Autoregressive text generation with temperature, top-k, and top-p sampling.
    """
    if device is None:
        device = next(model.parameters()).device
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    if not prompt.strip():
        input_ids = torch.tensor(
            [[tokenizer.vocab.get("<|endoftext|>", 0)]],
            dtype=torch.long,
            device=device,
        )
    else:
        ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)

    generated = list(input_ids[0].tolist())

    for _ in range(max_new_tokens - 1):
        max_len = model.config.max_seq_length
        if len(generated) > max_len:
            input_ids = torch.tensor([generated[-max_len:]], dtype=torch.long, device=device)
        else:
            input_ids = torch.tensor([generated], dtype=torch.long, device=device)

        logits, _ = model(input_ids=input_ids, labels=None)
        logits = logits[0, -1, :] / max(temperature, 1e-8)
        logits = top_k_filter(logits.unsqueeze(0), top_k).squeeze(0)
        logits = top_p_filter(logits.unsqueeze(0), top_p).squeeze(0)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_id)
        eos_id = tokenizer.vocab.get("<|endoftext|>", None)
        if eos_id is not None and next_id == eos_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)


# ---------------------------------------------------------------------------
# Feedback for learning from user
# ---------------------------------------------------------------------------

def save_feedback_entry(
    feedback_path: str,
    user_prompt: str,
    model_response: str,
    corrected_response: Optional[str] = None,
) -> None:
    """Append one interaction to feedback.jsonl (for later fine-tuning)."""
    path = Path(feedback_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "user_prompt": user_prompt,
        "model_response": model_response,
        "corrected_response": corrected_response,
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_chat(
    model: TransformerLM,
    tokenizer,
    device: torch.device,
    feedback_file: str = FEEDBACK_FILE,
    max_new_tokens: int = 80,
    temperature: float = 0.8,
    top_k: int = 40,
    top_p: float = 1.0,
    prompt_prefix: str = "You: ",
    response_prefix: str = "Model: ",
) -> None:
    """
    Interactive chat in the terminal. After each model response, the user may
    optionally provide a correction; it is stored in feedback_file for later learning.
    Type 'quit' or 'exit' to end the session.
    """
    print("Interactive chat. Type 'quit' or 'exit' to stop.\n")
    while True:
        try:
            user_input = input(prompt_prefix).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break
        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            print("Goodbye.")
            break

        response = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=user_input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            device=device,
            seed=None,
        )
        # Strip prompt from response if we only want the continuation
        if response.startswith(user_input):
            response = response[len(user_input):].strip()
        print(response_prefix + response)

        correction = input("Correction (or Enter to skip): ").strip()
        save_feedback_entry(
            feedback_path=feedback_file,
            user_prompt=user_input,
            model_response=response,
            corrected_response=correction if correction else None,
        )
        if correction:
            print("Feedback saved. You can run 'python main.py learn' later to update the model.\n")


def main():
    parser = argparse.ArgumentParser(description="Generate text from trained LLM")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=DEFAULT_GENERATE_CONFIG.max_new_tokens)
    parser.add_argument("--temperature", type=float, default=DEFAULT_GENERATE_CONFIG.temperature)
    parser.add_argument("--top-k", type=int, default=DEFAULT_GENERATE_CONFIG.top_k)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    device = torch.device(args.device if (args.device == "cuda" and torch.cuda.is_available()) else "cpu")
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    tokenizer_path = args.tokenizer or str(ckpt_path.parent / "tokenizer")
    tokenizer = get_tokenizer(kind="bpe")
    if not Path(tokenizer_path).exists():
        raise FileNotFoundError(
            f"Tokenizer not found at {tokenizer_path}. Train first or pass --tokenizer /path/to/tokenizer"
        )
    tokenizer.load(tokenizer_path)

    model, _, _ = load_checkpoint(str(ckpt_path), device)
    model.eval()

    out = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
        seed=args.seed,
    )
    print(out)


if __name__ == "__main__":
    main()
