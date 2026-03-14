"""
Training loop for the LLM: cross-entropy loss, batching, checkpointing, and logging.
"""

import argparse
from pathlib import Path
from typing import Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from config import ModelConfig, TrainConfig, DEFAULT_MODEL_CONFIG, DEFAULT_TRAIN_CONFIG
from model import TransformerLM
from tokenizer import SimpleBPETokenizer, CharacterTokenizer, get_tokenizer
from dataset import get_dataloader, build_tokenizer_from_data


def get_linear_warmup_scheduler(optimizer, warmup_steps: int, total_steps: int):
    """Linear warmup then constant LR."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 1.0

    return LambdaLR(optimizer, lr_lambda)


def train(
    model_config: ModelConfig | None = None,
    train_config: TrainConfig | None = None,
) -> None:
    model_config = model_config or DEFAULT_MODEL_CONFIG
    train_config = train_config or DEFAULT_TRAIN_CONFIG

    device = torch.device(train_config.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Tokenizer ---
    tokenizer_path = Path(train_config.tokenizer_path) if train_config.tokenizer_path else None
    if tokenizer_path and tokenizer_path.exists():
        tokenizer = get_tokenizer(kind="bpe")
        tokenizer.load(str(tokenizer_path))
        print(f"Loaded tokenizer from {tokenizer_path}, vocab_size={tokenizer.vocab_size}")
    else:
        print("Building tokenizer from data...")
        tokenizer = build_tokenizer_from_data(
            data_path=train_config.data_path,
            tokenizer_type="bpe",
            vocab_size=train_config.vocab_size,
            save_path=str(Path(train_config.checkpoint_dir) / "tokenizer"),
        )
        print(f"Tokenizer built, vocab_size={tokenizer.vocab_size}")

    # Override model vocab size to match tokenizer
    model_config.vocab_size = tokenizer.vocab_size
    model_config.max_seq_length = model_config.max_seq_length  # keep as is

    # --- Data ---
    # Chunk length = block_size + 1; keep it <= max_seq_length for positional encoding
    block_size = model_config.max_seq_length - 1
    dataloader = get_dataloader(
        data_path=train_config.data_path,
        tokenizer=tokenizer,
        block_size=block_size,
        batch_size=train_config.batch_size,
        shuffle=True,
        num_workers=0,
    )
    print(f"Data loaded: {len(dataloader.dataset)} chunks, {len(dataloader)} batches per epoch")

    # --- Model ---
    model = TransformerLM(model_config).to(device)
    print(f"Model parameters: {model.get_num_params():,}")

    optimizer = AdamW(
        model.parameters(),
        lr=train_config.learning_rate,
        weight_decay=train_config.weight_decay,
    )
    scheduler = get_linear_warmup_scheduler(
        optimizer,
        warmup_steps=train_config.warmup_steps,
        total_steps=train_config.max_steps,
    )

    checkpoint_dir = Path(train_config.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    model.train()
    global_step = 0
    running_loss = 0.0

    while global_step < train_config.max_steps:
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            # Next-token prediction: labels = input_ids (model shifts internally)
            labels = input_ids.clone()
            labels[attention_mask == 0] = -100  # don't compute loss on padding

            optimizer.zero_grad()
            logits, loss = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            if loss is None:
                continue
            loss.backward()
            if train_config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_config.grad_clip)
            optimizer.step()
            scheduler.step()

            running_loss += loss.item()
            global_step += 1

            if global_step % train_config.log_every == 0:
                avg_loss = running_loss / train_config.log_every
                lr = scheduler.get_last_lr()[0]
                print(f"Step {global_step} | loss={avg_loss:.4f} | lr={lr:.2e}")
                running_loss = 0.0

            if global_step % train_config.save_every == 0:
                ckpt_path = checkpoint_dir / f"checkpoint_step_{global_step}.pt"
                save_checkpoint(
                    ckpt_path,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    step=global_step,
                    model_config=model_config,
                )
                print(f"Saved checkpoint to {ckpt_path}")

            if global_step >= train_config.max_steps:
                break

    # Final save
    final_path = checkpoint_dir / "checkpoint_final.pt"
    save_checkpoint(
        final_path,
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        step=global_step,
        model_config=model_config,
    )
    print(f"Training complete. Final checkpoint: {final_path}")


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR,
    step: int,
    model_config: ModelConfig,
) -> None:
    torch.save(
        {
            "step": step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "model_config": model_config,
        },
        path,
    )


def load_checkpoint(path: str, device: torch.device) -> Tuple[TransformerLM, ModelConfig, int]:
    """Load model and config from checkpoint. Returns (model, config, step)."""
    # weights_only=False: checkpoint contains ModelConfig (trusted); required for PyTorch 2.6+
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config = ckpt.get("model_config")
    if config is None:
        config = DEFAULT_MODEL_CONFIG
    model = TransformerLM(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    step = ckpt.get("step", 0)
    return model, config, step


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the LLM")
    parser.add_argument("--data", type=str, default=DEFAULT_TRAIN_CONFIG.data_path, help="Path to training text file")
    parser.add_argument("--checkpoint-dir", type=str, default=DEFAULT_TRAIN_CONFIG.checkpoint_dir)
    parser.add_argument("--tokenizer", type=str, default=None, help="Path to saved tokenizer (optional)")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_TRAIN_CONFIG.batch_size)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_TRAIN_CONFIG.max_steps)
    parser.add_argument("--lr", type=float, default=DEFAULT_TRAIN_CONFIG.learning_rate)
    parser.add_argument("--device", type=str, default=DEFAULT_TRAIN_CONFIG.device)
    args = parser.parse_args()

    train_config = TrainConfig(
        data_path=args.data,
        checkpoint_dir=args.checkpoint_dir,
        tokenizer_path=args.tokenizer,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        device=args.device,
    )
    train(train_config=train_config)
