"""
Dataset and DataLoader for training the LLM.
Supports multiple text files, automatic cleaning/preprocessing, and dataset statistics.
"""

import re
from pathlib import Path
from typing import Optional, Union, List

import torch
from torch.utils.data import Dataset, DataLoader

from tokenizer import SimpleBPETokenizer, CharacterTokenizer, get_tokenizer


# ---------------------------------------------------------------------------
# Text cleaning and preprocessing
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean and normalize text before tokenization.
    - Normalize whitespace (multiple spaces/newlines -> single space or newline)
    - Strip per line and remove empty lines (optional: keep paragraph breaks)
    - Remove control characters
    """
    # Remove control characters (except newline and tab)
    text = "".join(c for c in text if c == "\n" or c == "\t" or (ord(c) >= 32 and ord(c) != 127))
    # Normalize line breaks to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse multiple blank lines to at most two (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces to one
    text = re.sub(r"[ \t]+", " ", text)
    # Strip each line and remove lines that are only whitespace
    lines = [line.strip() for line in text.split("\n") if line.strip()]
    return "\n".join(lines)


def load_and_clean_files(paths: Union[str, List[str]], clean: bool = True) -> str:
    """
    Load one or more text files and return concatenated text (optionally cleaned).
    paths: single path string, or list of paths, or comma-separated string.
    """
    if isinstance(paths, str):
        paths = [p.strip() for p in paths.split(",") if p.strip()]
    all_text = []
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Data file not found: {path}")
        raw = p.read_text(encoding="utf-8", errors="replace")
        all_text.append(clean_text(raw) if clean else raw)
    return "\n\n".join(all_text)


def get_dataset_stats(num_tokens: int, block_size: int) -> dict:
    """Compute dataset statistics from token count and block size."""
    num_chunks = max(0, num_tokens - block_size)
    return {
        "num_tokens": num_tokens,
        "num_chunks": num_chunks,
        "block_size": block_size,
    }


def print_dataset_stats(stats: dict) -> None:
    """Log dataset statistics before training."""
    print("  Dataset statistics:")
    print(f"    Tokens: {stats['num_tokens']:,}")
    print(f"    Chunks (context length {stats['block_size']}): {stats['num_chunks']:,}")


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """
    Dataset that loads text from one or more files, cleans it, tokenizes,
    and returns contiguous chunks of length block_size + 1 for next-token prediction.
    """

    def __init__(
        self,
        data_path: Union[str, List[str]],
        tokenizer: Union[SimpleBPETokenizer, CharacterTokenizer],
        block_size: int,
        max_samples: Optional[int] = None,
        clean: bool = True,
    ):
        """
        Args:
            data_path: Single path, list of paths, or comma-separated paths.
            tokenizer: Fitted tokenizer.
            block_size: Context length (chunk will be block_size + 1 tokens).
            max_samples: Cap number of chunks (for debugging).
            clean: If True, run clean_text on loaded content.
        """
        self.block_size = block_size
        self.tokenizer = tokenizer
        text = load_and_clean_files(data_path, clean=clean)
        self.ids = tokenizer.encode(text, add_special_tokens=False)
        self.length = max(0, len(self.ids) - block_size)
        if max_samples is not None and self.length > max_samples:
            self.length = max_samples

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> dict:
        start = idx
        end = start + self.block_size + 1
        chunk = self.ids[start:end]
        return {"input_ids": torch.tensor(chunk, dtype=torch.long)}


def collate_fn(batch: list) -> dict:
    """Stack batches and build attention_mask."""
    input_ids = torch.stack([b["input_ids"] for b in batch])
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def get_dataloader(
    data_path: Union[str, List[str]],
    tokenizer: Union[SimpleBPETokenizer, CharacterTokenizer],
    block_size: int,
    batch_size: int = 32,
    max_samples: Optional[int] = None,
    shuffle: bool = True,
    num_workers: int = 0,
    clean: bool = True,
) -> DataLoader:
    """Build DataLoader; uses CPU-friendly defaults (num_workers=0, no pin_memory on CPU)."""
    dataset = TextDataset(
        data_path=data_path,
        tokenizer=tokenizer,
        block_size=block_size,
        max_samples=max_samples,
        clean=clean,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=False,  # Safer on CPU / low RAM
    )


def build_tokenizer_from_data(
    data_path: Union[str, List[str]],
    tokenizer_type: str = "bpe",
    vocab_size: int = 8192,
    save_path: Optional[str] = None,
    clean: bool = True,
) -> Union[SimpleBPETokenizer, CharacterTokenizer]:
    """Build tokenizer from one or more text files; optionally save."""
    text = load_and_clean_files(data_path, clean=clean)
    tokenizer = get_tokenizer(kind=tokenizer_type)
    tokenizer.fit(text, vocab_size=vocab_size)
    if save_path:
        tokenizer.save(save_path)
    return tokenizer
