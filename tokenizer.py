"""
Tokenizer for the LLM: Byte Pair Encoding (BPE) with optional character-level fallback.
BPE merges frequent pairs of tokens to build a subword vocabulary.
"""

import json
import re
from pathlib import Path
from typing import List, Optional, Dict, Tuple


# Special tokens (same convention as many GPT tokenizers)
PAD_TOKEN = "<|pad|>"
BOS_TOKEN = "<|endoftext|>"  # Often used as start; we use it as generic special
EOS_TOKEN = "<|endoftext|>"
UNK_TOKEN = "<|unk|>"

SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


class SimpleBPETokenizer:
    """
    A simple Byte Pair Encoding tokenizer built from scratch.
    - First builds a base vocabulary of characters (and special tokens).
    - Then repeatedly merges the most frequent pair of adjacent tokens.
    """

    def __init__(
        self,
        vocab: Optional[Dict[str, int]] = None,
        merges: Optional[List[Tuple[str, str]]] = None,
    ):
        """
        Args:
            vocab: token -> id mapping. If None, you must call fit() or load().
            merges: list of (token1, token2) merge pairs in order of application.
        """
        self.vocab = vocab or {}
        self.merges = merges or []
        self.inverse_vocab: Dict[int, str] = {v: k for k, v in self.vocab.items()}

    def _get_pairs(self, word: List[str]) -> List[Tuple[str, str]]:
        """Get all adjacent pairs in a list of tokens."""
        pairs = set()
        for i in range(len(word) - 1):
            pairs.add((word[i], word[i + 1]))
        return list(pairs)

    def _tokenize_word(self, word: List[str]) -> List[str]:
        """Tokenize a single word (list of characters) by applying merges."""
        if len(word) == 1:
            return word
        word = list(word)
        while True:
            pairs = self._get_pairs(word)
            if not pairs:
                break
            # Find the merge that appears in our merges list (earliest has priority)
            bigram = min(pairs, key=lambda p: self.merges.index(p) if p in self.merges else 1e9)
            if bigram not in self.merges:
                break
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and (word[i], word[i + 1]) == bigram:
                    new_word.append(word[i] + word[i + 1])
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = new_word
        return word

    def _tokenize(self, text: str) -> List[str]:
        """Split text by whitespace and punctuation, then BPE-tokenize each part."""
        # Simple regex: split on whitespace but keep words; split words into chars for BPE
        # We use a pattern that splits on spaces and keeps words as units, then we split each word into chars
        words = re.findall(r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+", text)
        tokens = []
        for word in words:
            if word.strip() == "":
                tokens.append(word)  # keep spaces as single "token" if they were split
            else:
                chars = list(word)
                tokens.extend(self._tokenize_word(chars))
        return [t for t in tokens if t]

    def fit(self, text: str, vocab_size: int = 8192, min_frequency: int = 2) -> None:
        """
        Build BPE vocabulary from text.
        Args:
            text: Raw corpus text.
            vocab_size: Target vocabulary size (including special tokens).
            min_frequency: Minimum pair frequency to consider for merging.
        """
        # Start with special tokens and all characters
        self.vocab = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        for c in sorted(set(text)):
            if c not in self.vocab:
                self.vocab[c] = len(self.vocab)
        self.merges = []

        # Split text into words (by whitespace), then each word into characters
        words = re.findall(r"'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?\d+| ?[^\s\w]+|\s+", text)
        tokenized = [list(w) for w in words]

        while len(self.vocab) < vocab_size:
            # Count pairs
            pair_counts: Dict[Tuple[str, str], int] = {}
            for word in tokenized:
                for i in range(len(word) - 1):
                    pair = (word[i], word[i + 1])
                    pair_counts[pair] = pair_counts.get(pair, 0) + 1
            # Filter by min_frequency
            pair_counts = {p: c for p, c in pair_counts.items() if c >= min_frequency}
            if not pair_counts:
                break
            # Merge the most frequent pair
            best_pair = max(pair_counts, key=pair_counts.get)
            new_token = best_pair[0] + best_pair[1]
            if new_token in self.vocab:
                break
            self.vocab[new_token] = len(self.vocab)
            self.merges.append(best_pair)
            # Apply this merge to tokenized words
            new_tokenized = []
            for word in tokenized:
                i = 0
                new_word = []
                while i < len(word):
                    if i < len(word) - 1 and (word[i], word[i + 1]) == best_pair:
                        new_word.append(new_token)
                        i += 2
                    else:
                        new_word.append(word[i])
                        i += 1
                new_tokenized.append(new_word)
            tokenized = new_tokenized

        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        """
        Encode text to list of token ids.
        Args:
            text: Input string.
            add_special_tokens: If True, prepend BOS (optional; dataset/train can add it).
        """
        tokens = self._tokenize(text)
        ids = []
        for t in tokens:
            ids.append(self.vocab.get(t, self.vocab.get(UNK_TOKEN, 0)))
        if add_special_tokens:
            ids = [self.vocab[BOS_TOKEN]] + ids
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        """Decode list of token ids back to string."""
        tokens = []
        for i in ids:
            t = self.inverse_vocab.get(i, UNK_TOKEN)
            if skip_special_tokens and t in SPECIAL_TOKENS:
                continue
            tokens.append(t)
        return "".join(tokens)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: str) -> None:
        """Save vocabulary and merges to a directory (e.g. tokenizer/)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)
        with open(path / "merges.txt", "w", encoding="utf-8") as f:
            for a, b in self.merges:
                f.write(f"{a} {b}\n")

    def load(self, path: str) -> None:
        """Load vocabulary and merges from a directory."""
        path = Path(path)
        with open(path / "vocab.json", "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.merges = []
        merges_file = path / "merges.txt"
        if merges_file.exists():
            with open(merges_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        parts = line.split(" ", 1)
                        if len(parts) == 2:
                            self.merges.append((parts[0], parts[1]))
        self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}


class CharacterTokenizer:
    """
    Fallback character-level tokenizer. Maps each character to an id.
    Useful for very small experiments or when BPE is not needed.
    """

    def __init__(self, vocab: Optional[Dict[str, int]] = None):
        self.vocab = vocab or {}
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def fit(self, text: str, vocab_size: Optional[int] = None) -> None:
        """Build character vocabulary from text. vocab_size is ignored (we use all chars)."""
        self.vocab = {t: i for i, t in enumerate(SPECIAL_TOKENS)}
        for c in sorted(set(text)):
            if c not in self.vocab:
                self.vocab[c] = len(self.vocab)
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids = [self.vocab.get(c, self.vocab.get(UNK_TOKEN, 0)) for c in text]
        if add_special_tokens:
            ids = [self.vocab[BOS_TOKEN]] + ids
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        chars = []
        for i in ids:
            c = self.inverse_vocab.get(i, UNK_TOKEN)
            if skip_special_tokens and c in SPECIAL_TOKENS:
                continue
            chars.append(c)
        return "".join(chars)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "vocab.json", "w", encoding="utf-8") as f:
            json.dump(self.vocab, f, ensure_ascii=False)

    def load(self, path: str) -> None:
        path = Path(path)
        with open(path / "vocab.json", "r", encoding="utf-8") as f:
            self.vocab = json.load(f)
        self.inverse_vocab = {int(v): k for k, v in self.vocab.items()}


def get_tokenizer(
    kind: str = "bpe",
    path: Optional[str] = None,
    vocab: Optional[Dict] = None,
    merges: Optional[List[Tuple[str, str]]] = None,
):
    """
    Get a tokenizer by type and optionally load from path.
    kind: "bpe" or "char"
    """
    if kind == "char":
        tok = CharacterTokenizer(vocab=vocab)
    else:
        tok = SimpleBPETokenizer(vocab=vocab, merges=merges)
    if path:
        tok.load(path)
    return tok
