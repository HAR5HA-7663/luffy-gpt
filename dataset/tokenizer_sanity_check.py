#!/usr/bin/env python3
"""Compare token counts across character-level and tiktoken BPE tokenizers."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="dataset/processed/train.txt")
    args = parser.parse_args()

    path = Path(args.input)
    text = path.read_text(encoding="utf-8", errors="ignore")

    chars = len(text)
    unique_chars = len(set(text))

    print("=== Tokenization Sanity Check ===")
    print(f"Input:            {path}")
    print(f"Characters:       {chars:,}")
    print(f"Unique chars:     {unique_chars:,}")

    try:
        import tiktoken
    except Exception as exc:
        print("tiktoken unavailable; install it to run BPE comparison.")
        print(f"Import error: {exc}")
        return

    encodings = ["gpt2", "cl100k_base", "o200k_base"]
    for name in encodings:
        try:
            enc = tiktoken.get_encoding(name)
            token_count = len(enc.encode(text))
            ratio = chars / token_count if token_count else 0
            print(f"{name:16} tokens={token_count:10,} | chars/token={ratio:0.2f}")
        except Exception as exc:
            print(f"{name:16} unavailable ({exc})")


if __name__ == "__main__":
    main()
