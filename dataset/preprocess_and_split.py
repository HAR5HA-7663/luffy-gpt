#!/usr/bin/env python3
"""Clean, deduplicate, and split dialogue corpus for autoregressive LM training."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

SPEAKER_RE = re.compile(r"^[A-Za-z0-9 .,'+\-()&/]+:$")
URL_RE = re.compile(r"https?://\S+|www\.\S+")
HTML_RE = re.compile(r"<[^>]+>")
WS_RE = re.compile(r"\s+")

QUOTE_TRANSLATION = str.maketrans(
    {
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "—": "-",
        "–": "-",
        "…": "...",
        "\u00A0": " ",
    }
)


def clean_line(line: str) -> str:
    line = line.translate(QUOTE_TRANSLATION)
    line = URL_RE.sub("", line)
    line = HTML_RE.sub("", line)
    line = WS_RE.sub(" ", line).strip()

    # Remove surrounding quotes if the full line is quoted.
    if len(line) >= 2 and (
        (line.startswith('"') and line.endswith('"'))
        or (line.startswith("'") and line.endswith("'"))
    ):
        line = line[1:-1].strip()

    # Remove simple bracketed stage directions.
    line = re.sub(r"^\[[^\]]*\]\s*", "", line)
    line = re.sub(r"\s*\[[^\]]*\]$", "", line).strip()

    # Keep only lines with at least one alphabetic char.
    if not re.search(r"[A-Za-z]", line):
        return ""
    return line


def parse_blocks(text: str) -> list[tuple[str, list[str]]]:
    blocks: list[tuple[str, list[str]]] = []
    current_speaker: str | None = None
    current_lines: list[str] = []

    def flush() -> None:
        nonlocal current_speaker, current_lines
        if current_speaker and current_lines:
            blocks.append((current_speaker, current_lines))
        current_speaker = None
        current_lines = []

    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            flush()
            continue

        if SPEAKER_RE.match(line):
            flush()
            current_speaker = line[:-1].strip()
            continue

        cleaned = clean_line(line)
        if cleaned and current_speaker:
            current_lines.append(cleaned)

    flush()
    return blocks


def dedupe_blocks(blocks: list[tuple[str, list[str]]]) -> list[tuple[str, list[str]]]:
    seen: set[str] = set()
    output: list[tuple[str, list[str]]] = []

    for speaker, lines in blocks:
        key = speaker + "\n" + "\n".join(lines)
        if key in seen:
            continue
        seen.add(key)
        output.append((speaker, lines))
    return output


def format_blocks(blocks: list[tuple[str, list[str]]]) -> str:
    parts = []
    for speaker, lines in blocks:
        parts.append(f"{speaker}:\n" + "\n".join(lines))
    return "\n\n".join(parts).strip() + "\n"


def split_blocks(
    blocks: list[tuple[str, list[str]]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[str, list[str]]], list[tuple[str, list[str]]], list[tuple[str, list[str]]]]:
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio must be between 0 and 1")
    if not 0 <= val_ratio < 1:
        raise ValueError("val_ratio must be between 0 and 1")
    if train_ratio + val_ratio >= 1:
        raise ValueError("train_ratio + val_ratio must be < 1")

    indices = list(range(len(blocks)))
    rng = random.Random(seed)
    rng.shuffle(indices)

    train_end = int(len(indices) * train_ratio)
    val_end = train_end + int(len(indices) * val_ratio)

    train_idx = set(indices[:train_end])
    val_idx = set(indices[train_end:val_end])

    train, val, test = [], [], []
    for i, block in enumerate(blocks):
        if i in train_idx:
            train.append(block)
        elif i in val_idx:
            val.append(block)
        else:
            test.append(block)
    return train, val, test


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="luffy_dataset.txt", help="Path to input corpus")
    parser.add_argument("--out-dir", default="dataset/processed", help="Output directory")
    parser.add_argument("--train-ratio", type=float, default=0.9)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    input_path = Path(args.input)
    out_dir = Path(args.out_dir)

    raw = input_path.read_text(encoding="utf-8", errors="ignore")
    parsed = parse_blocks(raw)
    deduped = dedupe_blocks(parsed)

    train_blocks, val_blocks, test_blocks = split_blocks(
        deduped,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    cleaned_text = format_blocks(deduped)
    train_text = format_blocks(train_blocks) if train_blocks else ""
    val_text = format_blocks(val_blocks) if val_blocks else ""
    test_text = format_blocks(test_blocks) if test_blocks else ""

    write_text(out_dir / "corpus_clean.txt", cleaned_text)
    write_text(out_dir / "train.txt", train_text)
    write_text(out_dir / "val.txt", val_text)
    write_text(out_dir / "test.txt", test_text)

    print("=== Preprocess Summary ===")
    print(f"Input file:        {input_path}")
    print(f"Raw chars:         {len(raw):,}")
    print(f"Parsed blocks:     {len(parsed):,}")
    print(f"Deduped blocks:    {len(deduped):,}")
    print(f"Train blocks:      {len(train_blocks):,}")
    print(f"Validation blocks: {len(val_blocks):,}")
    print(f"Test blocks:       {len(test_blocks):,}")
    print(f"Output dir:        {out_dir}")


if __name__ == "__main__":
    main()
