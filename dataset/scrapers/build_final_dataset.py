"""
Build the final training dataset for the Luffy GPT model.

Combines:
1. Luffy-specific dialogue lines (from wiki scraper) — placed at the top
2. All One Piece dialogue from One Pace subtitles — the bulk of the corpus

This mirrors TinyShakespeare's approach: the full corpus teaches the model
the One Piece dialogue style, which is naturally dominated by Luffy as the
main character.
"""

import os
import shutil

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(OUTPUT_DIR)

luffy_file = os.path.join(OUTPUT_DIR, 'luffy.txt')
onepace_file = os.path.join(OUTPUT_DIR, 'onepace_all_dialogue.txt')
final_file = os.path.join(OUTPUT_DIR, 'luffy_dataset.txt')


def main():
    parts = []

    # 1. Luffy-specific lines
    if os.path.exists(luffy_file):
        with open(luffy_file, 'r', encoding='utf-8') as f:
            luffy_lines = f.read().strip()
        if luffy_lines:
            parts.append(luffy_lines)
            print(f"Luffy-specific lines: {len(luffy_lines.splitlines())} lines")

    # 2. All One Pace dialogue
    if os.path.exists(onepace_file):
        with open(onepace_file, 'r', encoding='utf-8') as f:
            onepace_lines = f.read().strip()
        if onepace_lines:
            parts.append(onepace_lines)
            print(f"One Pace dialogue: {len(onepace_lines.splitlines())} lines")

    # Combine
    combined = '\n'.join(parts)
    with open(final_file, 'w', encoding='utf-8') as f:
        f.write(combined)

    size = os.path.getsize(final_file)
    line_count = len(combined.splitlines())
    print(f"\nFinal dataset: {final_file}")
    print(f"  Lines: {line_count:,}")
    print(f"  Size:  {size:,} bytes ({size/1024:.1f} KB / {size/1024/1024:.1f} MB)")

    # Copy to project root
    dest = os.path.join(PROJECT_DIR, 'luffy_dataset.txt')
    shutil.copy2(final_file, dest)
    print(f"\nCopied to project root: {dest}")

    # Stats
    print("\n=== Dataset Summary ===")
    print(f"  Total lines:    {line_count:,}")
    print(f"  Total size:     {size/1024/1024:.2f} MB")
    print(f"  TinyShakespeare comparison: {size/1115394:.1f}x the size")
    print(f"  Avg line length: {size/line_count:.0f} chars")


if __name__ == '__main__':
    main()
