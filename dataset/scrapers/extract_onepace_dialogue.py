"""
Extract ALL English dialogue from One Pace subtitle files (.ass).
This gives us the full One Piece dialogue corpus.
Since Luffy is the main character, his style dominates the text.
"""

import re
import os
import glob

SUBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'one-pace-subtitles', 'main')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def extract_dialogue_from_ass(filepath):
    """Extract dialogue text from an ASS subtitle file."""
    lines = []

    try:
        with open(filepath, 'r', encoding='utf-8-sig') as f:
            content = f.read()
    except UnicodeDecodeError:
        try:
            with open(filepath, 'r', encoding='latin-1') as f:
                content = f.read()
        except Exception:
            return []

    for line in content.split('\n'):
        line = line.strip()
        # Only process Dialogue lines (not Comments)
        if not line.startswith('Dialogue:'):
            continue

        # Parse ASS dialogue format:
        # Dialogue: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
        parts = line.split(',', 9)
        if len(parts) < 10:
            continue

        style = parts[3].strip()
        text = parts[9].strip()

        # Skip non-dialogue styles
        skip_styles = ['Karaoke', 'Lyrics', 'Warning', 'Credits', 'Title',
                       'Captions', 'Note', 'Sign', 'Caption']
        if any(s.lower() in style.lower() for s in skip_styles):
            continue

        # Clean the dialogue text
        cleaned = clean_ass_text(text)
        if cleaned and len(cleaned) > 1:
            lines.append(cleaned)

    return lines


def clean_ass_text(text):
    """Clean ASS subtitle formatting tags from text."""
    # Remove override tags like {\pos(x,y)} {\fad(x,y)} etc
    text = re.sub(r'\{[^}]*\}', '', text)
    # Remove text in curly braces that are comments/alternatives
    # e.g., {Alternative translation - CR}
    text = re.sub(r'\{[^}]*\}', '', text)
    # Replace \N (ASS newline) with space
    text = text.replace('\\N', ' ')
    text = text.replace('\\n', ' ')
    # Remove \h (hard space)
    text = text.replace('\\h', ' ')
    # Remove any remaining backslash commands
    text = re.sub(r'\\[a-zA-Z]+[^a-zA-Z\s]*', '', text)
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Skip if it's just punctuation or very short
    if len(text.strip('!?.…, ')) < 1:
        return ''
    return text


def main():
    # Find all English subtitle files
    # Pattern: files ending with " en.ass" (not "en cc.ass" to avoid duplicates)
    all_ass_files = glob.glob(os.path.join(SUBS_DIR, '**', '* en.ass'), recursive=True)
    # Sort by path to maintain arc order
    all_ass_files.sort()

    print(f"Found {len(all_ass_files)} English subtitle files")

    all_dialogue = []
    for i, filepath in enumerate(all_ass_files):
        arc_name = os.path.basename(os.path.dirname(os.path.dirname(filepath)))
        filename = os.path.basename(filepath)
        lines = extract_dialogue_from_ass(filepath)
        all_dialogue.extend(lines)
        if (i + 1) % 20 == 0 or i == len(all_ass_files) - 1:
            print(f"  [{i+1}/{len(all_ass_files)}] {len(all_dialogue)} total lines")

    print(f"\nTotal dialogue lines: {len(all_dialogue)}")

    # Write output
    output_file = os.path.join(OUTPUT_DIR, 'onepace_all_dialogue.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(all_dialogue))

    size = os.path.getsize(output_file)
    print(f"Saved to: {output_file}")
    print(f"File size: {size:,} bytes ({size/1024:.1f} KB / {size/1024/1024:.1f} MB)")

    # Show sample
    print("\nSample dialogue (first 15 lines):")
    for line in all_dialogue[:15]:
        print(f"  > {line}")
    print("\nSample dialogue (random mid-point):")
    mid = len(all_dialogue) // 2
    for line in all_dialogue[mid:mid+10]:
        print(f"  > {line}")


if __name__ == '__main__':
    main()
