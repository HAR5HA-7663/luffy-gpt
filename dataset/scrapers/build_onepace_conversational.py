"""
Convert One Pace subtitle dialogue into TinyShakespeare-style conversational format.

Since subtitles don't tag character names, we use the Narrator style tag
to separate narration from dialogue, and group consecutive dialogue lines
into natural conversation blocks.

Output matches TinyShakespeare format:
    Speaker:
    Dialogue text here spanning
    multiple lines if needed.

    Speaker:
    Response text.
"""

import re
import os
import glob

SUBS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        'one-pace-subtitles', 'main')
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(OUTPUT_DIR)


def extract_dialogue_with_style(filepath):
    """Extract dialogue with style info from ASS subtitle file."""
    entries = []

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
        if not line.startswith('Dialogue:'):
            continue

        parts = line.split(',', 9)
        if len(parts) < 10:
            continue

        style = parts[3].strip()
        name = parts[4].strip()  # Character name field (sometimes populated)
        text = parts[9].strip()

        # Skip non-dialogue styles
        skip_styles = ['Karaoke', 'Lyrics', 'Warning', 'Credits', 'Title',
                       'Captions', 'Note', 'Sign', 'Caption']
        if any(s.lower() in style.lower() for s in skip_styles):
            continue

        cleaned = clean_ass_text(text)
        if not cleaned or len(cleaned) < 2:
            continue

        # Determine speaker type from style
        style_lower = style.lower()
        if 'narrator' in style_lower:
            speaker_type = 'Narrator'
        elif 'flashback' in style_lower:
            speaker_type = 'Flashback'
        elif 'thought' in style_lower:
            speaker_type = 'Thoughts'
        elif 'secondary' in style_lower:
            speaker_type = 'Secondary'
        else:
            speaker_type = 'Main'

        entries.append({
            'text': cleaned,
            'speaker_type': speaker_type,
            'name': name if name and name != 'chptr' else '',
        })

    return entries


def clean_ass_text(text):
    """Clean ASS subtitle formatting."""
    text = re.sub(r'\{[^}]*\}', '', text)
    text = text.replace('\\N', '\n')
    text = text.replace('\\n', '\n')
    text = text.replace('\\h', ' ')
    text = re.sub(r'\\[a-zA-Z]+[^a-zA-Z\s]*', '', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = text.strip()
    if not re.search(r'[a-zA-Z]', text):
        return ''
    return text


def format_as_shakespeare(entries):
    """
    Format subtitle entries into TinyShakespeare-style blocks.
    Group consecutive same-speaker entries and separate with blank lines.
    """
    if not entries:
        return ''

    blocks = []
    current_speaker = None
    current_lines = []

    for entry in entries:
        speaker = entry['speaker_type']

        # Skip narrator lines for cleaner dialogue
        if speaker == 'Narrator':
            # Flush current block
            if current_lines and current_speaker:
                blocks.append(format_block(current_speaker, current_lines))
                current_speaker = None
                current_lines = []
            continue

        if speaker == current_speaker:
            # Same speaker continues
            current_lines.append(entry['text'])
        else:
            # New speaker - flush previous
            if current_lines and current_speaker:
                blocks.append(format_block(current_speaker, current_lines))
            current_speaker = speaker
            current_lines = [entry['text']]

    # Flush last block
    if current_lines and current_speaker:
        blocks.append(format_block(current_speaker, current_lines))

    return '\n\n'.join(blocks)


def format_block(speaker, lines):
    """Format a single speaker block."""
    # Use generic speaker labels since we don't know character names
    # This matches how Shakespeare uses "First Citizen:", "All:", etc.
    combined = '\n'.join(lines)
    return f"{speaker}:\n{combined}"


def main():
    all_ass_files = glob.glob(os.path.join(SUBS_DIR, '**', '* en.ass'),
                              recursive=True)
    all_ass_files.sort()
    print(f"Found {len(all_ass_files)} English subtitle files")

    all_text_parts = []

    for i, filepath in enumerate(all_ass_files):
        entries = extract_dialogue_with_style(filepath)
        if entries:
            formatted = format_as_shakespeare(entries)
            if formatted:
                all_text_parts.append(formatted)

        if (i + 1) % 50 == 0 or i == len(all_ass_files) - 1:
            print(f"  [{i+1}/{len(all_ass_files)}] processed", flush=True)

    # Combine all episodes with double newlines between them
    full_text = '\n\n'.join(all_text_parts)

    output_file = os.path.join(OUTPUT_DIR, 'onepace_conversational.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(full_text)

    size = os.path.getsize(output_file)
    line_count = full_text.count('\n') + 1
    print(f"\nSaved to: {output_file}")
    print(f"Lines: {line_count:,}")
    print(f"Size: {size:,} bytes ({size/1024:.1f} KB / {size/1024/1024:.1f} MB)")

    # Copy to project root
    import shutil
    dest = os.path.join(PROJECT_DIR, 'onepace_conversational.txt')
    shutil.copy2(output_file, dest)
    print(f"Copied to: {dest}")

    # Show sample
    print("\nSample (first 1000 chars):")
    print("-" * 40)
    print(full_text[:1000])


if __name__ == '__main__':
    main()
