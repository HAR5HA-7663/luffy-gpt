"""
Build final conversational dataset combining:
1. Luffy dialogue pairs from Episode 1 transcript (character-tagged)
2. All One Pace dialogue in TinyShakespeare format (bulk corpus)

This gives us a dataset in TinyShakespeare format where the model
learns One Piece conversational style.
"""

import requests
import re
import os
import shutil

WIKI_API = 'https://animetranscript.fandom.com/api.php'
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(OUTPUT_DIR)


def clean_dialogue(text):
    text = text.replace("''", "").replace("'''", "")
    text = re.sub(r'\[[^\]]*\]\s*', '', text)
    text = re.sub(r'\[\[[^\]]*\|([^\]]*)\]\]', r'\1', text)
    text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)
    text = re.sub(r'\[https?://\S+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://\S+\]', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'\{\{[^\}]*\}\}', '', text)
    text = re.sub(r'[─━═☆☽✦★♦♣♠♥●◆▶►▷➤➦➥⇐⇒⤴⤵↑↓←→]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.strip('*•·⁃–—').strip()
    return text


def scrape_all_transcribed_episodes():
    """Scrape ALL episodes that have transcripts (not just ep 1)."""
    # Get all episode page links
    params = {
        'action': 'parse', 'page': 'One Piece',
        'format': 'json', 'prop': 'links'
    }
    r = requests.get(WIKI_API, params=params, timeout=30)
    links = r.json().get('parse', {}).get('links', [])

    pages = []
    skip = ['Category:', 'Template:', 'User:', 'File:', 'MediaWiki:',
            'Module:', 'Special:', 'Talk:', 'Help:']
    for link in links:
        title = link.get('*', '')
        if title == 'One Piece' or any(s in title for s in skip):
            continue
        pages.append(title)

    print(f"Checking {len(pages)} episode pages for transcripts...")

    all_dialogues = []
    episodes_found = 0

    for i, page in enumerate(pages):
        try:
            params = {
                'action': 'parse', 'page': page,
                'format': 'json', 'prop': 'wikitext'
            }
            r = requests.get(WIKI_API, params=params, timeout=15)
            data = r.json()
            if 'error' in data:
                continue

            wikitext = data.get('parse', {}).get('wikitext', {}).get('*', '')
            if not wikitext:
                continue

            # Check for transcript section
            transcript_match = re.search(r'==\s*Transcript\s*==', wikitext, re.IGNORECASE)
            if not transcript_match:
                continue

            # Extract dialogue from transcript
            transcript_text = wikitext[transcript_match.end():]
            next_section = re.search(r'\n==\s*[^=]', transcript_text)
            if next_section:
                transcript_text = transcript_text[:next_section.start()]

            episode_dialogues = []
            for line in transcript_text.split('\n'):
                line = line.strip()
                match = re.match(r"(?:\*\s*)?'''([^']+?):'''\s*(.+)", line)
                if match:
                    char = match.group(1).strip()
                    dlg = clean_dialogue(match.group(2))
                    if char.lower() not in ['narrator', 'title card', 'title',
                                            'opening', 'ending', 'eyecatcher',
                                            'preview'] and dlg and len(dlg) > 1:
                        episode_dialogues.append((char, dlg))

            if episode_dialogues:
                all_dialogues.extend(episode_dialogues)
                episodes_found += 1
                print(f"  Found transcript: {page} ({len(episode_dialogues)} lines)")

        except Exception:
            pass

        # Progress
        if (i + 1) % 100 == 0:
            print(f"  [{i+1}/{len(pages)}] checked, {episodes_found} episodes with transcripts")

        import time
        time.sleep(0.2)

    return all_dialogues, episodes_found


def format_shakespeare_style(dialogues):
    """Format dialogue list into TinyShakespeare style."""
    blocks = []
    current_char = None
    current_lines = []

    for char, line in dialogues:
        if char == current_char:
            current_lines.append(line)
        else:
            if current_lines and current_char:
                blocks.append(f"{current_char}:\n" + '\n'.join(current_lines))
            current_char = char
            current_lines = [line]

    if current_lines and current_char:
        blocks.append(f"{current_char}:\n" + '\n'.join(current_lines))

    return '\n\n'.join(blocks)


def main():
    # Part 1: Get character-tagged dialogue from wiki transcripts
    print("=== Part 1: Scraping wiki transcripts ===")
    wiki_dialogues, ep_count = scrape_all_transcribed_episodes()
    print(f"\nFound {len(wiki_dialogues)} character-tagged lines from {ep_count} episodes")

    wiki_text = ''
    if wiki_dialogues:
        wiki_text = format_shakespeare_style(wiki_dialogues)
        print(f"Wiki section: {len(wiki_text):,} chars")

    # Part 2: Load One Pace conversational data
    print("\n=== Part 2: Loading One Pace conversational data ===")
    onepace_file = os.path.join(OUTPUT_DIR, 'onepace_conversational.txt')
    onepace_text = ''
    if os.path.exists(onepace_file):
        with open(onepace_file, 'r', encoding='utf-8') as f:
            onepace_text = f.read()
        print(f"One Pace section: {len(onepace_text):,} chars")

    # Combine: wiki (character-tagged) first, then One Pace (bulk)
    parts = []
    if wiki_text:
        parts.append(wiki_text)
    if onepace_text:
        parts.append(onepace_text)

    final_text = '\n\n'.join(parts)

    # Write final dataset
    output_file = os.path.join(PROJECT_DIR, 'luffy_dataset.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_text)

    size = os.path.getsize(output_file)
    line_count = final_text.count('\n') + 1
    print(f"\n{'='*60}")
    print(f"Final dataset: {output_file}")
    print(f"  Lines: {line_count:,}")
    print(f"  Size:  {size:,} bytes ({size/1024:.1f} KB / {size/1024/1024:.2f} MB)")
    tiny_size = os.path.getsize(os.path.join(PROJECT_DIR, 'tinyshakespeare.txt'))
    print(f"  vs TinyShakespeare: {size/tiny_size:.1f}x")

    # Show sample
    print("\nSample (first 1500 chars):")
    print("-" * 40)
    print(final_text[:1500])


if __name__ == '__main__':
    main()
