"""
Build a conversational Luffy dataset from the Anime Transcripts Wiki.
Format: dialogue pairs where another character speaks and Luffy responds.

Output format (matches TinyShakespeare):
    Koby:
    But that's impossible!

    Luffy:
    I've decided to be the King of the Pirates, so if I die fighting for that, that's fine with me!
"""

import requests
import re
import time
import json
import os

WIKI_API = 'https://animetranscript.fandom.com/api.php'
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(OUTPUT_DIR)


def get_all_episode_pages():
    """Get all One Piece episode page titles from the main index."""
    params = {
        'action': 'parse',
        'page': 'One Piece',
        'format': 'json',
        'prop': 'links'
    }
    r = requests.get(WIKI_API, params=params, timeout=30)
    data = r.json()
    links = data.get('parse', {}).get('links', [])

    episode_pages = []
    skip_keywords = [
        'Category:', 'Template:', 'User:', 'File:', 'MediaWiki:',
        'Module:', 'Special:', 'Talk:', 'Help:'
    ]
    for link in links:
        title = link.get('*', '')
        if any(kw in title for kw in skip_keywords):
            continue
        if title == 'One Piece':
            continue
        episode_pages.append(title)
    return episode_pages


def get_page_wikitext(page_title):
    """Fetch the raw wikitext for a page."""
    params = {
        'action': 'parse',
        'page': page_title,
        'format': 'json',
        'prop': 'wikitext'
    }
    r = requests.get(WIKI_API, params=params, timeout=30)
    data = r.json()
    if 'error' in data:
        return None
    return data.get('parse', {}).get('wikitext', {}).get('*', '')


def parse_dialogue_lines(wikitext):
    """
    Parse wikitext into a list of (character, dialogue) tuples.
    Wiki format: '''Character:''' ''dialogue text''
    """
    if not wikitext:
        return []

    dialogues = []

    # Find the transcript section
    transcript_match = re.search(r'==\s*Transcript\s*==', wikitext, re.IGNORECASE)
    if not transcript_match:
        return []

    # Get text after the transcript header
    transcript_text = wikitext[transcript_match.end():]
    # Stop at next major section
    next_section = re.search(r'\n==\s*[^=]', transcript_text)
    if next_section:
        transcript_text = transcript_text[:next_section.start()]

    lines = transcript_text.split('\n')

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Match: '''Character Name:''' ''dialogue''
        # Also handles: '''Character Name:''' dialogue
        # And: '''Character + Character:''' ''dialogue''
        match = re.match(
            r"(?:\*\s*)?'''([^']+?):'''\s*(.+)",
            line
        )
        if match:
            character = match.group(1).strip()
            dialogue = match.group(2).strip()

            # Skip narration, scene descriptions
            if character.lower() in ['narrator', 'title card', 'title',
                                      'opening', 'ending', 'eyecatcher',
                                      'preview']:
                continue

            # Clean the dialogue
            dialogue = clean_dialogue(dialogue)
            if dialogue and len(dialogue) > 1:
                dialogues.append((character, dialogue))

    return dialogues


def clean_dialogue(text):
    """Remove wiki markup from dialogue."""
    # Remove italic markers
    text = text.replace("''", "")
    # Remove bold markers
    text = text.replace("'''", "")
    # Remove [emotion/action] bracketed tags
    text = re.sub(r'\[[^\]]*\]\s*', '', text)
    # Remove wiki links [[text|display]] -> display
    text = re.sub(r'\[\[[^\]]*\|([^\]]*)\]\]', r'\1', text)
    text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)
    # Remove external links
    text = re.sub(r'\[https?://\S+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://\S+\]', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove {{templates}}
    text = re.sub(r'\{\{[^\}]*\}\}', '', text)
    # Remove special decorations
    text = re.sub(r'[─━═☆☽✦★♦♣♠♥●◆▶►▷➤➦➥⇐⇒⇑⇓⤴⤵↑↓←→]', '', text)
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    text = text.strip('*•·⁃–—')
    text = text.strip()
    # Remove if it's just punctuation
    if not re.search(r'[a-zA-Z]', text):
        return ''
    return text


def is_luffy(character):
    """Check if a character name refers to Luffy."""
    name = character.lower().strip()
    return name in ['luffy', 'monkey d. luffy', 'monkey d luffy',
                    'young luffy', 'kid luffy', 'child luffy',
                    'luffy (young)', 'luffy (kid)']


def build_conversation_pairs(dialogues):
    """
    Build conversation pairs: (other character's line, Luffy's response).
    Also captures multi-turn exchanges.
    """
    pairs = []

    for i in range(len(dialogues)):
        char_i, line_i = dialogues[i]

        if is_luffy(char_i):
            # Luffy is speaking - look for the preceding non-Luffy line
            if i > 0:
                char_prev, line_prev = dialogues[i - 1]
                if not is_luffy(char_prev):
                    pairs.append((char_prev, line_prev, line_i))
            else:
                # Luffy speaks first - use it as a standalone
                pairs.append((None, None, line_i))

            # Check if Luffy continues speaking (multi-line response)
            # Combine consecutive Luffy lines
            j = i + 1
            combined_luffy = line_i
            while j < len(dialogues) and is_luffy(dialogues[j][0]):
                combined_luffy += ' ' + dialogues[j][1]
                j += 1

            if combined_luffy != line_i and i > 0:
                char_prev, line_prev = dialogues[i - 1]
                if not is_luffy(char_prev):
                    pairs.append((char_prev, line_prev, combined_luffy))

    return pairs


def format_dataset(all_pairs):
    """
    Format conversation pairs matching TinyShakespeare format:

    Character Name:
    Dialogue text here.

    Luffy:
    Response text here.
    """
    lines = []

    for char_name, prompt, response in all_pairs:
        if char_name and prompt:
            lines.append(f"{char_name}:")
            lines.append(prompt)
            lines.append("")
            lines.append("Luffy:")
            lines.append(response)
            lines.append("")
        else:
            lines.append("Luffy:")
            lines.append(response)
            lines.append("")

    return '\n'.join(lines)


def main():
    print("Fetching episode page list...")
    pages = get_all_episode_pages()
    print(f"Found {len(pages)} episode pages\n")

    all_pairs = []
    episodes_with_content = 0

    for i, page_title in enumerate(pages):
        try:
            print(f"[{i+1}/{len(pages)}] {page_title}...", end=' ', flush=True)
            wikitext = get_page_wikitext(page_title)

            if not wikitext:
                print("no content")
                time.sleep(0.3)
                continue

            dialogues = parse_dialogue_lines(wikitext)
            if not dialogues:
                print("no dialogue")
                time.sleep(0.3)
                continue

            pairs = build_conversation_pairs(dialogues)
            if pairs:
                all_pairs.extend(pairs)
                episodes_with_content += 1
                print(f"{len(dialogues)} dialogue lines -> {len(pairs)} Luffy pairs")
            else:
                print(f"{len(dialogues)} lines but no Luffy pairs")

            time.sleep(0.3)

        except requests.exceptions.Timeout:
            print("TIMEOUT")
            continue
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    print(f"\n{'='*60}")
    print(f"Episodes with transcripts: {episodes_with_content}")
    print(f"Total conversation pairs: {len(all_pairs)}")

    # Deduplicate pairs
    seen = set()
    unique_pairs = []
    for pair in all_pairs:
        key = (pair[1], pair[2]) if pair[1] else (None, pair[2])
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    print(f"Unique pairs: {len(unique_pairs)}")

    # Format and save
    dataset_text = format_dataset(unique_pairs)
    output_file = os.path.join(OUTPUT_DIR, 'luffy_conversational.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(dataset_text)

    size = os.path.getsize(output_file)
    print(f"\nSaved to: {output_file}")
    print(f"File size: {size:,} bytes ({size/1024:.1f} KB)")

    # Also copy to project root
    import shutil
    dest = os.path.join(PROJECT_DIR, 'luffy_conversational.txt')
    shutil.copy2(output_file, dest)
    print(f"Copied to: {dest}")

    # Show samples
    print("\nSample conversation pairs:")
    print("-" * 40)
    sample = format_dataset(unique_pairs[:15])
    print(sample)


if __name__ == '__main__':
    main()
