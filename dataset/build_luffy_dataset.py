"""
Scrape Luffy's dialogue from Anime Transcripts Wiki (animetranscript.fandom.com)
and compile into a single luffy.txt dataset for GPT training.
"""

import requests
import re
import time
import json
import os

WIKI_API = 'https://animetranscript.fandom.com/api.php'
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_FILE = os.path.join(OUTPUT_DIR, 'luffy.txt')
PROGRESS_FILE = os.path.join(OUTPUT_DIR, 'scrape_progress.json')


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
        # Skip non-content pages
        if any(kw in title for kw in skip_keywords):
            continue
        # Skip the main "One Piece" page itself
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


def extract_luffy_lines(wikitext):
    """Extract Luffy's dialogue lines from wikitext."""
    if not wikitext:
        return []

    lines = wikitext.split('\n')
    luffy_lines = []

    # Patterns for Luffy's dialogue:
    # '''Luffy:''' ''text''
    # '''Luffy:''' text
    # '''Monkey D. Luffy:''' ''text''
    # Also handle variations like Luffy + Other: for shared lines
    luffy_patterns = [
        r"'''(?:Monkey D\. )?Luffy(?:\s*[\+\&]\s*\w+)*:'''\s*(.+)",
        r"\*\s*'''(?:Monkey D\. )?Luffy(?:\s*[\+\&]\s*\w+)*:'''\s*(.+)",
    ]

    for line in lines:
        line = line.strip()
        for pattern in luffy_patterns:
            match = re.search(pattern, line, re.IGNORECASE)
            if match:
                dialogue = match.group(1)
                # Clean up wiki markup
                dialogue = clean_dialogue(dialogue)
                if dialogue and len(dialogue) > 1:
                    luffy_lines.append(dialogue)
                break

    return luffy_lines


def clean_dialogue(text):
    """Remove wiki markup and formatting from dialogue text."""
    # Remove italic markers
    text = text.replace("''", "")
    # Remove bold markers
    text = text.replace("'''", "")
    # Remove [emotion/action] tags but keep the text readable
    text = re.sub(r'\[(?:Amazed|Shocked|Angry|Happy|Sad|Crying|Laughing|'
                  r'Smiling|Yelling|Screaming|Confused|Terrified|Excited|'
                  r'Determined|Serious|Surprised|Worried|Confident|Grinning|'
                  r'Thinking|Whispering|Annoyed|Frustrated|Relieved|Scared|'
                  r'Proud|Calm|Nervous|Panting|Eating|Running|Fighting|'
                  r'Standing|Sitting|Walking|Looking|Pointing|Holding|'
                  r'Grabbing|Punching|Kicking|Stretching|Jumping|Landing|'
                  r'Falling|Flying|Swimming|Climbing|Pulling|Pushing|'
                  r'Catching|Throwing|Breaking|Opening|Closing|Turning|'
                  r'Nodding|Shaking|Waving|Hugging|Smacking|Slapping|'
                  r'Hitting|Blocking|Dodging|Charging|Attacking|Defending|'
                  r'Remembering|Flashback|Sighing|Gasping|Coughing|Snoring|'
                  r'Waking|Sleeping|Drooling|Sniffing|Tasting|Chewing|'
                  r'Swallowing|Burping|Stretches|Grins|Smiles|Laughs|'
                  r'Cries|Yells|Screams|Shouts|Whispers|Mumbles|Mutters'
                  r')[^\]]*\]\s*', '', text, flags=re.IGNORECASE)
    # Remove remaining bracket tags that look like stage directions
    text = re.sub(r'\[[A-Z][a-z]+(?:\s+[a-z]+)*\]\s*', '', text)
    # Remove wiki links [[text|display]] -> display, [[text]] -> text
    text = re.sub(r'\[\[[^\]]*\|([^\]]*)\]\]', r'\1', text)
    text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)
    # Remove external links [url text] -> text
    text = re.sub(r'\[https?://\S+\s+([^\]]+)\]', r'\1', text)
    text = re.sub(r'\[https?://\S+\]', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove {{templates}}
    text = re.sub(r'\{\{[^\}]*\}\}', '', text)
    # Clean up \N (subtitle newlines)
    text = text.replace('\\N', ' ')
    # Remove special unicode decorations
    text = re.sub(r'[─━═☆☽✦★♦♣♠♥●◆▶►▷➤➦➥]', '', text)
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Remove leading/trailing punctuation artifacts
    text = text.strip('*•·⁃–—')
    text = text.strip()

    return text


def load_progress():
    """Load scraping progress to allow resuming."""
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {'completed': [], 'all_lines': []}


def save_progress(progress):
    """Save scraping progress."""
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)


def main():
    print("Fetching episode page list...")
    pages = get_all_episode_pages()
    print(f"Found {len(pages)} episode pages")

    progress = load_progress()
    completed = set(progress['completed'])
    all_lines = progress['all_lines']

    remaining = [p for p in pages if p not in completed]
    print(f"Already scraped: {len(completed)}, Remaining: {len(remaining)}")

    for i, page_title in enumerate(remaining):
        try:
            print(f"[{len(completed) + i + 1}/{len(pages)}] Scraping: {page_title}...", end=' ')
            wikitext = get_page_wikitext(page_title)
            if wikitext:
                lines = extract_luffy_lines(wikitext)
                all_lines.extend(lines)
                print(f"found {len(lines)} Luffy lines")
            else:
                print("no content")

            completed.add(page_title)

            # Save progress every 20 pages
            if (i + 1) % 20 == 0:
                progress['completed'] = list(completed)
                progress['all_lines'] = all_lines
                save_progress(progress)
                print(f"  [Progress saved: {len(all_lines)} total lines so far]")

            # Be polite to the server
            time.sleep(0.3)

        except requests.exceptions.Timeout:
            print("TIMEOUT - skipping")
            continue
        except Exception as e:
            print(f"ERROR: {e}")
            continue

    # Final save
    progress['completed'] = list(completed)
    progress['all_lines'] = all_lines
    save_progress(progress)

    # Write final dataset
    print(f"\n{'='*60}")
    print(f"Total Luffy lines extracted: {len(all_lines)}")

    # Deduplicate while preserving order
    seen = set()
    unique_lines = []
    for line in all_lines:
        if line not in seen:
            seen.add(line)
            unique_lines.append(line)

    print(f"Unique lines: {len(unique_lines)}")

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write('\n'.join(unique_lines))

    file_size = os.path.getsize(OUTPUT_FILE)
    print(f"Written to: {OUTPUT_FILE}")
    print(f"File size: {file_size:,} bytes ({file_size/1024:.1f} KB)")


if __name__ == '__main__':
    main()
