"""
Scrape Luffy's quoted dialogue from the One Piece Fandom Wiki.
The wiki has extensive history pages with dialogue woven into the narrative.
"""

import requests
import re
import json
import os
import time

WIKI_API = 'https://onepiece.fandom.com/api.php'
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Luffy's history is split across multiple pages on the wiki
LUFFY_PAGES = [
    'Monkey D. Luffy/History',
    'Monkey D. Luffy/Personality and Relationships',
    'Monkey D. Luffy',
]

# Also grab pages for each major arc where Luffy appears
# The wiki uses "X Arc" format
ARC_PAGES = [
    'Romance Dawn Arc', 'Orange Town Arc', 'Syrup Village Arc',
    'Baratie Arc', 'Arlong Park Arc', 'Loguetown Arc',
    'Reverse Mountain Arc', 'Whisky Peak Arc', 'Little Garden Arc',
    'Drum Island Arc', 'Arabasta Arc', 'Jaya Arc', 'Skypiea Arc',
    'Long Ring Long Land Arc', 'Water 7 Arc', 'Enies Lobby Arc',
    'Post-Enies Lobby Arc', 'Thriller Bark Arc', 'Sabaody Archipelago Arc',
    'Amazon Lily Arc', 'Impel Down Arc', 'Marineford Arc',
    'Post-War Arc', 'Return to Sabaody Arc', 'Fish-Man Island Arc',
    'Punk Hazard Arc', 'Dressrosa Arc', 'Zou Arc',
    'Whole Cake Island Arc', 'Levely Arc', 'Wano Country Arc',
    'Egghead Arc',
]


def get_wikitext(page_title):
    """Fetch raw wikitext from One Piece fandom wiki."""
    params = {
        'action': 'parse',
        'page': page_title,
        'format': 'json',
        'prop': 'wikitext'
    }
    try:
        r = requests.get(WIKI_API, params=params, timeout=30)
        data = r.json()
        if 'error' in data:
            return None
        return data.get('parse', {}).get('wikitext', {}).get('*', '')
    except Exception as e:
        print(f"  Error fetching {page_title}: {e}")
        return None


def extract_quoted_dialogue(wikitext, character='Luffy'):
    """Extract dialogue quotes attributed to a character from wiki narrative text."""
    if not wikitext:
        return []

    lines = []

    # Pattern 1: Direct speech in quotes attributed to Luffy
    # e.g., Luffy said "I'm gonna be King of the Pirates!"
    # e.g., Luffy declared, "text"
    # e.g., Luffy: "text"
    patterns = [
        # "quote" - Luffy said/declared/etc
        rf'"([^"]+)"\s*[-—]\s*{character}',
        rf'"([^"]+)"\s*[-—]\s*Monkey D\. {character}',
        # Luffy said "quote" / Luffy: "quote"
        rf'{character}\s*(?:said|says|declared|shouted|yelled|cried|screamed|exclaimed|replied|responded|asked|told|announced|stated|demanded|claimed|insisted|promised|vowed|whispered|muttered|called|called out)[,:]?\s*"([^"]+)"',
        # Luffy's catchphrases often appear in wiki formatting
        rf'{{{{qquote\|([^|}}]+)\|{character}',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, wikitext, re.IGNORECASE)
        for match in matches:
            cleaned = clean_text(match)
            if cleaned and len(cleaned) > 3:
                lines.append(cleaned)

    # Pattern 2: Look for quoted text near Luffy mentions
    # Find all quoted strings and check if Luffy is mentioned within 100 chars
    all_quotes = re.findall(r'"([^"]{5,300})"', wikitext)
    for quote in all_quotes:
        # Find the position of this quote in the text
        pos = wikitext.find(f'"{quote}"')
        if pos == -1:
            continue
        # Check surrounding context (200 chars before and after)
        context_start = max(0, pos - 200)
        context_end = min(len(wikitext), pos + len(quote) + 200)
        context = wikitext[context_start:context_end].lower()

        # Check if Luffy is speaking (mentioned before the quote)
        before_quote = wikitext[context_start:pos].lower()
        luffy_patterns = ['luffy', 'he said', 'he told', 'he declared',
                         'he shouted', 'he yelled', 'he cried', 'he replied',
                         'he exclaimed', 'he asked', 'he announced',
                         'he stated', 'he screamed', 'he called']

        is_luffy_speaking = any(p in before_quote[-150:] for p in luffy_patterns)

        if is_luffy_speaking:
            cleaned = clean_text(quote)
            if cleaned and len(cleaned) > 3:
                lines.append(cleaned)

    return lines


def clean_text(text):
    """Clean wiki markup from text."""
    # Remove wiki links
    text = re.sub(r'\[\[[^\]]*\|([^\]]*)\]\]', r'\1', text)
    text = re.sub(r'\[\[([^\]]*)\]\]', r'\1', text)
    # Remove templates
    text = re.sub(r'\{\{[^\}]*\}\}', '', text)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Remove ref tags content
    text = re.sub(r'<ref[^>]*>.*?</ref>', '', text, flags=re.DOTALL)
    text = re.sub(r'<ref[^>]*/>', '', text)
    # Clean whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def main():
    all_luffy_lines = []

    # Scrape Luffy-specific pages
    print("=== Scraping Luffy character pages ===")
    for page in LUFFY_PAGES:
        print(f"Fetching: {page}...")
        wikitext = get_wikitext(page)
        if wikitext:
            lines = extract_quoted_dialogue(wikitext)
            print(f"  Found {len(lines)} dialogue lines")
            all_luffy_lines.extend(lines)
        time.sleep(0.5)

    # Scrape arc pages for Luffy dialogue
    print("\n=== Scraping arc pages ===")
    for page in ARC_PAGES:
        print(f"Fetching: {page}...")
        wikitext = get_wikitext(page)
        if wikitext:
            lines = extract_quoted_dialogue(wikitext)
            print(f"  Found {len(lines)} dialogue lines")
            all_luffy_lines.extend(lines)
        time.sleep(0.5)

    # Deduplicate
    seen = set()
    unique = []
    for line in all_luffy_lines:
        if line.lower() not in seen:
            seen.add(line.lower())
            unique.append(line)

    print(f"\nTotal lines: {len(all_luffy_lines)}")
    print(f"Unique lines: {len(unique)}")

    # Save
    output_file = os.path.join(OUTPUT_DIR, 'luffy_fandom.txt')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(unique))

    size = os.path.getsize(output_file)
    print(f"Saved to: {output_file}")
    print(f"File size: {size:,} bytes ({size/1024:.1f} KB)")

    # Show sample
    print("\nSample lines:")
    for line in unique[:20]:
        print(f"  > {line}")


if __name__ == '__main__':
    main()
