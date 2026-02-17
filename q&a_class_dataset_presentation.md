# Dataset Presentation - Q&A Prep

## Project
**Training a GPT model to generate text in Monkey D. Luffy's style (One Piece)**

---

## Dataset Sources

| Source | URL | What I got |
|--------|-----|------------|
| Anime Transcripts Wiki | https://animetranscript.fandom.com/wiki/One_Piece | 193 character-tagged dialogue lines from Episode 1 (only 1 out of 1089 episodes was fully transcribed) |
| One Pace Public Subtitles (GitHub) | https://github.com/one-pace/one-pace-public-subtitles | 271 English ASS subtitle files covering the full anime. No character names, but has style tags (Main, Flashback, Narrator). Extracted 105,865 dialogue lines. |
| One Piece Fandom Wiki | https://onepiece.fandom.com/wiki/Monkey_D._Luffy | Scraped Luffy's history and arc pages for quoted dialogue. Low yield — wiki uses narrative prose, not direct quotes. |
| Figshare (ramazantrn dataset) | https://figshare.com/articles/dataset/30188161 | Character-tagged transcripts for Ep 382-777. Could not access — returned 403 Forbidden. |
| One Piece Transcripts (Kaggle) | https://www.kaggle.com/datasets/kaproquinji/one-piece-text-data | Only contained wiki summaries and character metadata, not dialogue. |

### Scraping Scripts (in `dataset/` folder)

- `build_luffy_dataset.py` - Scraped Anime Transcripts Wiki for Luffy-tagged lines
- `scrape_fandom_luffy.py` - Scraped One Piece Fandom Wiki for Luffy quotes
- `extract_onepace_dialogue.py` - Parsed ASS subtitle files, stripped formatting tags
- `build_onepace_conversational.py` - Reformatted subtitles into TinyShakespeare style
- `build_conversational_dataset.py` - Extracted character-tagged conversation pairs
- `build_final_conversational.py` - Combined all sources into final dataset

---

## Dataset Stats

| Metric | Value |
|--------|-------|
| Raw file | `luffy_dataset.txt` — 3.2 MB |
| Raw characters | 3,300,842 |
| Parsed blocks | 5,460 |
| After deduplication | 5,171 blocks |
| Train split (80%) | 4,136 blocks — 2.5 MB |
| Validation split (10%) | 517 blocks — 359 KB |
| Test split (10%) | 518 blocks — 300 KB |

---

## Anticipated Questions & Answers

### Q1: Why didn't you use Q&A format?

**A:** My dataset is style imitation, not Q&A — and that matches the TinyShakespeare approach from class. Q&A format requires clean question/answer pairs. One Piece doesn't have that — no one has tagged 1000+ episodes of anime dialogue with character names. I scraped 5+ sources and only found 1 fully transcribed episode out of 1089. Style imitation is the correct objective when you have a text corpus and want the model to learn a character's speaking patterns.

### Q2: Why are there "Main:" and "Flashback:" labels instead of character names?

**A:** The One Pace subtitle files use ASS format where each dialogue line has a style tag (Main, Flashback, Narrator, Thoughts) but no character name field. These are the only labels the data had. I used them as speaker labels because they still provide useful structure — they tell the model whether text is regular dialogue, a flashback, or narration. The actual dialogue text is real One Piece dialogue from 271 episodes.

### Q3: Only 523 lines have real character names. Isn't that a problem?

**A:** Yes, it's a limitation. Only Episode 1 on the Anime Transcripts Wiki had a full character-tagged transcript. The remaining 140K+ lines have style-based labels. However, the model doesn't learn from the labels alone — it learns from the text patterns. The first 523 lines teach it what the `Character:\nresponse` format looks like, and the rest teaches it how One Piece dialogue sounds. At inference, prompting with `Luffy:` generates text in the style the model learned from the full corpus.

### Q4: Is 3.2 MB enough data to train a GPT?

**A:** TinyShakespeare is 1.1 MB and it works well enough for a workshop-scale model. My dataset is 3x larger at 3.2 MB with 105K+ lines and 5,171 unique dialogue blocks. It's well above the "tens of thousands of lines" threshold. For a from-scratch character-level or small BPE model, this is a solid size. It won't produce GPT-4 quality, but it will learn recognizable One Piece dialogue patterns.

### Q5: How did you clean the data?

**A:** The `preprocess_and_split.py` script handles:
- Removing HTML tags and URLs
- Normalizing smart quotes, em dashes, and ellipses to ASCII
- Stripping extra whitespace
- Removing bracketed stage directions like `[Angry]` or `[Laughing]`
- Dropping lines with no alphabetic characters
- Deduplicating identical speaker+dialogue blocks
- This reduced 5,460 parsed blocks to 5,171 after dedup.

### Q6: How did you split the data?

**A:** 80/10/10 split (train/validation/test) with a fixed random seed of 42 for reproducibility. The split is done at the block level (speaker + their dialogue lines), not at the individual line level, so conversations aren't broken mid-exchange.

### Q7: Why Luffy / One Piece?

**A:** I wanted to train a model that generates text in Luffy's speaking style — short exclamations, food references, pirate king declarations, attack callouts. It's a unique and fun dataset compared to generic text, and the data pipeline I built (scraping wikis, parsing subtitle files, cleaning ASS formatting tags) demonstrates real data engineering work.

### Q8: What would you do differently with more time?

**A:**
- Find Yibis fansub ASS files — they had character names in the Name field for episodes 382-777, but the compiled dataset on Figshare was 403 blocked
- Use ML-based speaker diarization to label the unnamed subtitle lines with character names
- Manually annotate a few hundred more conversation pairs for the character-tagged section
- Train a larger model using the dual 3090 GPUs to see if the style imitation improves with more compute

### Q9: What tokenizer are you using?

**A:** Ran `tokenizer_sanity_check.py` on the train split (2.55 MB):

| Tokenizer | Tokens | Chars/Token | Vocab Size |
|-----------|--------|-------------|------------|
| Character-level | 2,550,667 | 1.00 | 96 |
| gpt2 (BPE) | 779,136 | 3.27 | 50,257 |
| cl100k_base (BPE) | 691,908 | 3.69 | 100,256 |
| o200k_base (BPE) | 655,915 | 3.89 | 200,019 |

Character-level gives a tiny vocabulary (96 unique chars) but 2.5M tokens — very long sequences. BPE compresses ~3-4x, so the model sees more context per training window. For a from-scratch model, character-level is simpler to implement but slower to train. BPE (gpt2 or cl100k_base) is the practical choice.

### Q10: Will this model actually talk like Luffy?

**A:** It will generate text that sounds like One Piece dialogue — short, energetic lines with the vocabulary and cadence of the show. Since Luffy is the main character and speaks the most across 271 episodes, his patterns naturally dominate the corpus. It won't understand questions or have real conversations — it's a text generator, not a chatbot. You prompt it with `Luffy:` and it predicts what comes next based on learned patterns.

### Q11: What is your vocabulary size and compression ratio?

**A:** Character-level vocab is 96 (all unique ASCII characters in the corpus). With gpt2 BPE, the compression ratio is 3.27x (2.55M chars → 779K tokens). With cl100k_base it's 3.69x, and o200k_base gives 3.89x. Higher compression = shorter sequences = more context per training window, but larger embedding tables.

### Q12: What context window (block size) are you planning?

**A:** Starting with block_size=256 for character-level (matches Karpathy's nanoGPT default). For BPE, 256 tokens covers ~940 characters (~3.7x compression), which is roughly 8-12 dialogue lines — enough for a full conversation exchange. Will experiment with 512 if the 3090s have enough VRAM.

### Q13: How are you formatting samples autoregressively?

**A:** The dataset is already in autoregressive format — just raw text. The model reads `Main:\nI refuse to lose!` and learns to predict each next token. At inference, you give it a prompt like `Luffy:\n` and it generates the continuation. No special `<question>` / `<answer>` tokens needed because this is style imitation, not instruction following.

### Q14: What do you expect the small model to learn?

**A:** Style, not reasoning. The model should learn One Piece vocabulary (attack names, character names, pirate terminology), sentence structure (short exclamations, dramatic declarations), and cadence (how dialogue flows between speakers). It won't understand plot, answer questions, or reason. It's a text generator that outputs text statistically similar to its training data.

### Q15: How will you evaluate quality on validation/test?

**A:** Primary metric is validation loss (cross-entropy) — lower means the model predicts the next token better. I'll track train vs val loss curves to detect overfitting. For qualitative evaluation, I'll generate samples with different temperature settings and manually check if they sound like One Piece dialogue (correct names, attack callouts, pirate vocabulary).

### Q16: Did you check for overfitting risk due to small data?

**A:** 3.2 MB is 3x TinyShakespeare, so it's adequate for a workshop-scale model. The main overfitting risk is the model memorizing specific dialogue blocks. Mitigations: dropout in the model, monitoring train vs val loss divergence, and the 10% held-out test set for final evaluation. If val loss starts increasing while train loss drops, I'll reduce model size or add regularization.

### Q17: Show one raw sample vs one cleaned sample.

**A:**
**Raw (from ASS subtitle):**
```
Dialogue: 0,0:12:34.56,0:12:36.78,Main,,0,0,0,,{\pos(640,500)\fad(200,200)}I'm gonna be\N King of the Pirates!
```

**Cleaned (in dataset):**
```
Main:
I'm gonna be King of the Pirates!
```

The cleaning strips ASS formatting tags (`{\pos(...)}`, `\fad(...)`), converts `\N` to line breaks, removes style overrides, and formats into Speaker:\nDialogue blocks.

---

## 5 Sample Records (Cleaned)

**Sample 1 — Character-tagged (Episode 1):**
```
Koby:
You mean Zoro? I heard he was being held prisoner at a navy base.

Luffy:
Aw, he's a weakling?
```

**Sample 2 — Character-tagged (Episode 1):**
```
Alvida:
Who's the most beautiful of all on these seas?

Koby:
T-this ship's captain, Alvida-sama, of course!
```

**Sample 3 — One Pace dialogue (Main style):**
```
Main:
I refuse to lose! Damn you!
I'm gonna eat you!
Part-Part Cannon!
Quit struggling!
Just accept your fate!
```

**Sample 4 — One Pace dialogue (Flashback style):**
```
Flashback:
Gum-Gum... Bazooka!
Retreat for now, men!
Everyone! Find any way out you can!
```

**Sample 5 — One Pace dialogue (conversation flow):**
```
Main:
Did we all make it?!
Are the people you were with
when you got off the ship here?!
Yes!
Yeah.
Yep!
```
