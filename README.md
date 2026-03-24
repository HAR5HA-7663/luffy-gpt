# Luffy GPT

A GPT transformer built from scratch in PyTorch, trained on One Piece (Luffy) dialogue. Fine-tuned with 5,004 conversation pairs to respond as Luffy.

Built as part of the [Build Fellowship](https://buildfellowship.com/) program.

## Quick start

```bash
# setup
git clone https://github.com/HAR5HA-7663/luffy-gpt.git
cd luffy-gpt
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# download model weights from HuggingFace
pip install huggingface_hub
python -c "from huggingface_hub import hf_hub_download; hf_hub_download('HAR5HA-YELLELA/luffy-gpt', 'luffy_gpt_gpt2tok_finetuned.pth', local_dir='.')"

# chat with Luffy
python gpt.py --input dataset/processed/corpus_clean.txt \
  --eval luffy_gpt_gpt2tok_finetuned.pth --interactive --style 1 \
  --temperature 0.8 --top-k 50 --top-p 0.9 --repetition-penalty 1.2 \
  --gpt2-tokenizer
```

## What this does

```
You: who are you?
Luffy: I'm Monkey D. Luffy!

You: are you scared?
Luffy: Nope.

You: are you okay?
Luffy: Not really.
```

Works best with direct questions. Not a perfect chatbot -- it's a 49M param model trained from scratch on limited data.

## Model specs

| | Base model | Final model |
|--|-----------|-------------|
| Parameters | 49.39M | 49.39M |
| Tokenizer | GPT-2 BPE (50,257 vocab) | GPT-2 BPE (50,257 vocab) |
| Context | 256 tokens (~100 words) | 256 tokens (~100 words) |
| Training | 25k steps on 3.2M char corpus | + 10k steps SFT on 5,004 Q&A pairs |
| Architecture | n_embd=384, n_head=6, n_layer=6 | + style embedding + EOS token |

## CLI usage

```bash
# train from scratch
python gpt.py --input dataset/processed/corpus_clean.txt \
  --train model.pth --epoch 25000 --gpt2-tokenizer

# fine-tune with SFT
python gpt.py --input dataset/processed/corpus_clean.txt \
  --finetune model_ft.pth --sft-input dataset/luffy_sft.txt \
  --pretrained model.pth --epoch 10000 --lr 5e-5 --gpt2-tokenizer

# evaluate
python gpt.py --input dataset/processed/corpus_clean.txt \
  --eval model_ft.pth --interactive --style 1 --gpt2-tokenizer
```

All flags: `python gpt.py --help`

## Project structure

```
gpt.py                  -- model classes, training, fine-tuning, inference CLI
util.py                 -- CharacterTokenizer, Dataset classes
loss.py                 -- loss estimation utility
metrics.py              -- perplexity, ROUGE, BERTScore, masked accuracy
app.py                  -- Gradio web app (HuggingFace Spaces)

transformer.ipynb       -- lecture 3: self-attention exploration
transformer_final.ipynb -- lecture 5: scaled pretraining
baseline.ipynb          -- bigram baseline model
tokenization.ipynb      -- tokenizer experiments
lecture6.ipynb          -- metrics, KV cache
lecture7.ipynb          -- SFT fine-tuning with style tokens

dataset/
  processed/corpus_clean.txt  -- training corpus (Luffy dialogue)
  luffy_sft.txt               -- 5,004 SFT conversation pairs
  luffy_sft.jsonl             -- same pairs in JSONL format
  generate_luffy_sft_v2.py    -- SFT dataset generator
  scrapers/                   -- data collection scripts

tokenizer/
  train_tokenizer.py          -- SentencePiece BPE trainer
  tokenizer.py                -- LuffyTokenizer wrapper
```

## Techniques used

Everything from the Build Fellowship lecture series:

- Decoder-only GPT from scratch (self-attention, multi-head, residual connections, LayerNorm, dropout)
- Three tokenizers tested: character-level, custom BPE (SentencePiece), GPT-2 BPE
- KV cache for faster inference
- Cosine LR decay with warmup, weight decay, early stopping
- Top-k, top-p, temperature, repetition penalty sampling
- Style token conditioning for SFT
- EOS token for controlled generation stopping
- Evaluation: perplexity, ROUGE, BERTScore, masked accuracy
- Multi-GPU training (DataParallel)

## Links

- Model weights: [huggingface.co/HAR5HA-YELLELA/luffy-gpt](https://huggingface.co/HAR5HA-YELLELA/luffy-gpt)
- Live demo: [huggingface.co/spaces/HAR5HA-YELLELA/luffy-gpt](https://huggingface.co/spaces/HAR5HA-YELLELA/luffy-gpt)
