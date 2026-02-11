# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Build Fellowship project: building and training a GPT-architecture Transformer language model from scratch. The project spans 8 weekly workshops covering data preprocessing, tokenization, model architecture, training, evaluation, and fine-tuning. The final deliverable is a trained generative language model with a command-line interface (CLI).

Mentor: Kacper Raczy (Research Engineer at Comma AI, data science fellow at Build Fellowship).

The default training dataset is `tinyshakespeare.txt` (~1MB of Shakespeare text). Participants may substitute their own text datasets.

## Environment Setup

- Python 3.11 (installed via `uv`, managed at `~/.local/bin`)
- Virtual environment at `venv/` (created with `uv venv --python 3.11`)

```bash
# Activate venv
source venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt
```

## Key Dependencies

- **torch ~=2.6.0** — model architecture and training (GPT decoder-only transformer)
- **tiktoken** — tokenizer (same BPE tokenizer used by OpenAI GPT models)
- **numpy, pandas, scipy, scikit-learn** — data processing and evaluation
- **matplotlib, seaborn, plotly** — visualization and training metrics
- **spacy** — NLP utilities
- **jupyter, notebook, ipython** — interactive experimentation
- **tqdm** — training progress bars

## Architecture Notes

This is a **decoder-only GPT model** (no encoder block). The focus is on autoregressive text generation, not classification. The model should learn to generate text that mimics the style of its training corpus.

Expected project structure as it develops:
- Data preprocessing / tokenization pipeline
- Model definition (transformer blocks: self-attention, feed-forward, layer norm)
- Training loop with loss tracking
- Text generation / inference
- CLI entry point for interacting with the trained model
