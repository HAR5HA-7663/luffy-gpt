import torch
from torch import nn
import torch.nn.functional as F


class Head(nn.Module):

    def __init__(self, head_size, n_embd, context_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(context_size, context_size)))
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, kv_cache=None):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)

        if kv_cache is not None:
            k_prev, v_prev = kv_cache
            k = torch.cat([k_prev, k], dim=1)
            v = torch.cat([v_prev, v], dim=1)

        new_cache = (k, v)
        T_full = k.shape[1]

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[T_full-T:T_full, :T_full] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v
        return out, new_cache


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size, n_embd, context_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, context_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, kv_cache=None):
        if kv_cache is None:
            kv_cache = [None] * len(self.heads)
        results = [h(x, c) for h, c in zip(self.heads, kv_cache)]
        outs, new_caches = zip(*results)
        out = torch.cat(list(outs), dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out, list(new_caches)


class FeedFoward(nn.Module):

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, n_embd * 4),
            nn.ReLU(),
            nn.Linear(n_embd * 4, n_embd),
            nn.Dropout(0.2),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):

    def __init__(self, n_embd, n_head, context_size):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size, n_embd, context_size)
        self.ffwd = FeedFoward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x, kv_cache=None):
        sa_out, new_cache = self.sa(self.ln1(x), kv_cache)
        x = x + sa_out
        x = x + self.ffwd(self.ln2(x))
        return x, new_cache


class GPTWithStyle(nn.Module):

    def __init__(self, vocab_size, n_embd=32, context_size=8, n_head=4, n_layer=4, n_styles=2):
        super().__init__()
        self.context_size = context_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_size, n_embd)
        self.style_embedding_table = nn.Embedding(n_styles, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head, context_size=context_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, style, targets=None, use_cache=False, kv_cache=None):
        if use_cache and kv_cache is not None and kv_cache[0] is not None:
            past_len = kv_cache[0][0][0].shape[1]
            emb = self.token_embedding_table(idx)
            pos = self.position_embedding_table(
                torch.arange(past_len, past_len + idx.shape[1], device=idx.device))
            x = emb + pos
        else:
            suffix = idx[:, -(self.context_size - 1):]
            T = suffix.shape[1] + 1
            style_emb = self.style_embedding_table(style).unsqueeze(1)
            tok_emb = self.token_embedding_table(suffix)
            emb = torch.cat([style_emb, tok_emb], dim=1)
            pos = self.position_embedding_table(torch.arange(T, device=idx.device))
            x = emb + pos

        if use_cache:
            if kv_cache is None:
                kv_cache = [None] * len(self.blocks)
            new_caches = []
            for block, cache in zip(self.blocks, kv_cache):
                x, nc = block(x, cache)
                new_caches.append(nc)
        else:
            new_caches = None
            for block in self.blocks:
                x, _ = block(x)

        logits = self.lm_head(self.ln_f(x))
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            loss = F.cross_entropy(logits.view(B*T, C), targets.view(B*T))
        if use_cache:
            return logits, loss, new_caches
        return logits, loss

    @torch.no_grad()
    def generate(self, start_idx, style, number_of_tokens,
                 temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0,
                 eos_tokens=None):
        self.eval()
        idx = start_idx
        prompt_len = start_idx.shape[1]
        kv_cache = None
        for _ in range(number_of_tokens):
            if kv_cache is not None and kv_cache[0][0][0].shape[1] >= self.context_size:
                kv_cache = None
            if kv_cache is not None:
                logits, _, kv_cache = self(idx[:, -1:], style, use_cache=True, kv_cache=kv_cache)
            else:
                idx_in = idx[:, -(self.context_size - 1):]
                logits, _, kv_cache = self(idx_in, style, use_cache=True)
            logits = _apply_sampling(logits, idx, temperature, top_k, top_p, repetition_penalty)
            probs = F.softmax(logits, dim=-1)
            next_tok = torch.multinomial(probs, 1)
            idx = torch.cat([idx, next_tok], dim=1)
            # stop at EOS if detected in generated text
            if eos_tokens is not None:
                generated = idx[0, prompt_len:].tolist()
                gen_str = ''.join([chr(t) if t < 128 else '' for t in generated])
                if '<EOS>' in gen_str:
                    break
        # return only the generated portion (not the prompt)
        return idx[:, prompt_len:]


class GPT(nn.Module):

    def __init__(self, vocab_size, n_embd=32, context_size=8, n_head=4, n_layer=4):
        super().__init__()
        self.context_size = context_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_size, n_embd)
        self.blocks = nn.ModuleList([Block(n_embd, n_head=n_head, context_size=context_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.ln_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None, use_cache=False, kv_cache=None):
        B, T = idx.shape

        past_len = 0
        if kv_cache is not None and kv_cache[0] is not None:
            past_len = kv_cache[0][0][0].shape[1]

        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(past_len, past_len + T, device=idx.device))
        x = tok_emb + pos_emb

        if use_cache:
            if kv_cache is None:
                kv_cache = [None] * len(self.blocks)
            new_caches = []
            for block, cache in zip(self.blocks, kv_cache):
                x, new_cache = block(x, cache)
                new_caches.append(new_cache)
        else:
            new_caches = None
            for block in self.blocks:
                x, _ = block(x)

        x = self.ln_f(x)
        logits = self.ln_head(x)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        if use_cache:
            return logits, loss, new_caches
        return logits, loss

    def generate(self, start_idx, number_of_tokens, use_cache=False,
                 temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0):
        idx = start_idx
        if use_cache:
            kv_cache = None
            for _ in range(number_of_tokens):
                if kv_cache is not None and kv_cache[0][0][0].shape[1] >= self.context_size:
                    kv_cache = None
                if kv_cache is not None:
                    logits, _, kv_cache = self(idx[:, -1:], use_cache=True, kv_cache=kv_cache)
                else:
                    idx_input = idx[:, -self.context_size:]
                    if idx_input.shape[1] < self.context_size:
                        logits, _, kv_cache = self(idx_input, use_cache=True)
                    else:
                        logits, _ = self(idx_input)
                logits = logits[:, -1, :]
                logits = _apply_sampling(logits, idx, temperature, top_k, top_p, repetition_penalty)
                probs = F.softmax(logits, dim=-1)
                idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        else:
            for _ in range(number_of_tokens):
                idx_cond = idx[:, -self.context_size:]
                logits, _ = self(idx_cond)
                logits = logits[:, -1, :]
                logits = _apply_sampling(logits, idx, temperature, top_k, top_p, repetition_penalty)
                probs = F.softmax(logits, dim=-1)
                idx = torch.cat((idx, torch.multinomial(probs, num_samples=1)), dim=1)
        return idx


def _apply_sampling(logits, generated_ids, temperature, top_k, top_p, repetition_penalty):
    if logits.dim() == 3:
        logits = logits[:, -1, :]
    if repetition_penalty != 1.0:
        for token_id in set(generated_ids[0].tolist()):
            if logits[0, token_id] > 0:
                logits[0, token_id] /= repetition_penalty
            else:
                logits[0, token_id] *= repetition_penalty

    # temperature: sharpen or flatten the distribution
    if temperature != 1.0:
        logits = logits / temperature

    # top-k: keep only the k highest-probability tokens
    if top_k > 0:
        values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < values[:, -1:]] = float('-inf')

    # top-p (nucleus): keep smallest set of tokens whose cumulative prob >= p
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
        sorted_logits[mask] = float('-inf')
        logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)

    return logits


def load_dataset(path, use_bpe=False):
    with open(path, 'r') as f:
        text = f.read()
    if use_bpe:
        import sys, os
        sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'tokenizer'))
        from tokenizer import LuffyTokenizer
        tok = LuffyTokenizer()
        encode = tok.encode
        decode = tok.decode
        vocab_size = tok.vocab_size
        print(f'tokenizer: BPE (vocab={vocab_size})')
    else:
        characters = sorted(list(set(text)))
        vocab_size = len(characters)
        char_to_idx = {ch: i for i, ch in enumerate(characters)}
        idx_to_char = {i: ch for i, ch in enumerate(characters)}
        encode = lambda xs: [char_to_idx[x] for x in xs if x in char_to_idx]
        decode = lambda xs: ''.join([idx_to_char[x] for x in xs])
        print(f'tokenizer: char-level (vocab={vocab_size})')
    return text, vocab_size, encode, decode


def get_batch(train_data, val_data, split, batch_size, context_size, device):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - context_size, (batch_size,))
    x = torch.stack([data[i:i+context_size] for i in ix])
    y = torch.stack([data[i+1:i+context_size+1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, train_data, val_data, batch_size, context_size, device, eval_iters=100):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(train_data, val_data, split, batch_size, context_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.mean().item()
        out[split] = losses.mean()
    model.train()
    return out


def train_model(model, train_data, val_data, steps, batch_size, context_size, device, lr,
                metrics_obj=None, decode=None, metric_interval=1000,
                warmup_steps=0, weight_decay=0.0):
    import math
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # cosine decay with warmup
    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda) if warmup_steps > 0 else None
    base = model.module if isinstance(model, nn.DataParallel) else model

    # build a minimal dataset-like object that metrics.py expects
    class _DataWrapper:
        def __init__(self, train_data, val_data, context_size, batch_size):
            self.train_data = train_data
            self.val_data = val_data
            self.context_size = context_size
            self.batch_size = batch_size
        def get_batch(self, split, device):
            data = self.train_data if split == 'train' else self.val_data
            ix = torch.randint(len(data) - self.context_size - 1, (self.batch_size,))
            x = torch.stack([data[i:i+self.context_size] for i in ix])
            y = torch.stack([data[i+1:i+self.context_size+1] for i in ix])
            return x.to(device), y.to(device)

    data_wrapper = _DataWrapper(train_data, val_data, context_size, batch_size)

    for step in range(steps):
        xb, yb = get_batch(train_data, val_data, 'train', batch_size, context_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.mean().backward()
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if step % 500 == 0 or step == steps - 1:
            losses = estimate_loss(model, train_data, val_data, batch_size, context_size, device)
            print(f'step {step:>5}  train loss: {losses["train"]:.4f}  val loss: {losses["val"]:.4f}')
            if metrics_obj is not None and decode is not None and (step % metric_interval == 0 or step == steps - 1):

                class _SimpleTokenizer:
                    def __init__(self, decode_fn):
                        self._decode = decode_fn
                    def decode(self, xs):
                        return self._decode(xs)

                results = metrics_obj(data_wrapper, base, _SimpleTokenizer(decode))
                for k, v in results.items():
                    print(f'  {k:20s}: {v:.4f}')
            print()


def generate_text(model, encode, decode, context_size, device, prompt='', num_tokens=300,
                  temperature=1.0, top_k=0, top_p=0.0, repetition_penalty=1.0):
    model.eval()
    if prompt:
        idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(num_tokens):
            idx_cond = idx[:, -context_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1:]
            logits = _apply_sampling(logits, idx, temperature, top_k, top_p, repetition_penalty)
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
    return decode(idx[0].tolist())


def build_model(vocab_size, args, device):
    model = GPT(
        vocab_size,
        n_embd=args.n_embd,
        context_size=args.context_size,
        n_head=args.n_head,
        n_layer=args.n_layer,
    ).to(device)
    if torch.cuda.device_count() > 1:
        print(f'using {torch.cuda.device_count()} GPUs')
        model = nn.DataParallel(model)
    return model


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Luffy GPT — train and generate text')
    parser.add_argument('--input', required=True, help='path to input txt dataset')
    parser.add_argument('--train', metavar='SAVE_PATH', help='train model and save checkpoint to this path')
    parser.add_argument('--eval', metavar='CHECKPOINT', help='load checkpoint and run inference')
    parser.add_argument('--epoch', type=int, default=5000, help='number of training steps (default: 5000)')
    parser.add_argument('--batch-size', type=int, default=64, dest='batch_size')
    parser.add_argument('--context-size', type=int, default=256, dest='context_size')
    parser.add_argument('--n-embd', type=int, default=384, dest='n_embd')
    parser.add_argument('--n-head', type=int, default=6, dest='n_head')
    parser.add_argument('--n-layer', type=int, default=6, dest='n_layer')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--prompt', type=str, default='', help='starting prompt for generation')
    parser.add_argument('--tokens', type=int, default=300, help='number of tokens to generate (default: 300)')
    parser.add_argument('--interactive', action='store_true', help='interactive prompt mode (use with --eval)')
    parser.add_argument('--custom-bpe', action='store_true', dest='custom_bpe', help='use custom BPE tokenizer instead of char-level')
    parser.add_argument('--metrics', action='store_true', help='report full metrics during training')
    parser.add_argument('--metric-interval', type=int, default=1000, dest='metric_interval')
    parser.add_argument('--temperature', type=float, default=1.0, help='sampling temperature (lower=sharper)')
    parser.add_argument('--top-k', type=int, default=0, dest='top_k', help='top-k sampling (0=disabled)')
    parser.add_argument('--top-p', type=float, default=0.0, dest='top_p', help='nucleus sampling threshold (0=disabled)')
    parser.add_argument('--repetition-penalty', type=float, default=1.0, dest='repetition_penalty', help='penalize repeated tokens (1.0=off)')
    parser.add_argument('--warmup-steps', type=int, default=0, dest='warmup_steps', help='LR warmup steps (0=disabled)')
    parser.add_argument('--weight-decay', type=float, default=0.0, dest='weight_decay')
    parser.add_argument('--finetune', metavar='SAVE_PATH', help='fine-tune with style tokens and save checkpoint')
    parser.add_argument('--sft-input', dest='sft_input', help='path to SFT text file (for --finetune)')
    parser.add_argument('--pretrained', help='pretrained checkpoint to load (for --finetune)')
    parser.add_argument('--sft-mix', type=float, default=0.8, dest='sft_mix', help='SFT sampling ratio (default: 0.8)')
    parser.add_argument('--n-styles', type=int, default=2, dest='n_styles')
    parser.add_argument('--style', type=int, default=1, help='style index for generation (0=corpus, 1=sft)')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    text, vocab_size, encode, decode = load_dataset(args.input, use_bpe=args.custom_bpe)
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(len(data) * 0.9)
    train_data = data[:n]
    val_data = data[n:]
    print(f'dataset: {len(text):,} chars  vocab size: {vocab_size}')

    if args.finetune:
        import math, numpy as np

        if not args.sft_input:
            print('error: --sft-input required for --finetune')
            exit(1)

        with open(args.sft_input) as f:
            sft_text = f.read()
        sft_data = torch.tensor(encode(sft_text), dtype=torch.long)
        sft_n = int(len(sft_data) * 0.9)
        sft_train, sft_val = sft_data[:sft_n], sft_data[sft_n:]
        print(f'SFT data: {len(sft_text):,} chars  tokens: {len(sft_data):,}')

        model = GPTWithStyle(vocab_size, n_embd=args.n_embd, context_size=args.context_size,
                             n_head=args.n_head, n_layer=args.n_layer, n_styles=args.n_styles).to(device)
        print(f'params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

        if args.pretrained:
            ckpt = torch.load(args.pretrained, map_location=device)
            if 'ln_head.weight' in ckpt:
                ckpt['lm_head.weight'] = ckpt.pop('ln_head.weight')
                ckpt['lm_head.bias'] = ckpt.pop('ln_head.bias')
            missing, unexpected = model.load_state_dict(ckpt, strict=False)
            print(f'loaded pretrained: {args.pretrained} (new: {missing})')

        # multi-style dataset
        class _DS:
            def __init__(self, tr, va, cs, bs):
                self.train_data, self.val_data = tr, va
                self.context_size, self.batch_size = cs, bs
            def get_batch(self, split, dev):
                d = self.train_data if split == 'train' else self.val_data
                ix = torch.randint(len(d) - self.context_size - 1, (self.batch_size,))
                x = torch.stack([d[i:i+self.context_size] for i in ix])
                y = torch.stack([d[i+1:i+self.context_size+1] for i in ix])
                return x.to(dev), y.to(dev)

        ds0 = _DS(train_data, val_data, args.context_size, args.batch_size)
        ds1 = _DS(sft_train, sft_val, args.context_size, args.batch_size)
        datasets = [ds0, ds1]
        probs = [1.0 - args.sft_mix, args.sft_mix]
        print(f'mix: {probs[0]:.0%} corpus / {probs[1]:.0%} SFT')
        print(f'fine-tuning for {args.epoch} steps, lr={args.lr}\n')

        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        warmup = args.warmup_steps
        def lr_lambda(step):
            if warmup > 0 and step < warmup:
                return step / warmup
            progress = (step - warmup) / max(1, args.epoch - warmup)
            return 0.5 * (1 + math.cos(math.pi * progress))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        best_val, best_state = float('inf'), None
        model.train()
        for step in range(args.epoch):
            di = np.random.choice(len(datasets), p=probs)
            x, y = datasets[di].get_batch('train', device)
            style = torch.full((x.size(0),), di, dtype=torch.long, device=device)
            _, loss = model(x, style, targets=y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step % 500 == 0 or step == args.epoch - 1:
                model.eval()
                results = {}
                for si, ds in enumerate(datasets):
                    for sp in ['train', 'val']:
                        ls = []
                        for _ in range(50):
                            xb, yb = ds.get_batch(sp, device)
                            st = torch.full((xb.size(0),), si, dtype=torch.long, device=device)
                            _, l = model(xb, st, targets=yb)
                            ls.append(l.item())
                        results[f's{si}_{sp}'] = sum(ls)/len(ls)
                model.train()
                lr_now = scheduler.get_last_lr()[0]
                parts = [f'{k}: {v:.4f}' for k, v in results.items()]
                print(f'step {step:>5}  lr={lr_now:.2e}  |  ' + '  '.join(parts))
                if results['s1_val'] < best_val:
                    best_val = results['s1_val']
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)
            print(f'\nrestored best checkpoint (s1_val: {best_val:.4f})')

        torch.save(model.state_dict(), args.finetune)
        print(f'saved -> {args.finetune}')

    elif args.train:
        model = build_model(vocab_size, args, device)
        print(f'params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
        print(f'training for {args.epoch} steps...\n')
        metrics_obj = None
        if args.metrics:
            from metrics import Metrics
            metrics_obj = Metrics(number_of_steps=3)
            print('metrics reporting enabled (ROUGE, BERTScore, masked accuracy)\n')
        if args.warmup_steps > 0:
            print(f'LR schedule: cosine decay with {args.warmup_steps} warmup steps')
        if args.weight_decay > 0:
            print(f'weight decay: {args.weight_decay}')
        train_model(model, train_data, val_data, args.epoch, args.batch_size, args.context_size, device, args.lr,
                    metrics_obj=metrics_obj, decode=decode, metric_interval=args.metric_interval,
                    warmup_steps=args.warmup_steps, weight_decay=args.weight_decay)
        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state, args.train)
        print(f'\nmodel saved → {args.train}')
        print('\n--- sample output ---')
        print(generate_text(model, encode, decode, args.context_size, device, args.prompt, args.tokens,
                            args.temperature, args.top_k, args.top_p, args.repetition_penalty))

    elif args.eval:
        # detect if checkpoint is a style model or base model
        ckpt = torch.load(args.eval, map_location=device)
        is_style = 'style_embedding_table.weight' in ckpt

        if is_style:
            model = GPTWithStyle(vocab_size, n_embd=args.n_embd, context_size=args.context_size,
                                 n_head=args.n_head, n_layer=args.n_layer, n_styles=args.n_styles).to(device)
            model.load_state_dict(ckpt)
            print(f'loaded finetuned model: {args.eval} (style={args.style})')
            style_t = torch.tensor([args.style], dtype=torch.long, device=device)
            eos_tokens = encode('<EOS>')

            if args.interactive:
                print('interactive chat — enter a message (ctrl+c to quit)\n')
                while True:
                    try:
                        user_input = input('You: ')
                        prompt = f'USER: {user_input}\nLUFFY:'
                        idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
                        out = model.generate(idx, style_t, args.tokens,
                                             args.temperature, args.top_k, args.top_p, args.repetition_penalty,
                                             eos_tokens=eos_tokens)
                        result = decode(out[0].tolist())
                        # clean up: remove <EOS> and anything after USER:
                        result = result.split('<EOS>')[0].split('\nUSER:')[0].strip()
                        print(f'Luffy: {result}\n')
                    except KeyboardInterrupt:
                        print('\nexiting')
                        break
            else:
                prompt = args.prompt or 'USER: Who are you?\nLUFFY:'
                idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
                out = model.generate(idx, style_t, args.tokens,
                                     args.temperature, args.top_k, args.top_p, args.repetition_penalty,
                                     eos_tokens=eos_tokens)
                result = decode(out[0].tolist())
                result = result.split('<EOS>')[0].split('\nUSER:')[0].strip()
                print(result)
        else:
            model = build_model(vocab_size, args, device)
            base = model.module if isinstance(model, nn.DataParallel) else model
            base.load_state_dict(ckpt)
            print(f'loaded checkpoint: {args.eval}')

            inference_model = model.module if isinstance(model, nn.DataParallel) else model
            if args.interactive:
                print('interactive mode — enter a prompt (ctrl+c to quit)\n')
                while True:
                    try:
                        prompt = input('prompt> ')
                        print(generate_text(inference_model, encode, decode, args.context_size, device, prompt, args.tokens,
                                            args.temperature, args.top_k, args.top_p, args.repetition_penalty))
                        print()
                    except KeyboardInterrupt:
                        print('\nexiting')
                        break
            else:
                print(generate_text(inference_model, encode, decode, args.context_size, device, args.prompt, args.tokens,
                                    args.temperature, args.top_k, args.top_p, args.repetition_penalty))

    else:
        parser.print_help()
