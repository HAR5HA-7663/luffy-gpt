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

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size, n_embd, context_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size, n_embd, context_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


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

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):

    def __init__(self, vocab_size, n_embd=32, context_size=8, n_head=4, n_layer=4):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(context_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head, context_size=context_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.ln_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.ln_head(x)

        if targets is not None:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)
        else:
            loss = None

        return logits, loss


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


def train_model(model, train_data, val_data, steps, batch_size, context_size, device, lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(steps):
        xb, yb = get_batch(train_data, val_data, 'train', batch_size, context_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.mean().backward()
        optimizer.step()
        if step % 500 == 0 or step == steps - 1:
            losses = estimate_loss(model, train_data, val_data, batch_size, context_size, device)
            print(f'step {step:>5}  train loss: {losses["train"]:.4f}  val loss: {losses["val"]:.4f}')


def generate_text(model, encode, decode, context_size, device, prompt='', num_tokens=300):
    model.eval()
    if prompt:
        idx = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    with torch.no_grad():
        for _ in range(num_tokens):
            idx_cond = idx[:, -context_size:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :]
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
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'device: {device}')

    text, vocab_size, encode, decode = load_dataset(args.input, use_bpe=args.custom_bpe)
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(len(data) * 0.9)
    train_data = data[:n]
    val_data = data[n:]
    print(f'dataset: {len(text):,} chars  vocab size: {vocab_size}')

    if args.train:
        model = build_model(vocab_size, args, device)
        print(f'params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')
        print(f'training for {args.epoch} steps...\n')
        train_model(model, train_data, val_data, args.epoch, args.batch_size, args.context_size, device, args.lr)
        state = model.module.state_dict() if isinstance(model, nn.DataParallel) else model.state_dict()
        torch.save(state, args.train)
        print(f'\nmodel saved → {args.train}')
        print('\n--- sample output ---')
        print(generate_text(model, encode, decode, args.context_size, device, args.prompt, args.tokens))

    elif args.eval:
        model = build_model(vocab_size, args, device)
        model.load_state_dict(torch.load(args.eval, map_location=device))
        print(f'loaded checkpoint: {args.eval}')

        if args.interactive:
            print('interactive mode — enter a prompt (ctrl+c to quit)\n')
            while True:
                try:
                    prompt = input('prompt> ')
                    print(generate_text(model, encode, decode, args.context_size, device, prompt, args.tokens))
                    print()
                except KeyboardInterrupt:
                    print('\nexiting')
                    break
        else:
            print(generate_text(model, encode, decode, args.context_size, device, args.prompt, args.tokens))

    else:
        parser.print_help()
