import sentencepiece as spm
import os

corpus = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../dataset/processed/corpus_clean.txt')
output_dir = os.path.dirname(os.path.abspath(__file__))

spm.SentencePieceTrainer.train(
    input=corpus,
    model_prefix=os.path.join(output_dir, 'luffy_bpe'),
    vocab_size=2000,
    model_type='bpe',
    character_coverage=1.0,       # cover all chars in dataset
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
)

print('tokenizer trained and saved to tokenizer/luffy_bpe.model')

# quick sanity check
sp = spm.SentencePieceProcessor()
sp.load(os.path.join(output_dir, 'luffy_bpe.model'))

test = "Luffy: I'm going to be King of the Pirates!"
print(f'\noriginal : {test}')
print(f'tokens   : {sp.encode(test, out_type=str)}')
print(f'ids      : {sp.encode(test)}')
print(f'decoded  : {sp.decode(sp.encode(test))}')
print(f'\nvocab size: {sp.get_piece_size()}')

# compare compression vs char-level
with open(corpus, 'r') as f:
    text = f.read()

char_tokens = len(text)
bpe_tokens = len(sp.encode(text))
print(f'\nchar-level tokens : {char_tokens:,}')
print(f'bpe tokens        : {bpe_tokens:,}')
print(f'compression ratio : {char_tokens / bpe_tokens:.2f}x fewer tokens')
