import sentencepiece as spm
import os

_MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'luffy_bpe.model')


class LuffyTokenizer:
    def __init__(self, model_path=_MODEL_PATH):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.vocab_size = self.sp.get_piece_size()

    def encode(self, text):
        return self.sp.encode(text)

    def decode(self, ids):
        return self.sp.decode(ids)

    def __repr__(self):
        return f'LuffyTokenizer(vocab_size={self.vocab_size})'
