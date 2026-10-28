class CharTokenizer:
    def __init__(self, chars):
        self.chars = list(chars); self.stoi = {c: i for i, c in enumerate(self.chars)}
    @classmethod
    def from_text(cls, text): return cls(sorted(set(text)))
    def encode(self, s): return [self.stoi[c] for c in s]
    def decode(self, ids): return "".join(self.chars[i] for i in ids)
    @property
    def vocab_size(self): return len(self.chars)
