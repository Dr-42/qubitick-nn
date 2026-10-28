import torch.nn as nn
class LMHead(nn.Module):
    def __init__(self, d, vocab): super().__init__(); self.proj = nn.Linear(d, vocab)
    def forward(self, h): return self.proj(h)
