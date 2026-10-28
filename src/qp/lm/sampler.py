import torch
class Sampler:
    def sample(self, logits, temperature=1.0, top_k=None):
        l = logits.float() / max(temperature, 1e-6)
        if top_k: v = l.topk(top_k).values[..., -1:]; l = l.masked_fill(l < v, float("-inf"))
        return int(torch.multinomial(torch.softmax(l, -1), 1))
