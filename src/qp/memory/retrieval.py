import torch, torch.nn.functional as F

class Retriever:
    def __init__(self, key_dim=384, k=8): self.key_dim, self.k = key_dim, k

    def score(self, q, keys):
        """Cosine similarity, fp32 mandatory (inputs may be fp16)."""
        with torch.autocast(device_type=q.device.type, enabled=False):
            return F.normalize(q.float(), dim=-1) @ F.normalize(keys.float(), dim=-1).t()

    def top_k(self, scores, k):
        v, i = scores.topk(min(k, scores.shape[-1]), dim=-1)
        return v, i
