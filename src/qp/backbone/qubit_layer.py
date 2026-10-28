import torch, torch.nn as nn, torch.nn.functional as F

def _rot(theta):
    c, s = torch.cos(theta), torch.sin(theta)
    return torch.stack([torch.stack([c, -s], -1), torch.stack([s, c], -1)], -2)

class QubitLayer(nn.Module):
    """Sector-banked qubit attention + FFN. Channels are grouped into P=d/2 'qubits' (pairs); every bank holds a
    few real params per qubit (7+3+3+3 = 16 -> ~3K per layer at d=384).
      bulk   (E++, 7): PSD-projected 2x2 (sym part, eigenvalues clamped >=0 in fp32) + antisym + SU(2)-like post rot/scale -> V
      rot    (E+-, 3): rotation angle + 2 scales -> Q (angular mixing; K is the raw qubit)
      create (E-+, 3): input-dependent feature injection
      kill   (E--, 3): input-dependent feature removal (soft gate on attention output)
    """
    def __init__(self, d, n_heads, ffn_mult=4):
        super().__init__()
        assert d % (2 * n_heads) == 0
        self.d, self.H, self.P = d, n_heads, d // 2
        P = self.P
        bulk = torch.zeros(P, 7); bulk[:, 0] = 1
        rot = torch.ones(P, 3); rot[:, 0] = torch.randn(P) * 0.3
        kill = torch.zeros(P, 3); kill[:, :2] = -4.0
        self.bulk, self.rot = nn.Parameter(bulk), nn.Parameter(rot)
        self.create, self.kill = nn.Parameter(torch.randn(P, 3) * 0.02), nn.Parameter(kill)
        self.mix = nn.Parameter(torch.full((4,), 2.0))
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d))

    def bulk_matrix(self):  # fp32 mandatory (spectral op)
        p0, p1, p2, p3, ph, s1, s2 = self.bulk.float().unbind(-1)
        r = torch.sqrt(p1 * p1 + p2 * p2 + 1e-8)
        lp, lm = F.relu(p0 + r), F.relu(p0 - r)
        a, b = (lp + lm) / 2, (lp - lm) / (2 * r)
        S = torch.stack([torch.stack([a + b * p2, b * p1 + p3], -1), torch.stack([b * p1 - p3, a - b * p2], -1)], -2)
        D = torch.diag_embed(torch.exp(torch.stack([s1, s2], -1).clamp(-3, 3)))
        return _rot(ph) @ D @ S

    def rot_matrix(self):
        th, s1, s2 = self.rot.float().unbind(-1)
        return _rot(th) @ torch.diag_embed(torch.stack([s1, s2], -1))

    def forward(self, x, active=None):
        B, S, d = x.shape; P, H = self.P, self.H
        a = torch.ones(4, device=x.device) if active is None else active.to(x.device)
        w = torch.sigmoid(self.mix.float()) * a                       # learned mixing x Poisson gating
        I = torch.eye(2, device=x.device)
        pr = self.ln1(x).view(B, S, P, 2)
        M = w[0] * self.bulk_matrix() + (1 - w[0]) * I
        R = w[1] * self.rot_matrix() + (1 - w[1]) * I
        v = torch.einsum("pij,bspj->bspi", M.to(pr.dtype), pr).reshape(B, S, d)
        q = torch.einsum("pij,bspj->bspi", R.to(pr.dtype), pr).reshape(B, S, d)
        k = pr.reshape(B, S, d)
        sh = lambda t: t.view(B, S, H, d // H).transpose(1, 2)
        att = F.scaled_dot_product_attention(sh(q), sh(k), sh(v), is_causal=True).transpose(1, 2).reshape(B, S, P, 2)
        n = pr.float().norm(dim=-1)                                   # (B,S,P)
        g = torch.sigmoid(self.kill[:, :2].float()) * torch.sigmoid(n - self.kill[:, 2].float()).unsqueeze(-1)
        att = att * (1 - w[3] * g).to(att.dtype)
        c = self.create[:, :2].float() * torch.tanh(self.create[:, 2].float() * n).unsqueeze(-1)
        x = x + (att + (w[2] * c).to(att.dtype)).reshape(B, S, d)
        return x + self.ffn(self.ln2(x))
