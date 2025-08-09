import torch
import torch.nn as nn
import torch.nn.functional as F

# Small residual MLP for vector inputs (59 -> hidden -> policy(3226), value(1))


class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim)
        self.fc2 = nn.Linear(dim, dim)
        self.ln2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h = self.ln1(F.silu(self.fc1(x)))
        h = self.dropout(h)
        h = self.ln2(F.silu(self.fc2(h)))
        return x + h


class YachtNNet(nn.Module):
    def __init__(self, input_len=59, action_size=3226, hidden=512, nblocks=6, dropout=0.2):
        super().__init__()
        self.input_len = input_len
        self.action_size = action_size

        self.inp = nn.Sequential(
            nn.Linear(input_len, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden, dropout) for _ in range(nblocks)])

        # Policy head
        self.pi_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, action_size),
        )
        # Value head
        self.v_head = nn.Sequential(
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, 128),
            nn.SiLU(),
            nn.Linear(128, 1),
        )

        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x):  # x: (B, 59) or (B, 1, 59)
        if x.ndim == 3:
            x = x.squeeze(1)
        h = self.inp(x)
        for blk in self.blocks:
            h = blk(h)
        pi = self.pi_head(h)              # logits
        v = torch.tanh(self.v_head(h))    # [-1, 1]
        return pi, v
