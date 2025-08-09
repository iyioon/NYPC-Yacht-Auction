#!/usr/bin/env python3
"""
Calculate YachtNNet model size for different configurations
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


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
    def __init__(self, input_len=59, action_size=3226, hidden=256, nblocks=6, dropout=0.3):
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
            nn.Linear(hidden, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.inp(x)
        for block in self.blocks:
            x = block(x)
        pi = self.pi_head(x)
        v = self.v_head(x)
        return torch.log_softmax(pi, dim=1), v


def calculate_model_size(hidden, nblocks):
    model = YachtNNet(hidden=hidden, nblocks=nblocks)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())

    # Estimate size in bytes (float32 = 4 bytes per parameter)
    size_bytes = total_params * 4
    size_mb = size_bytes / (1024 * 1024)

    return total_params, size_mb


# Test different configurations
configs = [
    (256, 4),   # Small
    (256, 6),   # Small-medium
    (384, 6),   # Medium
    (512, 8),   # Medium-large (original)
    (768, 12),  # Large (current)
    (512, 6),   # Balanced
]

print("Model Size Analysis:")
print("=" * 50)
for hidden, nblocks in configs:
    params, size_mb = calculate_model_size(hidden, nblocks)
    print(
        f"Hidden: {hidden:3d}, Blocks: {nblocks:2d} -> {params:,} params, {size_mb:.2f} MiB")
