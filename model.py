import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        """
        t: [B] integer or float
        return: [B, dim]
        """
        half_dim = self.dim // 2
        device = t.device
        emb_scale = math.log(10000) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_scale)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        return emb

class ResBlock1D(nn.Module):
    def __init__(self, channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(4, channels)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.time_proj = nn.Linear(time_dim, channels)

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(t_emb).unsqueeze(-1)
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h

class Simple1DUNet(nn.Module):
    def __init__(self, in_ch=2, base_ch=64, time_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

        self.in_conv = nn.Conv1d(in_ch, base_ch, kernel_size=3, padding=1)
        self.block1 = ResBlock1D(base_ch, time_dim)
        self.block2 = ResBlock1D(base_ch, time_dim)
        self.block3 = ResBlock1D(base_ch, time_dim)
        self.out_norm = nn.GroupNorm(4, base_ch)
        self.out_conv = nn.Conv1d(base_ch, in_ch, kernel_size=3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        h = self.in_conv(x)
        h = self.block1(h, t_emb)
        h = self.block2(h, t_emb)
        h = self.block3(h, t_emb)
        h = self.out_conv(F.silu(self.out_norm(h)))
        return h