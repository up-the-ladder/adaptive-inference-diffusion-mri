"""
U-Net backbone for the diffusion model.
Input:  (B, 2, H, W) — noisy image + undersampled image concatenated
Output: (B, 1, H, W) — predicted noise at current step
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


# ── Time embedding ─────────────────────────────────────────────────────────────

class SinusoidalTimeEmbedding(nn.Module):
    """
    Sinusoidal positional embedding for diffusion timestep t.
    Maps scalar t → vector of dim `embed_dim`.
    """
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Linear(embed_dim * 4, embed_dim * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) integer timesteps
        half = self.embed_dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / (half - 1)
        )
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        return self.mlp(embedding)  # (B, embed_dim*4)


# ── Building blocks ────────────────────────────────────────────────────────────

class ResBlock(nn.Module):
    """Residual block with time embedding injection."""
    def __init__(self, in_channels: int, out_channels: int, time_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_dim, out_channels)
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.act = nn.SiLU()
        # Skip connection
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(x))
        h = self.conv1(h)
        # Inject time embedding
        h = h + self.time_proj(self.act(t_emb))[:, :, None, None]
        h = self.act(self.norm2(h))
        h = self.conv2(h)
        return h + self.skip(x)


class AttentionBlock(nn.Module):
    """Self-attention block for bottleneck."""
    def __init__(self, channels: int, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(groups, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, C, H * W)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]
        attn = torch.softmax(torch.bmm(q.transpose(1, 2), k) * self.scale, dim=-1)
        h = torch.bmm(v, attn.transpose(1, 2)).reshape(B, C, H, W)
        return x + self.proj(h)


class DownBlock(nn.Module):
    """Encoder block: 2 ResBlocks + optional attention + downsample."""
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, use_attention: bool = False):
        super().__init__()
        self.res1 = ResBlock(in_ch, out_ch, time_dim)
        self.res2 = ResBlock(out_ch, out_ch, time_dim)
        self.attn = AttentionBlock(out_ch) if use_attention else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        x = self.attn(x)
        return self.down(x), x  # downsampled, skip


class UpBlock(nn.Module):
    """Decoder block: upsample + skip concat + 2 ResBlocks + optional attention."""
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, use_attention: bool = False):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)
        self.res1 = ResBlock(out_ch * 2, out_ch, time_dim)  # *2 for skip
        self.res2 = ResBlock(out_ch, out_ch, time_dim)
        self.attn = AttentionBlock(out_ch) if use_attention else nn.Identity()

    def forward(self, x: torch.Tensor, skip: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.res1(x, t_emb)
        x = self.res2(x, t_emb)
        return self.attn(x)


# ── Full U-Net ─────────────────────────────────────────────────────────────────

class UNet(nn.Module):
    """
    Conditional U-Net for diffusion-based MRI reconstruction.

    Takes as input:
        x_t:          noisy image at step t,        shape (B, 1, H, W)
        condition:    undersampled input image,      shape (B, 1, H, W)
        t:            diffusion timestep,            shape (B,)

    Returns:
        predicted noise, shape (B, 1, H, W)
    """

    def __init__(
        self,
        in_channels: int = 2,        # noisy image + condition
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mults: List[int] = [1, 2, 4, 8],
        time_embed_dim: int = 128,
        attention_levels: List[int] = [2, 3],  # which levels get attention
    ):
        super().__init__()
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)
        time_dim = time_embed_dim * 4

        channels = [base_channels * m for m in channel_mults]

        # Input projection
        self.input_conv = nn.Conv2d(in_channels, channels[0], 3, padding=1)

        # Encoder
        self.down_blocks = nn.ModuleList()
        in_ch = channels[0]
        for i, out_ch in enumerate(channels):
            use_attn = i in attention_levels
            self.down_blocks.append(DownBlock(in_ch, out_ch, time_dim, use_attn))
            in_ch = out_ch

        # Bottleneck
        self.mid_res1 = ResBlock(channels[-1], channels[-1], time_dim)
        self.mid_attn = AttentionBlock(channels[-1])
        self.mid_res2 = ResBlock(channels[-1], channels[-1], time_dim)

        # Decoder
        self.up_blocks = nn.ModuleList()
        rev_channels = list(reversed(channels))
        dec_in = channels[-1]
        for i, out_ch in enumerate(rev_channels):
            use_attn = (len(channels) - 1 - i) in attention_levels
            self.up_blocks.append(UpBlock(dec_in, out_ch, time_dim, use_attn))
            dec_in = out_ch

        # Output
        self.output_norm = nn.GroupNorm(8, channels[0])
        self.output_conv = nn.Conv2d(channels[0], out_channels, 3, padding=1)

    def forward(self, x_t: torch.Tensor, condition: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Concatenate noisy image with condition
        x = torch.cat([x_t, condition], dim=1)  # (B, 2, H, W)

        # Time embedding
        t_emb = self.time_embed(t)  # (B, time_dim)

        # Input projection
        x = self.input_conv(x)

        # Encoder — save skip connections
        skips = []
        for block in self.down_blocks:
            x, skip = block(x, t_emb)
            skips.append(skip)

        # Bottleneck
        x = self.mid_res1(x, t_emb)
        x = self.mid_attn(x)
        x = self.mid_res2(x, t_emb)

        # Decoder — use skip connections
        for block, skip in zip(self.up_blocks, reversed(skips)):
            x = block(x, skip, t_emb)

        # Output
        x = self.output_conv(F.silu(self.output_norm(x)))
        return x


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
