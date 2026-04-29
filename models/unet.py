import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Time Embedding ─────────────────────────────────────────────────────────────

def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000)
        * torch.arange(half, device=t.device, dtype=torch.float32)
        / (half - 1)
    )
    emb = t.float()[:, None] * freqs[None, :]
    return torch.cat([emb.sin(), emb.cos()], dim=-1)  # (B, dim)


class TimeEmbedding(nn.Module):
    def __init__(self, base_ch: int):
        super().__init__()
        self.dim = base_ch
        self.net = nn.Sequential(
            nn.Linear(base_ch, base_ch * 4),
            nn.SiLU(),
            nn.Linear(base_ch * 4, base_ch * 4),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.net(sinusoidal_embedding(t, self.dim))  # (B, 4*base_ch)


# ── Building Blocks ────────────────────────────────────────────────────────────

class ResBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_emb_dim: int, groups: int = 8):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(groups, in_ch), in_ch)
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(groups, out_ch), out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, padding=1)
        self.res_conv = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time_proj(self.act(t_emb))[:, :, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.res_conv(x)


class SelfAttention1D(nn.Module):
    def __init__(self, channels: int, num_heads: int = 8, groups: int = 8):
        super().__init__()
        self.norm = nn.GroupNorm(min(groups, channels), channels)
        self.attn = nn.MultiheadAttention(channels, num_heads, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, L = x.shape
        h = self.norm(x).permute(0, 2, 1)           # (B, L, C)
        h, _ = self.attn(h, h, h, need_weights=False)
        return x + h.permute(0, 2, 1)


class Downsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, 4, stride=2, padding=1)

    def forward(self, x):
        return self.conv(x)


# ── Calendar Embedding ────────────────────────────────────────────────────────

class CalendarEmbedding(nn.Module):
    """
    Encodes day-level calendar features into the same space as the time embedding
    so they can be added directly to t_emb inside every ResBlock.

    Input cal: (B, 4) long — [day_of_week (0-6), month (0-11), is_weekend, is_holiday]
    Output:    (B, base_ch * 4)
    """

    def __init__(self, base_ch: int):
        super().__init__()
        self.dow_emb   = nn.Embedding(7, 16)
        self.month_emb = nn.Embedding(12, 16)
        # 16 + 16 + 2 binary scalars = 34
        self.proj = nn.Sequential(
            nn.Linear(34, base_ch * 4),
            nn.SiLU(),
            nn.Linear(base_ch * 4, base_ch * 4),
        )

    def forward(self, cal: torch.Tensor) -> torch.Tensor:
        dow    = self.dow_emb(cal[:, 0])        # (B, 16)
        month  = self.month_emb(cal[:, 1])      # (B, 16)
        binary = cal[:, 2:].float()             # (B, 2)
        return self.proj(torch.cat([dow, month, binary], dim=-1))  # (B, 4C)


# ── U-Net ──────────────────────────────────────────────────────────────────────

class UNet1D(nn.Module):
    """
    1D U-Net for conditional noise prediction.

    Input:  (B, 2, 288) — [noisy_rt, da_upsampled]
    Output: (B, 1, 288) — predicted noise ε

    Encoder downsamples 288 → 144 → 72 → 36.
    Decoder upsamples back 36 → 72 → 144 → 288.
    Each decoder stage: Upsample → cat(skip) → ResBlocks.
    """

    def __init__(
        self,
        base_channels: int = 64,
        num_res_blocks: int = 2,
        attn_heads: int = 8,
        use_calendar: bool = False,
    ):
        super().__init__()
        C = base_channels
        time_emb_dim = C * 4
        self.use_calendar = use_calendar
        if use_calendar:
            self.cal_emb = CalendarEmbedding(C)
        # Channel widths: input projection + 3 encoder stages
        enc_chs = [C, C * 2, C * 4, C * 4]   # [64, 128, 256, 256]

        self.time_emb = TimeEmbedding(C)

        # Input projection: 2 input channels (noisy RT + DA cond) → C
        self.conv_in = nn.Conv1d(2, C, 3, padding=1)

        # ── Encoder ────────────────────────────────────────────────────────
        # enc_chs[i] → enc_chs[i+1], then downsample
        # skip saved BEFORE downsample (at enc_chs[i+1] channels)
        self.enc_blocks = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        for i in range(len(enc_chs) - 1):
            in_ch, out_ch = enc_chs[i], enc_chs[i + 1]
            blocks = nn.ModuleList([
                ResBlock1D(in_ch if j == 0 else out_ch, out_ch, time_emb_dim)
                for j in range(num_res_blocks)
            ])
            self.enc_blocks.append(blocks)
            self.downsamplers.append(Downsample(out_ch))

        # skip_chs[i] = channel count of skip at encoder stage i (= enc_chs[i+1])
        self.skip_chs = enc_chs[1:]   # [128, 256, 256]

        # ── Bottleneck ─────────────────────────────────────────────────────
        btn_ch = enc_chs[-1]   # 256
        self.mid_res1 = ResBlock1D(btn_ch, btn_ch, time_emb_dim)
        self.mid_attn = SelfAttention1D(btn_ch, attn_heads)
        self.mid_res2 = ResBlock1D(btn_ch, btn_ch, time_emb_dim)

        # ── Decoder ────────────────────────────────────────────────────────
        # Stage i: Upsample(h_ch) → cat(skip) → ResBlocks(h_ch+skip_ch → out_ch)
        #
        # h_ch before each stage:
        #   stage 0: btn_ch=256   skip_ch=256  out_ch=256  in_ch=512
        #   stage 1: 256          skip_ch=256  out_ch=128  in_ch=512
        #   stage 2: 128          skip_ch=128  out_ch=64   in_ch=256
        #
        # Then a final stage using the conv_in skip (C=64 channels):
        #   final:  64  skip_ch=64   out_ch=64   in_ch=128

        dec_specs = [
            # (h_ch_before_upsample, skip_ch, out_ch)
            (enc_chs[-1],     self.skip_chs[-1], enc_chs[-2]),   # 256, 256, 256
            (enc_chs[-2],     self.skip_chs[-2], enc_chs[-3]),   # 256, 256, 128
            (enc_chs[-3],     self.skip_chs[-3], enc_chs[-4]),   # 128, 128,  64
        ]

        self.upsamplers = nn.ModuleList()
        self.dec_blocks = nn.ModuleList()
        for h_ch, skip_ch, out_ch in dec_specs:
            self.upsamplers.append(Upsample(h_ch))
            blocks = nn.ModuleList([
                ResBlock1D(h_ch + skip_ch if j == 0 else out_ch, out_ch, time_emb_dim)
                for j in range(num_res_blocks)
            ])
            self.dec_blocks.append(blocks)

        # Final stage with conv_in skip (C channels)
        self.final_up = Upsample(enc_chs[-4])   # Upsample(C=64) — actually this stage doesn't upsample;
        # After last decoder stage h = (B, C, 288) — same length as conv_in skip. No more upsample needed.
        # We just cat with skip0 and process.
        self.final_blocks = nn.ModuleList([
            ResBlock1D(C + C if j == 0 else C, C, time_emb_dim)
            for j in range(num_res_blocks)
        ])

        # Output projection
        self.norm_out = nn.GroupNorm(8, C)
        self.conv_out = nn.Conv1d(C, 1, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cal: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        x:   (B, 2, 288)  — [noisy_rt, da_conditioning]
        t:   (B,) long    — diffusion timestep indices
        cal: (B, 4) long  — [day_of_week, month, is_weekend, is_holiday] (optional)
        Returns: (B, 1, 288) — predicted noise
        """
        t_emb = self.time_emb(t)    # (B, 4C)
        if self.use_calendar and cal is not None:
            t_emb = t_emb + self.cal_emb(cal)

        h = self.conv_in(x)          # (B, C, 288)
        skip0 = h                    # keep for final skip connection

        # Encoder
        skips = []
        for blocks, down in zip(self.enc_blocks, self.downsamplers):
            for block in blocks:
                h = block(h, t_emb)
            skips.append(h)          # save BEFORE downsample
            h = down(h)

        # Bottleneck
        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)

        # Decoder: upsample → cat skip → ResBlocks
        for up, blocks, skip in zip(self.upsamplers, self.dec_blocks, reversed(skips)):
            h = up(h)
            h = torch.cat([h, skip], dim=1)
            for block in blocks:
                h = block(h, t_emb)

        # Final: cat with conv_in skip (same spatial size, no upsample needed)
        h = torch.cat([h, skip0], dim=1)
        for block in self.final_blocks:
            h = block(h, t_emb)

        return self.conv_out(F.silu(self.norm_out(h)))   # (B, 1, 288)
