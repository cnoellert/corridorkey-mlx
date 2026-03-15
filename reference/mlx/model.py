"""
GreenFormer re-implemented in MLX.

Architecture mirrors nikopueringer/CorridorKey exactly so that converted
weights drop in without any key remapping beyond what convert.py handles.

MLX layout notes
----------------
• MLX Conv2d is NHWC natively. Weights are [O, kH, kW, I].
• Linear weights are [O, I] — identical to PyTorch.
• Resize uses upsample_linear/upsample_cubic (mx.image absent in MLX 0.31.x).
• DropPath is inference-only — replaced with identity.

Bug fixes vs initial draft
---------------------------
1. MaskUnitAttention windowed path: output transpose corrected for 5D tensor.
   Was: x.transpose(0, 3, 1, 2)  → silently wrong shape
   Fix: x.transpose(0, 3, 2, 1, 4)  (swap dims 1↔3, keep all 5 axes)

2. HieraEncoder: native img_size=2048 (matches checkpoint pos_embed [1,262144,112]).
   pos_embed is bicubic-interpolated in forward() when tile size differs.

3. FoldedBN replaces nn.BatchNorm: loaded from convert.py-folded weight/bias,
   no running stats needed at inference. Avoids the broken .scale→.weight remap.

4. self.blocks built as list then assigned (cleaner module introspection).
"""
from __future__ import annotations

import math
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _interpolate(x: mx.array, size: Tuple[int, int]) -> mx.array:
    """Bilinear resize. x: [N, H, W, C] → [N, h, w, C].
    Uses upsample_linear which handles both up- and downsampling.
    mx.image does not exist in MLX 0.31.x.
    """
    from mlx.nn.layers.upsample import upsample_linear
    sh, sw = x.shape[1], x.shape[2]
    th, tw = size
    return upsample_linear(x, (th / sh, tw / sw), align_corners=False)


def _interpolate_pos_embed(pos_embed: mx.array, num_tokens: int) -> mx.array:
    """
    Bicubic-resize pos_embed from its native grid to match num_tokens.
    pos_embed: [1, N_src, C]  →  [1, num_tokens, C]
    Used when tile size differs from the native 2048px training size.
    """
    if pos_embed.shape[1] == num_tokens:
        return pos_embed
    C = pos_embed.shape[2]
    src_side = int(math.isqrt(pos_embed.shape[1]))
    dst_side = int(math.isqrt(num_tokens))
    # Reshape to spatial: [1, src, src, C]
    pe_spatial = pos_embed.reshape(1, src_side, src_side, C)
    # Bicubic resize to [1, dst, dst, C]
    from mlx.nn.layers.upsample import upsample_cubic
    pe_resized = upsample_cubic(pe_spatial, (dst_side / src_side, dst_side / src_side), align_corners=False)
    return pe_resized.reshape(1, dst_side * dst_side, C)


# ---------------------------------------------------------------------------
# FoldedBN — replaces nn.BatchNorm at inference
# Weights are pre-folded by convert.py:
#   scale = gamma / sqrt(var + eps)
#   bias  = beta - mean * scale
# Stored as .weight (scale) and .bias in the .npz.
# ---------------------------------------------------------------------------

class FoldedBN(nn.Module):
    """Folded BatchNorm: just an affine transform y = x * weight + bias."""
    def __init__(self, num_features: int) -> None:
        super().__init__()
        self.weight = mx.ones((num_features,))
        self.bias   = mx.zeros((num_features,))

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.weight + self.bias


# ---------------------------------------------------------------------------
# Shared MLP (channel projection used inside DecoderHead)
# ---------------------------------------------------------------------------

class ChannelMLP(nn.Module):
    """Linear: C_in → C_out (operates on last dim)."""
    def __init__(self, input_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.proj(x)

# ---------------------------------------------------------------------------
# Decoder Head
# ---------------------------------------------------------------------------

class DecoderHead(nn.Module):
    """
    MLP-Mixer style decoder.
    Takes 4 NHWC feature maps, projects to common dim,
    upsamples to C1 resolution, fuses, then predicts.
    """
    def __init__(
        self,
        feature_channels: Optional[List[int]] = None,
        embedding_dim: int = 256,
        output_dim: int = 1,
    ) -> None:
        super().__init__()
        if feature_channels is None:
            feature_channels = [112, 224, 448, 896]

        self.linear_c1 = ChannelMLP(feature_channels[0], embedding_dim)
        self.linear_c2 = ChannelMLP(feature_channels[1], embedding_dim)
        self.linear_c3 = ChannelMLP(feature_channels[2], embedding_dim)
        self.linear_c4 = ChannelMLP(feature_channels[3], embedding_dim)

        # Fuse: 4*embedding_dim → embedding_dim  (1×1 conv, NHWC)
        self.linear_fuse = nn.Conv2d(
            embedding_dim * 4, embedding_dim, kernel_size=1, bias=False
        )
        self.bn = FoldedBN(embedding_dim)
        self.dropout = nn.Dropout(p=0.1)
        self.classifier = nn.Conv2d(embedding_dim, output_dim, kernel_size=1)

    def __call__(self, features: List[mx.array]) -> mx.array:
        """features: list of [N, H_i, W_i, C_i] NHWC tensors."""
        c1, c2, c3, c4 = features
        target_hw = (c1.shape[1], c1.shape[2])

        def _proj_upsample(feat: mx.array, mlp: ChannelMLP) -> mx.array:
            return _interpolate(mlp(feat), target_hw)

        _c1 = self.linear_c1(c1)
        _c2 = _proj_upsample(c2, self.linear_c2)
        _c3 = _proj_upsample(c3, self.linear_c3)
        _c4 = _proj_upsample(c4, self.linear_c4)

        _c = mx.concatenate([_c4, _c3, _c2, _c1], axis=-1)  # [N, H, W, 4*E]
        _c = self.linear_fuse(_c)
        _c = self.bn(_c)
        _c = nn.relu(_c)
        x = self.dropout(_c)
        x = self.classifier(x)
        return x


# ---------------------------------------------------------------------------
# CNN Refiner
# ---------------------------------------------------------------------------

class RefinerBlock(nn.Module):
    """Residual block with dilation + GroupNorm."""
    def __init__(self, channels: int, dilation: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.gn1 = nn.GroupNorm(8, channels, pytorch_compatible=True)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.gn2 = nn.GroupNorm(8, channels, pytorch_compatible=True)

    def __call__(self, x: mx.array) -> mx.array:
        residual = x
        out = nn.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return nn.relu(out + residual)


class CNNRefinerModule(nn.Module):
    """
    Dilated residual refiner. Receptive field ~65 px.
    Input: cat(rgb[N,H,W,3], coarse_pred[N,H,W,4]) → [N,H,W,7]
    Output: delta_logits [N,H,W,4]
    """
    def __init__(self, in_channels: int = 7, hidden_channels: int = 64, out_channels: int = 4) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_channels, pytorch_compatible=True),
            nn.ReLU(),
        )
        self.res1 = RefinerBlock(hidden_channels, dilation=1)
        self.res2 = RefinerBlock(hidden_channels, dilation=2)
        self.res3 = RefinerBlock(hidden_channels, dilation=4)
        self.res4 = RefinerBlock(hidden_channels, dilation=8)
        self.final = nn.Conv2d(hidden_channels, out_channels, kernel_size=1)

    def __call__(self, img: mx.array, coarse_pred: mx.array) -> mx.array:
        x = mx.concatenate([img, coarse_pred], axis=-1)
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        x = self.res4(x)
        return self.final(x) * 10.0

# ---------------------------------------------------------------------------
# Hiera backbone helpers
# ---------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    """Patch embedding: [N,H,W,C_in] → [N, H//s * W//s, embed_dim]."""
    def __init__(self, in_channels: int, embed_dim: int, kernel: int = 7, stride: int = 4, padding: int = 3) -> None:
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=kernel, stride=stride, padding=padding)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.proj(x)
        N, H, W, C = x.shape
        return x.reshape(N, H * W, C)


class Unroll(nn.Module):
    """Reorders tokens so patches within each mask-unit window are contiguous."""
    def __init__(self, input_size: Tuple[int,int], patch_stride: Tuple[int,int], unroll_schedule: List[Tuple[int,int]]) -> None:
        super().__init__()
        self.size = [i // s for i, s in zip(input_size, patch_stride)]
        self.schedule = unroll_schedule

    def __call__(self, x: mx.array) -> mx.array:
        B, N, C = x.shape
        # Derive actual spatial grid from token count (supports any tile size)
        side = int(math.isqrt(N))
        cur_size = [side, side]
        x = x.reshape(B, *cur_size, C)
        for strides in self.schedule:
            cur_size = [i // s for i, s in zip(cur_size, strides)]
            new_shape = [B] + sum([[i, s] for i, s in zip(cur_size, strides)], []) + [C]
            x = x.reshape(new_shape)
            L = len(new_shape)
            permute = [0] + list(range(2, L - 1, 2)) + list(range(1, L - 1, 2)) + [L - 1]
            x = x.transpose(permute)
            x = x.flatten(0, len(strides))
            B *= math.prod(strides)
        return x.reshape(-1, N, C)


class Reroll(nn.Module):
    """Undoes Unroll to restore spatial order for feature extraction."""
    def __init__(self, input_size: Tuple[int,int], patch_stride: Tuple[int,int],
                 unroll_schedule: List[Tuple[int,int]], stage_ends: List[int], q_pool: int) -> None:
        super().__init__()
        self.size = [i // s for i, s in zip(input_size, patch_stride)]
        self.schedule: dict = {}
        size = list(self.size)
        schedule = list(unroll_schedule)
        for i in range(stage_ends[-1] + 1):
            self.schedule[i] = (list(schedule), list(size))
            if i in stage_ends[:q_pool]:
                if schedule:
                    size = [n // s for n, s in zip(size, schedule[0])]
                    schedule = schedule[1:]

    def __call__(self, x: mx.array, block_idx: int) -> mx.array:
        schedule, native_size = self.schedule[block_idx]
        B, N, C = x.shape
        D = len(native_size)
        # Derive actual spatial size from token count (tile-size agnostic)
        # native_size is for 2048px; scale down proportionally
        native_tokens = math.prod(native_size)
        scale = math.isqrt(N // native_tokens) if N < native_tokens else math.isqrt(N // native_tokens) if N < native_tokens else 1
        # Simpler: just derive side from N directly
        # At each stage after q-pool the token count is native_tokens // (q_stride^pools_so_far)
        # We can back-compute size from N assuming square grid
        side = int(math.isqrt(N))
        size = [side, side]

        cur_mu_shape = [1] * D
        for strides in schedule:
            x = x.reshape(B, *strides, N // math.prod(strides), *cur_mu_shape, C)
            L = len(x.shape)
            permute = (
                [0, 1 + D]
                + sum([list(p) for p in zip(range(1, 1 + D), range(1 + D + 1, L - 1))], [])
                + [L - 1]
            )
            x = x.transpose(permute)
            for i in range(D):
                cur_mu_shape[i] *= strides[i]
            x = x.reshape(B, -1, *cur_mu_shape, C)
            N = x.shape[1]
        x = x.reshape(B, N, *cur_mu_shape, C)
        # undo_windowing → [B, H, W, C]
        num_MUs = [s // mu for s, mu in zip(size, cur_mu_shape)]
        x = x.reshape(B, *num_MUs, *cur_mu_shape, C)
        perm = ([0]
                + sum([list(p) for p in zip(range(1, 1+D), range(1+D, 1+2*D))], [])
                + [len(x.shape) - 1])
        x = x.transpose(perm)
        return x.reshape(B, *size, C)

# ---------------------------------------------------------------------------
# Hiera Attention + Block
# ---------------------------------------------------------------------------

class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5) -> None:
        super().__init__()
        self.gamma = mx.ones((dim,)) * init_values

    def __call__(self, x: mx.array) -> mx.array:
        return x * self.gamma


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(nn.gelu(self.fc1(x)))


class MaskUnitAttention(nn.Module):
    """
    Windowed or global attention with optional q-pooling (amax).
    Inference-only: no mask support.
    """
    def __init__(self, dim: int, dim_out: int, heads: int,
                 q_stride: int = 1, window_size: int = 0,
                 use_mask_unit_attn: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.heads = heads
        self.q_stride = q_stride
        self.head_dim = dim_out // heads
        self.scale = self.head_dim ** -0.5
        self.use_mask_unit_attn = use_mask_unit_attn
        self.window_size = window_size
        self.qkv = nn.Linear(dim, 3 * dim_out)
        self.proj = nn.Linear(dim_out, dim_out)

    def __call__(self, x: mx.array) -> mx.array:
        B, N, _ = x.shape
        if self.use_mask_unit_attn:
            num_windows = N // (self.q_stride * self.window_size)
            # [B, q_stride*window_size, num_windows, 3, heads, head_dim]
            qkv = self.qkv(x).reshape(
                B, -1, num_windows, 3, self.heads, self.head_dim
            ).transpose(3, 0, 4, 2, 1, 5)
            # qkv: [3, B, heads, num_windows, q_stride*window_size, head_dim]
            q, k, v = qkv[0], qkv[1], qkv[2]
            if self.q_stride > 1:
                # Pool q: [B, heads, num_windows, q_stride, window_size, head_dim] → max over q_stride
                q = q.reshape(B, self.heads, num_windows, self.q_stride, -1, self.head_dim).max(axis=3)
            attn = (q * self.scale) @ k.swapaxes(-1, -2)
            attn = mx.softmax(attn, axis=-1)
            x = attn @ v
            # x: [B, heads, num_windows, window_size, head_dim]  — 5D
            # timm: x.transpose(1, 3) swaps dims 1↔3
            # Must give all 5 axes explicitly in MLX
            x = x.transpose(0, 3, 2, 1, 4).reshape(B, -1, self.dim_out)
        else:
            qkv = self.qkv(x).reshape(B, N, 3, self.heads, self.head_dim).transpose(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]
            if self.q_stride > 1:
                q = q.reshape(B, self.heads, self.q_stride, -1, self.head_dim).max(axis=2)
            attn = (q * self.scale) @ k.swapaxes(-1, -2)
            attn = mx.softmax(attn, axis=-1)
            x = attn @ v
            # x: [B, heads, N, head_dim]  — 4D
            x = x.transpose(0, 2, 1, 3).reshape(B, -1, self.dim_out)
        return self.proj(x)


class HieraBlock(nn.Module):
    def __init__(self, dim: int, dim_out: int, heads: int, mlp_ratio: float = 4.0,
                 init_values: Optional[float] = None, q_stride: int = 1,
                 window_size: int = 0, use_expand_proj: bool = True,
                 use_mask_unit_attn: bool = False) -> None:
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.do_expand = (dim != dim_out)
        self.q_stride = q_stride
        self.proj: Optional[nn.Linear] = nn.Linear(dim, dim_out) if (self.do_expand and use_expand_proj) else None
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MaskUnitAttention(dim, dim_out, heads, q_stride, window_size, use_mask_unit_attn)
        self.ls1 = LayerScale(dim_out, init_values) if init_values is not None else nn.Identity()
        self.norm2 = nn.LayerNorm(dim_out)
        self.mlp = FeedForward(dim_out, int(dim_out * mlp_ratio))
        self.ls2 = LayerScale(dim_out, init_values) if init_values is not None else nn.Identity()

    def __call__(self, x: mx.array) -> mx.array:
        x_norm = self.norm1(x)
        if self.do_expand:
            if self.proj is not None:
                # Project dim → dim_out, then pool by q_stride via amax
                x = self.proj(x_norm)
            else:
                # No proj: concat max-pool + avg-pool (doubles channels)
                x = mx.concatenate([
                    x.reshape(x.shape[0], self.q_stride, -1, x.shape[-1]).max(axis=1),
                    x.reshape(x.shape[0], self.q_stride, -1, x.shape[-1]).mean(axis=1),
                ], axis=-1)
            x = x.reshape(x.shape[0], self.attn.q_stride, -1, x.shape[-1]).max(axis=1)
        x = x + self.ls1(self.attn(x_norm))
        x = x + self.ls2(self.mlp(self.norm2(x)))
        return x

# ---------------------------------------------------------------------------
# Hiera encoder  (Hiera Base Plus — fixed at native 2048px)
# ---------------------------------------------------------------------------

class HieraEncoder(nn.Module):
    """
    Hiera Base Plus for GreenFormer.

    Checkpoint was trained at 2048px: pos_embed is [1, 262144, 112]
    (512×512 patch grid at stride 4).  When running tiled inference at a
    smaller tile_size the pos_embed is bicubic-interpolated in forward().

    Native parameters (Hiera Base Plus):
      embed_dim=112, num_heads=2, stages=(2,3,16,3)
    """
    EMBED_DIM  = 112
    NUM_HEADS  = 2
    STAGES     = (2, 3, 16, 3)
    Q_POOL     = 3
    Q_STRIDE   = (2, 2)
    MU_SIZE    = (8, 8)
    MU_ATTN    = (True, True, False, False)
    DIM_MUL    = 2.0
    HEAD_MUL   = 2.0
    PATCH_K    = (7, 7)
    PATCH_S    = (4, 4)
    PATCH_P    = (3, 3)
    MLP_RATIO  = 4.0
    NATIVE_IMG = 2048  # matches checkpoint pos_embed

    def __init__(self, in_chans: int = 4) -> None:
        super().__init__()
        img_size_t   = (self.NATIVE_IMG, self.NATIVE_IMG)
        patch_stride = self.PATCH_S
        tokens_spatial = [i // s for i, s in zip(img_size_t, patch_stride)]
        num_tokens   = math.prod(tokens_spatial)   # 262144
        flat_mu_size = math.prod(self.MU_SIZE)
        flat_q_stride = math.prod(self.Q_STRIDE)

        self.stage_ends  = [sum(self.STAGES[:i]) - 1 for i in range(1, len(self.STAGES) + 1)]
        q_pool_blocks    = [x + 1 for x in self.stage_ends[:self.Q_POOL]]

        self.patch_embed = PatchEmbed(in_chans, self.EMBED_DIM,
                                      self.PATCH_K[0], self.PATCH_S[0], self.PATCH_P[0])
        # pos_embed stores the native-resolution version; interpolated in forward()
        self.pos_embed = mx.zeros((1, num_tokens, self.EMBED_DIM))

        self.unroll = Unroll(img_size_t, patch_stride, [self.Q_STRIDE] * len(self.stage_ends[:-1]))
        self.reroll = Reroll(img_size_t, patch_stride, [self.Q_STRIDE] * len(self.stage_ends[:-1]),
                             self.stage_ends, self.Q_POOL)

        depth      = sum(self.STAGES)
        embed_dim  = self.EMBED_DIM
        num_heads  = self.NUM_HEADS
        cur_stage  = 0
        mu_size    = flat_mu_size

        blocks: List[HieraBlock] = []
        for i in range(depth):
            dim_out  = embed_dim
            use_mua  = self.MU_ATTN[cur_stage]
            if i - 1 in self.stage_ends:
                dim_out   = int(embed_dim * self.DIM_MUL)
                num_heads = int(num_heads * self.HEAD_MUL)
                cur_stage += 1
            if i in q_pool_blocks:
                mu_size //= flat_q_stride
            blocks.append(HieraBlock(
                dim=embed_dim, dim_out=dim_out, heads=num_heads,
                mlp_ratio=self.MLP_RATIO,
                q_stride=(flat_q_stride if i in q_pool_blocks else 1),
                window_size=mu_size,
                use_expand_proj=True,
                use_mask_unit_attn=use_mua,
            ))
            embed_dim = dim_out

        self.blocks = blocks   # assign complete list (Bug 6 fix)

    def __call__(self, x: mx.array) -> List[mx.array]:
        """x: [N,H,W,4]  returns list of 4 NHWC feature maps."""
        x_tok = self.patch_embed(x)               # [N, T, E]
        # Interpolate pos_embed if tile size != 2048 (Bug 2 fix)
        pe = _interpolate_pos_embed(self.pos_embed, x_tok.shape[1])
        x_tok = x_tok + pe
        x_tok = self.unroll(x_tok)

        features: List[mx.array] = []
        for i, blk in enumerate(self.blocks):
            x_tok = blk(x_tok)
            if i in self.stage_ends:
                feat = self.reroll(x_tok, i)      # [N, H_i, W_i, C_i]
                features.append(feat)

        return features  # channels: [112, 224, 448, 896]

# ---------------------------------------------------------------------------
# GreenFormer  (top-level model)
# ---------------------------------------------------------------------------

class GreenFormer(nn.Module):
    """
    Full GreenFormer in MLX.

    Input : x  [N, H, W, 4]  (NHWC, linear light, values 0-1)
            Channel order: R G B  trimap/mask
    Output: {"alpha": [N,H,W,1], "fg": [N,H,W,3]}  (NHWC, 0-1)
    """
    def __init__(self, use_refiner: bool = True) -> None:
        super().__init__()
        self.encoder      = HieraEncoder(in_chans=4)
        feat_ch           = [112, 224, 448, 896]
        self.alpha_decoder = DecoderHead(feat_ch, 256, output_dim=1)
        self.fg_decoder    = DecoderHead(feat_ch, 256, output_dim=3)
        self.use_refiner   = use_refiner
        self.refiner: Optional[CNNRefinerModule] = CNNRefinerModule(7, 64, 4) if use_refiner else None

    def __call__(self, x: mx.array) -> dict:
        input_hw = (x.shape[1], x.shape[2])
        features = self.encoder(x)

        alpha_logits = self.alpha_decoder(features)   # [N, H/4, W/4, 1]
        fg_logits    = self.fg_decoder(features)      # [N, H/4, W/4, 3]

        alpha_logits_up = _interpolate(alpha_logits, input_hw)
        fg_logits_up    = _interpolate(fg_logits,    input_hw)

        alpha_coarse = mx.sigmoid(alpha_logits_up)
        fg_coarse    = mx.sigmoid(fg_logits_up)

        rgb         = x[:, :, :, :3]
        coarse_pred = mx.concatenate([alpha_coarse, fg_coarse], axis=-1)  # [N,H,W,4]

        if self.use_refiner and self.refiner is not None:
            delta_logits = self.refiner(rgb, coarse_pred)
        else:
            delta_logits = mx.zeros_like(coarse_pred)

        delta_alpha = delta_logits[:, :, :, 0:1]
        delta_fg    = delta_logits[:, :, :, 1:4]

        return {
            "alpha": mx.sigmoid(alpha_logits_up + delta_alpha),
            "fg":    mx.sigmoid(fg_logits_up    + delta_fg),
        }


# ---------------------------------------------------------------------------
# Smoke-test: python model.py
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import time
    print("Building GreenFormer …")
    model = GreenFormer()
    # Use a 512-tile (pos_embed will be bicubic-downsampled 2048→512)
    x = mx.zeros((1, 512, 512, 4))
    print("First forward pass (includes JIT compile) …")
    t0 = time.time()
    out = model(x)
    mx.eval(out["alpha"])
    print(f"Done: {time.time()-t0:.2f}s  alpha={out['alpha'].shape}  fg={out['fg'].shape}")
