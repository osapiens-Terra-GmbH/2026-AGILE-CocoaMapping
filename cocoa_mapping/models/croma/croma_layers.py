"""The code is based on the code from https://github.com/antofuller/CROMA/tree/main under MIT license.

Citation:
    @inproceedings{fuller2023croma,
        title={CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders},
        author={Fuller, Anthony and Millard, Koreen and Green, James R},
        booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
        year={2023}}
"""

from einops import rearrange
from torch import nn, einsum


class FFN(nn.Module):
    def __init__(self,
                 dim,
                 mult=4,
                 dropout=0.,
                 ):
        super().__init__()
        inner_dim = int(dim * mult)

        self.net = nn.Sequential(
            nn.Linear(dim, inner_dim),  # (BSZ, num_patches, inner_dim)
            nn.GELU(),  # (BSZ, num_patches, inner_dim)
            nn.Dropout(dropout),  # (BSZ, num_patches, inner_dim)
            nn.Linear(inner_dim, dim)  # (BSZ, num_patches, dim)
        )
        self.input_norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.input_norm(x)  # (BSZ, num_patches, dim)
        return self.net(x)  # (BSZ, num_patches, dim)


class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 dropout=0.,
                 ):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, 'dim must be evenly divisible by num_heads'
        dim_head = int(dim / num_heads)
        self.scale = dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, relative_position_bias):
        x = self.input_norm(x)  # (BSZ, num_patches, dim)
        q, k, v = self.to_qkv(x).chunk(3, dim=-1)  # (BSZ, num_patches, dim)

        # (BSZ, num_heads, num_patches, dim_head)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        attention_scores = einsum('b h i d, b h j d -> b h i j', q,
                                  k) * self.scale  # (BSZ, num_heads, num_patches, num_patches)
        # (BSZ, num_heads, num_patches, num_patches)
        attention_scores = attention_scores + relative_position_bias

        # (BSZ, num_heads, num_patches, num_patches)
        attn = attention_scores.softmax(dim=-1)
        attn = self.dropout(attn)  # (BSZ, num_heads, num_patches, num_patches)

        # (BSZ, num_heads, num_patches, dim_head)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')  # (BSZ, num_patches, dim)
        return self.to_out(out)  # (BSZ, num_patches, dim)


class CrossAttention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 dropout=0.,
                 ):
        super().__init__()
        self.num_heads = num_heads
        assert dim % num_heads == 0, 'dim must be evenly divisible by num_heads'
        dim_head = int(dim / num_heads)
        self.scale = dim_head ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=False)
        self.to_v = nn.Linear(dim, dim, bias=False)

        self.to_out = nn.Linear(dim, dim)
        self.input_norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, context, relative_position_bias):
        x = self.input_norm(x)  # (BSZ, num_patches, dim)
        context = self.input_norm(context)  # (BSZ, num_patches, dim)

        q = self.to_q(x)  # (BSZ, num_patches, dim)
        k = self.to_k(context)  # (BSZ, num_patches, dim)
        v = self.to_v(context)  # (BSZ, num_patches, dim)

        # (BSZ, num_heads, num_patches, dim_head)
        q = rearrange(q, 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(k, 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(v, 'b n (h d) -> b h n d', h=self.num_heads)

        attention_scores = einsum('b h i d, b h j d -> b h i j', q,
                                  k) * self.scale  # (BSZ, num_heads, num_patches, num_patches)
        # (BSZ, num_heads, num_patches, num_patches)
        attention_scores = attention_scores + relative_position_bias

        # (BSZ, num_heads, num_patches, num_patches)
        attn = attention_scores.softmax(dim=-1)
        attn = self.dropout(attn)  # (BSZ, num_heads, num_patches, num_patches)

        # (BSZ, num_heads, num_patches, dim_head)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')  # (BSZ, num_patches, dim)
        return self.to_out(out)  # (BSZ, num_patches, dim)

# pylint: disable=too-many-arguments


class BaseTransformer(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads=8,
                 attn_dropout=0.,
                 ff_dropout=0.,
                 ff_mult=4,
                 final_norm=True,
                 ):
        super().__init__()
        self.final_norm = final_norm
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, num_heads=num_heads, dropout=attn_dropout),
                FFN(dim=dim, mult=ff_mult, dropout=ff_dropout),
            ]))

        if self.final_norm:
            self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, relative_position_bias=0):
        for self_attn, ffn in self.layers:
            x = self_attn(x, relative_position_bias) + \
                x  # (BSZ, num_patches, dim)
            x = ffn(x) + x  # (BSZ, num_patches, dim)

        if self.final_norm:
            return self.norm_out(x)
        return x


class BaseTransformerCrossAttn(nn.Module):
    def __init__(self,
                 dim,
                 depth,
                 num_heads=8,
                 attn_dropout=0.,
                 ff_dropout=0.,
                 ff_mult=4,
                 ):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim=dim, num_heads=num_heads, dropout=attn_dropout),
                CrossAttention(dim=dim, num_heads=num_heads,
                               dropout=attn_dropout),
                FFN(dim=dim, mult=ff_mult, dropout=ff_dropout),
            ]))

        self.norm_out = nn.LayerNorm(dim)

    def forward(self, x, context, relative_position_bias):
        for self_attn, cross_attn, ffn in self.layers:
            x = self_attn(x, relative_position_bias) + \
                x  # (BSZ, num_patches, dim)
            x = cross_attn(x, context, relative_position_bias) + \
                x  # (BSZ, num_patches, dim)
            x = ffn(x) + x  # (BSZ, num_patches, dim)

        x = self.norm_out(x)
        return x  # (BSZ, num_patches, dim)


class ViT(nn.Module):
    def __init__(self, dim, depth, in_channels):
        super().__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.dim = dim
        self.num_heads = 16  # always 16, for base and large models
        self.patch_size = 8  # always 8, for base and large models

        pixels_per_patch = int(self.patch_size * self.patch_size * in_channels)
        self.linear_input = nn.Linear(pixels_per_patch, self.dim)
        self.transformer = BaseTransformer(dim=self.dim,
                                           depth=self.depth,
                                           num_heads=self.num_heads,
                                           )

    def forward(self, imgs, attn_bias):
        x = rearrange(imgs, 'b c (h i) (w j) -> b (h w) (c i j)',
                      i=self.patch_size, j=self.patch_size)
        # x is shape -> (bsz, num_patches, self.channels*self.patch_size*self.patch_size)

        x = self.linear_input(x)  # (bsz, num_patches, dim)
        x = self.transformer(x, relative_position_bias=attn_bias)
        return x
