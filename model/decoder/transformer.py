import math
import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat


class PositionEncoding2d(nn.Module):
    """
    Positional encoding for 2D inputs
    """

    def __init__(self, d_model, feature_size,
                 temperature=10000, scale=2 * math.pi, eps=1e-6):
        """
        Args:
            dim: the dimension of the encoded position
            feature_size: the size of the features' width and height
        """
        super().__init__()

        area = torch.ones(feature_size, feature_size)  # [h, w]
        y_embed = area.cumsum(0, dtype=torch.float32)
        x_embed = area.cumsum(1, dtype=torch.float32)

        y_embed = y_embed / (y_embed[-1:, :] + eps) * scale
        x_embed = x_embed / (x_embed[:, -1:] + eps) * scale

        dim_t = torch.arange(d_model // 2, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / (d_model // 2))

        pos_x = x_embed[:, :, None] / dim_t
        pos_y = y_embed[:, :, None] / dim_t

        pos_x = torch.stack(
            (pos_x[:, :, 0::2].sin(),
             pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
        pos_y = torch.stack(
            (pos_y[:, :, 0::2].sin(),
             pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

        pos = torch.cat((pos_y, pos_x), dim=2)
        pos = rearrange(pos, 'h w c -> (h w) c')

        # pos size: [H x W, C]
        self.register_buffer('pos', pos, persistent=False)

    def forward(self, x):
        """
        Args:
            x: [N, (H x W), C]
        """
        pos = repeat(self.pos, 'l c -> b l c', b=x.size(0))
        return x + pos


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads, head_dim, dropout=0.):
        super().__init__()
        inner_dim = head_dim * heads
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t:
                      rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads, head_dim, dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return x


class ViT(nn.Module):
    def __init__(self, num_classes, num_joints, feature_size,
                 dim, depth, heads, head_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.pos_embedding = PositionEncoding2d(dim, feature_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.transformer = Transformer(
            dim, depth, heads, head_dim, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

        self.simple_decoder = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(
                in_channels=dim,
                out_channels=num_joints,
                kernel_size=1,
                stride=1,
                padding=0
            )
        )

    def forward(self, x):
        b, c, h, w = x.shape

        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.pos_embedding(x)

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)

        x = self.transformer(x)

        cls_feat = x[:, 0]
        cls_out = self.mlp_head(cls_feat)

        hmap_feat = rearrange(x[:, 1:], 'b (h w) c -> b c h w', h=h, w=w)

        hmap_feat = F.interpolate(hmap_feat, scale_factor=(4, 4),
                                  mode='bilinear', align_corners=True)
        hmap_out = self.simple_decoder(hmap_feat)

        return cls_out, hmap_out
