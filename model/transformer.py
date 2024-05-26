import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat


# Pos embedding
def pos_emb_sincos_2d(h, w, dim, temperature=10000, dtype=torch.float32):
    """Pos embedding for 2D image"""
    y, x = torch.meshgrid(
        torch.arange(h), torch.arange(w), indexing="ij"
    )
    assert (dim % 4) == 0, "dimension must be divisible by 4"

    # 1D pos embedding
    omega = torch.arange(dim // 4, dtype=dtype)
    omega = 1.0 / (temperature ** omega)

    # 2D pos embedding
    y = y.flatten()[:, None] * omega[None, :]
    x = x.flatten()[:, None] * omega[None, :]

    # concat sin and cos
    pe = torch.cat((x.sin(), x.cos(), y.sin(), y.cos()), dim=1)
    return pe.type(dtype)


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
    def __init__(self, dim, heads, head_dim):
        super().__init__()
        inner_dim = head_dim * heads
        project_out = not (heads == 1 and head_dim == dim)

        self.heads = heads
        self.scale = head_dim ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        if project_out:
            self.to_out = nn.Linear(inner_dim, dim, bias=False)
        else:
            self.to_out = nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t:
                      rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = torch.softmax(dots, dim=-1)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)

        return out, attn


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, head_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads, head_dim),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            message, attnmap = attn(x)
            x = message + x
            x = ff(x) + x

        return x, attnmap


class ViT(nn.Module):
    def __init__(self, num_classes, num_joints, feature_size,
                 dim, depth, heads, head_dim, mlp_dim, dropout=0.):
        super().__init__()
        self.pos_embedding = pos_emb_sincos_2d(
            h=feature_size[0],
            w=feature_size[1],
            dim=dim,
        )
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

        # add positional embedding
        x += self.pos_embedding.to(x.device)

        cls_token = repeat(self.cls_token, '() n d -> b n d', b=b)

        x = torch.cat([cls_token, x], dim=1)
        x, attnmap = self.transformer(x)

        cls_feat, hmap_feat = x[:, 0], x[:, 1:]

        cls_out = self.mlp_head(cls_feat)

        hmap_feat = rearrange(hmap_feat, 'b (h w) c -> b c h w', h=h, w=w)

        hmap_feat = F.interpolate(hmap_feat, scale_factor=(4, 4),
                                  mode='bilinear', align_corners=True)
        hmap_out = self.simple_decoder(hmap_feat)

        return cls_out, hmap_out, attnmap
