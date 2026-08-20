"""Skate-RCA: relation-composition attention for skeleton features."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, trunc_normal_


class SkateRCAFeedForward(nn.Module):
    def __init__(self, channels, hidden_channels, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Linear(channels, hidden_channels)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_channels, channels)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        return self.drop2(self.fc2(self.drop1(self.act(self.fc1(x)))))


class SkateRCABlock(nn.Module):
    """Dynamic low-rank spatial-temporal relation composition.

    Components share a value projection, then compose joint attention and
    anchor-based temporal attention. Graph/local priors retain skeleton bias and
    a sample-dependent router fuses components. Layout is (B, C, T, V).
    """

    def __init__(
        self, in_channels, num_points=14, kernel_size=7, num_heads=32,
        type_1_size=(1, 1), type_2_size=(1, 1), type_3_size=(1, 1),
        type_4_size=(1, 1), attn_drop=0.0, drop=0.0, rel=True,
        drop_path=0.0, mlp_ratio=4.0, act_layer=nn.GELU,
        norm_layer=nn.LayerNorm, relation_rank=2, qk_ratio=0.25,
        temporal_anchors=16, ffn_ratio=None,
    ):
        super().__init__()
        del num_heads, type_1_size, type_2_size, type_3_size, type_4_size
        if relation_rank <= 0 or in_channels % relation_rank:
            raise ValueError("relation_rank must be positive and divide in_channels")
        if temporal_anchors <= 0:
            raise ValueError("temporal_anchors must be positive")
        self.in_channels, self.num_points = in_channels, num_points
        self.relation_rank = relation_rank
        self.rank_channels = in_channels // relation_rank
        self.temporal_anchors, self.kernel_size = temporal_anchors, kernel_size
        self.use_relation_priors = rel
        qkc = max(relation_rank * 8, int(round(in_channels * qk_ratio)))
        qkc = math.ceil(qkc / relation_rank) * relation_rank
        self.head_dim = qkc // relation_rank
        self.scale = self.head_dim ** -0.5

        self.norm1 = norm_layer(in_channels)
        self.value_proj = nn.Conv2d(in_channels, in_channels, 1)
        self.spatial_qk = nn.Conv1d(in_channels, 2 * qkc, 1)
        self.temporal_q = nn.Conv1d(in_channels, qkc, 1)
        self.temporal_k = nn.Conv1d(in_channels, qkc, 1)
        self.graph_bias = nn.Parameter(torch.zeros(relation_rank, num_points, num_points))
        trunc_normal_(self.graph_bias, std=0.02)
        with torch.no_grad():
            self.graph_bias.add_(0.5 * torch.eye(num_points).unsqueeze(0))
        self.graph_strength = nn.Parameter(torch.ones(relation_rank))
        self.temporal_local_strength = nn.Parameter(torch.ones(relation_rank))
        self.attn_drop = nn.Dropout(attn_drop)

        router_hidden = max(16, in_channels // 4)
        self.rank_router = nn.Sequential(
            nn.Linear(in_channels, router_hidden), act_layer(),
            nn.Linear(router_hidden, relation_rank),
        )
        nn.init.zeros_(self.rank_router[-1].weight)
        nn.init.zeros_(self.rank_router[-1].bias)
        self.out_proj = nn.Conv2d(in_channels, in_channels, 1)
        self.out_drop = nn.Dropout(drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        ratio = max(1.0, mlp_ratio * 0.5) if ffn_ratio is None else ffn_ratio
        self.norm2 = norm_layer(in_channels)
        self.ffn = SkateRCAFeedForward(
            in_channels, max(1, int(round(in_channels * ratio))),
            act_layer=act_layer, drop=drop,
        )

    def _spatial_attention(self, x):
        b, _, _, v = x.shape
        q, k = self.spatial_qk(x.mean(2)).chunk(2, 1)
        q = q.reshape(b, self.relation_rank, self.head_dim, v)
        k = k.reshape(b, self.relation_rank, self.head_dim, v)
        logits = torch.matmul(q.transpose(-2, -1), k) * self.scale
        if self.use_relation_priors:
            s = F.softplus(self.graph_strength).view(1, -1, 1, 1)
            logits = logits + s * self.graph_bias.unsqueeze(0)
        return self.attn_drop(logits.softmax(-1))

    def _temporal_attention(self, x):
        b, _, t, _ = x.shape
        summary = x.mean(3)
        anchor_summary = F.adaptive_avg_pool1d(summary, self.temporal_anchors)
        q = self.temporal_q(summary).reshape(
            b, self.relation_rank, self.head_dim, t)
        k = self.temporal_k(anchor_summary).reshape(
            b, self.relation_rank, self.head_dim, self.temporal_anchors)
        logits = torch.matmul(q.transpose(-2, -1), k) * self.scale
        if self.use_relation_priors:
            fp = torch.linspace(0., 1., t, device=x.device)
            ap = torch.linspace(0., 1., self.temporal_anchors, device=x.device)
            radius = max(1.0 / max(t - 1, 1), self.kernel_size / t)
            local = -(fp[:, None] - ap[None, :]).abs() / radius
            s = F.softplus(self.temporal_local_strength).view(1, -1, 1, 1)
            logits = logits + s * local.view(1, 1, t, self.temporal_anchors)
        return self.attn_drop(logits.softmax(-1))

    def _pool_values(self, values):
        b, r, c, t, v = values.shape
        values = values.permute(0, 1, 2, 4, 3).reshape(b * r, c * v, t)
        values = F.adaptive_avg_pool1d(values, self.temporal_anchors)
        return values.reshape(
            b, r, c, v, self.temporal_anchors).permute(0, 1, 2, 4, 3)

    def forward(self, x):
        b, c, t, v = x.shape
        if c != self.in_channels or v != self.num_points:
            raise ValueError(
                f"expected (*, {self.in_channels}, *, {self.num_points}), "
                f"got {tuple(x.shape)}")
        z = self.norm1(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        spatial, temporal = self._spatial_attention(z), self._temporal_attention(z)
        values = self.value_proj(z).reshape(
            b, self.relation_rank, self.rank_channels, t, v)
        values = torch.matmul(values, spatial.transpose(-2, -1).unsqueeze(2))
        anchors = self._pool_values(values)
        composed = torch.matmul(
            anchors.permute(0, 1, 2, 4, 3),
            temporal.transpose(-2, -1).unsqueeze(2),
        ).permute(0, 1, 2, 4, 3)
        gates = self.rank_router(z.mean((2, 3))).softmax(-1)
        composed *= (gates * self.relation_rank).view(
            b, self.relation_rank, 1, 1, 1)
        composed = composed.reshape(b, c, t, v)
        x = x + self.drop_path(self.out_drop(self.out_proj(composed)))
        ffn = self.ffn(self.norm2(x.permute(0, 2, 3, 1)))
        return x + self.drop_path(ffn.permute(0, 3, 1, 2))
