import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_


class GroupedConvFFN(nn.Module):
    def __init__(self, in_channels, hidden_channels, groups, act_layer=nn.GELU, drop=0.0):
        super().__init__()
        self.fc1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=1, groups=groups, bias=True)
        self.act = act_layer()
        self.drop1 = nn.Dropout(p=drop)
        self.fc2 = nn.Conv2d(hidden_channels, in_channels, kernel_size=1, groups=groups, bias=True)
        self.drop2 = nn.Dropout(p=drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class PartitionFreeBranchCollapsedBlock(nn.Module):
    """A lightweight SkateFormer block without explicit partition/reverse/cat."""

    def __init__(
        self,
        in_channels,
        num_points=50,
        kernel_size=7,
        num_heads=32,
        type_1_size=(1, 1),
        type_2_size=(1, 1),
        type_3_size=(1, 1),
        type_4_size=(1, 1),
        attn_drop=0.0,
        drop=0.0,
        rel=True,
        drop_path=0.0,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        del type_1_size, type_2_size, type_3_size, type_4_size, attn_drop, rel

        self.in_channels = in_channels
        self.num_points = num_points
        self.graph_heads = max(1, min(4, num_heads // 4))
        self.channel_groups = 8 if in_channels % 8 == 0 else 4 if in_channels % 4 == 0 else 1

        self.norm_1 = norm_layer(in_channels)
        self.pre_mix = nn.Linear(in_channels, in_channels, bias=True)

        self.graph = nn.Parameter(torch.zeros(self.graph_heads, num_points, num_points))
        trunc_normal_(self.graph, std=0.02)

        pad = (kernel_size - 1) // 2
        self.temporal_mixer = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            groups=in_channels,
            bias=True,
        )

        # Partition-free relation mixer driven by dynamic time/joint gates.
        self.time_gate = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=True,
        )
        self.joint_gate = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=3,
            padding=1,
            groups=in_channels,
            bias=True,
        )

        self.fusion_gate = nn.Conv2d(2 * in_channels, in_channels, kernel_size=1, bias=True)
        self.out_proj = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=1,
            groups=self.channel_groups,
            bias=True,
        )
        self.out_drop = nn.Dropout(p=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm_2 = norm_layer(in_channels)
        hidden_channels = int(max(1.0, mlp_ratio * 0.5) * in_channels)
        self.ffn = GroupedConvFFN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            groups=self.channel_groups,
            act_layer=act_layer,
            drop=drop,
        )

    def _graph_mix(self, x):
        chunks = torch.chunk(x, self.graph_heads, dim=1)
        mixed = []
        for idx, chunk in enumerate(chunks):
            mixed.append(torch.einsum("n c t u, v u -> n c t v", chunk, self.graph[idx]))
        return torch.cat(mixed, dim=1)

    def forward(self, input):
        skip = input

        shared = self.pre_mix(self.norm_1(input.permute(0, 2, 3, 1).contiguous()))
        shared = shared.permute(0, 3, 1, 2).contiguous()

        local = self._graph_mix(shared) + self.temporal_mixer(shared)

        time_context = shared.mean(dim=3)
        joint_context = shared.mean(dim=2)
        time_gate = torch.sigmoid(self.time_gate(time_context)).unsqueeze(-1)
        joint_gate = torch.sigmoid(self.joint_gate(joint_context)).unsqueeze(2)
        relation = shared * (0.5 * (time_gate + joint_gate))

        local_stat = local.mean(dim=(2, 3), keepdim=True)
        relation_stat = relation.mean(dim=(2, 3), keepdim=True)
        fusion = torch.sigmoid(self.fusion_gate(torch.cat([local_stat, relation_stat], dim=1)))

        mixed = local + fusion * relation
        output = skip + self.drop_path(self.out_drop(self.out_proj(mixed)))

        ffn_input = self.norm_2(output.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        ffn_output = self.ffn(ffn_input)
        output = output + self.drop_path(ffn_output)
        return output
