import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import Mlp, DropPath, trunc_normal_

from model.SkateFormer import (
    MultiHeadSelfAttention,
    get_relative_position_index_1d,
    type_1_partition,
    type_1_reverse,
    type_2_partition,
    type_2_reverse,
    type_3_partition,
    type_3_reverse,
    type_4_partition,
    type_4_reverse,
)


def branchwise_linear_sum(branches, linear):
    """Equivalent rewrite of linear(cat(branches))."""
    out = None
    start = 0
    bias = linear.bias
    for idx, branch in enumerate(branches):
        channels = branch.shape[1]
        branch_last = branch.permute(0, 2, 3, 1).contiguous()
        weight_slice = linear.weight[:, start:start + channels]
        piece = F.linear(branch_last, weight_slice, bias if idx == 0 else None)
        out = piece if out is None else out + piece
        start += channels
    return out


class SkateFormerBlockCatProjFreeExact(nn.Module):
    """A numerically equivalent SkateFormerBlock with cat+proj rewritten."""

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
        self.type_1_size = type_1_size
        self.type_2_size = type_2_size
        self.type_3_size = type_3_size
        self.type_4_size = type_4_size
        self.partition_function = [type_1_partition, type_2_partition, type_3_partition, type_4_partition]
        self.reverse_function = [type_1_reverse, type_2_reverse, type_3_reverse, type_4_reverse]
        self.partition_size = [type_1_size, type_2_size, type_3_size, type_4_size]
        self.rel_type = ["type_1", "type_2", "type_3", "type_4"]

        self.norm_1 = norm_layer(in_channels)
        self.mapping = nn.Linear(in_features=in_channels, out_features=2 * in_channels, bias=True)
        self.gconv = nn.Parameter(torch.zeros(num_heads // (2 * 2), num_points, num_points))
        trunc_normal_(self.gconv, std=0.02)
        self.tconv = nn.Conv2d(
            in_channels // (2 * 2),
            in_channels // (2 * 2),
            kernel_size=(kernel_size, 1),
            padding=((kernel_size - 1) // 2, 0),
            groups=num_heads // (2 * 2),
        )

        attention = []
        for i in range(len(self.partition_function)):
            attention.append(
                MultiHeadSelfAttention(
                    in_channels=in_channels // (len(self.partition_function) * 2),
                    rel_type=self.rel_type[i],
                    num_heads=num_heads // (len(self.partition_function) * 2),
                    partition_size=self.partition_size[i],
                    attn_drop=attn_drop,
                    rel=rel,
                )
            )
        self.attention = nn.ModuleList(attention)
        self.proj = nn.Linear(in_features=in_channels, out_features=in_channels, bias=True)
        self.proj_drop = nn.Dropout(p=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm_2 = norm_layer(in_channels)
        self.mlp = Mlp(
            in_features=in_channels,
            hidden_features=int(mlp_ratio * in_channels),
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, input):
        _, C, T, V = input.shape

        input = input.permute(0, 2, 3, 1).contiguous()
        skip = input

        f = self.mapping(self.norm_1(input)).permute(0, 3, 1, 2).contiguous()

        f_conv, f_attn = torch.split(f, [C // 2, 3 * C // 2], dim=1)
        y = []

        split_f_conv = torch.chunk(f_conv, 2, dim=1)
        y_gconv = []
        split_f_gconv = torch.chunk(split_f_conv[0], self.gconv.shape[0], dim=1)
        for i in range(self.gconv.shape[0]):
            z = torch.einsum("n c t u, v u -> n c t v", split_f_gconv[i], self.gconv[i])
            y_gconv.append(z)
        y.append(torch.cat(y_gconv, dim=1))

        y.append(self.tconv(split_f_conv[1]))

        split_f_attn = torch.chunk(f_attn, len(self.partition_function), dim=1)
        for i in range(len(self.partition_function)):
            c_i = split_f_attn[i].shape[1]
            input_partitioned = self.partition_function[i](split_f_attn[i], self.partition_size[i])
            input_partitioned = input_partitioned.view(
                -1,
                self.partition_size[i][0] * self.partition_size[i][1],
                c_i,
            )
            y.append(self.reverse_function[i](self.attention[i](input_partitioned), (T, V), self.partition_size[i]))

        output = branchwise_linear_sum(y, self.proj)
        output = self.proj_drop(output)
        output = skip + self.drop_path(output)

        output = output + self.drop_path(self.mlp(self.norm_2(output)))
        output = output.permute(0, 3, 1, 2).contiguous()
        return output
