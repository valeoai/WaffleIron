# Copyright 2022 - Valeo Comfort and Driving Assistance - Gilles Puy @ valeo.ai
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import torch
import warnings
import torch.nn as nn

from waffleiron import WI_SCATTER_REDUCE
if WI_SCATTER_REDUCE:
    from .helper_projection import projection_3d_to_2d_scatter_reduce as projection_3d_to_2d
    from .helper_projection import get_all_projections_scatter_reduce as get_all_projections
else:
    from .helper_projection import projection_3d_to_2d_sparse_matrix as projection_3d_to_2d
    from .helper_projection import get_all_projections_sparse_matrices as get_all_projections


class myLayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x):
        return super().forward(x.transpose(1, -1)).transpose(1, -1)

NORM_OPTIONS = {
	"batchnorm": nn.BatchNorm1d,
	"layernorm": myLayerNorm,
}


class DropPath(nn.Module):
    """
    Stochastic Depth

    Original code of this module is at:
    https://github.com/facebookresearch/dino/blob/main/vision_transformer.py
    """

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob
        self.keep_prob = 1 - drop_prob

    def extra_repr(self):
        return f"prob={self.drop_prob}"

    def forward(self, x):
        if not self.training or self.drop_prob == 0.0:
            return x
        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = self.keep_prob + torch.rand(
            shape, dtype=x.dtype, device=x.device
        )
        random_tensor.floor_()  # binarize
        output = x.div(self.keep_prob) * random_tensor
        return output


class ChannelMix(nn.Module):
    def __init__(self, channels, drop_path_prob, which_norm="batchnorm"):
        super().__init__()
        self.compressed = False
        self.which_norm = which_norm
        self.norm = NORM_OPTIONS[which_norm](channels)
        self.mlp = nn.Sequential(
            nn.Conv1d(channels, channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, 1),
        )
        self.scale = nn.Conv1d(
            channels, channels, 1, bias=False, groups=channels
        )  # Implement LayerScale
        self.drop_path = DropPath(drop_path_prob)

    def compress(self):
        if self.which_norm == "layernorm":
            warnings.warn(
                "Compression of ChannelMix layer has not been implemented " +
                "with layer norm. Skipping compression."
            )
            return
        # Join Batch norm and first conv
        norm_weight = self.norm.weight.data / torch.sqrt(
            self.norm.running_var.data + 1e-05
        )
        norm_bias = self.norm.bias.data - norm_weight * self.norm.running_mean.data
        # Careful the order of the two lines below should not be changed
        self.mlp[0].bias.data = (
            self.mlp[0].weight.data[:, :, 0] @ norm_bias + self.mlp[0].bias.data
        )
        self.mlp[0].weight.data = self.mlp[0].weight.data * norm_weight[None, :, None]
        # Join scale and last conv
        self.mlp[-1].weight.data = self.mlp[-1].weight.data * self.scale.weight.data
        self.mlp[-1].bias.data = (
            self.mlp[-1].bias.data * self.scale.weight.data[:, 0, 0]
        )
        # Flag
        self.compressed = True

    def forward(self, tokens):
        """tokens <- tokens + LayerScale( MLP( BN(tokens) ) )"""
        if self.compressed:
            assert not self.training
            return tokens + self.drop_path(self.mlp(tokens))
        else:
            return tokens + self.drop_path(self.scale(self.mlp(self.norm(tokens))))


class SpatialMix(nn.Module):
    def __init__(self, channels, grid_shape, drop_path_prob, which_norm="batchnorm"):
        super().__init__()
        self.compressed = False
        self.H, self.W = grid_shape
        self.norm = NORM_OPTIONS[which_norm](channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels),
        )
        self.scale = nn.Conv1d(
            channels, channels, 1, bias=False, groups=channels
        )  # Implement LayerScale
        self.grid_shape = grid_shape
        self.drop_path = DropPath(drop_path_prob)

    def extra_repr(self):
        return f"(grid): [{self.grid_shape[0]}, {self.grid_shape[1]}]"

    def compress(self):
        # Join scale and last conv
        self.ffn[-1].weight.data = (
            self.ffn[-1].weight.data * self.scale.weight.data[..., None]
        )
        self.ffn[-1].bias.data = (
            self.ffn[-1].bias.data * self.scale.weight.data[:, 0, 0]
        )
        # Flag
        self.compressed = True

    def forward_compressed(self, tokens, sp_mat):
        """tokens <- tokens + LayerScale( Inflate( FFN( Flatten( BN(tokens) ) ) )"""
        # Make sure we are not in training mode
        assert not self.training
        # Forward pass
        B, C, N = tokens.shape
        residual = self.norm(tokens)
        # Flatten
        residual = projection_3d_to_2d(residual, sp_mat, B, C, self.H, self.W)
        residual = residual.reshape(B, C, self.H, self.W)
        # FFN
        residual = self.ffn(residual)
        # Inflate
        residual = residual.reshape(B, C, self.H * self.W)
        residual = torch.gather(residual, 2, sp_mat["inflate"])
        return tokens + self.drop_path(residual)

    def forward(self, tokens, sp_mat):
        """tokens <- tokens + LayerScale( Inflate( FFN( Flatten( BN(tokens) ) ) )"""
        if self.compressed:
            return self.forward_compressed(tokens, sp_mat)
        #
        B, C, N = tokens.shape
        residual = self.norm(tokens)
        # Flatten
        residual = projection_3d_to_2d(residual, sp_mat, B, C, self.H, self.W)
        residual = residual.reshape(B, C, self.H, self.W)
        # FFN
        residual = self.ffn(residual)
        # LayerScale
        residual = residual.reshape(B, C, self.H * self.W)
        residual = self.scale(residual)
        # Inflate
        residual = torch.gather(residual, 2, sp_mat["inflate"])
        return tokens + self.drop_path(residual)


class WaffleIron(nn.Module):
    def __init__(self, channels, depth, grids_shape, drop_path_prob, which_norm="batchnorm"):
        super().__init__()
        self.depth = depth
        self.grids_shape = grids_shape
        self.channel_mix = nn.ModuleList(
            [ChannelMix(channels, drop_path_prob, which_norm) for _ in range(depth)]
        )
        self.spatial_mix = nn.ModuleList(
            [
                SpatialMix(channels, grids_shape[d % len(grids_shape)], drop_path_prob, which_norm)
                for d in range(depth)
            ]
        )

    def compress(self):
        for d in range(self.depth):
            self.channel_mix[d].compress()
            self.spatial_mix[d].compress()

    def forward(self, tokens, cell_ind, occupied_cell):
        # Build all 3D to 2D projection matrices
        batch_size, nb_feat, num_points = tokens.shape
        sp_mat = get_all_projections(
            cell_ind, nb_feat, batch_size, num_points, 
            occupied_cell, tokens.device, self.grids_shape, tokens.dtype,
        )

        # Actual backbone
        for d, (smix, cmix) in enumerate(zip(self.spatial_mix, self.channel_mix)):
            tokens = smix(tokens, sp_mat[d % len(sp_mat)])
            tokens = cmix(tokens)

        return tokens
