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


import torch
import torch.nn as nn


class Embedding(nn.Module):
    def __init__(self, channels_in, channels_out):
        super().__init__()

        #
        self.compressed = False
        self.channels_in, self.channels_out = channels_in, channels_out

        # Normalize inputs
        self.norm = nn.BatchNorm1d(channels_in)

        # Point Embedding
        self.conv1 = nn.Conv1d(channels_in, channels_out, 1)

        # Neighborhood embedding
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(channels_in),
            nn.Conv2d(channels_in, channels_out, 1, bias=False),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out, 1, bias=False),
        )

        # Merge point and neighborhood embeddings
        self.final = nn.Conv1d(2 * channels_out, channels_out, 1, bias=True, padding=0)

    def compress(self):
        # Recombine first batch norm and conv1
        first_norm_weight = self.norm.weight.data / torch.sqrt(
            self.norm.running_var.data + 1e-05
        )
        first_norm_bias = (
            self.norm.bias.data - first_norm_weight * self.norm.running_mean.data
        )
        conv1_weight = self.conv1.weight.data * first_norm_weight[None, :, None]
        conv1_bias = (
            self.conv1.weight.data[:, :, 0] @ first_norm_bias + self.conv1.bias.data
        )
        self.conv1.weight.data = conv1_weight
        self.conv1.bias.data = conv1_bias
        self.norm = nn.Identity()
        # Merge all batch norms and conv in local part
        # Trick in understanding the two line below is too realize that first_norm_bias has no influence because of
        # relative difference. Hence vector is just rescaled
        second_norm_weight = (
            first_norm_weight
            * self.conv2[0].weight.data
            / torch.sqrt(self.conv2[0].running_var.data + 1e-05)
        )
        second_norm_bias = (
            self.conv2[0].bias.data
            - (second_norm_weight / first_norm_weight) * self.conv2[0].running_mean
        )
        third_norm_weight = self.conv2[2].weight.data / torch.sqrt(
            self.conv2[2].running_var.data + 1e-05
        )
        third_norm_bias = (
            self.conv2[2].bias.data
            - third_norm_weight * self.conv2[2].running_mean.data
        )
        conv2_weight = (
            self.conv2[1].weight.data * second_norm_weight[None, :, None, None]
        )
        conv2_bias = self.conv2[1].weight.data[:, :, 0, 0] @ second_norm_bias
        conv2_weight = conv2_weight * third_norm_weight[:, None, None, None]
        conv2_bias = third_norm_weight * conv2_bias + third_norm_bias
        # Update layers
        self.new_conv2 = nn.Sequential(
            nn.Conv2d(self.channels_in, self.channels_out, 1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.channels_out, self.channels_out, 1, bias=False),
        )
        self.new_conv2[0].weight.data = conv2_weight
        self.new_conv2[0].bias.data = conv2_bias
        self.new_conv2[2].weight.data = self.conv2[4].weight.data
        self.conv2 = self.new_conv2
        # Flag
        self.compressed = True

    def forward(self, x, neighbors):
        """x: B x C_in x N. neighbors: B x K x N. Output: B x C_out x N"""
        if self.compressed:
            assert not self.training

        # Normalize input
        x = self.norm(x)

        # Point embedding
        point_emb = self.conv1(x)

        # Neighborhood embedding
        gather = []
        # Gather neighbors around each center point
        for ind_nn in range(
            1, neighbors.shape[1]
        ):  # Remove first neighbors which is the center point
            temp = neighbors[:, ind_nn : ind_nn + 1, :].expand(-1, x.shape[1], -1)
            gather.append(torch.gather(x, 2, temp).unsqueeze(-1))
        # Relative coordinates
        neigh_emb = torch.cat(gather, -1) - x.unsqueeze(-1)  # Size: (B x C x N) x K
        # Embedding
        neigh_emb = self.conv2(neigh_emb).max(-1)[0]

        # Merge both embeddings
        return self.final(torch.cat((point_emb, neigh_emb), dim=1))
