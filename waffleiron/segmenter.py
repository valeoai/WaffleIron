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

import warnings
import torch.nn as nn
from .backbone import WaffleIron
from .embedding import Embedding


class Segmenter(nn.Module):
    def __init__(
        self,
        input_channels,
        feat_channels,
        nb_class,
        depth,
        grid_shape,
        drop_path_prob=0,
        layer_norm=None, # To maintain comptability with previous version
        which_norm=None, # Preferred new option
    ):
        super().__init__()
        # Embedding layer
        self.embed = Embedding(input_channels, feat_channels)
        # WaffleIron backbone
        if which_norm is None and layer_norm is None:
            which_norm = "batchnorm"
        elif which_norm is None and layer_norm is not None:
            which_norm = "layernorm" if layer_norm else "batchnorm"
        else:
            if (which_norm == "layernorm" and not layer_norm) or \
                (which_norm == "batchnorm" and layer_norm):
                warnings.warn(
                    "Arguments which_norm={which_norm} and layer_norm={layer_norm} " +
                    "are in conflict. Creating the backbone using which_norm={which_norm}."
                )
        self.waffleiron = WaffleIron(feat_channels, depth, grid_shape, drop_path_prob, which_norm)
        # Classification layer
        self.classif = nn.Conv1d(feat_channels, nb_class, 1)

    def compress(self):
        self.embed.compress()
        self.waffleiron.compress()

    def forward(self, feats, cell_ind, occupied_cell, neighbors):
        tokens = self.embed(feats, neighbors)
        tokens = self.waffleiron(tokens, cell_ind, occupied_cell)
        return self.classif(tokens)
