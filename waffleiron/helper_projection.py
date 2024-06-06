# Copyright 2024 - Valeo Comfort and Driving Assistance - Gilles Puy @ valeo.ai
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
import numpy as np
from torch import autocast


def projection_3d_to_2d_scatter_reduce(feat, sp_mat, B, C, H, W):

    residual = torch.zeros(
        (B, C, H * W), 
        device=feat.device, 
        dtype=feat.dtype
    )
    residual.scatter_reduce_(
        2, 
        sp_mat["inflate"], 
        feat, 
        "mean", 
        include_self=False,
    )
    
    return residual


def projection_3d_to_2d_sparse_matrix(feat, sp_mat, *args, **kwargs):

    with autocast("cuda", enabled=False):
        residual = torch.bmm(
            sp_mat["flatten"], feat.transpose(1, 2).float()
        ).transpose(1, 2)
    
    return residual


def get_all_projections_scatter_reduce(
    cell_ind, nb_feat, *args, **kwargs
):
    sp_mat = [
        {"inflate": cell_ind[:, i:i+1].expand(-1, nb_feat, -1)}
        for i in range(cell_ind.shape[1])
    ]
    return sp_mat


def get_all_projections_sparse_matrices(
    cell_ind, nb_feat, batch_size, num_points, occupied_cell, device, grids_shape
):
    point_ind = (
        torch.arange(num_points, device=device)
        .unsqueeze(0)
        .expand(batch_size, -1)
        .reshape(1, -1)
    )
    batch_ind = (
        torch.arange(batch_size, device=device)
        .unsqueeze(1)
        .expand(-1, num_points)
        .reshape(1, -1)
    )
    non_zeros_ind = []
    for i in range(cell_ind.shape[1]):
        non_zeros_ind.append(
            torch.cat((batch_ind, cell_ind[:, i].reshape(1, -1), point_ind), axis=0)
        )
    sp_mat = [
        build_proj_matrix(
            id,
            occupied_cell,
            batch_size,
            np.prod(sh),
            cell_ind[:, i],
            nb_feat,
        )
        for i, (id, sh) in enumerate(zip(non_zeros_ind, grids_shape))
    ]

    return sp_mat


def build_proj_matrix(
    indices_non_zeros, occupied_cell, batch_size, num_2d_cells, inflate_ind, channels,
):
    """Build sparse matrix for 3D to 2D projection"""
    num_points = indices_non_zeros.shape[1] // batch_size
    matrix_shape = (batch_size, num_2d_cells, num_points)

    # Sparse projection matrix for Inflate step
    inflate = torch.sparse_coo_tensor(
        indices_non_zeros, occupied_cell.reshape(-1), matrix_shape
    ).transpose(1, 2)
    inflate_ind = inflate_ind.unsqueeze(1).expand(-1, channels, -1)

    # Count number of points in each cells (used in flatten step)
    with autocast("cuda", enabled=False):
        num_points_per_cells = torch.bmm(
            inflate, torch.bmm(inflate.transpose(1, 2), occupied_cell.unsqueeze(-1))
        )

    # Sparse projection matrix for Flatten step (projection & average in each 2d cells)
    weight_per_point = 1.0 / (num_points_per_cells.reshape(-1) + 1e-6)
    weight_per_point *= occupied_cell.reshape(-1)
    flatten = torch.sparse_coo_tensor(indices_non_zeros, weight_per_point, matrix_shape)

    return {"flatten": flatten, "inflate": inflate_ind}