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

if torch.__version__ == "1.11.0":
    WI_SCATTER_REDUCE = False
elif torch.__version__ == "2.2.2":
    WI_SCATTER_REDUCE = True
else:
    # Try torch.scatter_reduce for other non tested versions of pytorch
    WI_SCATTER_REDUCE = True
WI_SCATTER_REDUCE = bool(int(os.environ.get(
    "WI_SCATTER_REDUCE", WI_SCATTER_REDUCE
)))
if WI_SCATTER_REDUCE:
    print("Using torch.scatter_reduce for 3D to 2D projection.")
else:
    print("Using torch.sparse_coo_tensor for 3D to 2D projection.")

from .backbone import WaffleIron
from .segmenter import Segmenter


__all__ = [WaffleIron, Segmenter]
__version__ = "0.3.0"
