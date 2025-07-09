import os
from typing import Literal

import torch

from .cartnet import CartNet_layer, MatformerConv
from .model import UniCrystalFormer as _RawUniCrystalFormer
from .utils import RBFExpansion, SBFExpansion

_DEFAULT_CONFIG = {
    "v1": {
        "num_cart_layers": 3,
        "num_mat_layers": 3,
        "edge_features": 128,
        "hidden_dim": 128,
        "fc_features": 128,
        "output_features": 1,
        "num_heads": 4,
        "dropout": 0.1,
        "radius": 8.0,
        "use_att_fusion": False,
    },
    "v2": {
        "num_cart_layers": 3,
        "num_mat_layers": 3,
        "edge_features": 128,
        "hidden_dim": 128,
        "fc_features": 128,
        "output_features": 1,
        "num_heads": 4,
        "dropout": 0.1,
        "radius": 8.0,
        "use_att_fusion": True,
    }
}

class UniCrystalFormer(_RawUniCrystalFormer):
    def __init__(
        self,
        device: Literal["cpu", "gpu"] = "cpu",
        version: Literal["v1", "v2"] = "v1"
    ):
        super().__init__(**_DEFAULT_CONFIG.get(version))
        self.to(device)

        ckpt_path = os.path.join(os.path.dirname(__file__), "pretrained", f"{version}.pth")
        checkpoint = torch.load(ckpt_path, map_location=device)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.eval()
