import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_scatter import scatter
from torch_sparse import SparseTensor

from src.models.data_utils import MAX_ATOMIC_NUM
from src.models.unicrystalformer import (CartNet_layer)
from src.models.embedder import LatentConditionEmbedder
from src.models.basis_layers import (RadialBasis,
                                     CircularBasisLayer)

class CrystalDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        latent_dim=256,
        num_atom_types=MAX_ATOMIC_NUM,
        num_message_layers=6,
        cutoff=5.0,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cutoff = cutoff

        self.embedder = LatentConditionEmbedder(
            latent_dim=latent_dim,
            emb_size_atom=hidden_dim,
            emb_size_edge=hidden_dim,
            cutoff=self.cutoff,
        )

        self.layers = nn.ModuleList([
            CartNet_layer(hidden_dim, cutoff)
            for _ in range(num_message_layers)
        ])

        self.coord_head = nn.Linear(hidden_dim, 3)  # Predicts diff to coordinates
        self.type_head = nn.Linear(hidden_dim, num_atom_types)


    def forward(self, z, pred_frac_coords, pred_atom_types, num_atoms, lengths, angles, target_property):
        """
        args:
            z: (N_cryst, num_latent)
            frac_coords: (N_atoms, 3)
            atom_types: (N_atoms, ), need to use atomic number e.g. H = 1
            num_atoms: (N_cryst,)
            lengths: (N_cryst, 3)
            angles: (N_cryst, 3)
        """
        (
            h, # node features
            m, # edge features
            rbf3, # rbf for triplets
            cbf3, # angular embeddings
            rbf_h, # rbf global info
            rbf_out, # distant dependent output
            batch, # batch info
        ) = self.embedder(z, pred_frac_coords, pred_atom_types, num_atoms, lengths, angles, target_property)

        data = Data(
            x=node_feat,                  # node features
            edge_attr=edge_attr,          # edge features
            edge_index=edge_index,        # graph
            cart_dist=dist                # edge distance (used in CartNet_layer)
        )
            
        # Message passing
        for layer in self.layers:
            data = layer(data)

        # Outputs
        pred_cart_coord_diff = self.coord_head(data.x)
        pred_atom_type_logits = self.type_head(data.x)

        return pred_cart_coord_diff, pred_atom_type_logits
