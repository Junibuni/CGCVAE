import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_scatter import scatter

from src.models.data_utils import (build_radius_graph_with_pbc,
                                   cart_to_frac_coords, frac_to_cart_coords,
                                   min_distance_sqr_pbc)
from src.models.unicrystalformer import (CartNet_layer, RBFExpansion,
                                         SBFExpansion)


def build_triplets(edge_index, num_nodes):
    j, i = edge_index  # j -> i
    # For each edge j->i, find neighbors k of i
    value = torch.arange(j.size(0), device=j.device)  # edge IDs
    adj_t = [[] for _ in range(num_nodes)]
    for eid, (src, dst) in enumerate(zip(j.tolist(), i.tolist())):
        adj_t[dst].append((eid, src))

    triplet_indices = []
    for center in range(num_nodes):
        neighbors = adj_t[center]
        for eid_j, j in neighbors:
            for eid_k, k in neighbors:
                if j != k:
                    triplet_indices.append((eid_k, eid_j))
    if not triplet_indices:
        return torch.empty((2, 0), dtype=torch.long, device=j.device)
    return torch.tensor(triplet_indices, dtype=torch.long).t().contiguous()  # [2, T]


class CrystalDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        edge_dim=128,
        latent_dim=256,
        num_rbf=32,
        num_sbf=16,
        num_atom_types=119,
        num_message_layers=6,
        cutoff=5.0,
        use_sbf=True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.latent_dim = latent_dim
        self.cutoff = cutoff
        self.use_sbf = use_sbf

        self.atom_embed = nn.Embedding(num_atom_types, hidden_dim)
        self.latent_lin = nn.Linear(latent_dim, hidden_dim)

        self.rbf = nn.Sequential(
            RBFExpansion(vmin=0.0, vmax=cutoff, bins=num_rbf),
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.sbf = nn.Sequential(
            SBFExpansion(num_sbf=num_sbf),
            nn.Linear(num_sbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ) if use_sbf else None

        self.dir_lin = nn.Sequential(
            nn.Linear(3, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, hidden_dim)
        )

        self.layers = nn.ModuleList([
            CartNet_layer(hidden_dim, cutoff)
            for _ in range(num_message_layers)
        ])

        self.coord_head = nn.Linear(hidden_dim, 3)  # Predicts diff to coordinates
        self.type_head = nn.Linear(hidden_dim, num_atom_types)

    def forward(self, z, pred_frac_coords, pred_atom_types, num_atoms, lengths, angles):
        batch = torch.repeat_interleave(torch.arange(len(num_atoms), device=z.device), num_atoms)
        cart_coords = frac_to_cart_coords(pred_frac_coords, lengths, angles)

        # Build graph
        edge_index = build_radius_graph_with_pbc(cart_coords, self.cutoff, batch=batch, lengths=lengths, angles=angles)
        src, dst = edge_index

        # Compute edge vectors
        rel = cart_coords[src] - cart_coords[dst]
        rel = min_distance_sqr_pbc(rel, lengths, angles, batch[dst])  # PBC-aware displacement
        dist = torch.norm(rel, dim=-1)
        direction = F.normalize(rel, dim=-1)  # [E, 3]

        # Edge embedding
        dir_emb = self.dir_lin(direction)
        rbf_emb = self.rbf(dist)
        edge_attr = rbf_emb + dir_emb  # [E, hidden_dim]
        
        # Node embedding
        atom_types_clamped = torch.clamp(pred_atom_types, 0, self.atom_embed.num_embeddings - 1)
        node_feat = self.atom_embed(atom_types_clamped)
        node_feat += self.latent_lin(z[batch])  # Broadcast latent z to atoms

        # Build triplets (j→i←k)
        triplet_idx = build_triplets(edge_index, node_feat.size(0))

        j_edge = edge_index[0][triplet_idx[0]]
        k_edge = edge_index[0][triplet_idx[1]]
        center_atom = edge_index[1][triplet_idx[1]]

        vec1 = min_distance_sqr_pbc(cart_coords[k_edge] - cart_coords[center_atom], lengths, angles, batch[center_atom])
        vec2 = min_distance_sqr_pbc(cart_coords[j_edge] - cart_coords[center_atom], lengths, angles, batch[center_atom])

        vec1 = F.normalize(vec1, dim=-1)
        vec2 = F.normalize(vec2, dim=-1)
        cosine = (vec1 * vec2).sum(dim=-1).clamp(-1.0, 1.0)
        angle = torch.acos(cosine)
        
        if self.use_sbf and angle.numel() > 0:
            sbf_emb = self.sbf(angle)             # [T, hidden_dim]
            center_edge = triplet_idx[1]          # 중심 edge
            sbf_agg = scatter(sbf_emb, center_edge, dim=0, dim_size=edge_attr.size(0), reduce='mean')
            edge_attr = edge_attr + sbf_agg       # edge feature에 sbf 주입

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
