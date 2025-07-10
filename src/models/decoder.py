import math

import torch
import torch.nn as nn
from torch_geometric.data import Data
from torch_geometric.nn import GraphNorm

from src.models.data_utils import MAX_ATOMIC_NUM
from src.models.embedder import FiLMLayer, LatentConditionEmbedder
from src.models.unicrystalformer import CartNet_layer, MatformerConv

class EdgeFeatureModulator(nn.Module):
    def __init__(self, 
                 edge_dim: int, 
                 rbf_dim: int, 
                 rbf3_dim: int, 
                 cbf3_dim: int,
                 hidden_dim: int):
        super().__init__()
        
        # 2-body radial
        self.rbf_proj = nn.Linear(rbf_dim, hidden_dim, bias=False)
        
        # 3-body angular terms
        self.rbf3_proj = nn.Linear(rbf3_dim, hidden_dim, bias=False)
        self.cbf3_proj = nn.Linear(cbf3_dim, hidden_dim, bias=False)

        # Combine all features
        self.fuse = nn.Sequential(
            nn.Linear(edge_dim + 3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.reset_parameters()
    
    def reset_parameters(self):
        self.rbf_proj.reset_parameters()
        self.rbf3_proj.reset_parameters()
        self.cbf3_proj.reset_parameters()

        for layer in self.fuse:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
            
    def angular_pooling(self, rbf_W1, sph):
        # rbf_W1: [N_edges, H, L]
        # sph:    [N_edges, K, L]
        # Output: [N_edges, H]
        return torch.einsum("eil,ekl->eik", rbf_W1, sph).sum(dim=1)  # sum over K neighbors


    def forward(self, edge_attr, rbf_h, rbf3, cbf3, idx_s, idx_t):
        """
        edge_attr: [N_edges, edge_dim]
        rbf_h:     [N_edges, rbf_dim]
        rbf3:      [N_triplets, rbf3_dim]
        cbf3:      [N_triplets, cbf3_dim]
        idx_s, idx_t: [N_edges] (edge indices)
        """
        e_rbf = self.rbf_proj(rbf_h)          # [N_edges, H]

        # Aggregate triplet info to edges (mean-pooling for simplicity)
        N_edges = edge_attr.size(0)
        device = edge_attr.device

        rbf3_proj = self.rbf3_proj(rbf3)      # [N_triplets, H]
        rbf_W1, sph = cbf3
        cbf3_vector = self.angular_pooling(rbf_W1, sph)  # [N_edges, emb_size_interm]
        cbf3_proj = self.cbf3_proj(cbf3_vector)     # [N_edges, hidden_dim]

        rbf3_sum = torch.zeros(N_edges, rbf3_proj.size(-1), device=device)
        cbf3_sum = torch.zeros_like(rbf3_sum)
        counts = torch.zeros(N_edges, 1, device=device)

        # `idx_s` indexes into edges
        rbf3_sum.index_add_(0, idx_s, rbf3_proj)
        cbf3_sum.index_add_(0, idx_s, cbf3_proj)
        counts.index_add_(0, idx_s, torch.ones_like(counts))

        rbf3_avg = rbf3_sum / (counts + 1e-6)
        cbf3_avg = cbf3_sum / (counts + 1e-6)

        # Concatenate all
        edge_feat = torch.cat([edge_attr, e_rbf, rbf3_avg, cbf3_avg], dim=-1)  # [N_edges, D_total]
        edge_attr_enriched = self.fuse(edge_feat)                              # [N_edges, edge_dim]

        return edge_attr_enriched

class CrystalDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        latent_dim=256,
        rbf_dim=16,
        cbf_dim=16,
        num_atom_types=MAX_ATOMIC_NUM,
        num_message_layers=3,
        cutoff=5.0,
        dropout=0.1,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.cutoff = cutoff
        self.edge_feature_dim = hidden_dim
        self.num_layers = num_message_layers

        self.embedder = LatentConditionEmbedder(
            latent_dim=latent_dim,
            emb_size_atom=hidden_dim,
            emb_size_edge=hidden_dim,
            emb_size_rbf=rbf_dim,
            emb_size_cbf=cbf_dim,
            cutoff=self.cutoff,
        )
        
        self.edge_modulator = EdgeFeatureModulator(
            edge_dim=hidden_dim,
            rbf_dim=rbf_dim,
            rbf3_dim=rbf_dim,
            cbf3_dim=cbf_dim,
            hidden_dim=hidden_dim
        )
        
        self.cart_layers = nn.ModuleList([
            CartNet_layer(dim_in=hidden_dim, radius=self.cutoff) 
            for _ in range(self.num_layers)
        ])
        self.mat_layers = nn.ModuleList([
            MatformerConv(in_channels=hidden_dim, out_channels=hidden_dim,
                          heads=4, edge_dim=self.edge_feature_dim,
                          concat=False, beta=True)
            for _ in range(self.num_layers)
        ])
        
        self.norm_attn = nn.ModuleList([GraphNorm(hidden_dim) for _ in range(self.num_layers)])
        self.ffn_layers = nn.ModuleList([nn.Sequential(
                                    nn.Linear(hidden_dim, 2*hidden_dim),
                                    nn.SiLU(),
                                    nn.Linear(2*hidden_dim, hidden_dim)
                                    ) for _ in range(self.num_layers)])
        self.norm_ffn = nn.ModuleList([GraphNorm(hidden_dim) for _ in range(self.num_layers)])

        self.coord_head = nn.Linear(hidden_dim, 3)  # Predicts diff to coordinates
        self.type_head = nn.Linear(hidden_dim, num_atom_types)
        
        self.coord_head.reset_parameters()
        self.type_head.reset_parameters()
        
        self.film_layer = FiLMLayer(hidden_dim)
        self.dropout = nn.Dropout(p=dropout)

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
        condition = target_property.unsqueeze(1)  # (N_cryst, 1)
        
        (
            h, # node features
            m, # edge features
            rbf3, # rbf for triplets
            cbf3, # angular embeddings
            rbf_h, # rbf global info
            rbf_out, # distant dependent output
            batch, # batch info
            edge_index, D_st, V_st, idx_s, idx_t
        ) = self.embedder(z, pred_frac_coords, pred_atom_types, num_atoms, lengths, angles, target_property)

        edge_attr = self.edge_modulator(
            m, rbf_h, rbf3, cbf3, idx_s, idx_t
        )  # [N_edges, hidden_dim]
            
        # Message passing
        for i in range(self.num_layers):
            local_data = Data(
                x=h,                          # node features
                edge_attr=edge_attr,          # edge features
                edge_index=edge_index,        # graph
                cart_dist=D_st,               # edge distances
                batch=batch
            )           
            local_data = self.cart_layers[i](local_data)
            h_local = local_data.x # [N, hidden_dim]
            
            h_global = self.mat_layers[i](h, edge_index, edge_attr) # [N, hidden_dim]
            
            h_comb = h_local + h_global
            h_comb_norm = self.norm_attn[i](h_comb, batch)
            h_attn = h + self.dropout(h_comb_norm)
            
            h_ffn = self.ffn_layers[i](h_attn)
            h_ffn_norm = self.norm_ffn[i](h_ffn, batch)
            h = h_attn + self.dropout(h_ffn_norm)
            
            h = self.film_layer(h, condition, num_atoms)



        # Outputs
        pred_cart_coord_diff = self.coord_head(h)
        pred_atom_type_logits = self.type_head(h)

        return pred_cart_coord_diff, pred_atom_type_logits
