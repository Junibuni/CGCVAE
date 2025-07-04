from typing import Literal

import torch
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GraphNorm, Set2Set

from src.models.unicrystalformer.cartnet import CartNet_layer
from src.models.unicrystalformer.utils import RBFExpansion
from src.models.unicrystalformer.matformer import MatformerConv

class AtomEncoder(nn.Module):
    def __init__(self, num_atom_types, hidden_dim, dropout=0.1):
        super().__init__()
        self.atom_embedding = nn.Embedding(num_atom_types, hidden_dim)
        self.megnet_proj = nn.Linear(16, hidden_dim)
        self.gate_layer = nn.Linear(2 * hidden_dim, hidden_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, atomic_numbers, megnet_embed):
        emb = self.atom_embedding(atomic_numbers)
        megnet = self.megnet_proj(megnet_embed)
        combined = torch.cat([emb, megnet], dim=-1)
        gate = torch.sigmoid(self.gate_layer(combined))
        fused = gate * emb + (1 - gate) * megnet
        return self.out_proj(fused)

class CartNetBlock(nn.Module):
    def __init__(self, num_layers, hidden_dim, radius, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([CartNet_layer(dim_in=hidden_dim, radius=radius) for _ in range(num_layers)])
        self.norms = nn.ModuleList([GraphNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        for layer, norm in zip(self.layers, self.norms):
            batch = layer(batch)
            h_new = norm(batch.x, batch.batch)
            h = h + self.dropout(h_new)
            batch.x = h
        return batch

class MatformerBlock(nn.Module):
    def __init__(self, num_layers, hidden_dim, num_heads, edge_dim, dropout=0.1):
        super().__init__()
        self.convs = nn.ModuleList([MatformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            heads=num_heads,
            edge_dim=edge_dim,
            concat=False,
            beta=True
        ) for _ in range(num_layers)])
        self.ffns = nn.ModuleList([nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        ) for _ in range(num_layers)])
        self.norms = nn.ModuleList([GraphNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch):
        h = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        batch_idx = batch.batch
        for conv, ffn, norm in zip(self.convs, self.ffns, self.norms):
            h_new = conv(h, edge_index, edge_attr)
            h_new = ffn(h_new)
            h_new = norm(h_new, batch_idx)
            h = h + self.dropout(h_new)
        batch.x = h
        return batch

class AttentionFusionMixer(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score_mlp = nn.Linear(hidden_dim, 1)

    def forward(self, h_A, h_B):
        score_A = self.score_mlp(h_A)
        score_B = self.score_mlp(h_B)
        weights = torch.softmax(torch.cat([score_A, score_B], dim=1), dim=1)
        w_A = weights[:, 0].unsqueeze(1)
        w_B = weights[:, 1].unsqueeze(1)
        h_out = w_A * h_A + w_B * h_B
        return h_out


class UniCrystalFormer(nn.Module):
    def __init__(self,
                 num_cart_layers: int = 3,
                 num_mat_layers: int = 3,
                 edge_features: int = 128,    # RBF expansion size (bins)
                 hidden_dim: int = 128,
                 fc_features: int = 128,
                 output_features: int = 1,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 radius: float = 8.0,
                 use_att_fusion: bool = False):
        super().__init__()

        num_atom_types = 119
        self.atom_encoder = AtomEncoder(num_atom_types, hidden_dim, dropout)

        self.rbf = nn.Sequential(
            RBFExpansion(vmin=0.0, vmax=radius, bins=edge_features),
            nn.Linear(edge_features, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        self.cart_layers = nn.ModuleList([
            CartNet_layer(dim_in=hidden_dim, radius=radius) 
            for _ in range(num_cart_layers)
        ])
        self.mat_layers = nn.ModuleList([
            MatformerConv(in_channels=hidden_dim, out_channels=hidden_dim,
                          heads=num_heads, edge_dim=edge_features,
                          concat=False, beta=True)
            for _ in range(num_mat_layers)
        ])

        self.num_layers = max(num_cart_layers, num_mat_layers)
        self.norm_attn = nn.ModuleList([GraphNorm(hidden_dim) for _ in range(self.num_layers)])
        self.ffn_layers = nn.ModuleList([nn.Sequential(
                                            nn.Linear(hidden_dim, 2*hidden_dim),
                                            nn.SiLU(),
                                            nn.Linear(2*hidden_dim, hidden_dim)
                                         ) for _ in range(self.num_layers)])
        self.norm_ffn = nn.ModuleList([GraphNorm(hidden_dim) for _ in range(self.num_layers)])
        self.dropout = nn.Dropout(dropout)

        self.readout = Set2Set(hidden_dim, processing_steps=3)
        self.fc_out = nn.Sequential(
            nn.Linear(2 * hidden_dim, fc_features),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(fc_features, output_features)
        )

        self.use_attention_fusion = use_att_fusion
        if self.use_attention_fusion:
            self.attn_fuser = AttentionFusionMixer(hidden_dim)

        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def forward(self, data):
        h0 = self.atom_encoder(data.x, data.atom_megnet_embed)          # [N, hidden_dim]

        edge_vec = data.edge_attr                                       # [E, 3] if containing displacement vectors
        edge_dist = torch.norm(edge_vec, dim=1)                         # [E]
        edge_attr = self.rbf(edge_dist)                                 # [E, hidden_dim]

        batch_idx = data.batch                                          # [N]

        h = h0
        edge_index = data.edge_index
        cart_dist = data.cart_dist

        data.x = h
        data.edge_attr = edge_attr

        for i in range(self.num_layers):
            if i < len(self.cart_layers):
                local_data = Data(x=h, edge_index=edge_index, edge_attr=edge_attr, batch=batch_idx, cart_dist=cart_dist)
                local_data = self.cart_layers[i](local_data)
                h_local = local_data.x # [N, hidden_dim]
            else:
                h_local = None

            if i < len(self.mat_layers):
                h_global = self.mat_layers[i](h, edge_index, edge_attr) # [N, hidden_dim]
            else:
                h_global = None

            if h_local is not None and h_global is not None:
                if self.use_attention_fusion:
                    h_comb = self.attn_fuser(h_local, h_global)
                else:
                    h_comb = h_local + h_global
            elif h_local is not None:
                h_comb = h_local
            else:
                h_comb = h_global

            h_comb_norm = self.norm_attn[i](h_comb, batch_idx)
            h_attn = h + self.dropout(h_comb_norm)

            h_ffn = self.ffn_layers[i](h_attn)
            h_ffn_norm = self.norm_ffn[i](h_ffn, batch_idx)
            h = h_attn + self.dropout(h_ffn_norm)

            data.x = h

        graph_emb = self.readout(h, batch_idx) # [batch_graphs, 2*hidden_dim]
        out = self.fc_out(graph_emb)
        return out

    @classmethod
    def from_config(cls, config: dict):
        """Factory method to create model from a configuration dictionary."""
        return cls(
            num_cart_layers = config.get("num_cart_layers", 3),
            num_mat_layers  = config.get("num_mat_layers", 3),
            edge_features   = config.get("edge_features", 128),
            hidden_dim      = config.get("hidden_dim", 128),
            fc_features     = config.get("fc_features", 128),
            output_features = config.get("output_features", 1),
            num_heads       = config.get("num_heads", 4),
            dropout         = config.get("dropout", 0.1),
            radius          = config.get("radius", 8.0),
            use_att_fusion  = config.get("use_att_fusion", False)
        )
