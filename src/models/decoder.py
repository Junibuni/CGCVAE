import torch
from torch import nn

class GraphDecoder(nn.Module):
    def __init__(self, latent_dim, cond_dim, hidden_dim, ...):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + cond_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # self.node_predictor = nn.Linear(hidden_dim, num_atom_types)
        # self.edge_predictor = nn.Linear(hidden_dim, num_edge_types)

    def forward(self, z, condition):
        zy = torch.cat([z, condition], dim=-1)
        h = self.fc(zy)
        node_logits = self.node_predictor(h)
        edge_logits = self.edge_predictor(h)
        return node_logits, edge_logits
