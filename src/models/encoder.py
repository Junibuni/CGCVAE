import torch
from torch import nn

from src.models.unicrystalformer import UniCrystalFormer
from src.models.embedder import FiLMLayer

class UniCrystalEncoder(nn.Module):
    def __init__(self, device: str, version: str, latent_dim: int):
        super().__init__()
        self.backbone = UniCrystalFormer(
            device=device,
            version=version
        )
        self.backbone.fc_out = nn.Identity()
        
        self.embed_dim = 2 * self.backbone.readout.out_channels
        
        self.fc_mu = nn.Linear(self.embed_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.embed_dim, latent_dim)
        self.film_layer = FiLMLayer(self.embed_dim)

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True
        
    def forward(self, data, target_property):
        emb = self.backbone(data)
        
        target_property = target_property.unsqueeze(1)  # [N_cryst, 1]
        emb = self.film_layer(emb, target_property, num_atoms=torch.ones_like(target_property, dtype=torch.long))
        
        mu = self.fc_mu(emb)
        logvar = self.fc_logvar(emb)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z, mu, logvar
