import torch
from torch import nn

class UniCrystalEncoder(nn.Module):
    def __init__(self, backbone: UniCrystalFormer, latent_dim: int):
        super().__init__()
        self.backbone = backbone
        self.backbone.fc_out = nn.Identity()  # 원래 FC 제거
        self.fc_mu = nn.Linear(2 * backbone.readout.out_channels, latent_dim)
        self.fc_logvar = nn.Linear(2 * backbone.readout.out_channels, latent_dim)

    def forward(self, data):
        emb = self.backbone(data)
        mu = self.fc_mu(emb)
        logvar = self.fc_logvar(emb)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z, mu, logvar
