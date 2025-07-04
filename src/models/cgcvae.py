import torch
from torch import nn

from src.models.encoder import UniCrystalEncoder
from src.models.decoder import GraphDecoder

class ConditionalGraphVAE(nn.Module):
    def __init__(self, encoder: UniCrystalEncoder, decoder: GraphDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data, condition):
        z, mu, logvar = self.encoder(data)
        node_logits, edge_logits = self.decoder(z, condition)
        return node_logits, edge_logits, mu, logvar
