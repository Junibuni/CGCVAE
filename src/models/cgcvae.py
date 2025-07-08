import torch
from torch import nn
from abc import ABC, abstractmethod

from src.models.decoder import CrystalDecoder
from src.models.encoder import UniCrystalEncoder

class _BaseModel(nn.Module, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    @property
    def num_params(self):
        return sum(p.numel() for p in self.parameters())

    @property
    def model_size_bytes(self):
        return sum(p.numel() * p.element_size() for p in self.parameters())

    @property
    def model_size_mb(self):
        return self.model_size_bytes / (1024 ** 2)

class ConditionalGraphVAE(_BaseModel):
    def __init__(self, encoder: UniCrystalEncoder, decoder: CrystalDecoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, data, condition):
        z, mu, logvar = self.encoder(data)
        node_logits, edge_logits = self.decoder(z, condition)
        return node_logits, edge_logits, mu, logvar
