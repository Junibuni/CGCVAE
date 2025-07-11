from abc import ABC, abstractmethod

import torch
from torch import nn
import numpy as np

from src.models.decoder import CrystalDecoder
from src.models.encoder import UniCrystalEncoder

def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)

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
    def __init__(
        self,
        # Encoder
        encoder_version='v1',
        # Decoder
        hidden_dim=128,
        latent_dim=256,
        rbf_dim=16,
        cbf_dim=16,
        num_message_layers=3,
        cutoff=5.0,
        dropout=0.1,
    ):
        super().__init__()
        self.encoder = UniCrystalEncoder(
            encoder_version
        )
        self.decoder = CrystalDecoder(
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
            rbf_dim=rbf_dim,
            cbf_dim=cbf_dim,
            num_message_layers=num_message_layers,
            cutoff=cutoff,
            dropout=dropout,
        )
        
        self.fc_num_atoms = build_mlp(self.hparams.latent_dim, self.hparams.hidden_dim,
                                    self.hparams.fc_num_layers, self.hparams.max_atoms+1)
        
        sigmas = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.sigma_begin),
            np.log(self.hparams.sigma_end),
            self.hparams.num_noise_level)), dtype=torch.float32)

        self.sigmas = nn.Parameter(sigmas, requires_grad=False)

        type_sigmas = torch.tensor(np.exp(np.linspace(
            np.log(self.hparams.type_sigma_begin),
            np.log(self.hparams.type_sigma_end),
            self.hparams.num_noise_level)), dtype=torch.float32)

        self.type_sigmas = nn.Parameter(type_sigmas, requires_grad=False)

    def predict_num_atoms(self, z):
        return self.fc_num_atoms(z)
    
    def predict_lattice(self, z, num_atoms):
        self.lattice_scaler.match_device(z)
        pred_lengths_and_angles = self.fc_lattice(z)  # (N, 6)
        scaled_preds = self.lattice_scaler.inverse_transform(
            pred_lengths_and_angles)
        pred_lengths = scaled_preds[:, :3]
        pred_angles = scaled_preds[:, 3:]
        if self.hparams.data.lattice_scale_method == 'scale_length':
            pred_lengths = pred_lengths * num_atoms.view(-1, 1).float()**(1/3)
        return pred_lengths_and_angles, pred_lengths, pred_angles
    
    def predict_composition(self, z, num_atoms):
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        pred_composition_per_atom = self.fc_composition(z_per_atom)
        return pred_composition_per_atom
    
    def decode_stats(self, z, gt_num_atoms=None, gt_lengths=None, gt_angles=None, teacher_forcing=False):
        if gt_num_atoms is not None:
            num_atoms = self.predict_num_atoms(z)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, gt_num_atoms))
            composition_per_atom = self.predict_composition(z, gt_num_atoms)
            if self.hparams.teacher_forcing_lattice and teacher_forcing:
                lengths = gt_lengths
                angles = gt_angles
        else:
            num_atoms = self.predict_num_atoms(z).argmax(dim=-1)
            lengths_and_angles, lengths, angles = (
                self.predict_lattice(z, num_atoms))
            composition_per_atom = self.predict_composition(z, num_atoms)
        return num_atoms, lengths_and_angles, lengths, angles, composition_per_atom
    
    def forward(self, data, condition, teacher_forcing):
        z, mu, logvar = self.encoder(data)
        
        (pred_num_atoms, 
         pred_lengths_and_angles, 
         pred_lengths, 
         pred_angles,
         pred_composition_per_atom
         ) = self.decode_stats(
            z, data.num_atoms, data.lengths, data.angles, teacher_forcing
            )
        
        
        node_logits, edge_logits = self.decoder(z, condition)
        return node_logits, edge_logits, mu, logvar
