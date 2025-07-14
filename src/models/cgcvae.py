from abc import ABC, abstractmethod
from typing import Any

import hydra
import numpy as np
import pytorch_lightning as pl
import torch
from torch import nn
from torch.nn import functional as F
from torch_scatter import scatter

from src.models.data_utils import (MAX_ATOMIC_NUM, cart_to_frac_coords,
                                   frac_to_cart_coords, min_distance_sqr_pbc)
from src.models.decoder import CrystalDecoder
from src.models.encoder import UniCrystalEncoder


def build_mlp(in_dim, hidden_dim, fc_num_layers, out_dim):
    mods = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
    for i in range(fc_num_layers-1):
        mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
    mods += [nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*mods)
    
class _BaseModel(pl.LightningModule, ABC):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        pass
    
    def configure_optimizers(self):
        opt = hydra.utils.instantiate(
            self.hparams.optim.optimizer, params=self.parameters(), _convert_="partial"
        )
        if not self.hparams.optim.use_lr_scheduler:
            return [opt]
        scheduler = hydra.utils.instantiate(
            self.hparams.optim.lr_scheduler, optimizer=opt
        )
        return {"optimizer": opt, "lr_scheduler": scheduler, "monitor": "val_loss"}
    
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
    
    #-------------------COMPUTE-LOSS-----------------------------
    def num_atom_loss(self, pred_num_atoms, batch):
        return F.cross_entropy(pred_num_atoms, batch.num_atoms)

    def property_loss(self, z, batch):
        return F.mse_loss(self.fc_property(z), batch.y)

    def lattice_loss(self, pred_lengths_and_angles, batch):
        self.lattice_scaler.match_device(pred_lengths_and_angles)
        if self.hparams.data.lattice_scale_method == 'scale_length':
            target_lengths = batch.lengths / \
                batch.num_atoms.view(-1, 1).float()**(1/3)
        target_lengths_and_angles = torch.cat(
            [target_lengths, batch.angles], dim=-1)
        target_lengths_and_angles = self.lattice_scaler.transform(
            target_lengths_and_angles)
        return F.mse_loss(pred_lengths_and_angles, target_lengths_and_angles)

    def composition_loss(self, pred_composition_per_atom, target_atom_types, batch):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(pred_composition_per_atom,
                               target_atom_types, reduction='none')
        return scatter(loss, batch.batch, reduce='mean').mean()

    def coord_loss(self, pred_cart_coord_diff, noisy_frac_coords,
                   used_sigmas_per_atom, batch):
        noisy_cart_coords = frac_to_cart_coords(
            noisy_frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        target_cart_coords = frac_to_cart_coords(
            batch.frac_coords, batch.lengths, batch.angles, batch.num_atoms)
        _, target_cart_coord_diff = min_distance_sqr_pbc(
            target_cart_coords, noisy_cart_coords, batch.lengths, batch.angles,
            batch.num_atoms, self.device, return_vector=True)

        target_cart_coord_diff = target_cart_coord_diff / \
            used_sigmas_per_atom[:, None]**2
        pred_cart_coord_diff = pred_cart_coord_diff / \
            used_sigmas_per_atom[:, None]

        loss_per_atom = torch.sum(
            (target_cart_coord_diff - pred_cart_coord_diff)**2, dim=1)

        loss_per_atom = 0.5 * loss_per_atom * used_sigmas_per_atom**2
        return scatter(loss_per_atom, batch.batch, reduce='mean').mean()

    def type_loss(self, pred_atom_types, target_atom_types,
                  used_type_sigmas_per_atom, batch):
        target_atom_types = target_atom_types - 1
        loss = F.cross_entropy(
            pred_atom_types, target_atom_types, reduction='none')
        # rescale loss according to noise
        loss = loss / used_type_sigmas_per_atom
        return scatter(loss, batch.batch, reduce='mean').mean()

    def kld_loss(self, mu, log_var):
        kld_loss = torch.mean(
            -0.5 * torch.sum(1 + log_var - mu**2 - log_var.exp(), dim=1), dim=0)
        return kld_loss
    
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
        z, mu, logvar = self.encoder(data, condition)
        
        (pred_num_atoms, 
         pred_lengths_and_angles, 
         pred_lengths, 
         pred_angles,
         pred_composition_per_atom
         ) = self.decode_stats(
            z, data.num_atoms, data.lengths, data.angles, teacher_forcing
            )
        
        noise_level = torch.randint(0, self.sigmas.size(0),
                                    (data.num_atoms.size(0),),
                                    device=self.device)
        used_sigmas_per_atom = self.sigmas[noise_level].repeat_interleave(
            data.num_atoms, dim=0)

        type_noise_level = torch.randint(0, self.type_sigmas.size(0),
                                         (data.num_atoms.size(0),),
                                         device=self.device)
        used_type_sigmas_per_atom = (
            self.type_sigmas[type_noise_level].repeat_interleave(
                data.num_atoms, dim=0))

        # add noise to atom types and sample atom types.
        pred_composition_probs = F.softmax(
            pred_composition_per_atom.detach(), dim=-1)
        atom_type_probs = (
            F.one_hot(data.atom_types - 1, num_classes=MAX_ATOMIC_NUM) +
            pred_composition_probs * used_type_sigmas_per_atom[:, None])
        rand_atom_types = torch.multinomial(
            atom_type_probs, num_samples=1).squeeze(1) + 1

        # add noise to the cart coords
        cart_noises_per_atom = (
            torch.randn_like(data.frac_coords) *
            used_sigmas_per_atom[:, None])
        cart_coords = frac_to_cart_coords(
            data.frac_coords, pred_lengths, pred_angles, data.num_atoms)
        cart_coords = cart_coords + cart_noises_per_atom
        noisy_frac_coords = cart_to_frac_coords(
            cart_coords, pred_lengths, pred_angles, data.num_atoms)
        
        pred_cart_coord_diff, pred_atom_types = self.decoder(z, 
                                                noisy_frac_coords, 
                                                rand_atom_types, 
                                                data.num_atoms, 
                                                pred_lengths, 
                                                pred_angles, 
                                                condition)
        
        # Loss
        num_atom_loss = self.num_atom_loss(pred_num_atoms, data)
        lattice_loss = self.lattice_loss(pred_lengths_and_angles, data)
        composition_loss = self.composition_loss(
            pred_composition_per_atom, data.atom_types, data)
        coord_loss = self.coord_loss(
            pred_cart_coord_diff, noisy_frac_coords, used_sigmas_per_atom, data)
        type_loss = self.type_loss(pred_atom_types, data.atom_types,
                                   used_type_sigmas_per_atom, data)

        kld_loss = self.kld_loss(mu, logvar)

        if self.hparams.predict_property:
            property_loss = self.property_loss(z, data)
        else:
            property_loss = 0.

        return {
            'num_atom_loss': num_atom_loss,
            'lattice_loss': lattice_loss,
            'composition_loss': composition_loss,
            'coord_loss': coord_loss,
            'type_loss': type_loss,
            'kld_loss': kld_loss,
            'property_loss': property_loss,
            'pred_num_atoms': pred_num_atoms,
            'pred_lengths_and_angles': pred_lengths_and_angles,
            'pred_lengths': pred_lengths,
            'pred_angles': pred_angles,
            'pred_cart_coord_diff': pred_cart_coord_diff,
            'pred_atom_types': pred_atom_types,
            'pred_composition_per_atom': pred_composition_per_atom,
            'target_frac_coords': data.frac_coords,
            'target_atom_types': data.atom_types,
            'rand_frac_coords': noisy_frac_coords,
            'rand_atom_types': rand_atom_types,
            'z': z,
        }
        
    def compute_stats(self, batch, outputs, prefix):
        num_atom_loss = outputs['num_atom_loss']
        lattice_loss = outputs['lattice_loss']
        coord_loss = outputs['coord_loss']
        type_loss = outputs['type_loss']
        kld_loss = outputs['kld_loss']
        composition_loss = outputs['composition_loss']
        property_loss = outputs['property_loss']

        loss = (
            self.hparams.cost_natom * num_atom_loss +
            self.hparams.cost_lattice * lattice_loss +
            self.hparams.cost_coord * coord_loss +
            self.hparams.cost_type * type_loss +
            self.hparams.beta * kld_loss +
            self.hparams.cost_composition * composition_loss +
            self.hparams.cost_property * property_loss)
        
        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_natom_loss': num_atom_loss,
            f'{prefix}_lattice_loss': lattice_loss,
            f'{prefix}_coord_loss': coord_loss,
            f'{prefix}_type_loss': type_loss,
            f'{prefix}_kld_loss': kld_loss,
            f'{prefix}_composition_loss': composition_loss,
        }
        return log_dict, loss
       
    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        teacher_forcing = (
            self.current_epoch <= self.hparams.teacher_forcing_max_epoch)
        outputs = self(batch, teacher_forcing, training=True)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='train')
        self.log_dict(
            log_dict,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='val')
        self.log_dict(
            log_dict,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        return loss

    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        outputs = self(batch, teacher_forcing=False, training=False)
        log_dict, loss = self.compute_stats(batch, outputs, prefix='test')
        self.log_dict(
            log_dict,
        )
        return loss