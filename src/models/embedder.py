import math

import numpy as np
import torch
import torch.nn as nn
from torch_sparse import SparseTensor

from src.models.basis_layers import CircularBasisLayer, RadialBasis
from src.models.data_utils import (MAX_ATOMIC_NUM, frac_to_cart_coords,
                                   get_pbc_distances, inner_product_normalized,
                                   radius_graph_pbc, ragged_range,
                                   repeat_blocks)


class AtomEmbedding(torch.nn.Module):
    """
    Initial atom embeddings based on the atom type

    Parameters
    ----------
        emb_size: int
            Atom embeddings size
    """

    def __init__(self, emb_size):
        super().__init__()
        self.emb_size = emb_size

        self.embeddings = torch.nn.Embedding(MAX_ATOMIC_NUM, emb_size)

        torch.nn.init.uniform_(
            self.embeddings.weight, a=-np.sqrt(3), b=np.sqrt(3)
        )

    def forward(self, Z):
        """
        Returns
        -------
            h: torch.Tensor, shape=(nAtoms, emb_size)
                Atom embeddings.
        """
        h = self.embeddings(Z - 1)  # -1 because Z.min()=1 (==Hydrogen)
        return h


class EdgeEmbedding(torch.nn.Module):
    """
    Edge embedding based on the concatenation of atom embeddings and subsequent dense layer.

    Parameters
    ----------
        emb_size: int
            Embedding size after the dense layer.
        activation: str
            Activation function used in the dense layer.
    """

    def __init__(
        self,
        atom_features,
        edge_features,
        out_features,
    ):
        super().__init__()
        in_features = 2 * atom_features + edge_features
        self.dense = nn.Linear(
            in_features, out_features, bias=False
        )
        self.silu = nn.SiLU()
        self.scale_factor_silu = 1 / 0.6

    def reset_parameters(self):
        self.dense.reset_parameters()
    
    def forward(
        self,
        h,
        m_rbf,
        idx_s,
        idx_t,
    ):
        """

        Arguments
        ---------
        h
        m_rbf: shape (nEdges, nFeatures)
            in embedding block: m_rbf = rbf ; In interaction block: m_rbf = m_st
        idx_s
        idx_t

        Returns
        -------
            m_st: torch.Tensor, shape=(nEdges, emb_size)
                Edge embeddings.
        """
        h_s = h[idx_s]  # shape=(nEdges, emb_size)
        h_t = h[idx_t]  # shape=(nEdges, emb_size)

        m_st = torch.cat(
            [h_s, h_t, m_rbf], dim=-1
        )  # (nEdges, 2*emb_size+nFeatures)
        m_st = self.dense(m_st)  # (nEdges, emb_size)
        m_st = self.silu(m_st) * self.scale_factor_silu
        return m_st
    
class EfficientInteractionDownProjection(torch.nn.Module):
    """
    Down projection in the efficient reformulation.

    Parameters
    ----------
        emb_size_interm: int
            Intermediate embedding size (down-projection size).
        kernel_initializer: callable
            Initializer of the weight matrix.
    """

    def __init__(
        self,
        num_spherical: int,
        num_radial: int,
        emb_size_interm: int,
    ):
        super().__init__()

        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.emb_size_interm = emb_size_interm

        self.reset_parameters()

    def reset_parameters(self):
        self.weight = torch.nn.Parameter(
            torch.empty(
                (self.num_spherical, self.num_radial, self.emb_size_interm)
            ),
            requires_grad=True,
        )
        self.he_orthogonal_init(self.weight)
    
    def he_orthogonal_init(self, tensor):
        def _standardize(kernel):
            eps = 1e-6

            if len(kernel.shape) == 3:
                axis = [0, 1]  # last dimension is output dimension
            else:
                axis = 1

            var, mean = torch.var_mean(kernel, dim=axis, unbiased=True, keepdim=True)
            kernel = (kernel - mean) / (var + eps) ** 0.5
            return kernel
        
        tensor = torch.nn.init.orthogonal_(tensor)

        if len(tensor.shape) == 3:
            fan_in = tensor.shape[:-1].numel()
        else:
            fan_in = tensor.shape[1]

        with torch.no_grad():
            tensor.data = _standardize(tensor.data)
            tensor.data *= (1 / fan_in) ** 0.5

        return tensor

    def forward(self, rbf, sph, id_ca, id_ragged_idx):
        """

        Arguments
        ---------
        rbf: torch.Tensor, shape=(1, nEdges, num_radial)
        sph: torch.Tensor, shape=(nEdges, Kmax, num_spherical)
        id_ca
        id_ragged_idx

        Returns
        -------
        rbf_W1: torch.Tensor, shape=(nEdges, emb_size_interm, num_spherical)
        sph: torch.Tensor, shape=(nEdges, Kmax, num_spherical)
            Kmax = maximum number of neighbors of the edges
        """
        num_edges = rbf.shape[1]

        # MatMul: mul + sum over num_radial
        rbf_W1 = torch.matmul(rbf, self.weight)
        # (num_spherical, nEdges , emb_size_interm)
        rbf_W1 = rbf_W1.permute(1, 2, 0)
        # (nEdges, emb_size_interm, num_spherical)

        # Zero padded dense matrix
        # maximum number of neighbors, catch empty id_ca with maximum
        if sph.shape[0] == 0:
            Kmax = 0
        else:
            Kmax = torch.max(
                torch.max(id_ragged_idx + 1),
                torch.tensor(0).to(id_ragged_idx.device),
            )

        sph2 = sph.new_zeros(num_edges, Kmax, self.num_spherical)
        sph2[id_ca, id_ragged_idx] = sph

        sph2 = torch.transpose(sph2, 1, 2)
        # (nEdges, num_spherical/emb_size_interm, Kmax)

        return rbf_W1, sph2

class FiLMLayer(nn.Module):
    def __init__(self, emb_dim, condition_dim=1):
        super().__init__()
        self.film = nn.Linear(condition_dim, emb_dim * 2)
        self.reset_parameters()

    def reset_parameters(self):
        self.film.reset_parameters()
        
    def forward(self, x, condition, num_atoms):
        gamma_beta = self.film(condition)  # (N_cryst, 2 * emb_dim)
        gamma, beta = gamma_beta.chunk(2, dim=1)
        gamma = gamma.repeat_interleave(num_atoms, dim=0)
        beta = beta.repeat_interleave(num_atoms, dim=0)
        return gamma * x + beta

class LatentConditionEmbedder(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        num_spherical: int = 7,
        num_radial: int = 128,
        emb_size_atom: int = 512,
        emb_size_edge: int = 512,
        emb_size_rbf: int = 16,
        emb_size_cbf: int = 16,
        cutoff: float = 6.0,
        rbf: dict = {"name": "gaussian"},
        envelope: dict = {"name": "polynomial", "exponent": 5},
        cbf: dict = {"name": "spherical_harmonics"},
        activation: str = "swish",
    ):
        super().__init__()


        self.atom_emb = AtomEmbedding(emb_size_atom)
        self.atom_latent_emb = nn.Linear(emb_size_atom + latent_dim, emb_size_atom)
        self.edge_emb = EdgeEmbedding(
            emb_size_atom, num_radial, emb_size_edge, activation=activation
        )

        ### ---------------------------------- Basis Functions ---------------------------------- ###
        self.radial_basis = RadialBasis(
            num_radial=num_radial,
            cutoff=cutoff,
            rbf=rbf,
            envelope=envelope,
        )

        radial_basis_cbf3 = RadialBasis(
            num_radial=num_radial,
            cutoff=cutoff,
            rbf=rbf,
            envelope=envelope,
        )
        self.cbf_basis3 = CircularBasisLayer(
            num_spherical,
            radial_basis=radial_basis_cbf3,
            cbf=cbf,
            efficient=True,
        )
        
        self.mlp_rbf3 = nn.Linear(
            num_radial,
            emb_size_rbf,
            bias=False,
        )
        self.mlp_cbf3 = EfficientInteractionDownProjection(
            num_spherical, num_radial, emb_size_cbf
        )

        self.mlp_rbf_h = nn.Linear(
            num_radial,
            emb_size_rbf,
            bias=False,
        )
        self.mlp_rbf_out = nn.Linear(
            num_radial,
            emb_size_rbf,
            bias=False,
        )
        
        self.film_layer = FiLMLayer(emb_size_atom)
        
        self.reset_parameters()
        
    def reset_parameters(self):  
        self.atom_latent_emb.reset_parameters()       
        self.edge_emb.reset_parameters()

        self.mlp_rbf3.reset_parameters()
        self.mlp_cbf3.reset_parameters()

        self.mlp_rbf_h.reset_parameters()
        self.mlp_rbf_out.reset_parameters()

        self.film_layer.reset_parameters()


    def build_triplets(self, edge_index: torch.Tensor, batch: torch.Tensor, num_nodes: int):
        """
        Efficiently build triplets (k → i ← j) for batched graphs with PBC.
        
        Args:
            edge_index: [2, E] edge indices (j → i)
            batch: [num_nodes] assignment of each node to a structure
            num_nodes: total number of nodes
        
        Returns:
            triplet_index: [2, T] tensor where each column is (eid_k, eid_j)
        """
        j, i = edge_index
        E = j.size(0)
        device = edge_index.device
        eid = torch.arange(E, device=device)

        # Assign each edge to its corresponding structure via its target node i
        edge_batch = batch[i]  # [E]
        
        # Sort edges by batch then target i
        batch_i = edge_batch
        i_sorted, perm_i = torch.sort(i + batch_i * num_nodes)  # unique per structure
        j_sorted = j[perm_i]
        eid_sorted = eid[perm_i]
        edge_batch_sorted = edge_batch[perm_i]

        # Find segments per structure + target node
        unique_keys, counts = torch.unique_consecutive(i_sorted, return_counts=True)
        ptr = torch.cat([torch.tensor([0], device=device), counts.cumsum(0)])  # [num_groups + 1]

        row = []
        col = []

        for start, end in zip(ptr[:-1], ptr[1:]):
            eids = eid_sorted[start:end]
            if eids.size(0) < 2:
                continue
            k, j_ = torch.meshgrid(eids, eids, indexing='ij')
            mask = k != j_
            row.append(k[mask])
            col.append(j_[mask])

        if not row:
            return torch.empty((2, 0), dtype=torch.long, device=device)

        eid_k = torch.cat(row, dim=0)
        eid_j = torch.cat(col, dim=0)
        
        return torch.stack([eid_k, eid_j], dim=0)  # [2, T]

    def select_symmetric_edges(self, tensor, mask, reorder_idx, inverse_neg):
        # Mask out counter-edges
        tensor_directed = tensor[mask]
        # Concatenate counter-edges after normal edges
        sign = 1 - 2 * inverse_neg
        tensor_cat = torch.cat([tensor_directed, sign * tensor_directed])
        # Reorder everything so the edges of every image are consecutive
        tensor_ordered = tensor_cat[reorder_idx]
        return tensor_ordered

    def reorder_symmetric_edges(
        self, edge_index, cell_offsets, neighbors, edge_dist, edge_vector
    ):
        """
        Reorder edges to make finding counter-directional edges easier.

        Some edges are only present in one direction in the data,
        since every atom has a maximum number of neighbors. Since we only use i->j
        edges here, we lose some j->i edges and add others by
        making it symmetric.
        We could fix this by merging edge_index with its counter-edges,
        including the cell_offsets, and then running torch.unique.
        But this does not seem worth it.
        """

        # Generate mask
        mask_sep_atoms = edge_index[0] < edge_index[1]
        # Distinguish edges between the same (periodic) atom by ordering the cells
        cell_earlier = (
            (cell_offsets[:, 0] < 0)
            | ((cell_offsets[:, 0] == 0) & (cell_offsets[:, 1] < 0))
            | (
                (cell_offsets[:, 0] == 0)
                & (cell_offsets[:, 1] == 0)
                & (cell_offsets[:, 2] < 0)
            )
        )
        mask_same_atoms = edge_index[0] == edge_index[1]
        mask_same_atoms &= cell_earlier
        mask = mask_sep_atoms | mask_same_atoms

        # Mask out counter-edges
        edge_index_new = edge_index[mask[None, :].expand(2, -1)].view(2, -1)

        # Concatenate counter-edges after normal edges
        edge_index_cat = torch.cat(
            [
                edge_index_new,
                torch.stack([edge_index_new[1], edge_index_new[0]], dim=0),
            ],
            dim=1,
        )

        # Count remaining edges per image
        batch_edge = torch.repeat_interleave(
            torch.arange(neighbors.size(0), device=edge_index.device),
            neighbors,
        )
        batch_edge = batch_edge[mask]
        neighbors_new = 2 * torch.bincount(
            batch_edge, minlength=neighbors.size(0)
        )

        # Create indexing array
        edge_reorder_idx = repeat_blocks(
            neighbors_new // 2,
            repeats=2,
            continuous_indexing=True,
            repeat_inc=edge_index_new.size(1),
        )

        # Reorder everything so the edges of every image are consecutive
        edge_index_new = edge_index_cat[:, edge_reorder_idx]
        cell_offsets_new = self.select_symmetric_edges(
            cell_offsets, mask, edge_reorder_idx, True
        )
        edge_dist_new = self.select_symmetric_edges(
            edge_dist, mask, edge_reorder_idx, False
        )
        edge_vector_new = self.select_symmetric_edges(
            edge_vector, mask, edge_reorder_idx, True
        )

        return (
            edge_index_new,
            cell_offsets_new,
            neighbors_new,
            edge_dist_new,
            edge_vector_new,
        )

    def get_triplets(self, edge_index, num_atoms):
        """
        Get all b->a for each edge c->a.
        It is possible that b=c, as long as the edges are distinct.

        Returns
        -------
        id3_ba: torch.Tensor, shape (num_triplets,)
            Indices of input edge b->a of each triplet b->a<-c
        id3_ca: torch.Tensor, shape (num_triplets,)
            Indices of output edge c->a of each triplet b->a<-c
        id3_ragged_idx: torch.Tensor, shape (num_triplets,)
            Indices enumerating the copies of id3_ca for creating a padded matrix
        """
        idx_s, idx_t = edge_index  # c->a (source=c, target=a)

        value = torch.arange(
            idx_s.size(0), device=idx_s.device, dtype=idx_s.dtype
        )
        # Possibly contains multiple copies of the same edge (for periodic interactions)
        adj = SparseTensor(
            row=idx_t,
            col=idx_s,
            value=value,
            sparse_sizes=(num_atoms, num_atoms),
        )
        adj_edges = adj[idx_t]

        # Edge indices (b->a, c->a) for triplets.
        id3_ba = adj_edges.storage.value()
        id3_ca = adj_edges.storage.row()

        # Remove self-loop triplets
        # Compare edge indices, not atom indices to correctly handle periodic interactions
        mask = id3_ba != id3_ca
        id3_ba = id3_ba[mask]
        id3_ca = id3_ca[mask]

        # Get indices to reshape the neighbor indices b->a into a dense matrix.
        # id3_ca has to be sorted for this to work.
        num_triplets = torch.bincount(id3_ca, minlength=idx_s.size(0))
        id3_ragged_idx = ragged_range(num_triplets)

        return id3_ba, id3_ca, id3_ragged_idx
    
    def generate_interaction_graph(self, cart_coords, lengths, angles,
                                   num_atoms, edge_index, to_jimages,
                                   num_bonds):

        # Generate Graph On The Fly
        edge_index, to_jimages, num_bonds = radius_graph_pbc(
            cart_coords, lengths, angles, num_atoms, self.cutoff, self.max_neighbors,
            device=num_atoms.device)

        out = get_pbc_distances(
            cart_coords,
            edge_index,
            lengths,
            angles,
            to_jimages,
            num_atoms,
            num_bonds,
            coord_is_cart=True,
            return_offsets=True,
            return_distance_vec=True,
        )

        edge_index = out["edge_index"]
        D_st = out["distances"]
        V_st = -out["distance_vec"] / D_st[:, None]

        (
            edge_index,
            cell_offsets,
            neighbors,
            D_st,
            V_st,
        ) = self.reorder_symmetric_edges(
            edge_index, to_jimages, num_bonds, D_st, V_st
        )

        # Indices for swapping c->a and a->c (for symmetric MP)
        block_sizes = neighbors // 2
        id_swap = repeat_blocks(
            block_sizes,
            repeats=2,
            continuous_indexing=False,
            start_idx=block_sizes[0],
            block_inc=block_sizes[:-1] + block_sizes[1:],
            repeat_inc=-block_sizes,
        )

        id3_ba, id3_ca, id3_ragged_idx = self.get_triplets(
            edge_index, num_atoms=num_atoms.sum(),
        )

        return (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
        )

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
        cart_coords = frac_to_cart_coords(pred_frac_coords, lengths, angles, num_atoms)
        batch = torch.arange(num_atoms.size(0),device=num_atoms.device).repeat_interleave(num_atoms, dim=0)
        
        (
            edge_index,
            neighbors,
            D_st,
            V_st,
            id_swap,
            id3_ba,
            id3_ca,
            id3_ragged_idx,
        ) = self.generate_interaction_graph(
            cart_coords, lengths, angles, num_atoms, edge_index=None, to_jimages=None, num_bonds=None)
        idx_s, idx_t = edge_index

        # Calculate triplet angles
        cos_theta_cab = inner_product_normalized(V_st[id3_ca], V_st[id3_ba])
        rad_cbf3, cbf3 = self.cbf_basis3(D_st, cos_theta_cab, id3_ca)

        rbf = self.radial_basis(D_st)

        # Embedding block
        h = self.atom_emb(pred_atom_types)
        # Merge z and atom embedding
        z_per_atom = z.repeat_interleave(num_atoms, dim=0)
        h = torch.cat([h, z_per_atom], dim=1)
        h = self.atom_latent_emb(h)

        # Target property control
        condition = target_property.unsqueeze(1)  # (N_cryst, 1)
        h = self.film_layer(h, condition, num_atoms)

        # (nAtoms, emb_size_atom)
        m = self.edge_emb(h, rbf, idx_s, idx_t)  # (nEdges, emb_size_edge)

        rbf3 = self.mlp_rbf3(rbf)
        cbf3 = self.mlp_cbf3(rad_cbf3, cbf3, id3_ca, id3_ragged_idx)

        rbf_h = self.mlp_rbf_h(rbf)
        rbf_out = self.mlp_rbf_out(rbf)

        return (
            h, # node features
            m, # edge features
            rbf3, # rbf for triplets
            cbf3, # angular embeddings
            rbf_h, # rbf global info
            rbf_out, # distant dependent output
            batch, # batch info
            edge_index, D_st, V_st, idx_s, idx_t
        )
