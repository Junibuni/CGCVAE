import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_scatter import scatter
from torch_sparse import SparseTensor

from src.models.data_utils import (build_radius_graph_with_pbc,
                                   frac_to_cart_coords, get_pbc_distances, repeat_blocks,
                                   min_distance_sqr_pbc, radius_graph_pbc, ragged_range, MAX_ATOMIC_NUM)
from src.models.unicrystalformer import (CartNet_layer)

class CrystalDecoder(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        edge_dim=128,
        latent_dim=256,
        num_rbf=32,
        num_sbf=16,
        num_atom_types=MAX_ATOMIC_NUM,
        num_message_layers=6,
        cutoff=5.0,
        use_sbf=True
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.edge_dim = edge_dim
        self.latent_dim = latent_dim
        self.cutoff = cutoff
        self.use_sbf = use_sbf

        self.atom_embed = AtomEmbedding(hidden_dim)
        self.latent_lin = nn.Linear(latent_dim, hidden_dim)

        self.rbf = nn.Sequential(
            RBFExpansion(vmin=0.0, vmax=cutoff, bins=num_rbf),
            nn.Linear(num_rbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.sbf = nn.Sequential(
            SBFExpansion(num_sbf=num_sbf),
            nn.Linear(num_sbf, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        ) if use_sbf else None

        self.dir_lin = nn.Sequential(
            nn.Linear(3, edge_dim),
            nn.SiLU(),
            nn.Linear(edge_dim, hidden_dim)
        )

        self.layers = nn.ModuleList([
            CartNet_layer(hidden_dim, cutoff)
            for _ in range(num_message_layers)
        ])

        self.coord_head = nn.Linear(hidden_dim, 3)  # Predicts diff to coordinates
        self.type_head = nn.Linear(hidden_dim, num_atom_types)
    
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

        # Build graph
        edge_index = build_radius_graph_with_pbc(cart_coords, self.cutoff, batch=batch, lengths=lengths, angles=angles)
        src, dst = edge_index

        # Compute edge vectors
        rel = cart_coords[src] - cart_coords[dst]
        rel = min_distance_sqr_pbc(rel, lengths, angles, batch[dst])  # PBC-aware displacement
        dist = torch.norm(rel, dim=-1)
        direction = F.normalize(rel, dim=-1)  # [E, 3]

        # Edge embedding
        dir_emb = self.dir_lin(direction)
        rbf_emb = self.rbf(dist)
        edge_attr = rbf_emb + dir_emb  # [E, hidden_dim]
        
        # Node embedding
        node_feat = self.atom_emb(pred_atom_types)

        if z is not None:
            node_feat += self.latent_lin(z[batch])  # Safe and non-redundant

        # Build triplets (j→i←k)
        triplet_idx = self.build_triplets(edge_index, batch, node_feat.size(0))

        j_edge = edge_index[0][triplet_idx[0]]
        k_edge = edge_index[0][triplet_idx[1]]
        center_atom = edge_index[1][triplet_idx[1]]

        vec1 = min_distance_sqr_pbc(cart_coords[k_edge] - cart_coords[center_atom], lengths, angles, batch[center_atom])
        vec2 = min_distance_sqr_pbc(cart_coords[j_edge] - cart_coords[center_atom], lengths, angles, batch[center_atom])

        vec1 = F.normalize(vec1, dim=-1)
        vec2 = F.normalize(vec2, dim=-1)
        cosine = (vec1 * vec2).sum(dim=-1).clamp(-1.0 + 1e-10, 1.0 - 1e-10)
        angle = torch.acos(cosine)
        
        if self.use_sbf and angle.numel() > 0:
            sbf_emb = self.sbf(angle)             # [T, hidden_dim]
            center_edge = triplet_idx[1]          # 중심 edge
            sbf_agg = scatter(sbf_emb, center_edge, dim=0, dim_size=edge_attr.size(0), reduce='mean')
            edge_attr = edge_attr + sbf_agg       # edge feature에 sbf 주입

        data = Data(
            x=node_feat,                  # node features
            edge_attr=edge_attr,          # edge features
            edge_index=edge_index,        # graph
            cart_dist=dist                # edge distance (used in CartNet_layer)
        )
            
        # Message passing
        for layer in self.layers:
            data = layer(data)

        # Outputs
        pred_cart_coord_diff = self.coord_head(data.x)
        pred_atom_type_logits = self.type_head(data.x)

        return pred_cart_coord_diff, pred_atom_type_logits
