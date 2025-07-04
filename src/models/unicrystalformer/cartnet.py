# Copyright Universitat Politècnica de Catalunya 2024 https://imatge.upc.edu
# Distributed under the MIT License.
# (See accompanying file README.md file or copy at http://opensource.org/licenses/MIT)

import torch
import torch_geometric.nn as pyg_nn
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter
from torch_geometric.nn import JumpingKnowledge
from src.models.unicrystalformer.utils import ExpNormalSmearing, CosineCutoff


class CartNet(torch.nn.Module):
    """
    CartNet model from Cartesian Encoding Graph Neural Network for Crystal Structures Property Prediction: Application to Thermal Ellipsoid Estimation.
    Args:
        dim_in (int): Dimensionality of the input features.
        dim_rbf (int): Dimensionality of the radial basis function embeddings.
        num_layers (int): Number of CartNet layers in the model.
        radius (float, optional): Radius cutoff for neighbor interactions. Default is 5.0.
        invariant (bool, optional): If `True`, enforces rotational invariance in the encoder. Default is `False`.
        temperature (bool, optional): If `True`, includes temperature information in the encoder. Default is `True`.
        use_envelope (bool, optional): If `True`, applies an envelope function to the interactions. Default is `True`.
        cholesky (bool, optional): If `True`, uses a Cholesky head for the output. If `False`, uses a scalar head. Default is `True`.
        jk_mode (str, optional): Jumping knowledge aggregation strategy. One of ['cat', 'lstm', 'max', 'last']. Default is `'cat'`.
        layer_type (str, optional): Layer architecture type. One of ['transformer', 'default']. `'transformer'` enables Q/K/V attention with optional residuals. Default is `'transformer'`.
        use_residual (bool, optional): If `True`, adds residual connections in Transformer layers. Only applies when `layer_type='transformer'`. Default is `True`.
        num_heads (int, optional): Number of attention heads for Transformer-style layers. Only applies when `layer_type='transformer'`. Default is 4.

    Methods:
        forward(batch):
            Performs a forward pass of the model.
            Args:
                batch: A batch of input data.
            Returns:
                pred: The model's predictions.
                true: The ground truth values corresponding to the input batch.
    """


    def __init__(self, 
        dim_in: int, 
        dim_rbf: int, 
        num_layers: int,
        radius: float = 5.0,
        invariant: bool = False,
        temperature: bool = True, 
        use_envelope: bool = True,
        atom_types: bool = True,
        cholesky: bool = True,
        jk_mode='cat',
        layer_type='transformer',       # NEW: 'transformer' or 'default'
        dim_hidden=128,
        use_residual=True,              # NEW: only applies if layer_type == 'transformer'
        num_heads=4):                   # NEW: only applies if layer_type == 'transformer'
        
        super().__init__()
        LAYER_FACTORY = {
            'transformer': CartNetTransformerLayer,
            'default': CartNet_layer
        }
        self.encoder = Encoder(dim_in, dim_rbf, radius, invariant, temperature, atom_types)
        self.dim_in = dim_in
        self.cholesky = cholesky
        self.jk_mode = jk_mode
        self.layer_type = layer_type
        self.use_residual = use_residual
        self.num_heads = num_heads

        LayerClass = LAYER_FACTORY[layer_type]
        layer_kwargs = {
            'dim_in': dim_in,
            'radius': radius
        }

        if layer_type == 'transformer':
            layer_kwargs.update({
                'dim_hidden': dim_in,
                'num_heads': num_heads,
                'use_residual': use_residual,
                'dim_hidden': dim_hidden
            })
        elif layer_type == 'default':
            layer_kwargs.update({
                'use_envelope': use_envelope
            })

        self.layers = nn.ModuleList([LayerClass(**layer_kwargs) for _ in range(num_layers)])

        if jk_mode in ['cat', 'max', 'lstm']:
            self.jk = JumpingKnowledge(mode=jk_mode)
            jk_dim = dim_in * num_layers if jk_mode == 'cat' else dim_in
        elif jk_mode == 'last':
            self.jk = None  # no JumpingKnowledge module
            jk_dim = dim_in
        else:
            raise ValueError(f"Invalid jk_mode: {jk_mode}. Choose from ['cat', 'max', 'lstm', 'last']")

        if cholesky:
            self.head = Cholesky_head(jk_dim)
        else:
            self.head = Scalar_head(jk_dim)

    def forward(self, batch):
        batch = self.encoder(batch)
        x_layers = []

        for layer in self.layers:
            batch = layer(batch)
            x_layers.append(batch.x)

        if self.jk is not None:
            batch.x = self.jk(x_layers)
        else:
            batch.x = x_layers[-1]

        return self.head(batch)
    
    @classmethod
    def from_config(cls, config):
        return cls(
            dim_in=config.dim_in, 
            dim_rbf=config.dim_rbf, 
            num_layers=config.num_layers,
            radius = config.get("radius", 5.0),
            invariant = config.get("invariant", False),
            temperature = config.get("temperature", False), 
            use_envelope = config.get("use_envelope", True),
            atom_types = config.get("atom_types", True),
            cholesky = config.get("cholesky", True),
            jk_mode=config.get("jk_mode", 'cat'),
            layer_type=config.get("layer_type",' transformer'),
            dim_hidden=config.get("dim_hidden", 128),
            use_residual=config.get("use_residual", True),
            num_heads=config.get("num_heads", 4)
        )

class Encoder(torch.nn.Module):
    """
    Encoder module for the CartNet model.
    This module encodes node and edge features for input into the CartNet model, incorporating optional temperature information and rotational invariance.
    Args:
        dim_in (int): Dimension of the input features after embedding.
        dim_rbf (int): Dimension of the radial basis function used for edge attributes.
        radius (float, optional): Cutoff radius for neighbor interactions. Defaults to 5.0.
        invariant (bool, optional): If True, the encoder enforces rotational invariance by excluding directional information from edge attributes. Defaults to False.
        temperature (bool, optional): If True, includes temperature data in the node embeddings. Defaults to True.
    Attributes:
        dim_in (int): Dimension of the input features.
        invariant (bool): Indicates if rotational invariance is enforced.
        temperature (bool): Indicates if temperature information is included.
        embedding (nn.Embedding): Embedding layer mapping atomic numbers to feature vectors.
        temperature_proj_atom (pyg_nn.Linear): Linear layer projecting temperature to embedding dimensions (used if temperature is True).
        bias (nn.Parameter): Bias term added to embeddings (used if temperature is False).
        activation (nn.Module): Activation function (SiLU).
        encoder_atom (nn.Sequential): Sequential network encoding node features.
        encoder_edge (nn.Sequential): Sequential network encoding edge features.
        rbf (ExpNormalSmearing): Radial basis function for encoding distances.
    """
    
    def __init__(
        self,
        dim_in: int,
        dim_rbf: int,
        radius: float = 5.0,
        invariant: bool = False, 
        temperature: bool = True,
        atom_types: bool = True
    ):
        super(Encoder, self).__init__()
        self.dim_in = dim_in
        self.invariant = invariant
        self.temperature = temperature
        self.atom_types = atom_types
        if self.atom_types:
            self.embedding = nn.Embedding(119, self.dim_in*2)
            torch.nn.init.xavier_uniform_(self.embedding.weight.data)
        elif not self.temperature:
            self.embedding = nn.Embedding(1, self.dim_in)

        if self.temperature:
            self.temperature_proj_atom = pyg_nn.Linear(1, self.dim_in*2, bias=True)
        elif self.atom_types:
            self.bias = nn.Parameter(torch.zeros(self.dim_in*2))
        self.activation = nn.SiLU(inplace=True)
        
        if self.temperature or self.atom_types:
            self.encoder_atom = nn.Sequential(self.activation,
                                        pyg_nn.Linear(self.dim_in*2, self.dim_in),
                                        self.activation)
        if self.invariant:
            dim_edge = dim_rbf
        else:
            dim_edge = dim_rbf + 3
        
        self.encoder_edge = nn.Sequential(pyg_nn.Linear(dim_edge, self.dim_in*2),
                                        self.activation,
                                        pyg_nn.Linear(self.dim_in*2, self.dim_in),
                                        self.activation)

        self.rbf = ExpNormalSmearing(0.0,radius,dim_rbf,False)  
        
        

    def forward(self, batch):

        if self.temperature and self.atom_types:
            x = self.embedding(batch.x) + self.temperature_proj_atom(batch.temperature.unsqueeze(-1))[batch.batch]
        elif not self.temperature and self.atom_types:
            x = self.embedding(batch.x) + self.bias
        elif self.temperature and not self.atom_types:
            x = self.temperature_proj_atom(batch.temperature.unsqueeze(-1))[batch.batch]
        else:
            batch.x = self.embedding.weight.repeat(batch.x.shape[0],1)
        
        if self.temperature or self.atom_types:
            batch.x = self.encoder_atom(x)

        if self.invariant:
            batch.edge_attr = self.encoder_edge(self.rbf(batch.cart_dist))
        else:
            batch.edge_attr = self.encoder_edge(torch.cat([self.rbf(batch.cart_dist), batch.cart_dir], dim=-1))

        return batch

class CartNet_layer(pyg_nn.conv.MessagePassing):
    """
    The message-passing layer used in the CartNet architecture.
    Parameters:
        dim_in (int): Dimension of the input node features.
        use_envelope (bool, optional): If True, applies an envelope function to the distances. Defaults to True.
    Attributes:
        dim_in (int): Dimension of the input node features.
        activation (nn.Module): Activation function (SiLU) used in the layer.
        MLP_aggr (nn.Sequential): MLP used for aggregating messages.
        MLP_gate (nn.Sequential): MLP used for computing gating coefficients.
        norm (nn.BatchNorm1d): Batch normalization applied to the gating coefficients.
        norm2 (nn.BatchNorm1d): Batch normalization applied to the aggregated messages.
        use_envelope (bool): Indicates if the envelope function is used.
        envelope (CosineCutoff): Envelope function applied to the distances.
    """
    
    def __init__(self, 
        dim_in: int,
        radius: float,
        use_envelope: bool = True
    ):
        super().__init__()
        self.dim_in = dim_in
        self.activation = nn.SiLU(inplace=True) 
        self.MLP_aggr = nn.Sequential(
            pyg_nn.Linear(dim_in*3, dim_in, bias=True),
            self.activation,
            pyg_nn.Linear(dim_in, dim_in, bias=True),
        )
        self.MLP_gate = nn.Sequential(
            pyg_nn.Linear(dim_in*3, dim_in, bias=True),
            self.activation,
            pyg_nn.Linear(dim_in, dim_in, bias=True),
        )
        
        self.norm = nn.BatchNorm1d(dim_in)
        self.norm2 = nn.BatchNorm1d(dim_in)
        self.use_envelope = use_envelope
        self.envelope = CosineCutoff(0, radius)
        

    def forward(self, batch):

        x, e, edge_index, dist = batch.x, batch.edge_attr, batch.edge_index, batch.cart_dist
        """
        x               : [n_nodes, dim_in]
        e               : [n_edges, dim_in]
        edge_index      : [2, n_edges]
        dist            : [n_edges]
        batch           : [n_nodes]
        """
        
        x_in = x
        e_in = e

        x, e = self.propagate(edge_index,
                                Xx=x, Ee=e,
                                He=dist,
                            )
 
        batch.x = self.activation(x) + x_in
        
        batch.edge_attr = e_in + e 

        return batch


    def message(self, Xx_i, Ee, Xx_j, He):
        """
        x_i           : [n_edges, dim_in]
        x_j           : [n_edges, dim_in]
        e             : [n_edges, dim_in]
        """

        e_ij = self.MLP_gate(torch.cat([Xx_i, Xx_j, Ee], dim=-1))
        e_ij = F.sigmoid(self.norm(e_ij))
        
        if self.use_envelope:
            sigma_ij = self.envelope(He).unsqueeze(-1)*e_ij
        else:
            sigma_ij = e_ij
        
        self.e = sigma_ij
        return sigma_ij

    def aggregate(self, sigma_ij, index, Xx_i, Xx_j, Ee, Xx):
        """
        sigma_ij        : [n_edges, dim_in]  ; is the output from message() function
        index           : [n_edges]
        x_j           : [n_edges, dim_in]
        """
        dim_size = Xx.shape[0]  

        sender = self.MLP_aggr(torch.cat([Xx_i, Xx_j, Ee], dim=-1))
        

        out = scatter(sigma_ij*sender, index, 0, None, dim_size,
                                   reduce='sum')

        return out

    def update(self, aggr_out):
        """
        aggr_out        : [n_nodes, dim_in] ; is the output from aggregate() function after the aggregation
        x             : [n_nodes, dim_in]
        """
        x = self.norm2(aggr_out)
       
        e_out = self.e
        del self.e

        return x, e_out

class PositionalEncoding(nn.Module):
    def __init__(self, dim_edge):
        super().__init__()
        self.pos_mlp = nn.Sequential(
            nn.Linear(1 + 3, dim_edge),
            nn.SiLU(),
            nn.Linear(dim_edge, dim_edge)  # Output dim_edge instead of 1
        )
    def forward(self, dist, direction):
        pos = torch.cat([dist.unsqueeze(-1), direction], dim=-1)
        return self.pos_mlp(pos)
    
class CartNetTransformerLayer(pyg_nn.conv.MessagePassing):
    def __init__(self, dim_in, dim_hidden, radius, num_heads=4, use_residual=True):
        super().__init__(aggr=None)
        self.dropout = nn.Dropout(0.1)
        self.dim_in = dim_in
        self.dim_hidden = dim_hidden
        self.num_heads = num_heads
        self.head_dim = dim_hidden // num_heads
        self.use_residual = use_residual

        assert dim_hidden % num_heads == 0, f"dim_hidden({dim_hidden}) must be divisible by num_heads({num_heads})"

        # Q/K/V projection
        self.q_proj = nn.Linear(dim_in, dim_hidden)
        self.k_proj = nn.Linear(dim_in, dim_hidden)
        self.v_proj = nn.Linear(dim_in * 2, dim_hidden)

        self.out_proj = nn.Linear(dim_hidden, dim_in)

        self.pos_encoder = PositionalEncoding(self.head_dim)
        self.norm = nn.BatchNorm1d(dim_in)

    def forward(self, batch):
        x, e = batch.x, batch.edge_attr
        dist, direction = batch.cart_dist, batch.cart_dir
        edge_index = batch.edge_index

        q_all = self.q_proj(x)  # [618, 128]
        k_all = self.k_proj(x)  # [618, 128]
        x_j = x[edge_index[1]]
        v_input = torch.cat([x_j, e], dim=-1)
        v_all = self.v_proj(v_input)  # [1399, 128]

        pos_encoding = self.pos_encoder(dist, direction)

        x_out = self.propagate(
            edge_index=edge_index,
            q_all=q_all,
            k_all=k_all,
            v_all=v_all,
            pos_encoding=pos_encoding,
            size=(x.size(0), x.size(0))
        )

        x_out = self.out_proj(x_out)

        if self.use_residual:
            batch.x = self.norm(x + F.silu(x_out))
        else:
            batch.x = F.silu(x_out)

        return batch

    def message(self, q_all_i, k_all_j, v_all, pos_encoding):
        # reshape here to [n_edges, num_heads, head_dim]
        q_i = q_all_i.view(-1, self.num_heads, self.head_dim)
        k_j = k_all_j.view(-1, self.num_heads, self.head_dim)
        v = v_all.view(-1, self.num_heads, self.head_dim)

        pos_encoding = pos_encoding.unsqueeze(1)  # [n_edges, 1, head_dim]
        k_j = k_j + pos_encoding  # [n_edges, num_heads, head_dim]

        # Compute attention scores
        score = (q_i * k_j).sum(dim=-1) / self.head_dim ** 0.5  # [n_edges, num_heads]
        attn = F.softmax(score, dim=-2)  # Normalize over edges per head

        return (attn.unsqueeze(-1) * v).reshape(-1, self.dim_hidden)

    def aggregate(self, inputs, index, ptr=None, dim_size=None):
        return scatter(inputs, index, dim=0, dim_size=dim_size, reduce='sum')

    def update(self, aggr_out):
        return aggr_out

class Cholesky_head(torch.nn.Module):
    """
    The Cholesky head used in the CartNet model.
    It enforce the positive definiteness of the output covariance matrix.

    Args:
        dim_in (int): The input dimension of the features.
    """
    
    def __init__(self, 
        dim_in: int
    ):
        super(Cholesky_head, self).__init__()
        self.MLP = nn.Sequential(pyg_nn.Linear(dim_in, dim_in//2),
                                nn.SiLU(inplace=True), 
                                pyg_nn.Linear(dim_in//2, 6))

    def forward(self, batch):
        pred = self.MLP(batch.x[batch.non_H_mask])

        diag_elements = F.softplus(pred[:, :3])

        i,j = torch.tensor([0,1,2,0,0,1]), torch.tensor([0,1,2,1,2,2])
        L_matrix = torch.zeros(pred.size(0),3,3, device=pred.device, dtype=pred.dtype)
        L_matrix[:,i[:3], i[:3]] = diag_elements
        L_matrix[:,i[3:], j[3:]] = pred[:,3:]

        U = torch.bmm(L_matrix.transpose(1, 2), L_matrix)
        
        return U

class Scalar_head(torch.nn.Module):
    """
    A head to predict scalar values.
    Args:
        dim_in (int): The dimension of the input features.
    """
    
    def __init__(self,
        dim_in
    ):
        super(Scalar_head, self).__init__()

        self.MLP = nn.Sequential(pyg_nn.Linear(dim_in, dim_in//2), 
                                nn.SiLU(inplace=True), 
                                pyg_nn.Linear(dim_in//2, 1))

    def forward(self, batch):
        dim_size = int(batch.batch.max().item() + 1)
        batch.x = self.MLP(batch.x)
        batch.x = scatter(batch.x, batch.batch, dim=0, reduce="mean", dim_size=dim_size)
        return batch.x