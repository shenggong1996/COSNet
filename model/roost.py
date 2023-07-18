import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add, scatter_max, scatter_mean

class MeanPooling(nn.Module):
    """Mean pooling"""

    def __init__(self):
        super().__init__()

    def forward(self, x, index):
        return scatter_mean(x, index, dim=0)

    def __repr__(self):
        return self.__class__.__name__

class AttentionPooling(nn.Module):
    """
    softmax attention layer
    """

    def __init__(self, gate_nn, message_nn):
        """
        Args:
            gate_nn: Variable(nn.Module)
            message_nn
        """
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn

    def forward(self, x, index):
        gate = self.gate_nn(x)

        gate = gate - scatter_max(gate, index, dim=0)[0][index]
        gate = gate.exp()
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-10)

        x = self.message_nn(x)
        out = scatter_add(gate * x, index, dim=0)

        return out

    def __repr__(self):
        return self.__class__.__name__


class WeightedAttentionPooling(nn.Module):
    """
    Weighted softmax attention layer
    """

    def __init__(self, gate_nn, message_nn):
        """
        Inputs
        ----------
        gate_nn: Variable(nn.Module)
        """
        super().__init__()
        self.gate_nn = gate_nn
        self.message_nn = message_nn
        self.pow = torch.nn.Parameter(torch.randn(1))

    def forward(self, x, index, weights):
        gate = self.gate_nn(x)

        gate = gate - scatter_max(gate, index, dim=0)[0][index]
        gate = (weights ** self.pow) * gate.exp()
        # gate = weights * gate.exp()
        # gate = gate.exp()
        gate = gate / (scatter_add(gate, index, dim=0)[index] + 1e-10)

        x = self.message_nn(x)
        out = scatter_add(gate * x, index, dim=0)

        return out

    def __repr__(self):
        return self.__class__.__name__


class SimpleNetwork(nn.Module):
    """
    Simple Feed Forward Neural Network
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layer_dims,
        activation=nn.SiLU,
        batchnorm=True,
    ):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)
        """
        super().__init__()

        dims = [input_dim] + hidden_layer_dims

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        if batchnorm:
            self.bns = nn.ModuleList(
                [nn.BatchNorm1d(dims[i + 1]) for i in range(len(dims) - 1)]
            )
        else:
            self.bns = nn.ModuleList([nn.Identity() for i in range(len(dims) - 1)])

        self.acts = nn.ModuleList([activation() for _ in range(len(dims) - 1)])

        self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for fc, bn, act in zip(self.fcs, self.bns, self.acts):
            x = act(bn(fc(x)))

        return self.fc_out(x)

    def __repr__(self):
        return self.__class__.__name__

    def reset_parameters(self):
        for fc in self.fcs:
            fc.reset_parameters()

        self.fc_out.reset_parameters()

class ResidualNetwork(nn.Module):
    """
    Feed forward Residual Neural Network
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        hidden_layer_dims,
        activation=nn.SiLU,
        batchnorm=True,
        return_features=False,
    ):
        """
        Inputs
        ----------
        input_dim: int
        output_dim: int
        hidden_layer_dims: list(int)
        """
        super().__init__()

        dims = [input_dim] + hidden_layer_dims

        self.fcs = nn.ModuleList(
            [nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)]
        )

        if batchnorm:
            self.bns = nn.ModuleList(
                [nn.BatchNorm1d(dims[i + 1]) for i in range(len(dims) - 1)]
            )
        else:
            self.bns = nn.ModuleList([nn.Identity() for i in range(len(dims) - 1)])

        self.res_fcs = nn.ModuleList(
            [
                nn.Linear(dims[i], dims[i + 1], bias=False)
                if (dims[i] != dims[i + 1])
                else nn.Identity()
                for i in range(len(dims) - 1)
            ]
        )
        self.acts = nn.ModuleList([activation() for _ in range(len(dims) - 1)])

        self.return_features = return_features
        if not self.return_features:
            self.fc_out = nn.Linear(dims[-1], output_dim)

    def forward(self, x):
        for fc, bn, res_fc, act in zip(self.fcs, self.bns, self.res_fcs, self.acts):
            x = act(bn(fc(x))) + res_fc(x)

        if self.return_features:
            return x
        else:
            return self.fc_out(x)

    def __repr__(self):
        return self.__class__.__name__

class MessageLayer(nn.Module):
    """
    Massage Layers are used to propagate information between nodes in
    the stoichiometry graph.
    """

    def __init__(self, elem_fea_len, elem_heads, elem_gate, elem_msg):
        """
        """
        super().__init__()

        # Pooling and Output
        self.pooling = nn.ModuleList(
            [
                WeightedAttentionPooling(
                    gate_nn=SimpleNetwork(2 * elem_fea_len, 1, elem_gate),
                    message_nn=SimpleNetwork(2 * elem_fea_len, elem_fea_len, elem_msg),
                )
                for _ in range(elem_heads)
            ]
        )

    def forward(self, elem_weights, elem_in_fea, self_fea_idx, nbr_fea_idx):
        """
        Forward pass
        Parameters
        ----------
        N: Total number of elements (nodes) in the batch
        M: Total number of pairs (edges) in the batch
        C: Total number of crystals (graphs) in the batch
        Inputs
        ----------
        elem_weights: Variable(torch.Tensor) shape (N,)
            The fractional weights of elems in their materials
        elem_in_fea: Variable(torch.Tensor) shape (N, elem_fea_len)
            Element hidden features before message passing
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the first element in each of the M pairs
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of the second element in each of the M pairs
        Returns
        -------
        elem_out_fea: nn.Variable shape (N, elem_fea_len)
            Element hidden features after message passing
        """
        # construct the total features for passing
        elem_nbr_weights = elem_weights[nbr_fea_idx, :]
        elem_nbr_fea = elem_in_fea[nbr_fea_idx, :]
        elem_self_fea = elem_in_fea[self_fea_idx, :]
        fea = torch.cat([elem_self_fea, elem_nbr_fea], dim=1)

        # sum selectivity over the neighbours to get elems
        head_fea = []
        for attnhead in self.pooling:
            head_fea.append(
                attnhead(fea, index=self_fea_idx, weights=elem_nbr_weights)
            )

        # average the attention heads
        fea = torch.mean(torch.stack(head_fea), dim=0)

        return fea + elem_in_fea

    def __repr__(self):
        return self.__class__.__name__

class CompositionNetwork(nn.Module):
    """
    The Composition Network is the message passing section of the
    Roost Model.
    """

    def __init__(
        self,
        composition_net_params
    ):
        """
        """
        super().__init__()
        elem_emb_len = composition_net_params["elem_emb_len"]
        elem_fea_len = composition_net_params["elem_fea_len"]
        n_graph = composition_net_params["n_graph"]
        elem_heads = composition_net_params["elem_heads"]
        elem_gate = composition_net_params["elem_gate"]
        elem_msg = composition_net_params["elem_msg"]
        cry_heads = composition_net_params["cry_heads"]
        cry_gate = composition_net_params["cry_gate"]
        cry_msg = composition_net_params["cry_msg"]

        # apply linear transform to the input to get a trainable embedding
        # -1 here so we can add the weights as a node feature
        self.embedding = nn.Linear(elem_emb_len, elem_fea_len - 1)

        # create a list of Message passing layers
        self.graphs = nn.ModuleList(
            [
                MessageLayer(
                    elem_fea_len=elem_fea_len,
                    elem_heads=elem_heads,
                    elem_gate=elem_gate,
                    elem_msg=elem_msg,
                )
                for i in range(n_graph)
            ]
        )

        # define a global pooling function for materials
        self.cry_pool = nn.ModuleList(
            [
                WeightedAttentionPooling(
                    gate_nn=SimpleNetwork(elem_fea_len, 1, cry_gate),
                    message_nn=SimpleNetwork(elem_fea_len, elem_fea_len, cry_msg),
                )
                for _ in range(cry_heads)
            ]
        )

    def forward(self, composition_input):
        elem_weights, elem_fea, self_fea_idx, nbr_fea_idx, cry_elem_idx = composition_input
        """
        Forward pass
        Parameters
        ----------
        N: Total number of elements (nodes) in the batch
        M: Total number of pairs (edges) in the batch
        C: Total number of crystals (graphs) in the batch
        Inputs
        ----------
        elem_weights: Variable(torch.Tensor) shape (N)
            Fractional weight of each Element in its stoichiometry
        elem_fea: Variable(torch.Tensor) shape (N, orig_elem_fea_len)
            Element features of each of the N elems in the batch
        self_fea_idx: torch.Tensor shape (M,)
            Indices of the first element in each of the M pairs
        nbr_fea_idx: torch.Tensor shape (M,)
            Indices of the second element in each of the M pairs
        cry_elem_idx: list of torch.LongTensor of length C
            Mapping from the elem idx to crystal idx
        Returns
        -------
        cry_fea: nn.Variable shape (C,)
            Material representation after message passing
        """

        # embed the original features into a trainable embedding space
        elem_fea = self.embedding(elem_fea)

        # add weights as a node feature
        elem_fea = torch.cat([elem_fea, elem_weights], dim=1)

        # apply the message passing functions
        for graph_func in self.graphs:
            elem_fea = graph_func(elem_weights, elem_fea, self_fea_idx, nbr_fea_idx)

        # generate crystal features by pooling the elemental features
        head_fea = []
        for attnhead in self.cry_pool:
            head_fea.append(
                attnhead(elem_fea, index=cry_elem_idx, weights=elem_weights)
            )

        # head_fea = [
        #     head(elem_fea, index=cry_elem_idx, weights=elem_weights)
        #     for head in self.cry_pool
        # ]

        return torch.mean(torch.stack(head_fea), dim=0)

    def __repr__(self):
        return self.__class__.__name__