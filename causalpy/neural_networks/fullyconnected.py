from typing import Iterable, Union

import torch
import numpy as np
from .basemodel import NeuralBaseNet


class FCNet(NeuralBaseNet):
    """ Standard feed forward neural network with fully connected layers and ReLUs"""

    def __init__(self, *layer_nodes):
        super().__init__(layer_nodes[0], layer_nodes[-1])
        self.net = torch.nn.Sequential()

        # Skip Layer -- just a linear mapping
        self.skip_layer = torch.nn.Linear(layer_nodes[0], layer_nodes[-1])

        i = 0
        for l1, l2 in zip(layer_nodes, layer_nodes[1:-1]):
            self.net.add_module("Linear Layer {}".format(i), torch.nn.Linear(l1, l2))
            self.net.add_module("ReLU Layer {}".format(i), torch.nn.ReLU())
            i += 1

        self.net.add_module(
            "Linear Layer {}".format(i),
            torch.nn.Linear(layer_nodes[-2], layer_nodes[-1]),
        )

    def forward(self, x):
        return self.net(x)


class MatrixSampler(torch.nn.Module):
    """
    Matrix Sampler, following a Bernoulli distribution with trainable parameters.

    Parameters
    ----------
    graph_size: int or Iterable,
        shape of the matrix to sample. If it's an int, it samples a square matrix.
    mask: torch.Tensor,
        Allows to forbid some elements to be sampled. Defaults to ``1 - torch.eye()``.
    gumbel: bool,
        Use either gumbel softmax (True) or gumbel sigmoid (False)

    Attributes
    ----------
    weights: int or iterable,
        the learnable weights of the module of shape `(graph_size x graph_size)`, if the input was `int`,
        else `(*graph_size)`.
    """

    def __init__(self, graph_size: Union[int, Iterable], mask=None, gumbel=False):
        super().__init__()
        if isinstance(graph_size, int):
            self.graph_size = (graph_size, graph_size)
        elif isinstance(graph_size, Iterable):
            self.graph_size = graph_size
        else:
            raise ValueError(f"'graph_size' type needs to be either int or iterable. "
                             f"Provided type {graph_size} not supported")
        self.weights = torch.nn.Parameter(torch.zeros(*self.graph_size), requires_grad=True)
        if mask is None:
            mask = -torch.eye(*self.graph_size) + 1
        if not (type(mask) == bool and not mask):
            self.register_buffer("mask", mask)
        self.gumbel = gumbel

        ones_tensor = torch.ones(*self.graph_size)
        self.register_buffer("ones_tensor", ones_tensor)

        zeros_tensor = torch.zeros(*self.graph_size)
        self.register_buffer("zeros_tensor", zeros_tensor)

    def forward(self, tau=1, drawhard=True):
        """Return a sampled graph."""

        if self.gumbel:
            drawn_proba = gumbel_softmax(
                torch.stack([self.weights.view(-1), -self.weights.view(-1)], 1),
                tau=tau,
                hard=drawhard,
            )[:, 0].view(*self.graph_size)
        else:
            drawn_proba = gumbel_sigmoid(
                2 * self.weights,
                self.ones_tensor,
                self.zeros_tensor,
                tau=tau,
                hard=drawhard,
            )

        if hasattr(self, "mask"):
            return self.mask * drawn_proba
        else:
            return drawn_proba

    def get_proba(self):
        if hasattr(self, "mask"):
            return torch.sigmoid(2 * self.weights) * self.mask
        else:
            return torch.sigmoid(2 * self.weights)

    def set_skeleton(self, mask):
        self.register_buffer("mask", mask)


def functional_linear3d(input, weight, bias=None):
    r"""
    Apply a linear transformation to the incoming data: :math:`y = xA^T + b`.
    Shape:
        - Input: :math:`(N, *, in\_features)` where `*` means any number of
          additional dimensions
        - Weight: :math:`(out\_features, in\_features)`
        - Bias: :math:`(out\_features)`
        - Output: :math:`(N, *, out\_features)`
    """
    output = input.transpose(0, 1).matmul(weight)
    if bias is not None:
        output += bias.unsqueeze(1)
    return output.transpose(0, 1)


class Linear3D(torch.nn.Module):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`.
    Broadcasts following a 3rd dimension. If input is 2d, input is repeated over
    all channels. This layer is a linear layer with 3D parameters.
    Args:
        sizes: Triplet of int values defining the shape of the 3D tensor:
            (channels, in_features, out_features)
        bias: If set to False, the layer will not learn an additive bias.
            Default: ``True``
    Attributes:
        weight (torch.Tensor): the learnable weights of the module of shape
          `(out_features x in_features)`
        bias (torch.Tensor): the learnable bias of the module of shape `(out_features)`
    Shape:
        - Input: :math:`(N, *, in\_features)` where :math:`*` means number of
          channels or no additional dimension.
        - Output: :math:`(N, channels, out\_features)`.
    Examples::
        >>> Linear3D(3, 20, 30)
        >>> input = torch.randn(128, 20)
        >>> output = m(input)
        >>> print(output.size())
    """

    def __init__(self, sizes, bias=True):
        super(Linear3D, self).__init__()
        self.in_features = sizes[1]
        self.out_features = sizes[2]
        self.channels = sizes[0]
        self.weight = torch.nn.Parameter(
            data=torch.Tensor((self.channels, self.in_features, self.out_features))
        )
        if bias:
            self.bias = torch.nn.Parameter(
                data=torch.Tensor((self.channels, self.out_features))
            )
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1.0 / np.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, noise=None, adj_matrix=None):

        if input.dim() == 2:
            if noise is None:
                input = input.unsqueeze(1).expand(
                    [input.shape[0], self.channels, self.in_features]
                )
            else:
                input = torch.cat(
                    [
                        input.unsqueeze(1).expand(
                            [input.shape[0], self.channels, self.in_features - 1]
                        ),
                        noise.unsqueeze(2),
                    ],
                    2,
                )
        if adj_matrix is not None:
            input = input * adj_matrix.t().unsqueeze(0)

        return functional_linear3d(input, self.weight, self.bias)

    def extra_repr(self):
        return "in_features={}, out_features={}, bias={}".format(
            self.in_features, self.out_features, self.bias is not None
        )


class AgnosticModel(object):
    pass
