import itertools
from typing import Optional, Callable, Type, Tuple, List, Union

import torch
from .utils import get_jacobian
from abc import ABC, abstractmethod
import numpy as np


class CouplingBase(torch.nn.Module, ABC):
    def __init__(
        self,
        dim: int,
        dim_condition: int,
        nr_layers: int = 16,
        conditional_net: Optional[torch.nn.Module] = None,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ):
        super().__init__()
        self.device = device
        self.log_jacobian_cache = 0
        self.dim = dim
        self.dim_condition = dim_condition
        self.nr_layers = nr_layers
        self.nr_inverse_iters = 100

        self.mat_like_params = [
            torch.randn(1, dim, nr_layers, requires_grad=False),
            torch.randn(1, dim, nr_layers, requires_grad=False),
            torch.randn(1, dim, nr_layers, requires_grad=False),
        ]
        self.vec_like_params = [
            torch.randn(1, dim, requires_grad=False),  # bias2
            torch.ones(1, dim, requires_grad=False) * -1,  # eps
            torch.zeros(1, dim, requires_grad=False),  # alpha
        ]

        self.is_conditional = dim_condition > 0
        if self.is_conditional:
            self.matrix_shape = (self.dim, self.nr_layers)
            self.bias_shape = (self.dim,)

            if conditional_net is None:
                self.conditional_net: torch.nn.Module = torch.nn.Sequential(
                    torch.nn.Linear(dim_condition, 50),
                    torch.nn.ReLU(),
                    torch.nn.Linear(50, 3 * dim * (nr_layers + 1)),
                )
            else:
                # conditional net needs to fulfill:
                # ( dimension in = dim_condition, dimension out = 3 * dim * (nr_layers + 1) )
                with torch.no_grad():
                    out = conditional_net(torch.ones(1, dim_condition))
                    assert out.size() == (1, 3 * dim * (nr_layers + 1),), (
                        f"Size mismatch: Conditional network needs to output size "
                        f"[batch_size, 3*dim*(nr_layers+1) = {3 * dim * (nr_layers + 1)}]. "
                        f"Provided was [out] = [{tuple([*out.size()])}]."
                    )
                self.conditional_net: torch.nn.Module = conditional_net
        else:
            # only in the case of no conditional neural network are these trainable parameters.
            # Otherwise their data will be provided by the condition generating network.
            self.mat_like_params = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(tensor, requires_grad=True)
                    for tensor in self.mat_like_params
                ]
            )
            self.vec_like_params = torch.nn.ParameterList(
                [
                    torch.nn.Parameter(tensor, requires_grad=True)
                    for tensor in self.vec_like_params
                ]
            )

    @property
    def mat1(self):
        return self.mat_like_params[0]

    @property
    def bias1(self):
        return self.mat_like_params[1]

    @property
    def mat2(self):
        return self.mat_like_params[2]

    @property
    def bias2(self):
        return self.vec_like_params[0]

    @property
    def eps(self):
        return self.vec_like_params[1]

    @property
    def alpha(self):
        return self.vec_like_params[2]

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        rev: bool = False,
    ):
        if self.is_conditional:
            self.set_conditional_params(condition)

        if not rev:
            self.log_jacobian_cache = torch.log(self.transform_deriv(x))
            return self.transform(x)
        else:
            z = self.transform_inverse(x, nr_iters=self.nr_inverse_iters)
            self.log_jacobian_cache = -torch.log(self.transform_deriv(z))
            return z

    @abstractmethod
    def transform(self, x: torch.Tensor):
        raise NotImplementedError

    @abstractmethod
    def transform_deriv(self, x: torch.Tensor):
        raise NotImplementedError

    def transform_inverse(self, y: torch.Tensor, nr_iters=10):
        """
        Newton method for iterative approximation of the inverse g of the transformation f at point y: g(y).
        """
        yn = y  # * torch.exp(-self.alpha) - self.bias2
        for i in range(nr_iters):
            yn = yn - (self.transform(yn) - y) / self.transform_deriv(yn)
        return yn

    def set_conditional_params(self, condition: torch.Tensor):
        """
        In the case of a conditional INN set the parameters of the bijective map for the condition.

        Notes
        -----
        If this method was called although the INN is unconditional, uncaught errors will be raised.
        """
        assert (
            self.is_conditional
        ), "Conditional parameters can only be set when the INN is conditional. They are learned otherwise."
        size = condition.size(0)
        new_params = self.conditional_net(condition)

        begin = 0

        def reshape(param_list, new_shape):
            nonlocal begin
            for i, param in enumerate(param_list):
                # i * nr_previous_params_elements
                end = begin + np.prod(param.size()[1:])  # ignore batch size of last run
                param_list[i] = torch.reshape(new_params[:, begin:end], new_shape)
                begin = end

        reshape(self.mat_like_params, (size,) + self.matrix_shape)
        reshape(self.vec_like_params, (size,) + self.bias_shape)

    def jacobian(self, x: Optional[torch.Tensor] = None, rev: bool = False):
        if x is None:
            return self.log_jacobian_cache
        else:
            return get_jacobian(self, x, dim_in=1, dim_out=1, device=self.device)


class CouplingMonotone(CouplingBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.elu = torch.nn.ELU()

    def activation_func(self, x):
        return self.elu(x)

    def activation_deriv(self, x):
        return 1 + self.activation_func(-torch.relu(-x))

    def transform(self, x):
        internal_sum = torch.sum(
            self.mat2
            * self.activation_func(
                x.unsqueeze(2).expand(-1, -1, self.nr_layers) * self.mat1 + self.bias1
            ),
            dim=2,
        )
        divisor = torch.sum(torch.relu(-self.mat1 * self.mat2), dim=2) + 1
        return (
            self.alpha.exp()
            * (x + 0.8 * torch.sigmoid(self.eps) * internal_sum / divisor)
            + self.bias2
        )

    def transform_deriv(self, x):
        internal_sum = torch.sum(
            self.activation_deriv(
                x.unsqueeze(2).expand(-1, -1, self.nr_layers) * self.mat1 + self.bias1
            )
            * self.mat1
            * self.mat2,
            dim=2,
        )
        divisor = torch.sum(torch.relu(-self.mat1 * self.mat2), dim=2) + 1
        return self.alpha.exp() * (
            1 + 0.8 * torch.sigmoid(self.eps) * internal_sum / divisor
        )


class CouplingGeneral(CouplingBase):
    @staticmethod
    def activation_func(x):
        # return torch.tanh(x)
        return torch.exp(-(x ** 2)) * 1.16

    @staticmethod
    def activation_deriv(x):
        # return 1 - self.activation_func(x) ** 2
        return -2 * x * CouplingGeneral.activation_func(x)

    def transform(self, x: torch.Tensor):
        internal_sum = torch.sum(
            self.activation_func(
                x.unsqueeze(2).expand(-1, -1, self.nr_layers) * self.mat1 + self.bias1
            )
            * self.mat2,
            dim=2,
        )
        divisor = torch.sum(torch.abs(self.mat1 * self.mat2), dim=2) + 1
        return (
            torch.exp(self.alpha)
            * (x + 0.8 * torch.sigmoid(self.eps) * internal_sum / divisor)
            + self.bias2
        )

    def transform_deriv(self, x: torch.Tensor):
        internal_sum = torch.sum(
            self.activation_deriv(
                x.unsqueeze(2).expand(-1, -1, self.nr_layers) * self.mat1 + self.bias1
            )
            * self.mat1
            * self.mat2,
            dim=2,
        )
        divisor = torch.sum(torch.abs(self.mat1 * self.mat2), dim=2) + 1
        return torch.exp(self.alpha) * (
            1 + 0.8 * torch.sigmoid(self.eps) * internal_sum / divisor
        )


class cINN(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        dim_condition,
        nr_blocks: int = 3,
        nr_layers: int = 16,
        conditional_net: Optional[torch.nn.Module] = None,
    ):
        super().__init__()
        self.nr_blocks = nr_blocks
        mods = []
        for i in range(nr_blocks):
            mods.append(
                CouplingGeneral(
                    dim=dim,
                    nr_layers=nr_layers,
                    dim_condition=dim_condition,
                    conditional_net=conditional_net,
                )
            )
            mods.append(
                CouplingMonotone(
                    dim=dim,
                    nr_layers=nr_layers,
                    dim_condition=dim_condition,
                    conditional_net=conditional_net,
                )
            )
        self.blocks = torch.nn.ModuleList(mods)
        self.log_jacobian_cache = torch.zeros(dim)

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        rev: bool = False,
    ):
        self.log_jacobian_cache = 0.0
        if not rev:
            for block in self.blocks:
                x = block(x=x, condition=condition)
                self.log_jacobian_cache += block.jacobian()
            return x
        else:
            for block in self.blocks[::-1]:
                x = block(x=x, condition=condition, rev=True)
                self.log_jacobian_cache += block.jacobian()
            return x

    def jacobian(self, x: Optional[torch.Tensor], rev: bool = False):
        if x is None:
            return self.log_jacobian_cache
        else:
            return get_jacobian(
                self, x, dim_in=1, dim_out=1, device=self.device, rev=rev
            )


class CN(torch.nn.Module):
    def __init__(self, n_blocks=1, n_dim=1, ls=16, n_condim=0, subnet_constructor=None):
        super().__init__()
        self.n_blocks = n_blocks
        mods = []
        for i in range(n_blocks):
            # mods.append(GlowLikeCouplingBlock(dims_in=n_dim, subnet_constructor=subnet_constructor, net=PfndTanh))
            mods.append(
                GlowLikeCouplingBlock(
                    dims_in=n_dim,
                    subnet_constructor=subnet_constructor,
                    net=CouplingMonotone,
                )
            )
        self.blocks = torch.nn.ModuleList(mods)
        self.log_jacobian = torch.zeros(n_dim)

    def forward(self, x, y=None, rev=False):
        self.log_jacobian = 0.0
        if not rev:
            for block in self.blocks:
                x = block.forward(x, y)
                self.log_jacobian += block.jacobian(x)
            return x
        else:
            for block in self.blocks[::-1]:
                x = block.forward(x, y, rev=True)
                self.log_jacobian += block.jacobian(x)
            return x

    def jacobian(self, x):
        return self.log_jacobian


class GlowLikeCouplingBlock(torch.nn.Module):
    def __init__(
        self,
        dims_in,
        dims_c=[],
        subnet_constructor=None,
        clamp=5.0,
        net=CouplingMonotone,
    ):
        super().__init__()
        # channels = dims_in[0][0]
        # self.ndims = len(dims_in[0])
        channels = dims_in
        self.split_len1 = channels // 2
        self.split_len2 = channels - channels // 2
        self.log_jacobian = torch.tensor([0])
        self.clamp = clamp

        self.conditional = len(dims_c) > 0
        condition_length = sum([dims_c[i][0] for i in range(len(dims_c))])

        self.s1 = net(
            dim=self.split_len2,
            ls=100,
            subnet_constructor=subnet_constructor,
            n_condim=self.split_len1 + condition_length,
        )
        self.s2 = net(
            dim=self.split_len1,
            ls=100,
            subnet_constructor=subnet_constructor,
            n_condim=self.split_len2 + condition_length,
        )

    def forward(self, x, c=[], rev=False):
        x1, x2 = (x[:, : self.split_len1], x[:, self.split_len1 :])

        if not rev:
            y1 = self.s1(x1, torch.cat([x2, *c], 1) if self.conditional else x2)
            y2 = self.s2.forward(
                x2, torch.cat([y1.detach(), *c], 1) if self.conditional else y1.detach()
            )
            self.log_jacobian = torch.cat(
                (self.s1.log_jacobian, self.s2.log_jacobian), dim=1
            )

        else:  # names of x and y are swapped!
            y2 = self.s2.forward(
                x2, torch.cat([x1, *c], 1) if self.conditional else x1, rev=True
            )
            y1 = self.s1(
                x1, torch.cat([y2, *c], 1) if self.conditional else y2, rev=True
            )

            self.log_jacobian = torch.cat(
                (self.s1.log_jacobian, self.s2.log_jacobian), dim=1
            )
        return torch.cat((y1, y2), 1)

    def jacobian(self, x, c=[], rev=False):
        return self.log_jacobian

    def output_dims(self, input_dims):
        return input_dims


class bignet(torch.nn.Module):
    def __init__(
        self, n_dim=3, n_dimcon=0, n_blocks=1, ls=32, net=cINN, subnet_constructor=None
    ):
        super(bignet, self).__init__()
        self.con = n_dimcon > 0
        self.nets = []
        self.n_dim = n_dim
        self.n_dimcon = n_dimcon
        self.log_jacobian = 0
        for i in range(n_dim):
            self.nets.append(
                net(
                    n_dim=1,
                    ls=ls,
                    n_condim=i + n_dimcon,
                    n_blocks=n_blocks,
                    subnet_constructor=subnet_constructor,
                )
            )
        self.nets = torch.nn.ModuleList(self.nets)

    def forward(self, x, y=None, rev=False):
        z = torch.zeros_like(x)
        # baaaaad
        if y is None:
            y = torch.zeros_like(x[:, :1])
        if self.con:
            data = torch.cat((y, x), dim=1)
        else:
            data = x
        self.log_jacobian = 0
        z[:, 0] = (
            self.nets[0]
            .forward(x=data[:, self.n_dimcon].unsqueeze(1), y=y, rev=rev)
            .squeeze()
        )
        self.log_jacobian += self.nets[0].log_jacobian
        for i in range(1, self.n_dim):
            if rev:
                if self.con:
                    data[:, : self.n_dimcon + i] = torch.cat((y, z[:, :i]), dim=1)
                else:
                    data[:, :i] = z[:, :i]
            z[:, i] = (
                self.nets[i]
                .forward(
                    x=data[:, self.n_dimcon + i].unsqueeze(1),
                    y=data[:, : self.n_dimcon + i],
                    rev=rev,
                )
                .squeeze()
            )
            self.log_jacobian = torch.cat(
                (self.log_jacobian, self.nets[i].log_jacobian), dim=1
            )
        return z

    def jacobian(self, x):
        return torch.sum(self.log_jacobian, dim=1)
