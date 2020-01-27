from typing import Optional, Callable, Type

import torch
from .utils import get_jacobian
from abc import ABC, abstractmethod


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
        self.nr_layers = nr_layers
        self.nr_inverse_iters = 10

        self.mat1 = torch.randn(1, dim, nr_layers, requires_grad=False)
        self.bias1 = torch.randn(1, dim, nr_layers, requires_grad=False)
        self.mat2 = torch.randn(1, dim, nr_layers, requires_grad=False)
        self.bias2 = torch.randn(1, dim, requires_grad=False)
        self.eps = torch.ones(1, dim, requires_grad=False) * -1
        self.alpha = torch.zeros(1, dim, requires_grad=False)

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
                # ( dimension in = dim_condition, dimension out = 3 * dim * (layers + 1) )
                self.conditional_net: torch.nn.Module = conditional_net
        else:
            self.set_conditional_params(torch.zeros((1, dim_condition)))

            # only in the case of no conditional neural network are these trainable parameters.
            # Otherwise their data will be provided by the condition generating network.
            self.mat1 = torch.nn.Parameter(self.mat1, requires_grad=True)
            self.bias1 = torch.nn.Parameter(self.bias1, requires_grad=True)
            self.mat2 = torch.nn.Parameter(self.mat2, requires_grad=True)
            self.bias2 = torch.nn.Parameter(self.bias2, requires_grad=True)
            self.eps = torch.nn.Parameter(self.eps, requires_grad=True)
            self.alpha = torch.nn.Parameter(self.alpha, requires_grad=True)

    def forward(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        rev: bool = False,
    ):
        if self.is_conditional:
            self.set_conditional_params(condition)

        if not rev:
            self.log_jacobian_cache = torch.log(self.abl(x))
            return self.bijective_map(x)
        else:
            z = self.inverse(x, n=self.nr_inverse_iters)
            self.log_jacobian_cache = -torch.log(self.abl(z))
            return z

    @abstractmethod
    def inverse(self, x, n):
        raise NotImplementedError

    @abstractmethod
    def abl(self, x):
        raise NotImplementedError

    def set_conditional_params(self, condition_tensor: torch.Tensor):
        """
        In the case of a conditional INN set the parameters of the bijective map for the condition.

        Notes
        -----
        If this method was called although the INN is unconditional, this method will raise uncaught errors.
        """

        size = condition_tensor.size(0)
        new_params = self.conditional_net(condition_tensor)

        shapes_params = [
            (self.matrix_shape, self.mat1),
            (self.matrix_shape, self.mat2),
            (self.matrix_shape, self.bias1),
            (self.bias_shape, self.bias2),
            (self.bias_shape, self.eps),
            (self.bias_shape, self.alpha)
        ]
        for i, (new_shape, param) in enumerate(shapes_params):
            if i == 0:
                start = 0
            else:
                # i * nr_previous_params_elements
                start = i * shapes_params[i-1][1].nelement() // size
            stop = (i + 1) * param.nelement()

            param.data = torch.reshape(new_params[:, slice(start, stop)], new_shape).repeat(size, *new_shape)
        #
        # self.mat1 = torch.reshape(
        #     new_params[:, : self.dim * self.nr_layers], (size, self.dim, self.nr_layers)
        # )
        # self.bias1 = torch.reshape(
        #     new_params[:, self.dim * self.nr_layers : 2 * self.dim * self.nr_layers],
        #     (size, self.dim, self.nr_layers),
        # )
        # self.mat2 = torch.reshape(
        #     new_params[:, 2 * self.dim * self.nr_layers : 3 * self.dim * self.nr_layers],
        #     (size, self.dim, self.nr_layers),
        # )
        # self.bias2 = torch.reshape(
        #     new_params[
        #         :,
        #         3 * self.dim * self.nr_layers : 3 * self.dim * self.nr_layers
        #         + self.dim,
        #     ],
        #     (size, self.dim),
        # )
        # self.eps = (
        #     torch.reshape(
        #         new_params[
        #             :,
        #             3 * self.dim * self.nr_layers
        #             + self.dim : 3 * self.dim * self.nr_layers
        #             + 2 * self.dim,
        #         ],
        #         (size, self.dim),
        #     )
        #     / 10.0
        # )
        # self.alpha = (
        #     torch.reshape(
        #         new_params[:, 3 * self.dim * self.nr_layers + 2 * self.dim :],
        #         (size, self.dim),
        #     )
        #     / 10.0
        # )

    def jacobian(self, x: Optional[torch.Tensor] = None, rev: bool = False):
        if x is None:
            return self.log_jacobian_cache
        else:
            return get_jacobian(self, x, dim_in=1, dim_out=1, device=self.device)


class CouplingMonotone(CouplingBase):
    def __init__(self, dim=1, nr_layers=16, dim_condition=4, conditional_net=None):
        super().__init__(dim, nr_layers, dim_condition, conditional_net)
        self.elu = torch.nn.ELU()

    def activation_func(self, x):
        return self.elu(x)
        # return torch.nn.ReLU()(x)

    def activation_deriv(self, x):
        return 1 + self.activation_func(-torch.relu(-x))
        # return (x > torch.zeros_like(x)).float()

    def bijective_map(self, x):
        return (
            torch.exp(self.alpha)
            * (
                x
                + 0.8
                * torch.sigmoid(self.eps)
                * torch.sum(
                    self.activation_func(
                        x.unsqueeze(2).expand(-1, -1, self.nr_layers) * self.mat1
                        + self.bias1
                    )
                    * self.mat2,
                    dim=2,
                )
                / (torch.sum(torch.relu(-self.mat1 * self.mat2), dim=2) + 1)
            )
            + self.bias2
        )

    def abl(self, x):
        return torch.exp(self.alpha) * (
            1
            + 0.8
            * torch.sigmoid(self.eps)
            * torch.sum(
                self.activation_deriv(
                    x.unsqueeze(2).expand(-1, -1, self.nr_layers) * self.mat1
                    + self.bias1
                )
                * self.mat1
                * self.mat2,
                dim=2,
            )
            / (torch.sum(torch.nn.ReLU()(-self.mat1 * self.mat2), dim=2) + 1)
        )

    def inverse(self, y, n=10):
        yn = y  # * torch.exp(-self.alpha) - self.bias2
        for i in range(n):
            yn = yn - (self.bijective_map(yn) - y) / self.abl(yn)
        return yn


class CouplingGeneral(CouplingBase):
    def __init__(self, dim=1, nr_layers=16, dim_condition=4, conditional_net=None):
        super().__init__(dim, nr_layers, dim_condition, conditional_net)
        self.clamp = 5

    def actfunc(self, x):
        # return torch.tanh(x)
        return torch.exp(-(x ** 2)) * torch.tensor(1.16)

    def actderiv(self, x):
        # return 1-self.actfunc(x)**2
        return -2 * x * self.actfunc(x)

    def bijective_map(self, x):
        return (
            torch.exp(self.alpha)
            * (
                x
                + 0.8
                * torch.sigmoid(self.eps)
                * torch.sum(
                    self.actfunc(
                        x.unsqueeze(2).expand(-1, -1, self.nr_layers) * self.mat1
                        + self.bias1
                    )
                    * self.mat2,
                    dim=2,
                )
                / (torch.sum(torch.abs(self.mat1 * self.mat2), dim=2) + 1)
            )
            + self.bias2
        )

    def abl(self, x):
        return torch.exp(self.alpha) * (
            1
            + 0.8
            * torch.sigmoid(self.eps)
            * torch.sum(
                self.actderiv(
                    x.unsqueeze(2).expand(-1, -1, self.nr_layers) * self.mat1
                    + self.bias1
                )
                * self.mat1
                * self.mat2,
                dim=2,
            )
            / (torch.sum(torch.abs(self.mat1 * self.mat2), dim=2) + 1)
        )

    def inverse(self, y, n=5):
        yn = y  # *torch.exp(-self.alpha)-self.bias2
        for i in range(n):
            yn = yn - (self.function(yn) - y) / self.abl(yn)
        return yn


class INN(torch.nn.Module):
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
                x = block.forward(x=x, condition=condition)
                self.log_jacobian_cache += block.jacobian()
            return x
        else:
            for block in self.blocks[::-1]:
                x = block.forward(x=x, condition=condition, rev=True)
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
        self, n_dim=3, n_dimcon=0, n_blocks=1, ls=32, net=INN, subnet_constructor=None
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
