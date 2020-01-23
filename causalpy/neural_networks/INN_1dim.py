import torch


class PfndBase(torch.nn.Module):
    def __init__(self, dim=1, ls=16, n_condim=4, subnet_constructor=None):
        super().__init__()
        self.sig = torch.nn.Sigmoid()
        self.log_jacobian = 0
        self.log_jacobian_latest = None
        self.dim = dim
        self.ls = ls
        self.inviter = 10
        if n_condim > 0:
            self.con = True
            if subnet_constructor is None:
                self.subnet = torch.nn.Sequential(
                    torch.nn.Linear(n_condim, 50),
                    torch.nn.ReLU(),
                    torch.nn.Linear(50, 3 * dim * (ls + 1)),
                )
            else:
                self.subnet = subnet_constructor(n_condim, 3 * dim * (ls + 1))
            self.getParas(torch.zeros(1, n_condim))
        else:
            self.con = False
            self.mat1 = torch.nn.Parameter(torch.randn(1, dim, ls), True)
            self.bias1 = torch.nn.Parameter(torch.randn(1, dim, ls), True)
            self.mat2 = torch.nn.Parameter(torch.randn(1, dim, ls), True)
            self.bias2 = torch.nn.Parameter(torch.randn(1, dim), True)
            self.eps = torch.nn.Parameter(torch.zeros(1, dim) - 1, True)
            self.alpha = torch.nn.Parameter(torch.zeros(1, dim), True)

    def actfunc(self, x):
        raise NotImplementedError

    def actderiv(self, x):
        raise NotImplementedError

    def forward(self, x, y=None, rev=False):
        if self.con:
            self.getParas(y)
        if not rev:
            self.log_jacobian_latest = torch.log(self.abl(x))
            return self.function(x)
        else:
            z = self.inv(x, n=self.inviter)
            self.log_jacobian_latest = -torch.log(self.abl(z))
            return z

    def getParas(self, y):
        if self.con:
            size = len(y)
            param = self.subnet(y)
            dim_x_ls = self.dim * self.ls
            self.mat1 = torch.reshape(param[:, :dim_x_ls], (size, self.dim, self.ls))
            self.bias1 = torch.reshape(
                param[:, dim_x_ls : 2 * dim_x_ls], (size, self.dim, self.ls),
            )
            self.mat2 = torch.reshape(
                param[:, 2 * dim_x_ls : 3 * dim_x_ls], (size, self.dim, self.ls),
            )
            self.bias2 = torch.reshape(
                param[:, 3 * dim_x_ls : 3 * dim_x_ls + self.dim], (size, self.dim),
            )
            self.eps = (
                torch.reshape(
                    param[:, 3 * dim_x_ls + self.dim : 3 * dim_x_ls + 2 * self.dim],
                    (size, self.dim),
                )
                / 10.0
            )
            self.alpha = (
                torch.reshape(param[:, 3 * dim_x_ls + 2 * self.dim :], (size, self.dim))
                / 10.0
            )

    def jacobian(self, x, rev=False):
        return self.log_jacobian

    def abl(self, x):
        return torch.exp(self.alpha) * (
            1
            + 0.8
            * self.sig(self.eps)
            * torch.sum(
                self.actderiv(
                    x.unsqueeze(2).expand(-1, -1, self.ls) * self.mat1 + self.bias1
                )
                * self.mat1
                * self.mat2,
                dim=2,
            )
            / (torch.sum(torch.abs(self.mat1 * self.mat2), dim=2) + 1)
        )

    def inv(self, y, n=5):
        yn = y  # *torch.exp(-self.alpha)-self.bias2
        for i in range(n):
            yn = yn - (self.function(yn) - y) / self.abl(yn)
        return yn


class PfndELU(PfndBase):
    def __init__(self, dim=1, ls=16, n_condim=4, subnet_constructor=None):
        super().__init__(dim, ls, n_condim, subnet_constructor)

    def actfunc(self, x):
        return torch.nn.ELU()(x)
        # return torch.nn.ReLU()(x)

    def actderiv(self, x):
        return 1 + self.actfunc(-torch.relu(-x))
        # return (x > torch.zeros_like(x)).float()

    def function(self, x):
        return (
            torch.exp(self.alpha)
            * (
                x
                + 0.8
                * self.sig(self.eps)
                * torch.sum(
                    self.actfunc(
                        x.unsqueeze(2).expand(-1, -1, self.ls) * self.mat1 + self.bias1
                    )
                    * self.mat2,
                    dim=2,
                )
                / (torch.sum(torch.relu(-self.mat1 * self.mat2), dim=2) + 1)
            )
            + self.bias2
        )


class PfndTanh(PfndBase):
    def __init__(self, dim=1, ls=16, n_condim=4, subnet_constructor=None):
        super().__init__(dim, ls, n_condim, subnet_constructor)
        self.clamp = 5

    def actfunc(self, x):
        # return torch.tanh(x)
        return torch.exp(-(x ** 2)) * torch.tensor(1.16)

    def actderiv(self, x):
        # return 1-self.actfunc(x)**2
        return -2 * x * self.actfunc(x)

    def function(self, x):
        return (
            torch.exp(self.alpha)
            * (
                x
                + 0.8
                * self.sig(self.eps)
                * torch.sum(
                    self.actfunc(
                        x.unsqueeze(2).expand(-1, -1, self.ls) * self.mat1 + self.bias1
                    )
                    * self.mat2,
                    dim=2,
                )
                / (torch.sum(torch.abs(self.mat1 * self.mat2), dim=2) + 1)
            )
            + self.bias2
        )


class Net(torch.nn.Module):
    def __init__(self, n_blocks=3, n_dim=1, ls=16, n_condim=4, subnet_constructor=None):
        super().__init__()
        self.n_blocks = n_blocks
        mods = []
        for i in range(n_blocks):
            mods.append(
                PfndTanh(
                    dim=n_dim,
                    ls=ls,
                    n_condim=n_condim,
                    subnet_constructor=subnet_constructor,
                )
            )
            mods.append(
                PfndELU(
                    dim=n_dim,
                    ls=ls,
                    n_condim=n_condim,
                    subnet_constructor=subnet_constructor,
                )
            )
        self.blocks = torch.nn.ModuleList(mods)
        self.log_jacobian_latest = torch.zeros(n_dim)
        self.compute_jacobian_iteratively = True

    def forward(self, x, y=None, rev=False):
        # rev = reverse mode
        self.log_jacobian_latest = 0.0
        if not rev:
            for block in self.blocks:
                x = block.forward(x=x, y=y)
                if self.training:
                    self.log_jacobian_latest += block.log_jacobian_latest
            return x
        else:
            for block in self.blocks[::-1]:
                x = block.forward(x=x, y=y, rev=True)
                if self.training:
                    self.log_jacobian_latest += block.log_jacobian_latest
            return x


class CN(torch.nn.Module):
    def __init__(self, n_blocks=1, n_dim=1, ls=16, n_condim=0, subnet_constructor=None):
        super().__init__()
        self.n_blocks = n_blocks
        mods = []
        for i in range(n_blocks):
            # mods.append(GlowLikeCouplingBlock(dims_in=n_dim, subnet_constructor=subnet_constructor, net=PfndTanh))
            mods.append(
                GlowLikeCouplingBlock(
                    dims_in=n_dim, subnet_constructor=subnet_constructor, net=PfndELU
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
        self, dims_in, dims_c=None, subnet_constructor=None, clamp=5.0, net=PfndELU
    ):
        super().__init__()
        # channels = dims_in[0][0]
        # self.ndims = len(dims_in[0])
        channels = dims_in
        if dims_c is None:
            dims_c = []
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
        self, n_dim=3, n_dimcon=0, n_blocks=1, ls=32, net=Net, subnet_constructor=None
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
