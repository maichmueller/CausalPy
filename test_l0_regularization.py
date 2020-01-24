import itertools

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm.auto import tqdm
import matplotlib.pyplot as plt

from causalpy.neural_networks import L0Mask, FCNet

from visdom import Visdom


class VisdomLinePlotter(object):
    """Plots to Visdom"""
    def __init__(self, env_name='main'):
        self.viz = Visdom()
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x,x]), Y=np.array([y,y]), env=self.env, opts=dict(
                legend=[split_name],
                title=title_name,
                xlabel='Epochs',
                ylabel=var_name
            ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name], name=split_name, update = 'append')

limit_a, limit_b, epsilon = -.1, 1.1, 1e-6
import math
from torch.nn import Parameter, Module, init
import torch.nn.functional as F
class L0Dense(Module):
    """Implementation of L0 regularization for the input units of a fully connected layer"""

    def __init__(self, in_features, out_features, bias=True, weight_decay=1., droprate_init=0.5, temperature=2./3.,
                 lamba=1., local_rep=False, **kwargs):
        """
        :param in_features: Input dimensionality
        :param out_features: Output dimensionality
        :param bias: Whether we use a bias
        :param weight_decay: Strength of the L2 penalty
        :param droprate_init: Dropout rate that the L0 gates will be initialized to
        :param temperature: Temperature of the concrete distribution
        :param lamba: Strength of the L0 penalty
        :param local_rep: Whether we will use a separate gate sample per element in the minibatch
        """
        super(L0Dense, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_prec = weight_decay
        self.weights = Parameter(torch.Tensor(in_features, out_features))
        self.qz_loga = Parameter(torch.Tensor(in_features))
        self.temperature = temperature
        self.droprate_init = droprate_init if droprate_init != 0. else 0.5
        self.lamba = lamba
        self.use_bias = False
        self.local_rep = local_rep
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
            self.use_bias = True
        self.floatTensor = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor
        self.reset_parameters()
        print(self)

    def reset_parameters(self):
        init.kaiming_normal(self.weights, mode='fan_out')

        self.qz_loga.data.normal_(math.log(1 - self.droprate_init) - math.log(self.droprate_init), 1e-2)

        if self.use_bias:
            self.bias.data.fill_(0)

    def constrain_parameters(self, **kwargs):
        self.qz_loga.data.clamp_(min=math.log(1e-2), max=math.log(1e2))

    def cdf_qz(self, x):
        """Implements the CDF of the 'stretched' concrete distribution"""
        xn = (x - limit_a) / (limit_b - limit_a)
        logits = math.log(xn) - math.log(1 - xn)
        return F.sigmoid(logits * self.temperature - self.qz_loga).clamp(min=epsilon, max=1 - epsilon)

    def quantile_concrete(self, x):
        """Implements the quantile, aka inverse CDF, of the 'stretched' concrete distribution"""
        y = F.sigmoid((torch.log(x) - torch.log(1 - x) + self.qz_loga) / self.temperature)
        return y * (limit_b - limit_a) + limit_a

    def _reg_w(self):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        logpw_col = torch.sum(- (.5 * self.prior_prec * self.weights.pow(2)) - self.lamba, 1)
        logpw = torch.sum((1 - self.cdf_qz(0)) * logpw_col)
        logpb = 0 if not self.use_bias else - torch.sum(.5 * self.prior_prec * self.bias.pow(2))
        return logpw + logpb

    def regularization(self):
        return self._reg_w()

    def count_expected_flops_and_l0(self):
        """Measures the expected floating point operations (FLOPs) and the expected L0 norm"""
        # dim_in multiplications and dim_in - 1 additions for each output neuron for the weights
        # + the bias addition for each neuron
        # total_flops = (2 * in_features - 1) * out_features + out_features
        ppos = torch.sum(1 - self.cdf_qz(0))
        expected_flops = (2 * ppos - 1) * self.out_features
        expected_l0 = ppos * self.out_features
        if self.use_bias:
            expected_flops += self.out_features
            expected_l0 += self.out_features
        return expected_flops.data[0], expected_l0.data[0]

    def get_eps(self, size):
        """Uniform random numbers for the concrete distribution"""
        eps = self.floatTensor(size).uniform_(epsilon, 1-epsilon)
        eps.requires_grad_(True)
        return eps

    def sample_z(self, batch_size, sample=True):
        """Sample the hard-concrete gates for training and use a deterministic value for testing"""
        if sample:
            eps = self.get_eps(self.floatTensor(batch_size, self.in_features))
            z = self.quantile_concrete(eps)
            return F.hardtanh(z, min_val=0, max_val=1)
        else:  # mode
            pi = F.sigmoid(self.qz_loga).view(1, self.in_features).expand(batch_size, self.in_features)
            return F.hardtanh(pi * (limit_b - limit_a) + limit_a, min_val=0, max_val=1)

    def sample_weights(self):
        z = self.quantile_concrete(self.get_eps(self.floatTensor(self.in_features)))
        mask = F.hardtanh(z, min_val=0, max_val=1)
        return mask.view(self.in_features, 1) * self.weights

    def forward(self, input):
        if self.local_rep or not self.training:
            z = self.sample_z(input.size(0), sample=self.training)
            xin = input.mul(z)
            output = xin.mm(self.weights)
        else:
            weights = self.sample_weights()
            output = input.mm(weights)
        if self.use_bias:
            output.add_(self.bias)
        return output

    def __repr__(self):
        s = ('{name}({in_features} -> {out_features}, droprate_init={droprate_init}, '
             'lamba={lamba}, temperature={temperature}, weight_decay={prior_prec}, '
             'local_rep={local_rep}')
        if not self.use_bias:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)


import torch


class Single(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.layer = torch.nn.Parameter(0.5 * torch.ones(dim), requires_grad=True)

    def forward(self, input):
        return input * self.layer


if __name__ == "__main__":
    global plotter
    plotter = VisdomLinePlotter(env_name='Tutorial Plots')

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # l0 = L0Dense(6, 1).to(dev)
    # x = torch.rand(1000, 6).to(dev)
    # y = (2*x[:, 1]).view(-1, 1)
    # dataset = TensorDataset(x, y)
    # l0 = L0Mask(fc_inp).to(dev)
    # # # params = [fc.parameters(), l0.parameters()]
    # # # params = [fc.parameters()]
    # optimizer = torch.optim.Adam(itertools.chain(l0.parameters()), lr=0.01)
    # optimizer.zero_grad()
    # loss_f = torch.nn.MSELoss()
    # losses = []
    # # # print(*l0.sample_mask(1, deterministic=True), sep="\n")
    # # # print("FC params")
    # # # print(*fc.parameters(), sep="\n")
    # for epoch in tqdm(range(1000)):
    #     for data, target in DataLoader(dataset, 10000):
    #         loss = 0
    #         mcs_size = 10
    #         for i in range(mcs_size):
    #             loss += loss_f(fc_inp(data), target)
    #         loss /= mcs_size
    #         loss += 100 * l0.l2_regularization()
    #         # loss = loss_f(fc(data), target)
    #         # print(loss)
    #         if loss < 0.1:
    #             print(l0.sample_z(1, sample=False))
    #         losses.append(loss.detach().item())
    #         plotter.plot('loss', 'train', 'Class Loss', epoch, np.array(losses)[-1:].mean())
    #         loss.backward()
    #         # for pa in l0.log_alphas:
    #         #     print(pa.grad)
    #         optimizer.step()

    # fc_inp = FCNet(6, 1)
    # # fc = FCNet(3, 20, 10, 1).to(dev)
    # x = torch.rand(10000, 6).to(dev)
    # y = (50 * x[:, 1]).view(-1, 1)
    # y = (y - y.mean()) / y.std(unbiased=True)
    # dataset = TensorDataset(x, y)
    # l0 = L0Mask(fc_inp).to(dev)
    # # # params = [fc.parameters(), l0.parameters()]
    # # # params = [fc.parameters()]
    # optimizer = torch.optim.Adam(itertools.chain(l0.parameters()), lr=0.01)
    # optimizer.zero_grad()
    # loss_f = torch.nn.MSELoss()
    # losses = []
    # # # print(*l0.sample_mask(1, deterministic=True), sep="\n")
    # # # print("FC params")
    # # # print(*fc.parameters(), sep="\n")
    # for epoch in tqdm(range(1000)):
    #     for data, target in DataLoader(dataset, 10000):
    #         loss = 0
    #         mcs_size = 10
    #         evals = []
    #         for i in range(mcs_size):
    #             evals.append(fc_inp(data))
    #         for ev in evals:
    #             loss += loss_f(ev, target)
    #         loss /= mcs_size
    #
    #         loss += 0.01 * l0.l2_regularization()
    #         print(l0.l2_regularization())
    #         # loss = loss_f(fc(data), target)
    #         print(l0.sample_mask(1, deterministic=True)[0])
    #         losses.append(loss.detach().item())
    #         plotter.plot('loss', 'train', 'Class Loss', epoch, np.mean(losses))
    #         loss.backward()
    #         # for loga in  l0.log_alphas:
    #         #     print(loga.grad)
    #         # for pa in l0.log_alphas:
    #         #     print(pa.grad)
    #         optimizer.step()
    # # print(*l0.sample_mask(1, deterministic=True), sep="\n")
    # # print("FC params")
    # # print(*fc.parameters(), sep="\n")
    # plt.plot(losses)
    # plt.show()
    # gate = GateLayer(6)
    fc_inp = FCNet(6, 1)
    single = Single(6)
    l0 = L0Mask(single)
    # fc = FCNet(3, 20, 10, 1).to(dev)
    x = torch.rand(10000, 6).to(dev)
    y = (50 * x[:, 1]).view(-1, 1)
    y = (y - y.mean()) / y.std(unbiased=True)
    dataset = TensorDataset(x, y)
    # l0 = L0Mask(fc_inp).to(dev)
    # # params = [fc.parameters(), l0.parameters()]
    # # params = [fc.parameters()]
    optimizer = torch.optim.Adam(itertools.chain(fc_inp.parameters(), l0.parameters()), lr=0.01)
    optimizer.zero_grad()
    loss_f = torch.nn.MSELoss()
    losses = []
    # # print(*l0.sample_mask(1, deterministic=True), sep="\n")
    # # print("FC params")
    # # print(*fc.parameters(), sep="\n")
    for epoch in tqdm(range(1000)):
        for data, target in DataLoader(dataset, 10000):
            loss = 0
            mcs_size = 10
            evals = []
            # for i in range(mcs_size):
            #     # evals.append(fc_inp(data))
            loss += loss_f(fc_inp(single(data)), target)
            # loss /= mcs_size

            # loss += 0.05 * gate.complexity()
            loss += 0.05 * l0.l0_regularization()
            print("")
            # print(gate.final_layer().detach())
            # loss = loss_f(fc(data), target)
            print(l0.sample_mask(1, deterministic=True)[0])
            losses.append(loss.detach().item())
            plotter.plot('loss', 'train', 'Class Loss', epoch, np.array(losses)[-1:].mean())
            loss.backward()
            # for loga in  l0.log_alphas:
            #     print(loga.grad)
            # for pa in l0.log_alphas:
            #     print(pa.grad)
            optimizer.step()
