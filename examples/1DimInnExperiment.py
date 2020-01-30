import itertools
from typing import Union, Collection

import visdom
import torch
from torch.nn import Parameter, Module
from torch.utils.data import DataLoader, TensorDataset
import math

from tqdm.auto import tqdm

from causalpy.neural_networks import cINN
import pandas as pd
import numpy as np
from examples.simulation_linear import simulate
from causalpy.neural_networks import L0InputGate
from causalpy.causal_prediction.interventional import ICPredictor


def s(x):
    return x.squeeze().cpu()


def std(x):
    return 1 / math.sqrt(2 * math.pi) * torch.exp(-(x ** 2) / 2)


def subnet_fc(c_in, c_out):
    return torch.nn.Sequential(
        torch.nn.Linear(c_in, 100), torch.nn.ReLU(), torch.nn.Linear(100, c_out)
    )


class MMDStatistic:
    r"""The *unbiased* MMD test of :cite:`gretton2012kernel`.
    The kernel used is equal to:
    .. math ::
        k(x, x') = \sum_{j=1}^k e^{-\alpha_j\|x - x'\|^2},
    for the :math:`\alpha_j` proved in :py:meth:`~.MMDStatistic.__call__`.
    Arguments
    ---------
    n_1: int
        The number of points in the first sample.
    n_2: int
        The number of points in the second sample."""

    def __init__(self, n_1, n_2):
        self.n_1 = n_1
        self.n_2 = n_2

        # The three constants used in the test.
        self.a00 = 1.0 / (n_1 * (n_1 - 1))
        self.a11 = 1.0 / (n_2 * (n_2 - 1))
        self.a01 = -1.0 / (n_1 * n_2)

    def __call__(self, sample_1, sample_2, alphas, ret_matrix=False):
        r"""Evaluate the statistic.
        The kernel used is
        .. math::
            k(x, x') = \sum_{j=1}^k e^{-\alpha_j \|x - x'\|^2},
        for the provided ``alphas``.
        Arguments
        ---------
        sample_1: :class:`torch:torch.autograd.Variable`
            The first sample, of size ``(n_1, d)``.
        sample_2: variable of shape (n_2, d)
            The second sample, of size ``(n_2, d)``.
        alphas : list of :class:`float`
            The kernel parameters.
        ret_matrix: bool
            If set, the call with also return a second variable.
            This variable can be then used to compute a p-value using
            :py:meth:`~.MMDStatistic.pval`.
        Returns
        -------
        :class:`float`
            The test statistic.
        :class:`torch:torch.autograd.Variable`
            Returned only if ``ret_matrix`` was set to true."""
        sample_12 = torch.cat((sample_1, sample_2), 0)
        distances = self.pdist(sample_12, sample_12, norm=2)

        kernels = None
        for alpha in alphas:
            kernels_a = torch.exp(-alpha * distances ** 2)
            if kernels is None:
                kernels = kernels_a
            else:
                kernels = kernels + kernels_a

        k_1 = kernels[: self.n_1, : self.n_1]
        k_2 = kernels[self.n_1 :, self.n_1 :]
        k_12 = kernels[: self.n_1, self.n_1 :]

        mmd = (
            2 * self.a01 * k_12.sum()
            + self.a00 * (k_1.sum() - torch.trace(k_1))
            + self.a11 * (k_2.sum() - torch.trace(k_2))
        )
        if ret_matrix:
            return mmd, kernels
        else:
            return mmd

    def pval(self, distances, n_permutations=1000):
        r"""Compute a p-value using a permutation test.
        Arguments
        ---------
        matrix: :class:`torch:torch.autograd.Variable`
            The matrix computed using :py:meth:`~.MMDStatistic.__call__`.
        n_permutations: int
            The number of random draws from the permutation null.
        Returns
        -------
        float
            The estimated p-value."""
        if isinstance(distances, Parameter):
            distances = distances.data
        return self.permutation_test_mat(
            distances.cpu().numpy(),
            self.n_1,
            self.n_2,
            n_permutations,
            a00=self.a00,
            a11=self.a11,
            a01=self.a01,
        )

    @staticmethod
    def permutation_test_mat(
        matrix: np.ndarray,
        n_1: int,
        n_2: int,
        n_permutations: int,
        a00: float = 1,
        a11: float = 1,
        a01: float = 0,
    ):
        """Compute the p-value of the following statistic (rejects when high)

            \sum_{i,j} a_{\pi(i), \pi(j)} matrix[i, j].
        """
        n = n_1 + n_2

        pi = np.zeros(n, dtype=np.int8)
        pi[n_1:] = 1

        larger = 0.0
        statistic = 0.0
        for sample_n in range(1 + n_permutations):
            count = 0.0
            for i in range(n):
                for j in range(i, n):
                    mij = matrix[i, j] + matrix[j, i]
                    if pi[i] == pi[j] == 0:
                        count += a00 * mij
                    elif pi[i] == pi[j] == 1:
                        count += a11 * mij
                    else:
                        count += a01 * mij
            if sample_n == 0:
                statistic = count
            elif statistic <= count:
                larger += 1

            np.random.shuffle(pi)

        return larger / n_permutations

    @staticmethod
    def pdist(sample_1, sample_2, norm=2, eps=1e-5):
        r"""Compute the matrix of all squared pairwise distances.
        Arguments
        ---------
        sample_1 : torch.Tensor or Variable
            The first sample, should be of shape ``(n_1, d)``.
        sample_2 : torch.Tensor or Variable
            The second sample, should be of shape ``(n_2, d)``.
        norm : float
            The l_p norm to be used.
        Returns
        -------
        torch.Tensor or Variable
            Matrix of shape (n_1, n_2). The [i, j]-th entry is equal to
            ``|| sample_1[i, :] - sample_2[j, :] ||_p``."""
        n_1, n_2 = sample_1.size(0), sample_2.size(0)
        norm = float(norm)
        if norm == 2.0:
            norms_1 = torch.sum(sample_1 ** 2, dim=1, keepdim=True)
            norms_2 = torch.sum(sample_2 ** 2, dim=1, keepdim=True)
            norms = norms_1.expand(n_1, n_2) + norms_2.transpose(0, 1).expand(n_1, n_2)
            distances_squared = norms - 2 * sample_1.mm(sample_2.t())
            return torch.sqrt(eps + torch.abs(distances_squared))
        else:
            dim = sample_1.size(1)
            expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
            expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
            differences = torch.abs(expanded_1 - expanded_2) ** norm
            inner = torch.sum(differences, dim=2, keepdim=False)
            return (eps + inner) ** (1.0 / norm)


if __name__ == "__main__":

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    viz = visdom.Visdom()
    seed = 0
    np.random.seed(seed)

    nr_genes = 50
    scm = simulate(nr_genes, 2)
    target_gene = np.random.choice(
        list(scm.get_variables())[len(list(scm.get_variables())) // 2 :]
    )
    target_parents = list(scm.graph.predecessors(target_gene))
    scm.reseed(seed)
    environments = []
    sample_data = [scm.sample(100)]
    environments += [0] * 100
    for parent in target_parents:
        scm.do_intervention([parent], [0])
        sample_data.append(scm.sample(100))
        environments += [environments[-1] + 1] * 100
    data = pd.concat(sample_data, sort=True)
    environments = np.array(environments)
    target_data = data[target_gene]
    data = data.drop(columns=target_gene)
    envs_unique = np.unique(environments)
    env_map = {e: np.where(environments == e)[0] for e in envs_unique}

    data, target_data = list(map(lambda x: torch.as_tensor(x.to_numpy()).float().to(dev), (data, target_data)))

    dim_condition = data.shape[1]
    cINN_list = [
        cINN(nr_blocks=4, dim=1, nr_layers=30, dim_condition=dim_condition).to(dev)
        for _ in envs_unique
    ]
    l0mask = L0InputGate(data.shape[1], monte_carlo_sample_size=100).to(dev)

    optimizer = torch.optim.Adam(
        itertools.chain(l0mask.parameters(), *[c_inn.parameters() for c_inn in cINN_list]),
        lr=0.01,
        weight_decay=1e-5,
    )
    epochs = 100

    vis_wins = {"map": [], "density": []}

    for env, cinn in enumerate(cINN_list):
        x = torch.arange(-3, 3, 6/data.shape[0]).unsqueeze(1).to(dev)
        z = cinn(
            x=x, condition=torch.zeros(data.shape[0], dim_condition).to(dev), rev=False
        )
        vis_wins["map"].append(viz.line(X=s(x), Y=s(z), opts=dict(title=f"Forward Map Env {env}")))
        vis_wins["density"].append(viz.line(
            X=s(x),
            Y=s(std(z) * torch.exp(cinn.log_jacobian_cache)),
            opts=dict(title=f"Estimated Density Env {env}"),
            )
        )

    for i in tqdm(range(epochs)):
        optimizer.zero_grad()

        masked_conditional_data = l0mask(data)
        loss = torch.zeros(1).to(dev)
        for env, env_indices in env_map.items():
            env_masked_cond_data = masked_conditional_data[env_indices]
            env_data_in = torch.arange(-3, 3, 6/env_masked_cond_data.shape[0]).unsqueeze(1).to(dev)
            env_cinn = cINN_list[env]
            z = env_cinn(x=env_data_in, condition=env_masked_cond_data, rev=False)

            log_grad = env_cinn.log_jacobian_cache
            loss += (z ** 2 / 2 - log_grad).mean()

            if i % 1 == 0:
                viz.line(X=s(env_data_in), Y=s(z), win=vis_wins["map"][env], opts=dict(title=f"Forward Map Env {env}"))
                viz.line(
                    X=s(env_data_in),
                    Y=s(std(z) * torch.exp(env_cinn.log_jacobian_cache)),
                    win=vis_wins["density"][env],
                    opts=dict(title=f"Estimated Density Env {env}"),
                )

        loss.backward()
        optimizer.step()


