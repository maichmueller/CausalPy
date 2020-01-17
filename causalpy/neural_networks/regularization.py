from copy import copy
from functools import partial
from typing import (
    Iterator,
    Union,
    List,
    Callable,
    Optional,
    Tuple,
    Any,
    Iterable,
    Collection,
)

import torch
from torch.nn import Hardtanh
from torch.distributions import RelaxedBernoulli, Distribution
import numpy as np
from numpy.random import Generator
from .basemodel import NeuralBaseNet


def hard_sigmoid(x: Union[np.ndarray, torch.Tensor]):
    if isinstance(x, np.ndarray):
        x = torch.as_tensor(x)
    return torch.min(torch.ones_like(x), torch.max(torch.zeros_like(x), x))


class BinaryConcreteDist:
    def __init__(
        self,
        alpha: Union[float, Iterable] = 0.5,
        beta: Union[float, Iterable] = 1.0,
        seed: Optional[int] = None,
    ):
        if isinstance(alpha, torch.Tensor):
            pass  # preserve potential torch.nn.Parameter without copying them into standard Tensor
        elif isinstance(alpha, (float, Iterable)):
            alpha = torch.tensor(alpha)
        else:
            raise ValueError("Parameter 'alpha' needs to be either float or iterable.")

        if isinstance(alpha, torch.Tensor):
            pass  # preserve potential torch.nn.Parameter without copying them into standard Tensor
        elif isinstance(beta, float):
            beta = torch.tensor(beta)
        else:
            raise ValueError("Parameter 'beta' needs to be either float or iterable.")

        assert torch.all(alpha > 0), "Location parameter 'alpha' needs to be > 0."
        assert torch.all(beta > 0), "Temperature parameter 'beta' needs to be > 0."

        self.alpha = alpha  # the 'location'
        self.log_alpha = torch.log(alpha)
        self.beta = beta  # the 'temperature'
        self.uniform_dist = np.random.default_rng(seed).uniform

    def rsample(self, size: Tuple = (1,)):
        uniform_sample = torch.as_tensor(self.uniform_dist(size=size))
        logit_sample = torch.log(uniform_sample) - torch.log(-uniform_sample + 1)
        bin_concrete_sample = torch.sigmoid((logit_sample + self.log_alpha) / self.beta)
        return bin_concrete_sample

    def pdf(self, x: Union[np.ndarray, torch.Tensor, float]):
        beta_min_1 = -self.beta - 1
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x)
        numerator = torch.pow(x, beta_min_1) * torch.pow(-x + 1, beta_min_1)
        denominator = np.power(x, -self.beta) + np.power(1 - x, -self.beta)
        return (self.beta / self.alpha) * numerator / torch.pow(denominator, 2)

    def cdf(self, x: Union[np.ndarray, float]):
        logits = np.log(x) - np.log(1 - x)
        return torch.sigmoid(torch.as_tensor(logits * self.beta - self.log_alpha))

    def quantile(self, p: Union[np.ndarray, float]):
        p = torch.as_tensor(p)
        logits = torch.log(p) - torch.log(-p + 1)
        return torch.sigmoid((logits + self.log_alpha / self.beta)).clamp(min=0, max=1)


class HardConcreteDist:
    def __init__(
        self,
        q_dist: Any = BinaryConcreteDist(0.5, 0.1),
        gamma: Optional[float] = 0.5,
        zeta: Optional[float] = 2.0,
        seed: Optional[int] = None,
    ):
        self.q_dist = q_dist
        self.gamma = gamma
        self.zeta = zeta
        rs = np.random.default_rng(seed)
        self.uniform = rs.uniform(0, 1)

        # the function h(x) = min(1, max(0, x)) is implemented in torch as Hardtanh(0, 1)(x)
        self.hard_tanh = Hardtanh(0, 1)

    def get_base_dist(self):
        return self.q_dist

    def rsample(self, size: Tuple = (1,)):
        s = self.q_dist.rsample(size=size)
        s_tilde = s * (self.zeta - self.gamma) + self.gamma
        return self.hard_tanh(s_tilde)

    def pdf(self, x: Union[np.ndarray, torch.Tensor, float]):
        if 0 < x < 1:
            edge_probs = self.q_dist.quantile(np.ndarray([0, 1]))
            return (edge_probs[1] - edge_probs[0]) * self.q_dist.pdf(x)
        elif x == 1:
            return 1 - self.q_dist.quantile(1)
        elif x == 0:
            return self.q_dist.quantile(0)
        else:
            return 0

    def cdf(self, x: Union[Iterable, float]):
        raise NotImplementedError

    def quantile(self, p: Union[Iterable, float]):
        raise NotImplementedError


class L0Mask(torch.nn.Module):
    """
    The variable names in this class are chosen to resemble the names given to them in the original paper:
    'LEARNING SPARSE NEURAL NETWORKS THROUGH L0 REGULARIZATION' (https://arxiv.org/pdf/1712.01312.pdf)
    """

    EPS = 1e-8

    def __init__(
        self,
        parameters: Iterable,
        dimensions_in: Optional[int] = None,
        dimensions_out: Optional[int] = None,
        temperature: float = 2.0 / 3.0,
        gamma: float = -0.1,
        zeta: float = 1.1,
        initial_sparsity_rate: float = 0.5,
        device: Optional[torch.device] = None,
        transformation_f: Callable = torch.nn.Softplus(),
        seed: Optional[int] = None,
    ):
        super().__init__()
        if 0 < initial_sparsity_rate < 1:
            self.initial_sparsity_rate = initial_sparsity_rate
        else:
            self.initial_sparsity_rate = 0.5
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # the log alpha locations are learnable parameters
        log_alphas = []
        for params in parameters:
            log_alphas.append(
                torch.nn.Parameter(
                    torch.empty(params.shape, device=self.device), requires_grad=True
                )
            )

        self.nr_masks = len(log_alphas)
        self.log_alphas = torch.nn.ParameterList(log_alphas)

        self.initialize_weights()

        self.hard_concrete_dists = [
            HardConcreteDist(
                q_dist=BinaryConcreteDist(
                    alpha=log_alpha.exp(), beta=temperature, seed=seed
                ),
                gamma=gamma,
                zeta=zeta,
                seed=seed,
            )
            for log_alpha in self.log_alphas
        ]

        # the function named g in the paper, which computes z. It's mathematical form is:
        # g(x) = min(1, max(0, x))
        self.hard_tanh = Hardtanh(min_val=0, max_val=1)

        self.random_state = np.random.default_rng(seed)

    def initialize_weights(self):
        self.log_alphas.data.normal_(
            mean=np.log(1 - self.initial_sparsity_rate)
            - np.log(self.initial_sparsity_rate),
            std=1e-2,
        )

    def forward(self, batch_size: int):
        z = self.sample_mask(batch_size, deterministic=self.training)
        return z

    def _sample_epsilon(self, size=(1,), low=EPS, high=1 - EPS):
        """
        distribution of the noise 'epsilon'
        """
        return torch.as_tensor(self.random_state.uniform(low=low, high=high, size=size))

    def _cdf_stretched_concrete(self, x, layer_nr, *args, **kwargs):
        return (
            self.hard_concrete_dist[layer_nr]
            .get_base_dist()
            .cdf((x - self.gamma) / (self.zeta - self.gamma), *args, **kwargs)
        )

    def _quantile_stretched_concrete(self, p, layer_nr, *args, **kwargs):
        return (
            self.hard_concrete_dist[layer_nr]
            .get_base_dist()
            .quantile(p, *args, **kwargs)
            * (self.zeta - self.gamma)
            + self.gamma
        )

    def sample_mask(
        self,
        batch_size: int,
        weight_layer_indices: Optional[Iterable[int]] = None,
        deterministic: bool = False,
    ):
        r"""
        Sample the hard-concrete gates for training and use a deterministic value for testing

        Notes
        -----
        Quote paper (p.5):

        "at test time the following estimator is used for the final parameters:

        .. math:: \hat{z} = \min (1, \max(0, Sigmoid(\log \alpha)(\zeta - \gamma) + \gamma),
        .. math:: \theta^{\star} = \tilde{\theta}^{\star} \cdot \hat{z}

        """
        layers = (
            range(self.nr_masks)
            if weight_layer_indices is None
            else weight_layer_indices
        )

        masks = []
        if not deterministic:
            # this should be reached at train time
            for layer_idx in layers:
                z = self._quantile_stretched_concrete(
                    self._sample_epsilon(
                        (batch_size,) + self.log_alphas[layer_idx].shape
                    )
                )
                masks.append(self.hard_tanh(z))
        else:
            # this should be reached for test time:
            for layer_idx in layers:
                pi = (
                    torch.sigmoid(self.log_alphas[layer_idx])
                    .view(1, self.dim_in)
                    .expand(batch_size, self.dim_in)
                )
                masks.append(self.hard_tanh(pi * (self.zeta - self.gamma) + self.gamma))
        return masks

    def compute_loss(
        self,
        data_in: torch.Tensor,
        target: torch.Tensor,
        loss_func: Callable,
        model: torch.nn.Module,
        mcs_size: int,
    ):
        backups = []
        for mask_sample, weights in zip(
            self.sample_mask(batch_size=mcs_size), model.parameters()
        ):
            backups.append(
                weights.clone().detach().data
            )  # store a backup for reassigning after evaluation.
            weights.data = (
                weights.data.repeat((mcs_size,) + tuple(1 for _ in weights.data.shape))
                * mask_sample
            )
        loss = loss_func(model(data_in), target).mean()
        for backup, weights in zip(backups, model.parameters()):
            weights.data = backup
        return loss

    def l0_regularization(self):
        """
        Expected L0 norm under the stochastic gates
        """
        logpw = sum(
            (
                torch.sum((1 - self.hard_concrete_dists[i].get_base_dist().cdf(0)))
                for i in range(self.nr_masks)
            )
        )
        return logpw

    def l2_regularization(self, weights_iter: Iterable):
        """
        Expected L2 norm under the stochastic gates
        """
        logpw = sum(
            (
                torch.sum(
                    (1 - self._cdf_stretched_concrete(0, layer_nr=i)) * weights.pow(2)
                )
                for i, weights in zip(range(self.nr_masks), weights_iter)
            )
        )
        return logpw
