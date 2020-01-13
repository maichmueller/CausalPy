from functools import partial
from typing import Iterator, Union, List, Callable, Optional, Tuple, Any, Iterable

import torch
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
        self, alpha: float = 0.5, beta: float = 1.0, seed: Optional[int] = None
    ):
        assert alpha > 0, "Location parameter 'alpha' needs to be > 0."
        assert beta > 0, "Temperature parameter 'beta' needs to be > 0."

        self.alpha = alpha  # the 'location'
        self.log_alpha = np.log(alpha)
        self.beta = beta  # the 'temperature'
        self.uniform_dist = np.random.default_rng(seed).uniform

    def rsample(self, size: Tuple = (1,)):
        uniform_sample = torch.as_tensor(
            np.atleast_1d(self.uniform_dist(size=np.prod(size))).reshape(size)
        )
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

    def quantile(self, x: Union[np.ndarray, float]):
        x = torch.as_tensor(x)
        logits = torch.log(x) - torch.log(-x + 1)
        return torch.sigmoid(logits * self.beta - self.log_alpha).clamp(min=0, max=1)


class HardConcreteDist:
    def __init__(
        self,
        base_distribution: Any = BinaryConcreteDist(0.5, 0.1),
        gamma: Optional[float] = 0.5,
        zeta: Optional[float] = 2.0,
        seed: Optional[int] = None,
    ):
        self.base_distribution = base_distribution
        self.gamma = gamma
        self.zeta = zeta
        rs = np.random.default_rng(seed)
        self.uniform = rs.uniform(0, 1)

    def rsample(self, size: Tuple = (1,)):
        s = self.base_distribution.rsample(size=size)
        s_tilde = s * (self.zeta - self.gamma) + self.gamma
        hard_z = torch.min(
            torch.ones_like(s_tilde), torch.max(torch.zeros_like(s_tilde), s_tilde)
        )
        return hard_z


class L0Regularizer(torch.nn.Module):
    """
    The variable names in this class are chosen to resemble the names given to them in the original paper:
    'LEARNING SPARSE NEURAL NETWORKS THROUGH L0 REGULARIZATION' (https://arxiv.org/pdf/1712.01312.pdf)
    """

    EPS = 1e-8

    def __init__(
        self,
        parameters: Iterable,
        initial_sparsity_rate: float = 0.5,
        device: Optional[torch.device] = None,
        seed=None,
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

        self.weights = []
        for params in parameters:
            init_weights = torch.nn.Parameter(
                torch.empty(params.shape, device=self.device), requires_grad=True
            )
            init_weights.data.normal_(
                mean=np.log(1 - self.initial_sparsity_rate)
                - np.log(self.initial_sparsity_rate),
                std=1e-2,
            )
            self.weights.append(init_weights)
        self.weights = torch.nn.ParameterList(self.weights)

        self.random_state = np.random.default_rng(seed)

        self.transformation_f = (
            torch.nn.Softplus
        )  # the deterministic and differentiable transformation f

    def sample_epsilon(self, size=(1,), low=EPS, high=1-EPS):
        """
        distribution of the noise 'epsilon'
        """
        return torch.as_tensor(
            np.atleast_1d(self.random_state.uniform(low=low, high=high, size=np.prod(size))).reshape(size)
        )

    def _reg_w(self):
        """Expected L0 norm under the stochastic gates, takes into account and re-weights also a potential L2 penalty"""
        logpw_col = torch.sum(- (.5 * self.prior_prec * self.weights.pow(2)) - self.lamba, 1)
        logpw = torch.sum((1 - self.cdf_qz(0)) * logpw_col)
        logpb = 0 if not self.use_bias else - torch.sum(.5 * self.prior_prec * self.bias.pow(2))
        return logpw + logpb

    @staticmethod
    def g(z: torch.Tensor):
        return torch.min(torch.ones_like(z), torch.max(torch.zeros_like(z), z))
