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


class BinaryConcreteDist:
    def __init__(
        self,
        log_alpha: Union[float, Iterable] = 0.0,
        beta: Union[float, Iterable] = 1.0,
        seed: Optional[int] = None,
    ):
        if isinstance(log_alpha, torch.Tensor):
            pass  # preserve potential torch.nn.Parameter without copying them into standard Tensor
        elif isinstance(log_alpha, (float, Iterable)):
            log_alpha = torch.tensor(log_alpha)
        else:
            raise ValueError("Parameter 'alpha' needs to be either float or iterable.")

        if isinstance(beta, torch.Tensor):
            pass  # preserve potential torch.nn.Parameter without copying them into standard Tensor
        elif isinstance(beta, float):
            beta = torch.tensor(beta)
        else:
            raise ValueError("Parameter 'beta' needs to be either float or iterable.")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        assert torch.all(
            torch.as_tensor(beta, device=self.device) > 0
        ), "Temperature parameter 'beta' needs to be > 0."

        self.log_alpha = log_alpha.to(device=self.device)
        self.beta = beta  # the 'temperature'
        self.uniform_dist = np.random.default_rng(seed).uniform

    def rsample(self, size: Tuple = (1,)):
        uniform_sample = torch.as_tensor(
            self.uniform_dist(size=size), device=self.device
        )
        uniform_sample.requires_grad_(True)
        logit_sample = torch.log(uniform_sample) - torch.log(-uniform_sample + 1)
        bin_concrete_sample = torch.sigmoid((logit_sample + self.log_alpha) / self.beta)
        return bin_concrete_sample

    def pdf(self, x: Union[np.ndarray, torch.Tensor, float]):
        if isinstance(x, np.ndarray):
            x = torch.as_tensor(x, dtype=torch.float, device=self.device)
        beta_min_1 = -self.beta - 1
        numerator = torch.pow(x, beta_min_1) * torch.pow(-x + 1, beta_min_1)
        denominator = np.power(x, -self.beta) + np.power(1 - x, -self.beta)
        return (
            (self.beta / torch.exp(self.log_alpha))
            * numerator
            / torch.pow(denominator, 2)
        )

    def cdf(self, x: Union[torch.Tensor, np.ndarray, float]):
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(x, dtype=torch.float, device=self.device)
        # TODO check whether this needs numeric adaptation or you simply didnt fully understand the paper here for the
        # TODO complexity loss!
        return torch.sigmoid(
            (torch.log(x) - torch.log(-x + 1)) * self.beta - self.log_alpha
        )

    def quantile(self, p: Union[torch.Tensor, np.ndarray, float]):
        p = torch.as_tensor(p, dtype=torch.float, device=self.device)
        return torch.sigmoid(
            (torch.log(p) - torch.log(-p + 1) + self.log_alpha / self.beta)
        ).clamp(min=0, max=1)


class HardConcreteDist:
    def __init__(
        self,
        binary_concrete_dist: Optional[BinaryConcreteDist] = None,
        gamma: Optional[float] = 0.5,
        zeta: Optional[float] = 2.0,
        seed: Optional[int] = None,
    ):

        self.bc_dist = (
            BinaryConcreteDist(0.5, 0.1)
            if binary_concrete_dist is None
            else binary_concrete_dist
        )

        self.gamma = gamma
        self.zeta = zeta
        rs = np.random.default_rng(seed)
        self.uniform = rs.uniform(0, 1)

        # the function h(x) = min(1, max(0, x)) is implemented in torch as Hardtanh(0, 1)(x)
        self.hard_tanh = Hardtanh(0, 1)

    def rsample(self, size: Tuple = (1,)):
        s = self.bc_dist.rsample(size=size)
        s_tilde = s * (self.zeta - self.gamma) + self.gamma
        return self.hard_tanh(s_tilde)

    def pdf(self, x: Union[np.ndarray, torch.Tensor, float]):
        if 0 < x < 1:
            edge_probs = self.bc_dist.quantile(np.ndarray([0.0, 1.0]))
            return (edge_probs[1] - edge_probs[0]) * self.bc_dist.pdf(x)
        elif x == 1:
            return -self.bc_dist.quantile(1.0) + 1
        elif x == 0:
            return self.bc_dist.quantile(0.0)
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
        module: torch.nn.Module,
        gamma: float = -0.1,
        zeta: float = 1.1,
        initial_sparsity_rate: float = 0.5,
        device: Optional[torch.device] = None,
        seed: Optional[int] = None,
    ):
        super().__init__()
        if 0 < initial_sparsity_rate < 1:
            self.initial_sparsity_rate = initial_sparsity_rate
        else:
            self.initial_sparsity_rate = 0.5
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        # the log alpha locations are learnable parameters
        self.log_alphas = torch.nn.ParameterList()
        for params in module.parameters():
            self.log_alphas.append(
                torch.nn.Parameter(
                    torch.empty(params.shape, device=self.device), requires_grad=True
                )
            )
        self.beta = torch.nn.Parameter(
            2.0 / 3.0 * torch.ones(1, dtype=torch.float), requires_grad=True
        )
        self.nr_param_tensors = len(self.log_alphas)
        self.backups = []
        # register the hooks for the wrapped module.
        self.masked_module = module
        self.handle_forward = module.register_forward_pre_hook(
            self.forward_hook_factory()
        )
        self.handle_backward = module.register_backward_hook(
            self.backward_hook_factory()
        )

        self.initialize_weights()

        self.hc_dists = [
            HardConcreteDist(
                binary_concrete_dist=BinaryConcreteDist(
                    log_alpha=log_alpha, beta=self.beta, seed=seed
                ),
                gamma=gamma,
                zeta=zeta,
                seed=seed,
            )
            for log_alpha in self.log_alphas
        ]

        # the function named g in the paper, which computes z. It's mathematical form is:
        # g(x) = min(1, max(0, x))
        self.hard_sigmoid = Hardtanh(min_val=0, max_val=1)

        self.random_state = np.random.default_rng(seed)

    def initialize_weights(self):
        for log_alpha in self.log_alphas:
            log_alpha.data.normal_(
                mean=np.log(1 - self.initial_sparsity_rate)
                - np.log(self.initial_sparsity_rate),
                std=1e-2,
            )

    def forward(self, batch_size: int):
        z = self.sample_mask(batch_size, deterministic=self.training)
        return z

    def remove_module_hooks(self):
        self.handle_backward.remove()
        self.handle_forward.remove()

    def _sample_epsilon(
        self, size: Tuple[int] = (1,), low: float = EPS, high: float = 1 - EPS
    ):
        """
        distribution of the noise 'epsilon'. Implemented as uniform(0,1).
        """
        eps = torch.as_tensor(
            self.random_state.uniform(low=low, high=high, size=size),
            dtype=torch.float32,
            device=self.device,
        )
        eps.requires_grad_(True)
        return eps

    def _cdf_stretched_concrete(
        self, x: Union[float, Iterable[float]], layer_nr: int, *args, **kwargs
    ):
        """
        Compute the cdf of the stretched concrete distribution for the given samples x and layer number.
        """
        hard_concrete_dist = self.hc_dists[layer_nr]
        gamma, zeta = hard_concrete_dist.gamma, hard_concrete_dist.zeta
        return hard_concrete_dist.bc_dist.cdf(
            (x - gamma) / (zeta - gamma), *args, **kwargs
        )

    def _quantile_stretched_concrete(
        self, p: Union[float, Iterable[float]], layer_nr: int, *args, **kwargs
    ):
        """
        Compute the quantile, aka the inverse cdf, of the stretched concrete distribution for the given probability p
        and layer number.
        """
        hard_concrete_dist = self.hc_dists[layer_nr]
        gamma, zeta = hard_concrete_dist.gamma, hard_concrete_dist.zeta
        return (
            hard_concrete_dist.bc_dist.quantile(p, *args, **kwargs) * (zeta - gamma)
            + gamma
        )

    def sample_mask(
        self,
        batch_size: int = 1,
        weight_layer_indices: Union[Iterable[int], int, None] = None,
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
            range(self.nr_param_tensors)
            if weight_layer_indices is None
            else np.atleast_1d(weight_layer_indices)
        )

        masks = []
        if not deterministic:
            # this should be reached at train time
            for layer_idx in layers:
                epsilon = self._sample_epsilon(
                    (batch_size,) + self.log_alphas[layer_idx].shape
                )
                z = self._quantile_stretched_concrete(epsilon, layer_idx)
                masks.append(self.hard_sigmoid(z))
        else:
            # this should be reached for test time:
            for layer_idx in layers:
                log_alpha = self.log_alphas[layer_idx]
                hard_concrete_dist = self.hc_dists[layer_idx]
                gamma, zeta = hard_concrete_dist.gamma, hard_concrete_dist.zeta
                pi = (
                    torch.sigmoid(log_alpha)
                    .view(1, *log_alpha.size())
                    .expand(batch_size, *log_alpha.size())
                ).detach()
                masks.append(self.hard_sigmoid(pi * (zeta - gamma) + gamma))
        return masks

    def forward_hook_factory(self):
        def hook(module: torch.nn.Module, _):
            masks = self.sample_mask()
            if len(self.backups) != self.nr_param_tensors:
                for layer_idx, weights in enumerate(module.parameters()):
                    self.backups.append(
                        weights.data.clone()
                    )  # store a backup for reassigning after evaluation.
                    weights.data = self.backups[layer_idx] * masks[layer_idx].squeeze(0)
            else:
                for layer_idx, weights in enumerate(module.parameters()):
                    weights.data = self.backups[layer_idx] * masks[layer_idx].squeeze(0)

        return hook

    def backward_hook_factory(self):
        def hook(module: torch.nn.Module, grad_input, grad_output):
            for backup, weights in zip(self.backups, module.parameters()):
                weights.data = backup.data

        # reset the backup
        self.backups = []
        return hook

    def estimate_loss(
        self,
        data_in: torch.Tensor,
        target: torch.Tensor,
        loss_func: Callable,
        mc_sample_size: int = 1,
        output_transformation: Optional[Callable] = lambda tensor: tensor,
    ):
        """
        Perform a monte carlo sampling estimation of the loss of the original model.
        This method will sample `mcs_size` many parameter masks, apply them on the parameters (all in forward_hook),
        and run the data through the model with the respective masks.
        The average of these losses will be the estimate of the overall loss evaluation given the current weights and
        mask parameters.
        Parameters
        ----------
        data_in: torch.Tensor,
            the data tensor for the model to evaluate.
        target: torch.Tensor,
            the target, to which we compare the output of the model.
        loss_func: Callable,
            the loss function.
        mc_sample_size: int,
            the number of monte carlo samples N, that we are to compute.
        output_transformation: (optional) Callable,
            a function to transform the model's output to an appropriate value (e.g. taking the argmax of the output).
        """
        loss = 0
        for mcs_run in range(mc_sample_size):
            loss += loss_func(
                output_transformation(self.masked_module(data_in)), target
            )
        return loss / mc_sample_size

    def l0_regularization(self):
        """
        Expected L0 norm under the stochastic gates.
        """
        reg_loss = 0
        for layer_idx in range(self.nr_param_tensors):
            hc_dist = self.hc_dists[layer_idx]
            zeta_over_gamma = torch.as_tensor(hc_dist.zeta / hc_dist.gamma)
            reg_loss += torch.sum(
                torch.sigmoid(
                    self.log_alphas[layer_idx]
                    - torch.sigmoid(self.beta) * torch.log(-zeta_over_gamma)
                )
            )
        return reg_loss

    def l2_regularization(
        self, weight_decay=0.5,
    ):
        """
        Expected L2 (combined with L0) norm under the stochastic gates.
        """
        reg_loss = 0
        for layer_idx, weights in zip(
            range(self.nr_param_tensors), self.masked_module.parameters()
        ):
            hc_dist = self.hc_dists[layer_idx]
            zeta_over_gamma = torch.as_tensor(hc_dist.zeta / hc_dist.gamma)
            reg_loss += torch.sum(
                torch.sigmoid(
                    self.log_alphas[layer_idx]
                    - torch.sigmoid(self.beta) * torch.log(-zeta_over_gamma)
                )
                * weights.pow(2)
                * weight_decay
            )
        return reg_loss


class L0InputGate(torch.nn.Module):
    """
    Single Layer network which masks by element-wise multiplication with a relaxed binary mask, mimicing an input
    L0 loss. General idea from the paper:
    """

    def __init__(
        self,
        dim_input: int = 10,
        monte_carlo_sample_size: int = 1,
        gamma: float = -0.1,
        zeta: float = 1.1,
        device: Optional[torch.device] = None,
    ):
        super(L0InputGate, self).__init__()
        self.n_dim = dim_input

        self.device = (
            torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device is None
            else device
        )

        self.log_alpha = torch.nn.Parameter(
            0.5 + 1e-2 * torch.randn(1, dim_input), requires_grad=True
        )
        self.beta = torch.nn.Parameter(0.5 * torch.ones(1), requires_grad=True)
        self.gamma = gamma
        self.zeta = zeta
        self.mcs_size = monte_carlo_sample_size
        self.gates = None

        self.hardTanh = torch.nn.Hardtanh(0, 1)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        batch_size = x.shape[0]

        if mask is None:
            self.gates = self.create_gates()
        else:
            self.gates = mask

        x = x.repeat(self.mcs_size, 1).view(self.mcs_size, batch_size, self.n_dim)

        x *= self.gates

        x = x.view(self.mcs_size * batch_size, self.n_dim)

        return x

    def complexity_loss(self):
        beta_sig = self.sigmoid(self.beta)

        return self.sigmoid(
            self.log_alpha - beta_sig * torch.log(-torch.tensor(self.gamma) / self.zeta)
        ).sum()

    def hard_sigmoid(self, x):
        return self.hardTanh(x)

    @staticmethod
    def stretch(s: float, gamma: float, chi: float):
        return s * (chi - gamma) + gamma

    def reparameterize(self, u, log_alpha, beta):
        """ Transform uniform distribution to CONCRETE distr."""
        return self.sigmoid((torch.log(u) - torch.log(1 - u) + log_alpha) / beta)

    def final_layer(self):
        """Estimates the gates at test time (i.e. after finished training)."""
        return self.hard_sigmoid(self.log_alpha * (self.zeta - self.gamma) + self.gamma)

    def create_gates(self):
        """Creates input gates using parameters log_alpha, beta"""
        dim = self.log_alpha.shape[0]
        m = torch.distributions.uniform.Uniform(
            torch.tensor([0.0]), torch.tensor([1.0])
        )
        u = (
            m.sample(sample_shape=(self.mcs_size, dim))
            .view(self.mcs_size, dim)
            .to(self.device)
        )
        s = self.reparameterize(u, self.log_alpha, self.sigmoid(self.beta))
        s_strech = self.stretch(s, self.gamma, self.zeta)
        return self.hard_sigmoid(s_strech).view(self.mcs_size, 1, self.n_dim)
