from .icpbase import ICPredictor
from causalpy.neural_networks import NeuralBaseNet, FCNet

from typing import *

from operator import mul
from collections import defaultdict, namedtuple
from functools import reduce

import numpy as np
import pandas as pd

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import matplotlib.pyplot as plt

from tqdm.auto import tqdm


Hyperparams = namedtuple("Hyperparams", "alpha beta gamma")


class ANMPredictor(ICPredictor):
    def __init__(
        self,
        network: torch.nn.Module,
        epochs: int = 100,
        batch_size: int = 1024,
        residual_equality_measure: Union[str, Callable] = "mmd",
        variable_independence_measure: str = "hsic",
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        hyperparams: Optional[Hyperparams] = None,
        log_level: bool = True,
    ):
        super().__init__(log_level=log_level)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.epochs = epochs
        self.network = network
        self.hyperparams = (
            Hyperparams(alpha=1, beta=1, gamma=1e-2)
            if hyperparams is None
            else hyperparams
        )

        if optimizer is None:
            self.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        if scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=50, gamma=0.9
            )

        if residual_equality_measure == "mmd":
            self.residuals_comparison = self.mmd_multiscale
        elif residual_equality_measure == "moments":
            self.residuals_comparison = self.moments
        elif isinstance(residual_equality_measure, Callable):
            self.residuals_comparison = residual_equality_measure
        else:
            raise ValueError(
                f"Dependency measure '{residual_equality_measure}' not supported."
            )

        if variable_independence_measure == "hsic":
            self.variable_independence_measure = self._hilbert_schmidt_independence

    def set_network(self, network: torch.nn.Module):
        self.network = network
        self.reset_optimizer()

    def reset_optimizer(self):
        self.optimizer.param_groups = []
        self.optimizer.add_param_group(
            {name: params for name, params in self.network.parameters()}
        )
        self.optimizer.zero_grad()

    def _get_jacobian(self, x: NeuralBaseNet, dim_in: int = None, dim_out: int = None):
        """
        Computes the Jacobian matrix for a batch of input data x with respect to the network of the class.

        Notes
        -----
        The output Jacobian for this method is potentially the transpose of the definition one might be familiar with.
        That is, the output Jacobian J is of the form:

        .. math:: J_{ij} = d_i f_j

        or in matix form

        .. math:: | d_1 f_1 \quad d_1 f_2 \quad \dots \quad d_1 f_m |
        .. math:: | d_2 f_1 \quad d_2 f_2 \quad \dots \quad d_2 f_m |
        .. math:: | ... \quad ... \quad ... \quad ... \quad ...  \quad ... \quad ... \quad ... \quad ... |
        .. math:: | d_n f_1 \quad d_n f_2 \quad \dots \quad d_n f_m |


        Parameters
        ----------
        x: Tensor,
            the data tensor of appropriate shape for then network. Can also be supplied in batches.
        dim_in: int,
            the input dimension for data intended for the neural network.
            If not provided by the user, will be inferred from the network.
        dim_in: int,
            the output dimension of the data returned by neural network.
            If not provided by the user, will be inferred from the network.

        Returns
        -------
        Jacobian: Tensor,
            the jacobian matrix for every entry of data in the batch.
        """
        if dim_in is None:
            dim_in = self.network.dim_in
        if dim_out is None:
            dim_out = self.network.dim_out

        x = x.squeeze()
        n = x.size(0)

        # ``torch.autograd.grad`` only returns the product J @ v with J = Jacobian, v = gradient vector, so as to allow
        # the user to provide external gradients.
        # Therefore we need to build a matrix of basis vectors v_i = (0, ..., 0, 1, 0, ..., 0), with 1 being on the
        # i-th position, in order to get each column of J (and with this every derivation of each input variable x_i
        # and each output function f_j).
        #
        # The output of the ``autograd.grad`` method apparently returns a tensor of last shape (..., ``dim_out), which
        # must mean, that the torch definition of J must be J_ij = d_i f_j, thus we need basis vectors of length
        # ``dim_out`` to get the correct aforementioned matrix product.

        # Unfortunately this also means that we will need to copy the input data ``dim_out`` times, in order to cover
        # all derivatives. This could be memory costly, if the input is big and might need to be replaced by iterative
        # calls to ``grad`` instead for such a case. However, if the data fits into memory, this should be the fastest
        # way for computing the Jacobian.

        x = x.repeat(dim_out, 1)
        x.requires_grad_(True)

        z = self.network(x)

        extraction_matrix = torch.zeros((dim_out * n, dim_out))

        for j in range(dim_out):
            jth_basis_vec = torch.zeros(dim_out)
            jth_basis_vec[j] = 1
            jth_basis_vec = jth_basis_vec.repeat(n, 1)
            extraction_matrix[j * n : (j + 1) * n] = jth_basis_vec

        z.backward(extraction_matrix)
        grad_data = x.grad.data

        # we now have the gradient data, but all the derivation vectors D f_j are spread out over ``dim_out`` many
        # repetitive computations. Therefore we need to find all the deriv. vectors belonging to the same function f_j
        # and put them into an output tensor of shape (batch_size, dim_in, dim_out). This will provide a Jacobian matrix
        # for every batch data entry.
        output = torch.zeros((n, dim_in, dim_out))
        for batch_entry in range(n):
            output[batch_entry] = torch.cat(
                tuple(
                    grad_data[batch_entry + j * n].view(-1, 1) for j in range(dim_out)
                ),
                1,
            )
        return output

    def _centering(self, K):
        n = K.shape[0]
        unit = torch.ones(n, n).to(self.device)
        I = torch.eye(n, device=self.device)
        Q = I - unit / n
        return torch.mm(torch.mm(Q, K), Q)

    @staticmethod
    def rbf(X, Y, sigma=None):
        GX = torch.mm(X, Y.t())
        KX = torch.diagonal(GX) - GX + (torch.diagonal(GX) - GX).t()

        if sigma is None:
            try:
                mdist = torch.median(KX[KX != 0])
                sigma = torch.sqrt(mdist)
            except:
                sigma = 1.2
        KX *= -0.5 / sigma / sigma
        KX = torch.exp(KX)
        return KX

    def _hilbert_schmidt_independence(self, X, Y):
        """ Hilbert Schmidt independence criterion -- kernel based measure for how dependent X and Y are"""

        out = (
            torch.sum(
                reduce(
                    mul,
                    map(
                        self._centering,
                        (ANMPredictor.rbf(X, X), ANMPredictor.rbf(Y, Y)),
                    ),
                )
            )
            / self.batch_size
        )
        return out

    @staticmethod
    def normalize_jointly(x, y):
        """ Normalize x and y separately with the mean and sd computed from both x and y"""
        xy = torch.cat((x, y), 0).detach()
        sd = torch.sqrt(xy.var(0))
        mean = xy.mean(0)
        x = (x - mean) / sd
        y = (y - mean) / sd
        return x, y

    def mmd_multiscale(self, x, y, normalize_j=True):
        """ MMD with rationale kernel"""
        # Normalize Inputs Jointly
        x, y = self.normalize_jointly(x, y)

        xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())

        rx = xx.diag().unsqueeze(0).expand_as(xx)
        ry = yy.diag().unsqueeze(0).expand_as(yy)

        dxx = rx.t() + rx - 2.0 * xx
        dyy = ry.t() + ry - 2.0 * yy
        dxy = rx.t() + ry - 2.0 * zz

        XX, YY, XY = (
            torch.zeros(xx.shape, device=self.device),
            torch.zeros(xx.shape, device=self.device),
            torch.zeros(xx.shape, device=self.device),
        )

        for a in [6e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1, 1.2, 1.5, 1.8, 2, 2.5]:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

        return torch.mean(XX + YY - 2.0 * XY)

    @staticmethod
    def moments(X, Y, order=2):
        """ Compares Expectation and Variance between two samples """
        if order == 2:
            a1 = (X.mean() - Y.mean()).abs()
            a2 = (X.var() - Y.var()).abs()

            return (a1 + a2) / 2

    @staticmethod
    def _null_data_hook(*args, **kwargs):
        """
        A null method to allow for data extraction during the training process.
        """
        pass

    def infer(self, obs, target_variable, envs, *args, **kwargs):

        data_hook = kwargs.pop("data_hook", self._null_data_hook)

        residual_ground_truth = kwargs.pop("ground_truth", None)
        if residual_ground_truth is not None:
            residual_ground_truth = np.array(residual_ground_truth)

        visualize = any(kwargs.pop(kwarg, False) for kwarg in ("plot", "visualize"))
        if visualize:
            losses = defaultdict(list)

        obs, target, environments = self.preprocess_input(obs, target_variable, envs)

        torch.autograd.set_detect_anomaly(True)
        self.reset_optimizer()

        pbar = tqdm(range(self.epochs), desc="Training Additive Noise Model Network")

        for epoch in pbar:
            # Compute Residuals for different contexts
            residuals = dict()
            for env, env_ind in environments.items():
                residuals[env] = target[env_ind] - self.network(obs[env_ind])

            # Regression Loss
            loss_regression = 0
            for residual in residuals:
                loss_regression = (residual ** 2).mean()

            # Compute Expectation of absolute values of the partial derivatives
            jacobians = dict()
            for env, env_ind in environments.items():
                jacobians[env] = self._get_jacobian(obs[env_ind])[0].abs().mean(0)

            # Loss that computes the dependence of parent variables and residuals per context
            # Loss is weighted by Expectation of absolute values of the partial derivatives
            loss_ind_par = sum(
                jacobian
                * sum(
                    self.variable_independence_measure(
                        residuals[env], obs[env_ind, var]
                    )
                    for env, env_ind in environments.items()
                )
                for var, jacobian in jacobians.items()
            )

            # Measure for the dependence of Residuals and Context
            # Formulated differently: Measure for how close the residuals are distributed across different contexts.
            # Note that we compare only residual_i to residual_{i+1}, which greatly reduces the computational amount.
            loss_identical_res = sum(
                self.residuals_comparison(residuals[i], residuals[i + 1])
                for i in range(len(residuals) - 1)
            )

            # Overall Loss
            loss = (
                loss_identical_res
                + self.hyperparams.gamma * loss_ind_par
                # + residuals[0].mean().abs()
            )

            loss.backward()
            self.optimizer.step()
            self.scheduler.step(epoch)

            data_hook(
                epoch,
                obs,
                target,
                environments,
                loss,
                loss_identical_res,
                loss_ind_par,
                jacobians,
            )

            if visualize:
                losses["residual"].append(loss_identical_res.item())
                losses["regression"].append(loss_regression.item())
                losses["total"].append(loss.item())

            if residual_ground_truth is not None:
                # Regression loss if the true causal model is used
                losses["regression_truth"].append(
                    sum(
                        (residual_ground_truth[env_ind] ** 2).mean()
                        for env_ind in environments.values()
                    )
                )
            if (epoch + 1) % 10 == 0:
                self.logger.info(str(losses))
            self.visualize_training(losses)

    @staticmethod
    def visualize_training(losses):
        rolling_window = 5

        loss_fig = plt.figure(num=4200, figsize=(10, 10))

        if len(losses["residual"]) > rolling_window:
            rolling_window = 1

        loss_fig.plot(
            range(len(losses["residual"])),
            pd.DataFrame(losses["residual"]).rolling(rolling_window).mean(),
            label="Residuals Loss",
        )
        loss_fig.plot(
            range(len(losses["regression"])),
            pd.DataFrame(losses["regression"]).rolling(rolling_window).mean(),
            label="Regression Loss",
        )
        loss_fig.plot(
            range(len(losses["total"])),
            pd.DataFrame(losses["total"]).rolling(rolling_window).mean(),
            label="Total Loss",
        )
        loss_fig.legend()
        loss_fig.title("Training Losses")

        plt.show(block=False)
