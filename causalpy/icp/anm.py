from .icpbase import ICP
from .utils import VisdomLinePlotter

from typing import *

from collections import defaultdict
from functools import reduce, partial

import numpy as np
import pandas as pd

import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler

import matplotlib.pyplot as plt

from tqdm.auto import tqdm
import logging


class ANModelICP(ICP):
    def __init__(
        self,
        network: torch.nn.Module,
        epochs: int = 100,
        batch_size: int = 1024,
        residual_equality_measure: Union[str, Callable] = "mmd",
        variable_independence_measure: str = "hsic",
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        hyperparams: Optional[Dict] = None,
        log_level: bool = True,
    ):
        super().__init__(log_level=log_level)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.epochs = epochs
        self.network = network
        self.hyperparams = {"gamma": 1e-2} if hyperparams is None else hyperparams

        if optimizer is None:
            self.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        if scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer, step_size=50, gamma=0.9
            )

        if residual_equality_measure == "mmd":
            self.residuals_comparison = lambda res_i, res_ip1: self.mmd_multiscale(
                res_i, res_ip1
            )
        elif residual_equality_measure == "moments":
            self.residuals_comparison = lambda res_i, res_ip1: self.moments(
                res_i, res_ip1
            )
        elif isinstance(residual_equality_measure, Callable):
            self.residuals_comparison = residual_equality_measure
        else:
            raise ValueError(
                f"Dependency measure '{residual_equality_measure}' not supported."
            )

        if variable_independence_measure == "hsic":
            self.variable_independence_measure = self._hilbert_schmidt_independence

    def set_network(self, network):
        self.network = network
        self.reset_optimizer()

    def reset_optimizer(self):
        self.optimizer.param_groups = []
        self.optimizer.add_param_group(
            {name: params for name, params in self.network.parameters()}
        )
        self.optimizer.zero_grad()

    def _get_jacobian(self, x):
        """ Compute Jacobian of net with respect to y"""
        n_outputs = 1
        x = x.squeeze()
        n = x.size()[0]
        x = x.repeat(n_outputs, 1)
        x.requires_grad_(True)
        y = self.network(x)
        jacobian = torch.autograd.grad(
            y, x, grad_outputs=torch.eye(n, device=self.device), create_graph=True
        )
        return jacobian

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
                    lambda x, y: x * y,
                    map(self._centering, (ANModelICP.rbf(X, X), ANModelICP.rbf(Y, Y))),
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
        A null method to allow for data extraction of the training process in the case of not needing the data.
        """
        return

    def infer(self, obs, target_variable, envs, *args, **kwargs):

        data_hook = kwargs.pop("data_hook", None)
        if data_hook is None:
            data_hook = self._null_data_hook

        residual_ground_truth = kwargs.pop("ground_truth", None)
        if residual_ground_truth is not None:
            residual_ground_truth = np.array(residual_ground_truth)

        visualize = any(kwargs.pop(kwarg, False) for kwarg in ("plot", "visualize"))
        if visualize:
            losses = defaultdict(list)

        obs, target, environments = self.preprocess_input(obs, target_variable, envs)

        torch.autograd.set_detect_anomaly(True)
        self.reset_optimizer()

        pbar = (
            tqdm(range(self.epochs), desc="Training Additive Noise Model Network")
            if self.log_level
            else range(self.epochs)
        )

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

            def sum_reduce(x, y):
                return x + y

            # Loss that computes the dependence of parent variables and residuals per context
            # Loss is weighted by Expectation of absolute values of the partial derivatives
            loss_ind_par = reduce(
                sum_reduce,
                [
                    jacobian
                    * reduce(
                        sum_reduce,
                        map(
                            self.variable_independence_measure,
                            (
                                (residuals[env], obs[env_ind, var])
                                for env, env_ind in environments.items()
                            ),
                        ),
                    )
                    for var, jacobian in jacobians.items()
                ],
            )

            # Measure for the dependence of Residuals and Context
            # Formulated differently: Measure for how close the residuals are distributed across different contexts.
            # Note that we compare only residual_i to residual_{i+1}, which greatly reduces the computational amount.
            loss_identical_res = reduce(
                sum_reduce,
                [
                    self.residuals_comparison(residuals[i], residuals[i + 1])
                    for i in range(len(residuals) - 1)
                ],
            )

            # Overall Loss
            # TODO: what does residuals[0] do here?
            loss = (
                loss_identical_res
                + self.hyperparams["gamma"] * loss_ind_par
                + residuals[0].mean().abs()
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
                        reduce(
                            sum_reduce,
                            [
                                (residual_ground_truth[env_ind] ** 2).mean()
                                for env_ind in environments.values()
                            ],
                        )
                    )
                if (epoch + 1) % 10 == 0:
                    self.log(str(losses))
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
