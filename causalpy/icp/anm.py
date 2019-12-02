from functools import reduce, partial
from typing import *

from .base import ICP

import numpy as np
import pandas as pd
import torch
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm.auto import tqdm
import logging


class ANModelICP(ICP):

    def __init__(
            self,
            network: torch.nn.Module,
            epochs: int = 100,
            batch_size: int = 1024,
            residual_equality_measure: Union[str, Callable] = "mmd",
            variable_importance_measure: str = "hsic",
            optimizer: Optional[Optimizer] = None,
            scheduler: Optional[_LRScheduler] = None,
            hyperparams: Optional[Dict] = None,
            verbose: bool = True
    ):
        super().__init__(verbose=verbose)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.epochs = epochs
        self.network = network
        self.hyperparams = {"gamma": 1e-2} if hyperparams is None else hyperparams

        if optimizer is None:
            self.optimizer = torch.optim.Adam(network.parameters(), lr=1e-3)

        if scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

        if residual_equality_measure == "mmd":
            self.residuals_comparison = lambda res_i, res_ip1: self.mmd_multiscale(res_i, res_ip1)
        elif residual_equality_measure == "moments":
            self.residuals_comparison = lambda res_i, res_ip1: self.moments(res_i, res_ip1)
        elif isinstance(residual_equality_measure, Callable):
            self.residuals_comparison = residual_equality_measure
        else:
            raise ValueError(f"Dependency measure '{residual_equality_measure}' not supported.")

        if variable_importance_measure == "hsic":
            self.variable_importance_measure = self._hilbert_schmidt_independence

    def set_network(self, network):
        self.network = network
        self.reset_optimizer()

    def reset_optimizer(self):
        self.optimizer.param_groups = []
        self.optimizer.add_param_group({name: params for name, params in self.network.parameters()})
        self.optimizer.zero_grad()

    def _get_jacobian(self, x):
        """ Compute Jacobian of net with respect to y"""
        n_outputs = 1
        x = x.squeeze()
        n = x.size()[0]
        x = x.repeat(n_outputs, 1)
        x.requires_grad_(True)
        y = self.network(x)
        jacobian = torch.autograd.grad(y, x, grad_outputs=torch.eye(n, device=self.device), create_graph=True)
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
        KX *= - 0.5 / sigma / sigma
        KX = torch.exp(KX)
        return KX

    def _hilbert_schmidt_independence(self, X, Y):
        """ Hilbert Schmidt independence criterion -- kernel based measure for how dependent X and Y are"""

        out = torch.sum(
            reduce(
                lambda x, y: x * y,
                map(
                    self._centering,
                    (ANModelICP.rbf(X, X), ANModelICP.rbf(Y, Y))
                )
            )
        ) / self.batch_size
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

        rx = (xx.diag().unsqueeze(0).expand_as(xx))
        ry = (yy.diag().unsqueeze(0).expand_as(yy))

        dxx = rx.t() + rx - 2. * xx
        dyy = ry.t() + ry - 2. * yy
        dxy = rx.t() + ry - 2. * zz

        XX, YY, XY = (torch.zeros(xx.shape, device=self.device),
                      torch.zeros(xx.shape, device=self.device),
                      torch.zeros(xx.shape, device=self.device))

        for a in [6e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1, 1.2, 1.5, 1.8, 2, 2.5]:
            XX += a ** 2 * (a ** 2 + dxx) ** -1
            YY += a ** 2 * (a ** 2 + dyy) ** -1
            XY += a ** 2 * (a ** 2 + dxy) ** -1

        return torch.mean(XX + YY - 2. * XY)

    @staticmethod
    def moments(X, Y, order=2):
        """ Compares Expectation and Variance between two samples """
        if order == 2:
            a1 = (X.mean() - Y.mean()).abs()
            a2 = (X.var() - Y.var()).abs()

            return (a1 + a2) / 2

    def infer(self, obs, target_variable, envs, *args, **kwargs):

        if any(kwargs.pop(kwarg, False) for kwarg in ("plot", "visualize")):
            self.visualize()

        target_ground_truth = kwargs.pop("ground_truth", None)

        obs, target, environments = self.preprocess_input(
            obs,
            target_variable,
            envs
        )

        self.reset_optimizer()

        pbar = tqdm(
            range(self.epochs),
            desc="Training Additive Noise Model Network"
        ) if self.verbose else range(self.epochs)

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
            jacobian = dict()
            for env, env_ind in environments.items():
                jacobian[env] = self._get_jacobian(obs[env_ind])[0].abs().mean(0)

            def sum_reduce(x, y):
                return x + y

            # Loss that computes the dependence of parent variables and residuals per context
            # Loss is weighted by Expectation of absolute values of the partial derivatives
            loss_ind_par = reduce(
                sum_reduce,
                [
                    reduce(
                        sum_reduce,
                        map(
                            lambda resid, data: self.variable_importance_measure(resid, data) * jacobian[var],
                            ((residuals[env], obs[env_ind, var]) for env, env_ind in environments.items())
                        )
                    )
                    for var in range(obs.shape[1])
                ]
            )

            # Measure for the dependence of Residuals and Context
            # Formulated differently: Measure for how close the residuals are distributed across different contexts
            # Note that we residuals (residual_i, residual_{i+1}) which greatly reduces the computational amount
            loss_identical_res = reduce(
                sum_reduce,
                [self.residuals_comparison(residuals[i], residuals[i + 1]) for i in range(len(residuals) - 1)]
            )

            # Overall Loss
            # TODO: what does residuals[0] do here?
            loss = loss_identical_res + self.hyperparams["gamma"] * loss_ind_par + residuals[0].mean().abs()

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if epoch % 10 == 0:
                losses['res'].append(loss_identical_res.item())
            losses['reg'].append(loss_regression.item())

            # Compute regression loss for ground truth model
            residual_1_true = y_1 - x_1[:, 1:2] - x_1[:, 2:3]
            residual_2_true = y_2 - x_2[:, 1:2] - x_2[:, 2:3]
            residual_3_true = y_3 - x_3[:, 1:2] - x_3[:, 2:3]
            residual_4_true = y_4 - x_4[:, 1:2] - x_4[:, 2:3]
            residual_5_true = y_5 - x_5[:, 1:2] - x_5[:, 2:3]

            # Regression loss if the causal model is used
            loss_reg_truth = (residual_1_true ** 2).mean() + (residual_2_true ** 2).mean() + (
                    residual_3_true ** 2).mean() + (residual_4_true ** 2).mean() + (residual_5_true ** 2).mean()
            losses['reg_truth'].append(loss_reg_truth.item())

            losses['exp_jac'][0].append(jacobian_1.detach()[0])
            losses['exp_jac'][1].append(jacobian_1.detach()[1])
            losses['exp_jac'][2].append(jacobian_1.detach()[2])
            losses['exp_jac'][3].append(jacobian_1.detach()[3])

            # plotter_loss.plot('loss', 'train_loss', 'loss all', iteration, loss.item())
            # if iteration % 10 == 0:
            #    print('Jacobian in context 1 in iteration {} '.format(iteration), jacobian_1.detach().cpu().numpy()

    def visualize(self):
        # import test for visdom. If it fails, either let the method fail or maybe substitute by matplotlib
        try:
            import visdom
            from visdom import Visdom
        except ImportError as e:
            logging.warn(f"Package 'Visdom' needs to be installed for visualization of training. Please install or"
                         f"call 'infer' without the visualization keyword.")
            raise e

        vlp = VisdomLinePlotter(Visdom())

class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, visdom_obj, env_name='main'):
        self.viz = visdom_obj
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(X=np.array([x, x]), Y=np.array([y, y]), env=self.env,
                                                 opts=dict(
                                                     legend=[split_name],
                                                     title=title_name,
                                                     xlabel='Epochs',
                                                     ylabel=var_name
                                                 ))
        else:
            self.viz.line(X=np.array([x]), Y=np.array([y]), env=self.env, win=self.plots[var_name],
                          name=split_name, update='append')
