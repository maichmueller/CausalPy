from .icpbase import ICPredictor
from causalpy.neural_networks import NeuralBaseNet, FCNet
from causalpy.neural_networks.utils import *

from typing import *

from operator import mul
from collections import defaultdict, namedtuple
from functools import reduce, partial

import numpy as np
import pandas as pd

import torch
from torch import Tensor
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

from tqdm.auto import tqdm

Hyperparams = namedtuple("Hyperparams", "alpha beta gamma")


class ANMPredictor(ICPredictor):
    def __init__(
        self,
        network: torch.nn.Module,
        epochs: int = 100,
        batch_size: int = 1024,
        loss_transform_res_to_par: str = "sum",
        loss_transform_res_to_res: str = "sum",
        compare_residuals_pairwise: bool = True,
        residual_equality_measure: Union[str, Callable] = "mmd",
        variable_independence_metric: str = "hsic",
        optimizer: Optional[Optimizer] = None,
        scheduler: Optional[_LRScheduler] = None,
        hyperparams: Optional[Hyperparams] = None,
        log_level: bool = True,
        **kwargs,
    ):
        super().__init__(log_level=log_level)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.epochs = epochs
        self.network = network
        self.hyperparams = (
            Hyperparams(alpha=1, beta=1, gamma=1)
            if hyperparams is None
            else hyperparams
        )

        self.optimizer = (
            torch.optim.Adam(network.parameters(), lr=1e-3)
            if optimizer is None
            else optimizer
        )

        if scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=50, gamma=0.9
            )

        self.compute_jacobian_iteratively = False
        self.compare_residuals_pairwise = compare_residuals_pairwise

        if loss_transform_res_to_par == "sum":
            self.loss_reduce_par = torch.sum
        elif loss_transform_res_to_par == "max":
            self.loss_reduce_par = torch.max
        elif loss_transform_res_to_par == "alpha_moment":
            self.loss_reduce_par = partial(alpha_moment, alpha=kwargs.pop("alpha", 1))
        else:
            args = ["sum", "max", "alpha_moment"]
            raise ValueError(
                f"Loss Transform function needs to be one of {args}. Provided: '{loss_transform_res_to_par}'."
            )

        if loss_transform_res_to_res == "sum":
            self.loss_reduce_res = torch.sum
        elif loss_transform_res_to_res == "max":
            self.loss_reduce_res = torch.max
        elif loss_transform_res_to_res == "alpha_moment":
            self.loss_reduce_res = partial(alpha_moment, alpha=kwargs.pop("alpha", 1))
        else:
            args = ["sum", "max", "alpha_moment"]
            raise ValueError(
                f"Loss Transform function needs to be one of {args}. Provided: '{loss_transform_res_to_res}'."
            )

        self.residual_equality_measure_str = residual_equality_measure
        if residual_equality_measure == "mmd":
            self.identical_distribution_metric = mmd_multiscale
        elif residual_equality_measure == "moments":
            self.identical_distribution_metric = moments
        elif isinstance(residual_equality_measure, Callable):
            self.identical_distribution_metric = residual_equality_measure
        else:
            raise ValueError(
                f"Dependency measure '{residual_equality_measure}' not supported."
            )

        self.variable_independence_metric_str = variable_independence_metric
        if variable_independence_metric == "hsic":
            self.variable_independence_metric = hsic

    def set_network(self, network: torch.nn.Module):
        self.network = network
        self.reset_optimizer()

    def reset_optimizer(self):
        self.optimizer.param_groups = []
        self.optimizer.add_param_group({"params": list(self.network.parameters())})
        self.optimizer.zero_grad()

    def _get_jacobian(self, x: Tensor, dim_in: int = None, dim_out: int = None):
        r"""
        Computes the Jacobian matrix for a batch of input data x with respect to the network of the class.

        Notes
        -----
        The output Jacobian for this method is potentially the transpose of the definition one might be familiar with.
        That is, the output Jacobian J is of the form:

        .. math:: J_{ij} = d_i f_j

        or in matrix form:

        .. math:: | d_1 f_1 \quad d_1 f_2 \quad \dots \quad d_1 f_m |
        .. math:: | d_2 f_1 \quad d_2 f_2 \quad \dots \quad d_2 f_m |
        .. math:: | ... \quad ... \quad ... \quad ... \quad ...  \quad ... \quad ... \quad ... \quad ... |
        .. math:: | d_n f_1 \quad d_n f_2 \quad \dots \quad d_n f_m |


        Parameters
        ----------
        x: Tensor,
            the data tensor of appropriate shape for the network. Can also be supplied in batches.
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
        if not self.compute_jacobian_iteratively:
            if dim_in is None:
                dim_in = self.network.dim_in
            if dim_out is None:
                dim_out = self.network.dim_out

            x = x.squeeze()
            n = x.size(0)

            # ``torch.autograd.grad`` only returns the product J @ v with J = Jacobian, v = gradient vector, so as to
            # allow the user to provide external gradients.
            # Therefore we need to build a matrix of basis vectors v_i = (0, ..., 0, 1, 0, ..., 0), with 1 being on the
            # i-th position, in order to get each column of J (and with this every derivation of each input variable x_i
            # and each output function f_j).
            #
            # The output of the ``autograd.grad`` method apparently returns a tensor of shape (..., ``dim_out), which
            # must mean, that the torch definition of J must be J_ij = d_i f_j, thus we need basis vectors of length
            # ``dim_out`` to get the correct aforementioned matrix product.

            # Unfortunately this also means that we will need to copy the input data ``dim_out`` times, in order to
            # cover all derivatives. This could be memory costly, if the input is big and might is replaced by iterative
            # calls to ``grad`` instead in case the memory allocation fails. However, if the data fits into memory, this
            # should be the faster than an iterative python call to ``autograd.grad``.
            try:
                x = x.repeat(dim_out, 1)
                x.requires_grad_(True)

                z = self.network(x)

                unit_vec_matrix = torch.zeros(
                    (dim_out * n, dim_out), device=self.device
                )

                for j in range(dim_out):
                    unit_vec_matrix[j * n : (j + 1) * n, j] = 1

                grad_data = torch.autograd.grad(
                    z,
                    x,
                    grad_outputs=unit_vec_matrix,
                    retain_graph=True,
                    create_graph=True,
                )[0]

                # we now have the gradient data, but all the derivation vectors D f_j are spread out over ``dim_out``
                # many repetitive computations. Therefore we need to find all the deriv. vectors belonging to the same
                # function f_j and put them into the jacobian batch tensor of shape (batch_size, dim_in, dim_out).
                # This will provide a Jacobian matrix for every batch data entry.
                jacobian = torch.zeros((n, dim_in, dim_out))
                for batch_entry in range(n):
                    jacobian[batch_entry] = torch.cat(
                        tuple(
                            grad_data[batch_entry + j * n].view(-1, 1)
                            for j in range(dim_out)
                        ),
                        1,
                    )
                return jacobian

            except MemoryError:
                self.compute_jacobian_iteratively = True
                return self._get_jacobian(x=x, dim_in=dim_in, dim_out=dim_out)

        else:
            y = self.network(x)
            unit_vectors = torch.eye(dim_out)
            result = [
                torch.autograd.grad(
                    outputs=[y], inputs=[x], grad_outputs=[unit_vec], retain_graph=True
                )[0]
                for unit_vec in unit_vectors
            ]
            jacobian = torch.stack(result, dim=0)
            return jacobian

    @staticmethod
    def _null_data_hook(*args, **kwargs):
        """
        A null method to allow for data extraction during the training process.
        """
        pass

    def infer(self, obs, target_variable, envs, *args, **kwargs):
        self.network.to(self.device)
        data_hook = kwargs.pop("data_hook", self._null_data_hook)

        residual_ground_truth = kwargs.pop("ground_truth", None)
        if residual_ground_truth is not None:
            residual_ground_truth = np.array(residual_ground_truth)

        visualize = any(kwargs.pop(kwarg, False) for kwarg in ("plot", "visualize"))
        losses = defaultdict(list)

        obs, target, environments = self.preprocess_input(obs, target_variable, envs)
        obs = torch.as_tensor(obs.astype(np.float32)).to(self.device)
        target = torch.as_tensor(target.astype(np.float32)).to(self.device)
        envs = torch.as_tensor(envs)
        torch.autograd.set_detect_anomaly(True)
        self.reset_optimizer()

        tensor_dataset = TensorDataset(obs, target, envs)
        dataloader = DataLoader(tensor_dataset, pin_memory=True)

        pbar = tqdm(range(self.epochs), desc="Training Additive Noise Model Network")

        for epoch in pbar:

            # Compute Residuals for different contexts
            residuals = dict()
            for env, env_ind in environments.items():
                residuals[env] = target[env_ind].view(-1, 1) - self.network(
                    obs[env_ind]
                )

            # Regression Loss
            loss_regression = 0
            for residual in residuals:
                loss_regression = (residual ** 2).mean()

            # Compute Expectation of absolute values of the partial derivatives
            # Expectation is estimated via the mean of each entry in the jacobian (i.e. gradient in this case, as f is
            # 1-dimensional in its output).
            gradient_abs_expectation = (
                self._get_jacobian(obs, dim_out=1).abs().mean(dim=0)
            )

            # Loss that computes the dependence of parent variables and residuals per context
            # loss is weighted by expectation of absolute values of the partial derivatives

            # Computing the sigma for when HSIC is being used, as do-interventional data (i.e. a constant value for a
            # whole environment for a variable) breaks the rbf methods computation.
            # if self.variable_independence_metric_str == "hsic":
            #     sigma = {var: self.rbf(obs[:, var], obs[:, var]) for var in range(self.p)}

            loss_ind_parents = sum(
                grad
                * self.loss_reduce_par(
                    torch.as_tensor(
                        list(
                            self.variable_independence_metric(
                                residuals[env], obs[env_ind, var]
                            )
                            for env, env_ind in environments.items()
                        )
                    )
                )
                for var, grad in zip(range(self.p), gradient_abs_expectation)
            )

            # Measure for the dependence of Residuals and Context
            # Formulated differently: Measure for how close the residuals are distributed across different contexts.
            if self.compare_residuals_pairwise:
                residuals_compared = list(
                    self.identical_distribution_metric(residuals[i], residuals[i + 1])
                    for i in range(len(residuals) - 1)
                )
            else:
                residuals_compared = list(
                    self.identical_distribution_metric(residuals[i], residuals[j])
                    for i, j in zip(range(len(residuals)), range(len(residuals)))
                    if i != j
                )
            loss_identical_res = self.loss_reduce_res(
                torch.as_tensor(residuals_compared)
            )

            # Overall Loss
            loss = loss_identical_res + self.hyperparams.gamma * loss_ind_parents

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
                loss_ind_parents,
                gradient_abs_expectation,
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
