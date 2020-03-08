import itertools
import os
from collections import namedtuple
from typing import Union, Optional, Callable, List, Dict, Type, Collection
import importlib

import numpy as np
import torch
import pandas as pd
import networkx as nx
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import Parameter, Module
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal
from tqdm import tqdm

from .icpbase import ICPredictor
import warnings
from causalpy.neural_networks import (
    AgnosticModel,
    Linear3D,
    MatrixSampler,
    L0Mask,
    L0InputGate,
    cINN,
)
from causalpy.neural_networks.utils import StratifiedSampler, wasserstein

Hyperparams = namedtuple("Hyperparams", "inn env l0 l2")


class MultiAgnosticPredictor(ICPredictor):
    def __init__(
        self,
        cinns: Optional[Collection[Module]] = None,
        cinn_params: Optional[Dict] = None,
        masker_network: Optional[Module] = None,
        masker_network_params: Optional[Dict] = None,
        epochs: int = 100,
        batch_size: int = 1024,
        mask_monte_carlo_size=1,
        device: Union[str, torch.device] = None,
        optimizer_type: Type[Optimizer] = torch.optim.Adam,
        optimizer_params: Optional[Dict] = None,
        scheduler_type: Type[_LRScheduler] = torch.optim.lr_scheduler.StepLR,
        scheduler_params: Optional[Dict] = None,
        hyperparams: Optional[Hyperparams] = None,
        visualize_with_visdom: bool = False,
        log_level: bool = True,
    ):
        super().__init__(log_level=log_level)
        if device is None:
            self.device = (
                torch.device("cuda")
                if torch.cuda.is_available()
                else torch.device("cpu")
            )
        else:
            self.device = device
        self.batch_size = batch_size
        self.epochs = epochs
        self.hyperparams = (
            Hyperparams(l0=0.2, env=5, inn=2, l2=0.005)
            if hyperparams is None
            else hyperparams
        )

        self.cinns = torch.nn.ModuleList(cinns) if cinns is not None else None
        self.cinn_params = (
            cinn_params
            if cinn_params is not None
            else dict(nr_blocks=3, dim=1, nr_layers=30, device=self.device)
        )

        self.masker_net = masker_network
        self.masker_net_params = (
            masker_network_params
            if masker_network_params is not None
            else dict(monte_carlo_sample_size=mask_monte_carlo_size, device=self.device)
        )

        self.optimizer = None
        self.optimizer_type = optimizer_type
        self.optimizer_params = (
            optimizer_params
            if optimizer_params is not None
            else dict(lr=1e-3, mask_lr=1e-2)
        )

        self.scheduler = None
        self.scheduler_type = scheduler_type
        self.scheduler_params = (
            dict(step_size=50, gamma=0.9)
            if scheduler_params is None
            else scheduler_params
        )

        self.use_visdom = visualize_with_visdom
        if self.use_visdom:
            try:
                import visdom

                # load plotly submodule dynamically
                globals()["go"] = importlib.import_module("plotly").__dict__[
                    "graph_objs"
                ]

                self.viz = visdom.Visdom()
            except ImportError as e:
                warnings.warn(
                    f"Packages Visdom and Plotly required for training visualization, but (at least one) was not found! "
                    f"Continuing without visualization.\n"
                    f"Exact error message was:\n{print(e)}"
                )
                self.use_visdom = True

    def _set_scheduler(self, force: bool = False):
        if self.scheduler is None or force:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self._set_optimizer(), **self.scheduler_params
            )
        return self.scheduler

    def _set_optimizer(self, force: bool = False):
        if self.optimizer is None or force:
            self.optimizer = self.optimizer_type(
                [
                    {
                        "params": self.masker_net.parameters(),
                        "lr": self.optimizer_params.pop("mask_lr", 1e-2),
                    },
                    {
                        "params": itertools.chain(
                            *(cinn.parameters() for cinn in self.cinns)
                        )
                    },
                ],
                lr=self.optimizer_params.pop("lr", 1e-3),
                **self.optimizer_params,
            )
        return self.optimizer

    def infer(
        self,
        obs: Union[pd.DataFrame, np.ndarray],
        envs: np.ndarray,
        target_variable: Union[int, str],
        show_epoch_progressbar=True,
        ground_truth_mask: Optional[Union[Tensor, np.ndarray]] = None,
    ):
        #########
        # Setup #
        #########

        obs, target, environments_map = self.preprocess_input(
            obs, target_variable, envs
        )
        nr_envs = max(environments_map.keys())
        obs, target = tuple(
            map(
                lambda data: torch.from_numpy(data).float().to(self.device),
                (obs, target),
            )
        )

        if self.masker_net is None:
            self.masker_net = L0InputGate(dim_input=self.p, **self.masker_net_params)
        self.masker_net.to(self.device)

        if self.cinns is None:
            self.cinns = torch.nn.ModuleList(
                [cINN(dim_condition=self.p, **self.cinn_params) for _ in range(nr_envs)]
            )
        else:
            for cinn in self.cinns:
                cinn.to(self.device)

        self._set_optimizer()

        if self.use_visdom:
            self._plot_data_approximation(obs, target, environments_map)
            self._plot_mask(
                self.masker_net.final_layer().detach().cpu().numpy().reshape(1, -1),
                self.get_parent_candidates(),
            )

        data_loader = DataLoader(
            dataset=TensorDataset(torch.arange(self.n), torch.zeros(self.n)),
            batch_size=self.batch_size,
            sampler=StratifiedSampler(
                data_source=np.arange(self.n),
                class_vector=torch.as_tensor(envs),
                batch_size=min(self.batch_size, self.n),
            ),
        )

        nr_masks = (
            self.masker_net_params["monte_carlo_sample_size"]
            if "monte_carlo_sample_size" in self.masker_net_params
            else 1
        )
        if ground_truth_mask is not None:
            ground_truth_mask = torch.as_tensor(
                ground_truth_mask, device=self.device
            ).view(1, 1, self.p)

        ############
        # Training #
        ############

        epoch_iter = (
            tqdm(range(self.epochs)) if show_epoch_progressbar else range(self.epochs)
        )
        epoch_losses = dict(total=[], invariance=[], cinn=[], l0_mask=[], l2=[])
        for epoch in epoch_iter:
            batch_losses = dict(total=[], invariance=[], cinn=[], l0_mask=[], l2=[])
            for batch_indices, _ in data_loader:
                self.optimizer.zero_grad()

                mask = self.get_mask(ground_truth_mask)

                batch_indices = torch.sort(batch_indices)[0]

                masked_batch = self.masker_net(obs[batch_indices], mask).view(
                    nr_masks, -1, self.p
                )
                target_batch = target[batch_indices].view(-1, 1)

                env_batch_indices = self._batch_indices_by_env(
                    batch_indices, environments_map
                )
                inn_loss = self._cinn_maxlikelihood_loss(
                    obs, target, batch_indices, environments_map, mask
                )
                env_loss, env_samples = self._environmental_invariance_loss(
                    obs, target, env_batch_indices, mask,
                )
                env_loss += self._pooled_gaussian_loss(
                    masked_batch, target_batch, nr_masks
                )
                l0_loss = self.masker_net.complexity_loss()
                l2_loss = self._l2_regularization()

                batch_loss = (
                    self.hyperparams.inn * inn_loss
                    + self.hyperparams.env * env_loss
                    + self.hyperparams.l0 * l0_loss
                    + self.hyperparams.l2 * l2_loss
                )

                batch_losses["invariance"].append(env_loss.item())
                batch_losses["cinn"].append(inn_loss.item())
                batch_losses["l0_mask"].append(l0_loss.item())
                batch_losses["l2"].append(l2_loss.item())
                batch_losses["total"].append(batch_loss.item())

                batch_loss.backward()
                self.optimizer.step()

            # store the various losses per batch into the epoch by averaging over batches
            for loss_acc, loss_list in epoch_losses.items():
                loss_list.append(np.mean(batch_losses[loss_acc]))

            ##################
            # Visualizations #
            ##################

            if self.use_visdom:
                self._plot_mask(
                    self.masker_net.final_layer().detach().cpu().numpy(),
                    self.get_parent_candidates(),
                    "mask_0",
                )

                self._plot_gaussian_histograms(env_samples)

                for loss_name, losses in epoch_losses.items():
                    self._plot_loss(
                        losses,
                        f"{loss_name}_loss",
                        f"{loss_name.capitalize()} Loss Movement",
                    )

                self._plot_data_approximation(obs, target, environments_map)

        mask = self.get_mask().detach().cpu().numpy().flatten()
        results = {
            var: mask
            for (var, mask) in sorted(
                {
                    parent_candidate: round(mask[idx], 2)
                    for idx, parent_candidate in self.index_to_varname.iteritems()
                }.items(),
                key=lambda x: x[0],
            )
        }
        return results

    def predict(self, obs: Union[pd.DataFrame]):
        obs = torch.as_tensor(
            obs[self.index_to_varname.values].to_numpy(),
            dtype=torch.float32,
            device=self.device,
        )
        mask = self.masker_net.create_gates(deterministic=True)
        obs *= mask
        predictions = torch.mean(
            torch.cat(
                [
                    cinn(
                        x=torch.randn(obs.size(0), 1, device=self.device),
                        condition=obs,
                        rev=True,
                    )
                    for cinn in self.cinns
                ],
                dim=1,
            ),
            dim=1,
        )
        return predictions

    def get_mask(self, ground_truth_mask: Optional[Tensor] = None, final=False):
        if ground_truth_mask is not None:
            return ground_truth_mask
        return self.masker_net.create_gates(deterministic=final)

    @staticmethod
    def _batch_indices_by_env(batch_indices: Union[Tensor, np.ndarray], env_map: Dict):
        batch_indices_by_env = dict()
        if isinstance(batch_indices, Tensor):
            batch_indices = batch_indices.numpy()
        for env, env_indices in env_map.items():
            batch_indices_by_env[env] = np.intersect1d(batch_indices, env_indices)
        return batch_indices_by_env

    def _l2_regularization(self):
        loss = torch.zeros(1, device=self.device)
        for cinn in self.cinns:
            for param in cinn.parameters():
                loss += (param ** 2).sum()
        return loss

    def _environmental_invariance_loss(
        self,
        obs: Tensor,
        target: Tensor,
        batch_indices_by_env: Dict[int, Tensor],
        mask: Tensor,
    ):
        """
        Compute the environmental similarity loss of the gauss mapping for each individual environment data only.
        This is done to enforce that the cinn maps every single environment individually to gauss and not just the
        complete data package together.
        """
        nr_masks = mask.size(0)
        env_samples = []
        env_loss = torch.zeros(1, device=self.device)
        for env_cinn, (env, env_indices) in zip(
            self.cinns, batch_indices_by_env.items()
        ):
            # split data by environmental affiliation
            target_notenv_batch = target[env_indices].view(-1, 1)
            masked_notenv_batch = self.masker_net(obs[env_indices], mask).view(
                nr_masks, -1, self.p
            )
            # compute the INN loss per mask
            per_mask_samples = []
            true_gauss_sample = torch.randn(len(env_indices), 1, device=self.device)
            for mask_nr in range(nr_masks):
                gauss_sample = env_cinn(
                    x=target_notenv_batch,
                    condition=masked_notenv_batch[mask_nr],
                    rev=False,
                )
                env_loss += wasserstein(gauss_sample, true_gauss_sample)
                per_mask_samples.append(gauss_sample)
            env_samples.append(torch.cat(per_mask_samples).view(nr_masks, -1, 1))
        env_loss = env_loss / nr_masks
        return env_loss, env_samples

    def _pooled_gaussian_loss(
        self, masked_batch: Tensor, target_batch: Tensor, nr_masks: int
    ):
        """
        Compute the environmental similarity loss of the gauss mapping for all environment data together.
        This is done to enforce that the cinn maps all environments combined to gauss. This seems redundant
        to the cINN maximum likelihood loss which enforces the same, yet this extra loss has proved to be
        necessary for stable training.
        """
        loss_per_mask = torch.zeros(1, device=self.device)

        true_gauss_sample = torch.randn(target_batch.size(0), 1, device=self.device)
        for mask_nr in range(nr_masks):
            for cinn in self.cinns:
                env_gauss_sample = cinn(
                    x=target_batch, condition=masked_batch[mask_nr], rev=False,
                ).view(nr_masks, -1, 1)

                loss_per_mask += wasserstein(
                    env_gauss_sample[mask_nr], true_gauss_sample
                )
        loss_per_mask /= nr_masks
        return loss_per_mask

    def _cinn_maxlikelihood_loss(
        self,
        obs: Tensor,
        target: Tensor,
        batch_indices_by_env: Dict[int, Tensor],
        mask: Tensor,
    ):
        """
        Compute the INN via maximum likelihood loss on the generated gauss samples of the cINN
        per mask, then average loss over the number of masks.
        This is the monte carlo approximation of the loss via multiple mask samples.
        """
        nr_masks = mask.size(0)
        inn_loss = torch.zeros(1, device=self.device)
        for env_cinn, (env, env_indices) in zip(
            self.cinns, batch_indices_by_env.items()
        ):
            # split data by environmental affiliation
            target_env_batch = target[env_indices].view(-1, 1).repeat(nr_masks, 1)
            masked_env_batch = self.masker_net(obs[env_indices], mask)
            gauss_sample = env_cinn(
                x=target_env_batch, condition=masked_env_batch, rev=False,
            ).view(nr_masks, -1, 1)
            inn_loss += torch.mean(
                torch.mean(
                    gauss_sample ** 2 / 2
                    - env_cinn.log_jacobian_cache.view(gauss_sample.shape),
                    dim=1,
                ),
                dim=0,
            )
        inn_loss /= nr_masks
        return inn_loss

    def _plot_data_approximation(
        self, obs: Tensor, target: Tensor, environments: Dict,
    ):
        with torch.no_grad():
            for env_cinn, (env, env_indices) in zip(self.cinns, environments.items()):
                target_approx_sample = env_cinn(
                    x=torch.randn(self.n, 1, device=self.device),
                    condition=(obs * self.get_mask(final=True)).view(-1, self.p),
                    rev=True,
                )

                targ_hist_fig = go.Figure()
                targ_hist_fig.add_trace(
                    go.Histogram(
                        x=target_approx_sample.detach().cpu().view(-1).numpy(),
                        name=r"Y_hat.",
                        histnorm="probability density",
                    )
                )
                targ_hist_fig.add_trace(
                    go.Histogram(
                        x=target[env_indices].detach().cpu().view(-1).numpy(),
                        name=r"Y",
                        histnorm="probability density",
                    )
                )
                # Overlay histograms
                targ_hist_fig.update_layout(barmode="overlay")
                # Reduce opacity to see all histograms
                targ_hist_fig.update_traces(opacity=0.5)
                targ_hist_fig.update_layout(
                    title=r"Marginal observ. Y vs approx. Y_hat Env. " + str(env)
                )
                self.viz.plotlyplot(targ_hist_fig, win=f"target_env_{env}")

    def _plot_gaussian_histograms(self, env_samples: Collection):
        fig = go.Figure()
        xr = torch.linspace(-5, 5, 100)
        for env, samples in enumerate(env_samples):
            fig.add_trace(
                go.Histogram(
                    x=samples[0].view(-1).cpu().detach().numpy(),
                    name=f"E = {env}",
                    histnorm="probability density",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=xr,
                y=Normal(0, 1).log_prob(xr).exp(),
                name="Gaussian",
                line={"color": "rgb(0,0,0)", "width": 2},
                line_shape="spline",
            )
        )
        # Overlay histograms
        fig.update_layout(barmode="overlay")
        # Reduce opacity to see all histograms
        fig.update_traces(opacity=0.5)
        fig.update_layout(title=f"Gaussian approximation per environment.",)
        self.viz.plotlyplot(fig, win="gaussian")

    def _plot_mask(self, curr_mask, possible_parents, window="mask_0"):
        if len(curr_mask) == len(possible_parents) == 1:
            curr_mask = np.append(curr_mask, 0)
            possible_parents = [possible_parents[0]] + [""]
        return self.viz.bar(
            X=curr_mask.reshape(1, -1),
            Y=np.array([n.replace("_", "") for n in possible_parents]).reshape(1, -1),
            win=window,
            opts=dict(title=f"Final Mask for variables", ytickmin=0, ytickmax=1),
        )

    def _plot_loss(self, losses, loss_win, title="Loss Behaviour"):
        losses = np.array(losses)
        if len(losses) > 1:
            ytickmax = np.mean(losses[losses < np.quantile(losses, 0.95)])
            ytickmax = ytickmax + abs(ytickmax)
            ytickmin = np.mean(losses[losses > np.quantile(losses, 0.01)])
            ytickmin = ytickmin - abs(ytickmin)
        else:
            ytickmin = ytickmax = None
        self.viz.line(
            X=np.arange(len(losses)),
            Y=losses,
            win=loss_win,
            opts=dict(title=title, ytickmin=ytickmin, ytickmax=ytickmax),
        )
