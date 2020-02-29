import os
from collections import namedtuple
from typing import Union, Optional, Callable, List, Dict, Type
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
from causalpy.neural_networks.utils import StratifiedSampler

Hyperparams = namedtuple("Hyperparams", "inn env l0")


class AgnosticPredictor(ICPredictor):
    def __init__(
        self,
        cinn: Optional[Module] = None,
        cinn_params: Optional[Dict] = None,
        masker_network: Optional[Module] = None,
        masker_network_params: Optional[Dict] = None,
        epochs: int = 100,
        batch_size: int = 1024,
        mask_monte_carlo_size=1,
        optimizer_type: Type[Optimizer] = torch.optim.Adam,
        optimizer_params: Optional[Dict] = None,
        scheduler_type: Type[_LRScheduler] = torch.optim.lr_scheduler.StepLR,
        scheduler_params: Optional[Dict] = None,
        hyperparams: Optional[Hyperparams] = None,
        visualize_with_visdom: bool = False,
        log_level: bool = True,
    ):
        super().__init__(log_level=log_level)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.epochs = epochs
        self.hyperparams = (
            Hyperparams(l0=0.2, env=5, inn=2) if hyperparams is None else hyperparams
        )

        self.cinn = cinn
        self.cinn_params = (
            cinn_params
            if cinn_params is not None
            else dict(nr_blocks=3, dim=1, nr_layers=30)
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
            else dict(lr=1e-3, masker_lr=1e-2)
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

                # load plotly module dynamically
                globals()["go"] = importlib.import_module("plotly").__dict__[
                    "graph_objs"
                ]

                self.viz = visdom.Visdom()
            except ImportError as e:
                warnings.warn(
                    f"Package Visdom and Plotly required for training visualization, but (at least one) was not found! "
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
                        "lr": self.optimizer_params.pop("masker_lr", 1e-2),
                    },
                    {"params": self.cinn.parameters()},
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
        obs, target, environments = self.preprocess_input(obs, target_variable, envs)

        obs, target = tuple(
            map(
                lambda data: torch.from_numpy(data).float().to(self.device),
                (obs, target),
            )
        )

        if self.masker_net is None:
            self.masker_net = L0InputGate(dim_input=self.p, **self.masker_net_params)
        self.masker_net.to(self.device)
        if self.cinn is None:
            self.cinn = cINN(dim_condition=self.p, **self.cinn_params).to(self.device)
        self.cinn.to(self.device)
        self._set_optimizer()

        if self.use_visdom:
            self._plot_data_approximation(obs, target, environments)
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
        if show_epoch_progressbar:
            epoch_pbar = tqdm(range(self.epochs))
        else:
            epoch_pbar = range(self.epochs)
        nr_masks = (
            self.masker_net_params["monte_carlo_sample_size"]
            if "monte_carlo_size" in self.masker_net_params
            else 1
        )
        if ground_truth_mask is not None:
            ground_truth_mask = torch.as_tensor(ground_truth_mask).view(1, 1, self.p)

        epoch_losses = dict(total=[], invariance=[], cinn=[], l0=[])

        for epoch in epoch_pbar:
            batch_losses = dict(total=[], invariance=[], cinn=[], l0=[])
            for batch_indices, _ in data_loader:
                self.optimizer.zero_grad()

                mask = self.get_mask(ground_truth_mask)

                batch_indices = torch.sort(batch_indices)[0]

                masked_batch = self.masker_net(obs[batch_indices], mask).view(
                    nr_masks, -1, self.p
                )
                target_batch = target[batch_indices].view(-1, 1)

                inn_loss = self._compute_cinn_loss(masked_batch, target_batch, nr_masks)
                # env_loss, env_samples = self._compute_environmental_loss(
                #     obs,
                #     target,
                #     batch_indices,
                #     environments,
                #     mask,
                #     save_samples=self.use_visdom,
                # )
                env_loss = self._compute_overall_gaussian_loss(
                    masked_batch, target_batch, nr_masks
                )
                l0_loss = self.masker_net.complexity_loss()

                batch_loss = (
                    self.hyperparams.inn * inn_loss
                    + self.hyperparams.env * env_loss
                    + self.hyperparams.l0 * l0_loss
                )

                batch_losses["invariance"].append(env_loss.item())
                batch_losses["cinn"].append(inn_loss.item())
                batch_losses["l0"].append(l0_loss.item())
                batch_losses["total"].append(batch_loss.item())

                batch_loss.backward()
                self.optimizer.step()

            # store the various losses per batch into the epoch by averaging over batches
            for loss_acc, loss_list in epoch_losses.items():
                loss_list.append(np.mean(batch_losses[loss_acc]))

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

                self._plot_data_approximation(obs, target, environments)

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

    def get_mask(self, ground_truth_mask: Optional[Tensor] = None, final=False):
        if ground_truth_mask is not None:
            return ground_truth_mask
        return self.masker_net.create_gates(deterministic=final)

    def _compute_environmental_loss(
        self, obs, target, batch_indices, environments, mask, save_samples
    ):
        """
        Compute the environmental similarity loss of the gauss mapping for each individual environment data only.
        This is done to enfore that the cinn maps every single environment invidually to gauss and not just the
        complete data package together.
        """
        nr_masks = mask.size(0)
        env_samples = []
        env_loss = torch.zeros(1, device=self.device)
        for env, env_indices in environments.items():
            env_batch_indices = np.intersect1d(
                batch_indices.detach().cpu().numpy(), env_indices
            )
            target_env_batch = target[env_batch_indices]
            masked_env_batch = self.masker_net(obs[env_batch_indices], mask)

            env_gauss_sample = self.cinn(
                x=target_env_batch.view(-1, 1).repeat(nr_masks, 1),
                condition=masked_env_batch,
                rev=False,
            ).view(nr_masks, -1, 1)

            if save_samples:
                env_samples.append(env_gauss_sample)

            # all environmental samples should be gaussian,
            # thus measure the deviation of their distributions.
            loss_per_mask = torch.zeros(1, device=self.device)
            for mask_nr in range(nr_masks):
                true_gauss_sample = torch.randn(
                    env_gauss_sample[mask_nr].size(0), 1, device=self.device
                )
                loss_per_mask += self._wasserstein(
                    env_gauss_sample[mask_nr], true_gauss_sample
                )
            env_loss += loss_per_mask
        env_loss = env_loss / nr_masks
        return env_loss, env_samples

    def _compute_overall_gaussian_loss(self, masked_batch, target_batch, nr_masks):
        """
        Compute the environmental similarity loss of the gauss mapping for all environment data together.
        This is done to enfore that the cinn maps all environments combined to gauss. This seems redundant
        to the cINN maximum likelihood loss which enforces the same, yet this extra loss has proved to be
        necessary for stable training.
        """
        loss_per_mask = torch.zeros(1, device=self.device)

        true_gauss_sample = torch.randn(target_batch.size(0), 1, device=self.device)
        for mask_nr in range(nr_masks):
            env_gauss_sample = self.cinn(
                x=target_batch, condition=masked_batch[mask_nr], rev=False,
            ).view(nr_masks, -1, 1)

            loss_per_mask += self._wasserstein(
                env_gauss_sample[mask_nr], true_gauss_sample
            )
        loss_per_mask /= nr_masks
        return loss_per_mask

    def _compute_cinn_loss(self, masked_batch, target_batch, nr_masks):
        # compute the INN loss per mask, then average loss over the number of masks.
        # This is the monte carlo approximation of the loss via multiple mask samples.
        inn_loss = torch.zeros(1, device=self.device)
        for mask_idx in range(nr_masks):
            gauss_sample = self.cinn(
                x=target_batch, condition=masked_batch[mask_idx], rev=False,
            )
            inn_loss += self._inn_mllhood_loss(
                gauss_sample, log_grad=self.cinn.log_jacobian_cache
            )
        inn_loss /= nr_masks
        return inn_loss

    @staticmethod
    def _inn_mllhood_loss(gauss_sample, log_grad):
        return (gauss_sample ** 2 / 2 - log_grad).mean()

    @staticmethod
    def _wasserstein(x, y, normalize=False):
        if normalize:
            x, y = AgnosticPredictor.normalize_jointly(x, y)
        sort_sq_diff = (torch.sort(x, dim=0)[0] - torch.sort(y, dim=0)[0]).pow(2)
        std_prod = torch.std(x) * torch.std(y)
        return torch.mean(sort_sq_diff) / std_prod

    @staticmethod
    def normalize_jointly(x, y):
        xy = torch.cat((x, y), 0).detach()
        sd = torch.sqrt(xy.var(0))
        mean = xy.mean(0)
        x = (x - mean) / sd
        y = (y - mean) / sd
        return x, y

    def _plot_data_approximation(
        self, obs: Tensor, target: Tensor, environments: Dict,
    ):
        with torch.no_grad():
            for env, env_indices in environments.items():
                environment_size = env_indices.size
                target_approx_sample = self.cinn(
                    x=torch.randn(environment_size, 1, device=self.device),
                    condition=(obs[env_indices] * self.get_mask(final=True)).view(
                        -1, self.p
                    ),
                    rev=False,
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
                    title=r"Observed Y and approximated Y_hat Env. " + str(env)
                )
                self.viz.plotlyplot(targ_hist_fig, win=f"target_env_{env}")

    def _plot_gaussian_histograms(self, env_samples):
        fig = go.Figure()
        xr = torch.linspace(-5, 5, 100)
        for env, samples in enumerate(env_samples):
            fig.add_trace(
                go.Histogram(
                    x=samples.view(-1).cpu().detach().numpy(),
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
