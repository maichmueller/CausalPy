import functools
import itertools
from abc import ABC, abstractmethod
from typing import Union, Optional, Dict, Type, Collection
import importlib
import numpy as np
import torch

import pandas as pd
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.nn import Module
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.distributions import Normal
from tqdm.auto import tqdm
from .icpbase import ICPredictor
import warnings

from causalpy.neural_networks import (
    L0InputGate,
    cINN,
)
from causalpy.neural_networks.utils import (
    Hyperparams,
    StratifiedSampler,
    mmd_multiscale,
    wasserstein,
    moments,
    hsic,
)
from causalpy.utils import TempFolder
from collections import namedtuple

import os
import re

go = None


class AgnosticPredictorBase(ICPredictor, ABC):
    """
    Model-less predictor for inferring causal parents of a target variable given its observed data
    under different environments.
    The theory to this predictor follows the paper called INSERT-TITLE-HERE (INSERT-LINK-HERE).
    """

    def __init__(
        self,
        masker_network: Optional[Module] = None,
        masker_network_params: Optional[Dict] = None,
        epochs: int = 100,
        batch_size: int = 1024,
        device: Union[str, torch.device] = None,
        optimizer_type: Type[Optimizer] = torch.optim.Adam,
        optimizer_params: Optional[Dict] = None,
        scheduler_type: Type[_LRScheduler] = torch.optim.lr_scheduler.StepLR,
        scheduler_params: Optional[Dict] = None,
        hyperparams: Optional[Dict[str, float]] = None,
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
        self.hyperparams = Hyperparams(l0=0.5, env=1, inn=1, independence=0.5, l2=0.01)
        if hyperparams is not None:
            self.hyperparams.update(**hyperparams)

        # to be defined by the subclass
        self.network_list = []
        self.network_params = []

        self.masker_net = masker_network
        self.masker_net_params = dict(
            monte_carlo_sample_size=1, initial_sparsity_rate=1, device=self.device
        )
        if masker_network_params is not None:
            self.masker_net_params.update(masker_network_params)

        self.optimizer = None
        self.optimizer_type = optimizer_type
        self.optimizer_params = dict(lr=1e-3, mask_lr=1e-1)
        if optimizer_params is not None:
            self.optimizer_params.update(optimizer_params)

        self.scheduler = None
        self.scheduler_type = scheduler_type
        self.scheduler_params = (
            dict(step_size_mask=50, step_size_model=100, mask_factor=2, model_factor=1)
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

            def mask_lr(epoch: int) -> float:
                exponent = epoch // self.scheduler_params["step_size_mask"]
                return self.scheduler_params["mask_factor"] ** min(3, exponent)

            def model_lr(epoch: int) -> float:
                exponent = epoch // self.scheduler_params["step_size_model"]
                return self.scheduler_params["model_factor"] ** exponent

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self._set_optimizer(), lr_lambda=[mask_lr, model_lr]
            )
        return self.scheduler

    def _set_optimizer(self, force: bool = False):
        raise NotImplementedError

    def reset(self):
        """
        Reset the neural network components for a fresh learning iteration.
        """
        if self.masker_net is None:
            self.masker_net = L0InputGate(dim_input=self.p, **self.masker_net_params)
        else:
            self.masker_net.reset_parameters()
        self.masker_net.to(self.device).train()
        for i, network in enumerate(self.network_list):
            if network is None:
                self.network_list[i] = cINN(
                    dim_condition=self.p, **self.network_params[0]
                )
            else:
                network.reset_parameters()
        self.network_list = (
            torch.nn.ModuleList(self.network_list).to(self.device).train()
        )
        self._set_optimizer()
        self._set_scheduler()

    def infer(
        self,
        obs: Union[pd.DataFrame, np.ndarray],
        envs: np.ndarray,
        target_variable: Union[int, str],
        nr_runs: int = 1,
        save_results: bool = False,
        results_folder: str = ".",
        results_filename: str = "results",
        **kwargs,
    ):

        with TempFolder("./temp_model_folder") as temp_folder:

            results_masks = []
            results_losses = []
            for run in range(nr_runs):
                final_mask, losses = self._infer(obs, envs, target_variable, **kwargs)
                results_masks.append(final_mask)
                results_losses.append(losses)
                for i, network in enumerate(self.network_list):
                    torch.save(
                        network.state_dict(),
                        os.path.join(temp_folder, f"run_{run}_network_{i}.pt"),
                    )
                torch.save(
                    self.masker_net.state_dict(),
                    os.path.join(temp_folder, f"run_{run}_masker.pt"),
                )

            best_run, lowest_loss, res_str = results_statistics(
                target_variable,
                self.get_parent_candidates(),
                results_losses,
                results_masks,
                results_filename,
                results_folder,
                save_results,
            )
            for i, network in enumerate(self.network_list):
                network.load_state_dict(
                    torch.load(
                        os.path.join(temp_folder, f"run_{best_run}_network_{i}.pt")
                    )
                )
            self.masker_net.load_state_dict(
                torch.load(os.path.join(temp_folder, f"run_{best_run}_masker.pt"))
            )

        return results_masks, results_losses, res_str

    def _infer(
        self,
        obs: Union[pd.DataFrame, np.ndarray],
        envs: np.ndarray,
        target_variable: Union[int, str],
        normalize: bool = False,
        show_epoch_progressbar: bool = True,
        ground_truth_mask: Optional[Union[Tensor, np.ndarray]] = None,
    ):

        #########
        # Setup #
        #########

        obs, target, environments_map = self.preprocess_input(
            obs, target_variable, envs, normalize
        )

        obs, target = tuple(
            map(
                lambda data: torch.from_numpy(data).float().to(self.device),
                (obs, target),
            )
        )

        self.reset()

        data_loader = DataLoader(
            dataset=TensorDataset(torch.arange(self.n), torch.as_tensor(envs)),
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
            if "monte_carlo_sample_size" in self.masker_net_params
            else 1
        )
        if ground_truth_mask is not None:
            ground_truth_mask = (
                self._validate_mask(ground_truth_mask)
                .to(self.device)
                .view(1, 1, self.p)
            )

        ############
        # Training #
        ############
        for param in self.masker_net.parameters():
            param.requires_grad = False
        mask_training_activated = False

        epoch_losses = dict(
            total=[], invariance=[], cinn=[], independence=[], l0_mask=[], l2=[]
        )
        for epoch in epoch_pbar:
            batch_losses = dict(
                total=[], invariance=[], cinn=[], independence=[], l0_mask=[], l2=[]
            )

            if epoch > 200 and not mask_training_activated:
                for param in self.masker_net.parameters():
                    param.requires_grad = True
                mask_training_activated = True

            for batch_indices, env_info in data_loader:
                self.optimizer.zero_grad()

                mask = self.get_mask(
                    ground_truth_mask, final=not mask_training_activated
                )
                batch_indices = torch.sort(batch_indices)[0]

                masked_batch = self.masker_net(obs[batch_indices], mask).view(
                    nr_masks, -1, self.p
                )
                target_batch = target[batch_indices].view(-1, 1)
                env_batch_indices = self._batch_indices_by_env(
                    batch_indices, environments_map
                )

                ##########
                # Losses #
                ##########

                inn_loss, gauss_sample = self._cinn_maxlikelihood_loss(
                    obs=obs,
                    target=target,
                    batch_indices=batch_indices,
                    env_batch_indices=env_batch_indices,
                    mask=mask,
                    masked_batch=masked_batch,
                    target_batch=target_batch,
                    nr_masks=nr_masks,
                )
                (
                    env_loss,
                    env_gauss_samples,
                    env_target_samples,
                ) = self._environmental_invariance_loss(
                    obs, target, env_batch_indices, mask, save_samples=self.use_visdom,
                )
                independence_loss = self._independence_loss(
                    obs, env_gauss_samples, env_batch_indices, mask
                )
                env_loss += self._pooled_gaussian_loss(gauss_sample)

                l0_loss = (
                    self.masker_net.complexity_loss()
                    if mask_training_activated
                    else torch.zeros(1, device=self.device)
                )
                l2_loss = self._l2_regularization()

                batch_loss = (
                    self.hyperparams.inn * inn_loss
                    + self.hyperparams.env * env_loss
                    + self.hyperparams.l0 * l0_loss
                    + self.hyperparams.l2 * l2_loss
                    + self.hyperparams.independence * independence_loss
                )

                batch_losses["invariance"].append(env_loss.item())
                batch_losses["independence"].append(independence_loss.item())
                batch_losses["cinn"].append(inn_loss.item())
                batch_losses["l0_mask"].append(l0_loss.item())
                batch_losses["l2"].append(l2_loss.item())
                batch_losses["total"].append(batch_loss.item())

                batch_loss.backward()
                self.optimizer.step()

            # update learning rates after last batch has finished
            self.scheduler.step()

            # store the various losses per batch into the epoch by averaging over batches
            for loss_acc, loss_list in epoch_losses.items():
                loss_list.append(np.mean(batch_losses[loss_acc]))

            ##################
            # Visualizations #
            ##################

            if self.use_visdom:
                self._plot_mask(
                    self.get_mask(final=True).detach().cpu().numpy(),
                    self.get_parent_candidates(),
                    "mask_0",
                )

                self._plot_gaussian_histograms(obs, target, environments_map)

                for loss_name, losses in epoch_losses.items():
                    self._plot_loss(
                        losses, f"{loss_name}_loss", f"{loss_name.capitalize()} Loss",
                    )

                self._plot_data_approximation(obs, target, environments_map)

        results = self._mask_dict()
        return results, epoch_losses

    def _mask_dict(self):
        mask = self.get_mask(final=True).view(-1).cpu().detach().numpy().round(2)
        results = {
            var: mask_val
            for (var, mask_val) in sorted(
                {
                    parent_candidate: mask[idx]
                    for idx, parent_candidate in self.index_to_varname.iteritems()
                }.items(),
                key=lambda x: x[0],
            )
        }
        return results

    @staticmethod
    def predict_input_validator(func):
        @functools.wraps(func)
        def wrapped_validate(
            self,
            obs: Union[pd.DataFrame, Tensor],
            mask: Optional[Union[pd.DataFrame, np.ndarray, Tensor]] = None,
        ):
            if isinstance(obs, pd.DataFrame):
                obs = torch.as_tensor(
                    obs[self.index_to_varname].to_numpy(),
                    dtype=torch.float32,
                    device=self.device,
                )
            if mask is None:
                self.masker_net.eval()
                mask = self.masker_net.create_gates(deterministic=True)
            else:
                mask = self._validate_mask(mask)
            for network in self.network_list:
                network.eval()

            return func(self, obs, mask)

        return wrapped_validate

    def predict(
        self,
        obs: Union[pd.DataFrame, Tensor],
        mask: Optional[Union[pd.DataFrame, np.ndarray, Tensor]] = None,
    ):
        """
        Predict samples from the target distribution.
        """
        raise NotImplementedError

    def _validate_mask(self, mask: Union[pd.DataFrame, np.ndarray, Tensor]):
        assert mask.shape == (
            1,
            self.p,
        ), f"Provided mask shape needs to be (1, {self.p})."
        if isinstance(mask, pd.DataFrame):
            no_real_column_names = mask.columns == range(self.p)
            if no_real_column_names:
                # masking values assumed to be in correct order
                mask = mask.to_numpy()
            else:
                # assert the column names fit the actual names
                assert np.all(
                    np.isin(mask.columns, self.index_to_varname.to_numpy())
                ), "All mask column names need to be integer indices or the correct variable names of the used data."

                mask = mask[self.index_to_varname].to_numpy()
        mask = torch.as_tensor(mask)
        return mask

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

    @abstractmethod
    def _environmental_invariance_loss(
        self,
        obs: Tensor,
        target: Tensor,
        batch_indices_by_env: Dict,
        mask: Tensor,
        save_samples: bool = False,
    ):
        """
        Compute the environmental similarity loss of the gauss/target mapping for each individual environment data only.
        This is done to enforce that the cINN maps every single environment individually to gauss and the target
        distribution and not just the complete data package together.
        """
        raise NotImplementedError

    @abstractmethod
    def _independence_loss(
        self,
        obs: Tensor,
        env_gauss_samples: Collection[Tensor],
        batch_indices_by_env: Dict,
        mask: Tensor,
    ):
        """
        Compute the dependence between the estimated gaussian noise and the predictors X of the target Y in each
        environment. This independence is an essential assumption in the construction of SCMs, thus should be enforced.
        """
        raise NotImplementedError

    @abstractmethod
    def _cinn_maxlikelihood_loss(self, *args, **kwargs):
        """
        Compute the maximum likelihood loss to a gauss sample.
        """
        raise NotImplementedError

    def _l2_regularization(self):
        loss = torch.zeros(1, device=self.device)
        for network in self.network_list:
            for param in network.parameters():
                loss += (param ** 2).sum()
        return loss

    def _pooled_gaussian_loss(self, gauss_samples: Collection[Tensor]):
        """
        Compute the environmental similarity loss of the gauss mapping for all environment data together.
        This is done to enforce that the cinn maps all environments combined to gauss. This seems redundant
        to the cINN maximum likelihood loss which enforces the same, yet this extra loss has proven to be
        necessary for stable training.
        """
        loss_per_mask = torch.zeros(1, device=self.device)
        for network, gauss_sample in zip(self.network_list, gauss_samples):
            nr_masks = gauss_sample.size(0)
            true_gauss_sample = torch.randn(gauss_sample.size(1), 1, device=self.device)

            for mask_nr in range(nr_masks):
                estimate = gauss_sample[mask_nr]
                loss_per_mask += wasserstein(estimate, true_gauss_sample)
                # loss_per_mask += mmd_multiscale(estimate, true_gauss_sample)
                loss_per_mask += moments(estimate, true_gauss_sample)
        loss_per_mask /= nr_masks
        return loss_per_mask

    @staticmethod
    def standardize(x: Tensor):
        return (x - torch.mean(x)) / torch.std(x)

    def _plot_data_approximation(
        self, obs: Tensor, target: Tensor, environments: Dict,
    ):
        with torch.no_grad():
            for env, env_indices in environments.items():
                target_approx_sample = self.predict(obs[env_indices])

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
                    title=r"Marginal observ. Y and approx. Y_hat Env. " + str(env)
                )
                self.viz.plotlyplot(targ_hist_fig, win=f"target_env_{env}")

    def _plot_gaussian_histograms(
        self, obs: Tensor, target: Tensor, environments_map: Dict[int, Tensor]
    ):
        fig = go.Figure()
        xr = torch.linspace(-5, 5, 100)
        mask = self.get_mask(final=True)
        nr_masks = mask.size(0)
        nr_envs = len(environments_map)
        subsample_size_per_mask = min(2000, target.size(0)) // nr_envs
        for env, env_indices in environments_map.items():
            target_env = target[env_indices].view(-1, 1)
            masked_env = self.masker_net(obs[env_indices], mask).view(-1, self.p)

            for i, network in enumerate(self.network_list):
                env_gauss_sample = self._subsample_from_sample(
                    network.normalizing_flow(x=target_env, condition=masked_env).view(
                        nr_masks, -1, 1
                    ),
                    nr_masks,
                    subsample_size_per_mask,
                )
                fig.add_trace(
                    go.Histogram(
                        x=env_gauss_sample.view(-1).cpu().detach().numpy(),
                        name=f"Net = {i}, E = {env}",
                        histnorm="probability density",
                        nbinsx=30,
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

    @staticmethod
    def _subsample_from_sample(env_gauss_sample, nr_masks, subsample_size_per_mask):
        """
        Sub-select `subsample_size_per_mask` many samples of each sample of each mask.

        This is done to limit the computational burden on the visualization.
        """
        potential_indices = np.arange(env_gauss_sample.size(1))
        indices = [
            np.random.choice(
                potential_indices, size=subsample_size_per_mask, replace=False,
            )
            for _ in range(nr_masks)
        ]
        env_gauss_sample = torch.cat(
            [env_gauss_sample[mask, indices] for mask, indices in enumerate(indices)]
        )
        return env_gauss_sample

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
        if not len(losses):
            ytickmax = np.quantile(losses, 0.99)
            ytickmax = ytickmax + 0.5 * abs(ytickmax)
            ytickmin = np.quantile(losses, 0)
            ytickmin = ytickmin - 0.5 * abs(ytickmin)
        else:
            ytickmin = ytickmax = None
        self.viz.line(
            X=np.arange(len(losses)),
            Y=losses,
            win=loss_win,
            opts=dict(
                title=title, ytickmin=ytickmin, ytickmax=ytickmax, xlabel="Epoch"
            ),
        )


class AgnosticPredictor(AgnosticPredictorBase):
    """
    Single inferential network model.
    """

    def __init__(
        self,
        network: Optional[Module] = None,
        network_params: Optional[Dict] = None,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)

        self.network_list.append(network)
        if network_params is not None:
            self.network_params.append(network_params)
        else:
            self.network_params.append(
                dict(nr_blocks=3, dim=1, nr_layers=30, device=self.device)
            )

    def _set_optimizer(self, force: bool = False):
        if self.optimizer is None or force:
            self.optimizer = self.optimizer_type(
                [
                    {
                        "params": self.masker_net.parameters(),
                        "lr": self.optimizer_params.pop("mask_lr", 1e-2),
                    },
                    {"params": self.network_list[0].parameters()},
                ],
                lr=self.optimizer_params.pop("lr", 1e-3),
                **self.optimizer_params,
            )
        return self.optimizer

    @AgnosticPredictorBase.predict_input_validator
    def predict(
        self,
        obs: Union[pd.DataFrame, Tensor],
        mask: Optional[Union[pd.DataFrame, np.ndarray, Tensor]] = None,
    ):
        obs *= mask
        gaussian_sample = torch.randn(obs.size(0), 1, device=self.device)
        predictions = self.network_list[0].generating_flow(
            z=gaussian_sample, condition=obs
        )
        return predictions

    def _cinn_maxlikelihood_loss(
        self, masked_batch: Tensor, target_batch: Tensor, nr_masks: int, **unused_kwargs
    ):
        """
        Compute the INN loss per mask, then average loss over the number of masks.
        This is the monte carlo approximation of the loss via multiple mask samples.
        """
        gauss_samples = []
        inn_loss = torch.zeros(1, device=self.device)
        for network in self.network_list:
            gauss_sample = network.normalizing_flow(
                x=target_batch.repeat(nr_masks, 1),
                condition=masked_batch.view(-1, self.p),
            ).view(nr_masks, -1, 1)
            gauss_samples.append(gauss_sample)
            # maximum likelihood of a standard gaussian
            inn_loss = torch.mean(
                torch.mean(
                    gauss_sample ** 2 / 2
                    - network.log_jacobian_cache.view(gauss_sample.shape),
                    dim=1,
                ),
                dim=0,
            )
        return inn_loss, gauss_samples

    def _environmental_invariance_loss(
        self,
        obs: Tensor,
        target: Tensor,
        batch_indices_by_env: Dict,
        mask: Tensor,
        save_samples: bool = False,
    ):
        """
        Compute the environmental similarity loss of the gauss/target mapping for each individual environment data only.
        This is done to enforce that the cINN maps every single environment individually to gauss and the target
        distribution and not just the complete data package together.
        """
        nr_masks = mask.size(0)
        env_gauss_samples, env_target_samples = [], []
        env_loss = torch.zeros(1, device=self.device)

        for env, env_batch_indices in batch_indices_by_env.items():
            target_env_batch = target[env_batch_indices].view(-1, 1)
            masked_env_batch = self.masker_net(obs[env_batch_indices], mask)

            env_gauss_sample = (
                self.network_list[0]
                .normalizing_flow(
                    x=target_env_batch.repeat(nr_masks, 1), condition=masked_env_batch
                )
                .view(nr_masks, -1, 1)
            )

            env_target_sample = (
                self.network_list[0]
                .generating_flow(
                    z=torch.randn(
                        target_env_batch.size(0), 1, device=self.device
                    ).repeat(nr_masks, 1),
                    condition=masked_env_batch,
                )
                .view(nr_masks, -1, 1)
            )

            if save_samples:
                env_gauss_samples.append(env_gauss_sample)
                env_target_samples.append(env_target_sample)

            # all environmental normalized samples should be gaussian,
            # all environmental target samples should follow the target marginal distribution,
            # thus measure the deviation of these distributions.
            loss_per_mask = torch.zeros(1, device=self.device)
            true_gauss = torch.randn(env_gauss_sample.size(1), 1, device=self.device)

            # compute the loss in the normalizing direction (gauss) and the generating direction (target).
            for mask_nr in range(nr_masks):
                estimate_gauss = env_gauss_sample[mask_nr]
                loss_per_mask += wasserstein(estimate_gauss, true_gauss)
                # loss_per_mask += mmd_multiscale(estimate_gauss, true_gauss)
                loss_per_mask += moments(estimate_gauss, true_gauss)

                estimate_target = env_target_sample[mask_nr]
                loss_per_mask += wasserstein(estimate_target, target_env_batch)
                # loss_per_mask += mmd_multiscale(estimate_target, target_env_batch)
                loss_per_mask += moments(estimate_target, target_env_batch)

            env_loss += loss_per_mask
        env_loss = env_loss / nr_masks
        return env_loss, env_gauss_samples, env_target_samples

    def _independence_loss(
        self,
        obs: Tensor,
        env_gauss_samples: Collection[Tensor],
        batch_indices_by_env: Dict,
        mask: Tensor,
    ):
        """
        Compute the dependence between the estimated gaussian noise and the predictors X of the target Y in each
        environment. This independence is an essential assumption in the construction of SCMs, thus should be enforced.
        """
        nr_masks = mask.size(0)
        independence_loss = torch.zeros(1, device=self.device)

        for gauss_sample, (env, env_batch_indices) in zip(
            env_gauss_samples, batch_indices_by_env.items()
        ):
            masked_env_batch = self.masker_net(obs[env_batch_indices], mask).view(
                nr_masks, -1, self.p
            )
            for mask_nr in range(nr_masks):
                # punish dependence between predictors and noise
                independence_loss += hsic(
                    gauss_sample[mask_nr],
                    masked_env_batch[mask_nr],
                    gauss_sample.size(1),
                )

        independence_loss = independence_loss / nr_masks
        return independence_loss

    def _pooled_gaussian_loss(self, gauss_samples: Collection[Tensor]):
        """
        Compute the environmental similarity loss of the gauss mapping for all environment data together.
        This is done to enforce that the cinn maps all environments combined to gauss. This seems redundant
        to the cINN maximum likelihood loss which enforces the same, yet this extra loss has proven to be
        necessary for stable training.
        """
        loss_per_mask = torch.zeros(1, device=self.device)
        nr_masks = gauss_samples[0].size(0)
        gauss_sample = gauss_samples[0]
        true_gauss_sample = torch.randn(gauss_sample.size(1), 1, device=self.device)

        for mask_nr in range(nr_masks):
            loss_per_mask += wasserstein(gauss_sample[mask_nr], true_gauss_sample)
        loss_per_mask /= nr_masks
        return loss_per_mask


class MultiAgnosticPredictor(AgnosticPredictorBase):
    """
    Multi inferential networks model.
    """

    def __init__(
        self,
        networks: Optional[Collection[Module]] = None,
        networks_params: Optional[Dict] = None,
        **base_kwargs,
    ):
        super().__init__(**base_kwargs)

        if networks is not None:
            for net in networks:
                self.network_list.append(net)
        self.network_params = (
            networks_params
            if networks_params is not None
            else dict(nr_blocks=3, dim=1, nr_layers=30, device=self.device)
        )

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
                            *(cinn.parameters() for cinn in self.network_list)
                        )
                    },
                ],
                lr=self.optimizer_params.pop("lr", 1e-3),
                **self.optimizer_params,
            )
        return self.optimizer

    @AgnosticPredictorBase.predict_input_validator
    def predict(
        self,
        obs: Union[pd.DataFrame, Tensor],
        mask: Optional[Union[pd.DataFrame, np.ndarray, Tensor]] = None,
    ):
        obs *= mask
        predictions = torch.mean(
            torch.cat(
                [
                    network.generating_flow(
                        z=torch.randn(obs.size(0), 1, device=self.device),
                        condition=obs,
                    )
                    for network in self.network_list
                ],
                dim=1,
            ),
            dim=1,
        )
        return predictions

    @staticmethod
    def _batch_indices_by_env(
        batch_indices: Union[Tensor, np.ndarray], env_map: Dict[int, np.ndarray]
    ) -> Dict[int, np.ndarray]:
        batch_indices_by_env = dict()
        if isinstance(batch_indices, Tensor):
            batch_indices = batch_indices.numpy()
        for env, env_indices in env_map.items():
            batch_indices_by_env[env] = np.intersect1d(
                batch_indices, env_indices, return_indices=False
            )
        return batch_indices_by_env

    def _environmental_invariance_loss(
        self,
        obs: Tensor,
        target: Tensor,
        batch_indices_by_env: Dict[int, np.ndarray],
        mask: Tensor,
        save_samples: bool = False,
    ):
        """
        Compute the environmental similarity loss of the gauss mapping for each individual environment data only.
        This is done to enforce that the cinn maps every single environment individually to gauss and not just the
        complete data package together.
        """
        nr_masks = mask.size(0)
        env_gauss_samples, env_target_samples = [], []
        env_loss = 0
        for env_cinn, (env, env_batch_indices) in zip(
            self.network_list, batch_indices_by_env.items()
        ):
            # split data by environmental affiliation
            target_env_batch = target[env_batch_indices].view(-1, 1)
            masked_env_batch = self.masker_net(obs[env_batch_indices], mask)
            # compute the INN loss per mask
            true_gauss_sample = torch.randn(
                len(env_batch_indices), 1, device=self.device
            )
            gauss_sample = env_cinn(
                x=target_env_batch.repeat(nr_masks, 1),
                condition=masked_env_batch,
                rev=False,
            ).view(nr_masks, -1, 1)
            target_sample = env_cinn(
                x=true_gauss_sample.repeat(nr_masks, 1),
                condition=masked_env_batch,
                rev=True,
            ).view(nr_masks, -1, 1)

            for mask_nr in range(nr_masks):
                for estimate, observation in zip(
                    (gauss_sample[mask_nr], target_sample[mask_nr]),
                    (true_gauss_sample, target_env_batch),
                ):
                    env_loss += wasserstein(estimate, observation)
                    # env_loss += mmd_multiscale(estimate, observation)
                    env_loss += moments(estimate, observation)

            if save_samples:
                env_gauss_samples.append(gauss_sample)
                env_target_samples.append(target_sample)
        env_loss = env_loss / nr_masks
        return env_loss, env_gauss_samples, env_target_samples

    @abstractmethod
    def _independence_loss(
        self,
        obs: Tensor,
        env_gauss_samples: Collection[Tensor],
        batch_indices_by_env: Dict,
        mask: Tensor,
    ):
        """
        Compute the dependence between the estimated gaussian noise and the predictors X of the target Y in each
        environment. This independence is an essential assumption in the construction of SCMs, thus should be enforced.
        """

    def _cinn_maxlikelihood_loss(
        self,
        obs: Tensor,
        target: Tensor,
        batch_indices: Union[Tensor, np.ndarray],
        batch_indices_by_env: Dict[int, np.ndarray],
        mask: Tensor,
        **unused_kwargs,
    ):
        """
        Compute the INN via maximum likelihood loss on the generated gauss samples of the cINN
        per mask, then average loss over the number of masks.
        This is the monte carlo approximation of the loss via multiple mask samples.
        """
        nr_masks = mask.size(0)
        inn_loss = 0
        for env_cinn, (env, env_indices) in zip(
            self.network_list, batch_indices_by_env.items()
        ):
            # split data by environmental affiliation
            notenv_batch_indices = torch.from_numpy(
                np.setdiff1d(batch_indices, env_indices)
            )
            target_notenv_batch = (
                target[notenv_batch_indices].view(-1, 1).repeat(nr_masks, 1)
            )
            masked_notenv_batch = self.masker_net(obs[notenv_batch_indices], mask)
            gauss_sample = env_cinn(
                x=target_notenv_batch, condition=masked_notenv_batch, rev=False,
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


def results_statistics(
    target_variable,
    candidates,
    results_losses,
    results_masks,
    results_filename,
    results_folder,
    save_results,
):
    full_results = {
        var: []
        for var in sorted(
            candidates, key=lambda x: int(x.split("_")[-1]) if "_" in x else x,
        )
    }
    best_invariance_result = -1
    lowest_invariance_loss = float("infinity")
    for i, (result_mask, result_loss) in enumerate(zip(results_masks, results_losses)):
        for var, mask in result_mask.items():
            full_results[var].append(mask)
            if result_loss["invariance"][-1] < lowest_invariance_loss:
                lowest_invariance_loss = result_loss["invariance"][-1]
                best_invariance_result = i
    statistics = dict()
    for var, values in full_results.items():
        stats_dict = dict()
        for func, args, kwargs in zip(
            (np.mean, np.min, np.max, np.var, np.quantile, np.quantile, np.quantile,),
            [values] * 7,
            (None, None, None, None, {"q": 0.25}, {"q": 0.5}, {"q": 0.75}),
        ):
            func_str = str(
                re.search(r"(?<=function\s)[a-zA-z\d]+(?=\s)", str(func)).group()
            )
            if kwargs is not None:
                func_str += ", " + ", ".join([f"{k}={v}" for k, v in kwargs.items()])
                stats_dict[func_str] = func(args, **kwargs).round(3)
            else:
                stats_dict[func_str] = func(args).round(3)
        statistics[var] = stats_dict
    res_str = "\nLearning outcomes:\n"
    res_str += f"Target Variable was: {target_variable}\n"
    res_str += f"Potential Causal Parents: {', '.join(candidates)}\n"
    for var, stat_dict in statistics.items():
        res_str += f"{var}\n"
        for func_str, value in stat_dict.items():
            res_str += f"\t{func_str}: {value}\n"
        res_str += f"\tresults: {full_results[var]}\n"
    res_str += "Best individual run by invariance loss:\n"
    res_str += f"\tRun: {best_invariance_result}\n"
    res_str += f"\tLoss: {lowest_invariance_loss}\n"
    val_str = ", ".join(
        [f"{var}: {val}" for var, val in results_masks[best_invariance_result].items()]
    )
    res_str += f"\tMask: {val_str}\n\n"
    if save_results:
        if not os.path.isdir(results_folder):
            os.mkdir(results_folder)
        results_df = pd.DataFrame.from_dict(statistics, orient="columns")
        results_df.loc["results_array"] = {
            var: str(values) for (var, values) in full_results.items()
        }
        results_df.to_csv(f"{os.path.join(results_folder, results_filename)}.csv")

    return best_invariance_result, lowest_invariance_loss, res_str
