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
from tqdm.auto import tqdm
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
from causalpy.neural_networks.utils import (
    StratifiedSampler,
    mmd_multiscale,
    wasserstein,
    moments,
    hsic,
)
from collections import namedtuple
import shutil
import os
import re


Hyperparams = namedtuple("Hyperparams", "inn env l0 l2")


class TempFolder:
    def __init__(self, folder: str):
        self.folder = folder

    def __enter__(self):
        i = 0
        while os.path.isdir(self.folder):
            self.folder += f"_{i}"
            i += 1

        os.mkdir(self.folder)

    def __exit__(self, type, value, traceback):
        shutil.rmtree(self.folder, ignore_errors=True)


class AgnosticPredictor(ICPredictor):
    def __init__(
        self,
        cinn: Optional[Module] = None,
        cinn_params: Optional[Dict] = None,
        masker_network: Optional[Module] = None,
        masker_network_params: Optional[Dict] = None,
        epochs: int = 100,
        batch_size: int = 1024,
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
            Hyperparams(l0=2, env=1, inn=1, l2=0.01)
            if hyperparams is None
            else hyperparams
        )

        self.cinn = cinn
        self.cinn_params = (
            cinn_params
            if cinn_params is not None
            else dict(nr_blocks=3, dim=1, nr_layers=30, device=self.device)
        )

        self.masker_net = masker_network
        self.masker_net_params = (
            masker_network_params
            if masker_network_params is not None
            else dict(
                monte_carlo_sample_size=1, initial_sparsity_rate=1, device=self.device
            )
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
                return self.scheduler_params["mask_factor"] ** min(4, exponent)

            def model_lr(epoch: int) -> float:
                exponent = epoch // self.scheduler_params["step_size_model"]
                return self.scheduler_params["model_factor"] ** exponent

            self.scheduler = torch.optim.lr_scheduler.LambdaLR(
                self._set_optimizer(), lr_lambda=[mask_lr, model_lr]
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
                    {"params": self.cinn.parameters()},
                ],
                lr=self.optimizer_params.pop("lr", 1e-3),
                **self.optimizer_params,
            )
        return self.optimizer

    def reset(self):
        """
        Reset the neural network components for a fresh learning iteration.
        """
        if self.cinn is None:
            self.masker_net = L0InputGate(dim_input=self.p, **self.masker_net_params)
        else:
            self.masker_net = type(self.masker_net)(
                dim_input=self.p, **self.masker_net_params
            )
        self.masker_net.to(self.device).train()
        if self.cinn is None:
            self.cinn = cINN(dim_condition=self.p, **self.cinn_params)
        else:
            self.cinn = type(self.cinn)(dim_condition=self.p, **self.cinn_params)
        self.cinn.to(self.device).train()
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
        temp_folder = "temp_model_folder"
        with TempFolder(temp_folder):

            results_masks = []
            results_losses = []
            for run in range(nr_runs):
                final_mask, losses = self._infer(obs, envs, target_variable, **kwargs)
                results_masks.append(final_mask)
                results_losses.append(losses)
                torch.save(
                    self.cinn.state_dict(),
                    os.path.join(".", temp_folder, f"run_{run}_model.pt"),
                )
                torch.save(
                    self.masker_net.state_dict(),
                    os.path.join(".", temp_folder, f"run_{run}_masker.pt"),
                )

            best_run, lowest_loss = self._results_statistics(
                results_losses,
                results_masks,
                results_filename,
                results_folder,
                save_results,
            )
            self.cinn = type(self.cinn)(dim_condition=self.p, **self.cinn_params)
            self.masker_net = type(self.masker_net)(
                dim_input=self.p, **self.masker_net_params
            )
            self.cinn.load_state_dict(
                torch.load(os.path.join(".", temp_folder, f"run_{best_run}_model.pt"))
            )
            self.masker_net.load_state_dict(
                torch.load(os.path.join(".", temp_folder, f"run_{best_run}_masker.pt"))
            )

        return results_masks, results_losses

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

        epoch_losses = dict(total=[], invariance=[], cinn=[], l0_mask=[], l2=[])
        for epoch in epoch_pbar:
            batch_losses = dict(total=[], invariance=[], cinn=[], l0_mask=[], l2=[])

            if epoch > 10 and not mask_training_activated:
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

                inn_loss, gauss_sample = self._cinn_maxlikelihood_loss(
                    masked_batch, target_batch, nr_masks
                )
                env_loss = self._environmental_invariance_loss(
                    obs, target, env_batch_indices, mask, save_samples=self.use_visdom,
                )[0]
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
                )

                batch_losses["invariance"].append(env_loss.item())
                batch_losses["cinn"].append(inn_loss.item())
                batch_losses["l0_mask"].append(l0_loss.item())
                batch_losses["l2"].append(l2_loss.item())
                batch_losses["total"].append(batch_loss.item())

                batch_loss.backward()
                self.optimizer.step()

            # update lr after last batch has finished
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
                        losses,
                        f"{loss_name}_loss",
                        f"{loss_name.capitalize()} Loss Movement",
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

    def predict(
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
        self.cinn.eval()

        obs *= mask
        gaussian_sample = torch.randn(obs.size(0), 1, device=self.device)
        predictions = self.cinn(x=gaussian_sample, condition=obs, rev=True)
        return predictions

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

    def _l2_regularization(self):
        loss = torch.zeros(1, device=self.device)
        for param in self.cinn.parameters():
            loss += (param ** 2).sum()
        return loss

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

            env_gauss_sample = self.cinn(
                x=target_env_batch.repeat(nr_masks, 1),
                condition=masked_env_batch,
                rev=False,
            ).view(nr_masks, -1, 1)

            env_target_sample = self.cinn(
                x=torch.randn(target_env_batch.size(0), 1, device=self.device).repeat(
                    nr_masks, 1
                ),
                condition=masked_env_batch,
                rev=True,
            ).view(nr_masks, -1, 1)

            if save_samples:
                env_gauss_samples.append(env_gauss_sample)
                env_target_samples.append(env_target_sample)

            # all environmental normalized samples should be gaussian,
            # all environmental target samples should follow the target marginal distribution,
            # thus measure the deviation of these distributions.
            loss_per_mask = torch.zeros(1, device=self.device)
            true_gauss = torch.randn(env_gauss_sample.size(1), 1, device=self.device)

            # compute the loss in the normalizing direction (gauss) and the inference direction (target).
            masked_env_batch = masked_env_batch.view(nr_masks, -1, self.p)
            for mask_nr in range(nr_masks):
                estimate_gauss = env_gauss_sample[mask_nr]
                loss_per_mask += wasserstein(estimate_gauss, true_gauss)
                # loss_per_mask += mmd_multiscale(estimate_gauss, true_gauss)
                loss_per_mask += moments(estimate_gauss, true_gauss)

                # punish dependence between predictors and noise
                loss_per_mask += hsic(
                    estimate_gauss, masked_env_batch[mask_nr], len(estimate_gauss)
                )

                estimate_target = env_target_sample[mask_nr]
                # loss_per_mask += wasserstein(estimate_target, target_env_batch)
                # loss_per_mask += mmd_multiscale(estimate_target, target_env_batch)
                loss_per_mask += moments(estimate_target, target_env_batch)

                # for estimate, observation in zip(
                #     (env_gauss_sample[mask_nr], env_target_sample[mask_nr]),
                #     (true_gauss, target_env_batch),
                # ):
                #     # all three measures enforce a quality approximation of the gauss, and of the
                #     # target marginal distribution.
                #     # (Should in theory suffice taking either mmd + moments or wasserstein)
                #     loss_per_mask += wasserstein(estimate, observation)
                #     loss_per_mask += mmd_multiscale(estimate, observation)
                #     loss_per_mask += moments(estimate, observation)
            env_loss += loss_per_mask
        env_loss = env_loss / nr_masks
        return env_loss, env_gauss_samples, env_target_samples

    # @staticmethod
    # def _batch_indices_by_env(env_batch_info: Union[Tensor]):
    #     batch_indices_by_env = dict()
    #
    #     # first sort the batch environment affiliation information by each entries env index.
    #     # we do this as x[argsort(x)] tells us the indices of the environments in order.
    #     sorted_env_info, argsort_env_info = torch.sort(env_batch_info)
    #     # To be done now, we only need to know where the breakpoints are in this list, i.e.
    #     # at which indices we are crossing into the next environment index.
    #     # To this end we compute the unique indices in the argsorted env info list, which will give us a tuple
    #     # of two lists:
    #     # 1st list: the unique, and sorted environments we can find in the total env list.
    #     # 2nd list: the list of index breaks. This will look like
    #     #   [start_idx_{env 0}, start_idx_{env 1}, start_idx_{env 2},...],
    #     # so that we can then find the full list by iterating over the environment names in the first returned list,
    #     # and taking all the indices in argsorted_env_list with the slice (start_idx{env k} : start_idx_{env k+1}).
    #     unique_envs_in_batch, indices_of_env_transitions = np.unique(
    #         env_batch_info[argsort_env_info].numpy(), return_index=True
    #     )
    #     for i, env in enumerate(unique_envs_in_batch[:-1]):
    #         batch_indices_by_env[env] = argsort_env_info[
    #             indices_of_env_transitions[i] : indices_of_env_transitions[i + 1]
    #         ]
    #
    #     # the last env entry are all the indices from the last unique entry pointer to the end.
    #     batch_indices_by_env[unique_envs_in_batch[-1]] = argsort_env_info[
    #         indices_of_env_transitions[-1] :
    #     ]
    #     return batch_indices_by_env

    def _pooled_gaussian_loss(self, gauss_sample: Tensor):
        """
        Compute the environmental similarity loss of the gauss mapping for all environment data together.
        This is done to enforce that the cinn maps all environments combined to gauss. This seems redundant
        to the cINN maximum likelihood loss which enforces the same, yet this extra loss has proven to be
        necessary for stable training.
        """
        loss_per_mask = torch.zeros(1, device=self.device)
        nr_masks = gauss_sample.size(0)
        true_gauss_sample = torch.randn(gauss_sample.size(1), 1, device=self.device)

        for mask_nr in range(nr_masks):
            loss_per_mask += wasserstein(gauss_sample[mask_nr], true_gauss_sample)
        loss_per_mask /= nr_masks
        return loss_per_mask

    def _cinn_maxlikelihood_loss(
        self, masked_batch: Tensor, target_batch: Tensor, nr_masks: int
    ):
        # compute the INN loss per mask, then average loss over the number of masks.
        # This is the monte carlo approximation of the loss via multiple mask samples.
        gauss_sample = self.cinn(
            x=target_batch.repeat(nr_masks, 1),
            condition=masked_batch.view(-1, self.p),
            rev=False,
        ).view(nr_masks, -1, 1)
        inn_loss = torch.mean(
            torch.mean(
                gauss_sample ** 2 / 2
                - self.cinn.log_jacobian_cache.view(gauss_sample.shape),
                dim=1,
            ),
            dim=0,
        )
        return inn_loss, gauss_sample

    @staticmethod
    def normalize(x: Tensor):
        return (x - torch.mean(x)) / torch.std(x)

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
        for env, env_indices in environments_map.items():
            target_env = target[env_indices].view(-1, 1)
            masked_env = self.masker_net(obs[env_indices], mask).view(-1, self.p)

            env_gauss_sample = self.cinn(
                x=target_env, condition=masked_env, rev=False,
            ).view(nr_masks, -1, 1)
            fig.add_trace(
                go.Histogram(
                    x=env_gauss_sample.view(-1).cpu().detach().numpy(),
                    name=f"E = {env}",
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

    def _results_statistics(
        self,
        results_losses,
        results_masks,
        results_filename,
        results_folder,
        save_results,
    ):
        full_results = {
            var: []
            for var in sorted(
                self.get_parent_candidates(),
                key=lambda x: int(x.split("_")[-1]) if "_" in x else x,
            )
        }
        best_invariance_result = -1
        lowest_invariance_loss = float("infinity")
        for i, (result_mask, result_loss) in enumerate(
            zip(results_masks, results_losses)
        ):
            for var, mask in result_mask.items():
                full_results[var].append(mask)
                if result_loss["invariance"][-1] < lowest_invariance_loss:
                    lowest_invariance_loss = result_loss["invariance"][-1]
                    best_invariance_result = i
        statistics = dict()
        for var, values in full_results.items():
            stats_dict = dict()
            for func, args, kwargs in zip(
                (
                    np.mean,
                    np.min,
                    np.max,
                    np.var,
                    np.quantile,
                    np.quantile,
                    np.quantile,
                ),
                [values] * 7,
                (None, None, None, None, {"q": 0.25}, {"q": 0.5}, {"q": 0.75}),
            ):
                func_str = str(
                    re.search(r"(?<=function\s)[a-zA-z\d]+(?=\s)", str(func)).group()
                )
                if kwargs is not None:
                    func_str += ", " + ", ".join(
                        [f"{k}={v}" for k, v in kwargs.items()]
                    )
                    stats_dict[func_str] = func(args, **kwargs).round(3)
                else:
                    stats_dict[func_str] = func(args).round(3)
            statistics[var] = stats_dict
        print("\nLearning outcomes:")
        for var, stat_dict in statistics.items():
            print(var)
            for func_str, value in stat_dict.items():
                print(
                    f"\t{func_str}:", value,
                )
            print(f"\tresults:", full_results[var])
        print("Best individual run by invariance loss:")
        print(f"\tRun: {best_invariance_result}")
        print(f"\tLoss: {lowest_invariance_loss}")
        print(
            f"\tMask: {', '.join([f'{var}: {val}' for var, val in results_masks[best_invariance_result].items()])}"
        )
        print("")
        if save_results:
            if not os.path.isdir(results_folder):
                os.mkdir(results_folder)
            results_df = pd.DataFrame.from_dict(statistics, orient="columns")
            results_df.loc["results_array"] = {
                var: str(values) for (var, values) in full_results.items()
            }
            results_df.to_csv(f"{os.path.join(results_folder, results_filename)}.csv")

        return best_invariance_result, lowest_invariance_loss

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
            opts=dict(title=title, ytickmin=ytickmin, ytickmax=ytickmax),
        )
