import itertools
import os
from functools import partial
from typing import Union, Collection, Optional

import visdom
import torch
from torch.nn import Parameter, Module
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import math

from tqdm.auto import tqdm

from causalpy import SCM, LinearAssignment, NoiseGenerator, DiscreteNoise
from causalpy.neural_networks import cINN, L0InputGate
import pandas as pd
import numpy as np
from examples.simulation_linear import simulate
from causalpy.causal_prediction.interventional import ICPredictor
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import wasserstein_distance
from plotly import graph_objs as go
from build_scm_funcs import *


import torch as th
import math
from torch.utils.data import Sampler


class StratifiedSampler(Sampler):
    """Stratified Sampling
    Provides equal representation of target classes in each batch
    """

    def __init__(self, data_source, class_vector, batch_size):
        """
        Arguments
        ---------
        class_vector : torch tensor
            a vector of class labels
        batch_size : integer
            batch_size
        """
        super().__init__(data_source=data_source)
        self.n_splits = int(class_vector.size(0) / batch_size)
        self.class_vector = class_vector

    def gen_sample_array(self):
        s = StratifiedShuffleSplit(n_splits=self.n_splits, test_size=0.5)
        X = th.randn(self.class_vector.size(0), 2).numpy()
        y = self.class_vector.numpy()
        s.get_n_splits(X, y)

        train_index, test_index = next(s.split(X, y))
        return np.hstack([train_index, test_index])

    def __iter__(self):
        return iter(self.gen_sample_array())

    def __len__(self):
        return len(self.class_vector)


def s(x):
    return x.squeeze().cpu()


def std(x):
    return 1 / math.sqrt(2 * math.pi) * torch.exp(-(x ** 2) / 2)


def subnet_fc(c_in, c_out):
    return torch.nn.Sequential(
        torch.nn.Linear(c_in, 100), torch.nn.ReLU(), torch.nn.Linear(100, c_out)
    )


def rbf(X: Tensor, Y: Optional[Tensor] = None, sigma: Optional[float] = None):
    # for computing the general 2-pairwise norm ||x_i - y_j||_2 ^ 2 for each
    # row i and j of the matrices X and Y the numpy code looks like the following:
    #   XY = X @ Y.transpose()
    #   XX_d = np.ones(XY.shape) * np.diag(X @ X.transpose())[:, np.newaxis]  # row wise mult of diagonal with mat
    #   YY_d = np.ones(XY.shape) * np.diag(Y @ Y.transpose())[np.newaxis, :]  # col wise mult of diagonal with mat
    #   pairwise_norm = XX_d + YY_d - 2 * XY
    if Y is not None:
        if X.dim() == 1:
            X.unsqueeze_(1)
        if Y.dim() == 1:
            Y.unsqueeze_(1)
        # adapted for torch:
        XY = X @ Y.t()
        XY_ones = torch.ones_like(XY)
        XX_d = XY_ones * torch.diagonal(X @ X.t()).unsqueeze(1)
        YY_d = XY_ones * torch.diagonal(Y @ Y.t()).unsqueeze(0)
        pairwise_norm = XX_d + YY_d - 2 * XY
    else:
        if X.dim() == 1:
            X.unsqueeze_(1)
        # one can save some time by not recomputing the same values in some steps
        XX = X @ X.t()
        XX_ones = torch.ones_like(XX)
        XX_diagonal = torch.diagonal(XX)
        XX_row_diag = XX_ones * XX_diagonal.unsqueeze(1)
        XX_col_diag = XX_ones * XX_diagonal.unsqueeze(0)
        pairwise_norm = XX_row_diag + XX_col_diag - 2 * XX

    if sigma is None:
        try:
            mdist = torch.median(pairwise_norm[pairwise_norm != 0])
            sigma = torch.sqrt(mdist)
        except RuntimeError:
            sigma = 1.0

    gaussian_rbf = torch.exp(pairwise_norm * (-0.5 / (sigma ** 2)))
    return gaussian_rbf


def centering(K: Tensor, device):
    n = K.shape[0]
    unit = torch.ones(n, n).to(device)
    I = torch.eye(n, device=device)
    Q = I - unit / n
    return torch.mm(torch.mm(Q, K), Q)


def normalize_jointly(x, y):
    xy = torch.cat((x, y), 0).detach()
    sd = torch.sqrt(xy.var(0))
    mean = xy.mean(0)
    x = (x - mean) / sd
    y = (y - mean) / sd
    return x, y


def mmd_multiscale(x, y, normalize_j=True):
    """ MMD with rationale kernel"""
    # Normalize Inputs Jointly
    if normalize_j:
        x, y = normalize_jointly(x, y)

    xx, yy, zz = x @ x.t(), y @ y.t(), x @ y.t()

    rx = xx.diag().unsqueeze(0).expand_as(xx)
    ry = yy.diag().unsqueeze(0).expand_as(yy)

    dxx = rx.t() + rx - 2.0 * xx
    dyy = ry.t() + ry - 2.0 * yy
    dxy = rx.t() + ry - 2.0 * zz

    XX, YY, XY = (
        torch.zeros_like(xx),
        torch.zeros_like(xx),
        torch.zeros_like(xx),
    )

    for a in [6e-2, 1e-1, 3e-1, 5e-1, 7e-1, 1, 1.2, 1.5, 1.8, 2, 2.5]:
        XX += a ** 2 * (a ** 2 + dxx) ** -1
        YY += a ** 2 * (a ** 2 + dyy) ** -1
        XY += a ** 2 * (a ** 2 + dxy) ** -1

    return torch.mean(XX + YY - 2.0 * XY)


def wasserstein(x, y):
    # x, y = normalize_jointly(x, y)
    sort_sq_diff = (torch.sort(x, dim=0)[0] - torch.sort(y, dim=0)[0]).pow(2)
    std_prod = torch.std(x) * torch.std(y)
    return torch.mean(sort_sq_diff) / std_prod


def generate_data_from_scm(
    scm,
    target_var=None,
    markovblanket_interv_only=True,
    countify=False,
    sample_size=100,
    seed=None,
):
    # scm = simulate(nr_genes, 2, seed=seed)
    rng = np.random.default_rng(seed + 10)
    # scm = build_scm_minimal(seed)
    variables = sorted(scm.get_variables())
    if target_var is None:
        target_var = rng.choice(variables[len(variables) // 2 :])
    other_variables = sorted(scm.get_variables())
    other_variables.remove(target_var)
    target_parents = sorted(scm.graph.predecessors(target_var))
    scm.reseed(seed)
    environments = []
    sample_size_per_env = sample_size
    sample = scm.sample(sample_size_per_env)
    sample_data = [sample[variables]]
    environments += [0] * sample_size_per_env
    if markovblanket_interv_only:
        interv_variables = set(target_parents)
        for child in scm.graph.successors(target_var):
            child_pars = set(scm.graph.predecessors(child))
            child_pars = child_pars.union([child])
            child_pars.remove(target_var)
            interv_variables = interv_variables.union(child_pars)
    else:
        interv_variables = other_variables

    # perform interventions on selected variables
    for parent in interv_variables:
        # interv_value = rng.choice([-1, 1]) * rng.random(1) * 10
        interv_value = 0
        scm.do_intervention([parent], [interv_value])
        print(
            f"Environment {environments[-1] + 1}: Intervention on variable {parent} for value {interv_value}."
        )
        sample_data.append(scm.sample(sample_size_per_env))
        environments += [environments[-1] + 1] * sample_size_per_env
        scm.undo_intervention()
    data = pd.concat(sample_data, sort=True)[variables]

    if countify:
        data = pd.DataFrame(
            np.random.poisson(
                torch.nn.functional.softplus(torch.as_tensor(data.to_numpy())).numpy()
            ),
            columns=data.columns,
        )
        # data += np.random.normal(0, 0.1, size=data.shape)

    # normalize
    data = (data) / data.std()

    environments = np.array(environments)
    target_data = data[target_var]
    data = data.drop(columns=[target_var])
    possible_parents = np.array(data.columns)
    envs_unique = np.unique(environments)
    env_map = {e: np.where(environments == e)[0] for e in envs_unique}

    data, target_data = list(
        map(
            lambda x: torch.as_tensor(x.to_numpy()).float().to(dev),
            (data[possible_parents], target_data),
        )
    )
    print(scm)
    print("Target Variable:", target_var)
    print("Actual Parents:", ", ".join(target_parents))
    print("Candidate Parents:", ", ".join(possible_parents))

    return (
        data,
        target_data,
        environments,
        envs_unique,
        env_map,
        scm,
        target_var,
        target_parents,
    )


class prohibit_model_grad(object):
    def __init__(self, model: torch.nn.Module):
        self.model = model

    def __enter__(self):
        for p in self.model.parameters():
            p.requires_grad = False
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for p in self.model.parameters():
            p.requires_grad = True
        return self


def visdom_plot_mask(curr_mask, possible_parents, window=None):
    if len(curr_mask) == len(possible_parents) == 1:
        curr_mask = np.append(curr_mask, 0)
        possible_parents = possible_parents + ["Dummy"]
    return viz.bar(
        X=curr_mask.reshape(1, -1),
        Y=np.array([n.replace("_", "") for n in possible_parents]).reshape(1, -1),
        win=window,
        opts=dict(title=f"Final Mask for variables"),
    )


def visdom_plot_loss(losses, loss_win, title="Loss Behaviour"):
    losses = np.array(losses)
    ytickmax = (
        np.mean(losses[losses < np.quantile(losses, 0.95)]) * 2
        if len(losses) > 1
        else None
    )
    viz.line(
        X=np.arange(len(losses)),
        Y=losses,
        win=loss_win,
        opts=dict(title=title, ytickmin=-1, ytickmax=ytickmax,),
    )


def inn_max_likelihood_loss(gauss_sample, log_grad):
    return (gauss_sample ** 2 / 2 - log_grad).mean()


def train(epochs, use_visdom):
    dim_condition = complete_data.shape[1]
    cinn = cINN(nr_blocks=3, dim=1, nr_layers=30, dim_condition=dim_condition).to(dev)
    l0_masker_net = L0InputGate(
        complete_data.shape[1], monte_carlo_sample_size=1, device=dev
    ).to(dev)
    mask_rep_size = l0_masker_net.mcs_size
    optimizer = torch.optim.Adam(
        [
            {"params": l0_masker_net.parameters(), "lr": 1e-2},
            {"params": cinn.parameters()},
        ],
        lr=0.001,
        # weight_decay=1e-5,
    )
    ####################
    #  Visdom Windows  #
    ####################

    if use_visdom:
        viz = visdom.Visdom()
        vis_wins = {"map": [], "density": []}
        loss_windows = dict()
        for win_name in ("total", "dist", "cinn", "l0"):
            loss_windows[win_name] = viz.line(
                X=np.arange(1).reshape(-1, 1),
                Y=np.array([1]).reshape(-1, 1),
                opts=dict(title=f"Loss"),
            )
        mask_win = visdom_plot_mask(
            l0_masker_net.final_layer().cpu().detach().numpy().reshape(1, -1),
            possible_parents,
        )
        quantiles_envs = dict()
        with torch.no_grad():
            for env in envs_unique:
                t_data = target_data[env_map[env]]
                quantiles_envs[env] = [np.quantile(t_data.cpu(), 0.01)]
                quantiles_envs[env].append(np.quantile(t_data.cpu(), 0.99))
                x = (
                    torch.as_tensor(
                        np.linspace(*quantiles_envs[env], complete_data.shape[0])
                    )
                    .unsqueeze(1)
                    .to(dev)
                )
                gauss_sample = cinn(
                    x=x,
                    condition=torch.zeros(complete_data.shape[0], dim_condition).to(
                        dev
                    ),
                    rev=False,
                )
                vis_wins["map"].append(
                    viz.line(
                        X=s(x),
                        Y=s(gauss_sample),
                        opts=dict(title=f"Forward Map Env {env}"),
                    )
                )
                vis_wins["density"].append(
                    viz.line(
                        X=s(x),
                        Y=s(std(gauss_sample) * torch.exp(cinn.log_jacobian_cache)),
                        opts=dict(title=f"Estimated Density Env {env}"),
                    )
                )
                vis_wins["ground_truth"] = viz.histogram(
                    X=t_data,
                    opts=dict(numbins=20, title=f"Target data histogram Env {env}"),
                )
    ##################
    # Model Training #
    ##################
    use_ground_truth_mask = False
    total_losses = []
    env_distance_losses = []
    cinn_losses = []
    l0_losses = []
    nr_total_samples = complete_data.shape[0]
    hyperparams = {"l0": 0.2, "env": 5, "inn": 2}
    epoch_pbar = tqdm(range(epochs))
    batch_size = min(1000, complete_data.shape[0])
    sampler = StratifiedSampler(
        data_source=np.arange(nr_total_samples),
        class_vector=torch.as_tensor(environments),
        batch_size=batch_size,
    )
    dataloader = DataLoader(
        dataset=TensorDataset(
            torch.arange(nr_total_samples), torch.zeros(nr_total_samples)
        ),
        batch_size=batch_size,
        sampler=sampler,
    )
    for epoch in epoch_pbar:
        batch_losses = []
        batch_distr_losses = []
        batch_inn_losses = []
        l0_batch_losses = []
        for batch_indices, _ in dataloader:
            optimizer.zero_grad()

            if not use_ground_truth_mask:
                l0_mask = l0_masker_net.create_gates()
            else:
                l0_mask = torch.cat(
                    [
                        torch.ones(mask_rep_size, 1, 2, device=dev),
                        # torch.zeros(mask_rep_size, 1, 1, device=dev),
                    ],
                    dim=2,
                )
            batch_indices = torch.sort(batch_indices)[0]

            masked_batch_data = l0_masker_net(
                complete_data[batch_indices], l0_mask
            ).view(mask_rep_size, -1, dim_condition)

            samples = []

            # we need to repeat the input by the number of monte carlo samples we generate in the
            # l0 masker network, as this is the number of repeated, different maskings of the data.
            target_batch_data = (
                target_data[batch_indices]
                .repeat(mask_rep_size, 1)
                .view(mask_rep_size, -1, 1)
            )
            # compute the INN loss per mask, then average over the loss outcome with respect to the number of masks.
            inn_loss = torch.zeros(1, device=dev)
            samples_per_mask = []
            for target_batch_data_for_mask, masked_condition_data in zip(
                target_batch_data, masked_batch_data
            ):
                samples_per_mask_per_env = []
                # for env, env_indices in env_map.items():
                gauss_sample = cinn(
                    x=target_batch_data_for_mask,
                    condition=masked_condition_data,
                    rev=False,
                )
                log_grad = cinn.log_jacobian_cache
                inn_loss += inn_max_likelihood_loss(gauss_sample, log_grad)
                # samples_per_mask_per_env.append(gauss_sample)
                samples_per_mask.append(gauss_sample)
            inn_loss /= mask_rep_size

            # all environmental distributions of Y should become equal, thus use
            # MMD to measure the distance of their distributions.
            env_distribution_distance = 0
            # generated_samples = [
            #     torch.cat([samples_per_mask[mask][i] for i in range(nr_envs)])
            #     for mask in range(mask_rep_size)
            # ]
            generated_samples = samples_per_mask
            for gen_samples_mask_i in generated_samples:
                loss_per_mask = 0
                gauss_sample = torch.randn(gen_samples_mask_i.size(0), 1, device=dev)
                loss_per_mask += wasserstein(gauss_sample, gen_samples_mask_i)

                env_distribution_distance += loss_per_mask
            env_distribution_distance = env_distribution_distance / mask_rep_size

            batch_distr_losses.append(env_distribution_distance.item())
            batch_inn_losses.append(inn_loss.item())

            l0_loss = l0_masker_net.complexity_loss()
            l0_batch_losses.append(l0_loss.item())

            batch_loss = (
                hyperparams["inn"] * inn_loss
                # + hyperparams["env"] * env_distribution_distance
                + hyperparams["l0"] * l0_loss
            )
            # batch_distr_losses.append(env_distribution_distance.item())
            batch_inn_losses.append(batch_loss.item())

            # batch_loss += env_distribution_distance

            if use_visdom:
                visdom_plot_mask(
                    l0_masker_net.final_layer().cpu().detach().numpy(),
                    possible_parents,
                    mask_win,
                )
            # print(loss)
            batch_losses.append(batch_loss.item())
            batch_loss.backward()
            optimizer.step()
        # Add the L0 regularization
        env_distance_losses.append(np.mean(batch_distr_losses))
        cinn_losses.append(np.mean(batch_inn_losses))
        total_losses.append(np.mean(batch_losses))
        l0_losses.append(np.mean(l0_batch_losses))
        if use_visdom:
            visdom_plot_mask(
                l0_masker_net.final_layer().cpu().detach().numpy(),
                possible_parents,
                mask_win,
            )
            # hist_data = [
            #     gen_samples_mask_i.detach().cpu().numpy() for gen_samples_mask_i in generated_samples
            # ]
            # group_labels = ["Mask " + str(i) for i in range(mask_rep_size)]
            #
            # fig = ff.create_distplot(
            #     hist_data, group_labels, bin_size=0.5, curve_type="normal"
            # )  # override default 'kde')
            fig = go.Figure()
            for i, gen_samples_mask_i in enumerate(generated_samples):
                fig.add_trace(
                    go.Histogram(
                        x=gen_samples_mask_i.view(-1).cpu().detach().numpy(),
                        name=f"Mask {i}",
                        histnorm="probability density",
                    )
                )

            gauss_density = lambda inp: np.exp(-(inp ** 2) / 2) / 2
            xr = np.linspace(-5, 5, 100)
            fig.add_trace(
                go.Scatter(
                    x=xr,
                    y=gauss_density(xr),
                    name="Gaussian density",
                    line_shape="spline",
                )
            )
            # Overlay both histograms
            fig.update_layout(barmode="overlay")
            # Reduce opacity to see both histograms
            fig.update_traces(opacity=0.5)
            fig.update_layout(
                title=f"Gaussian shape approximation by the environments.",
            )
            viz.plotlyplot(fig, win="gaussian")

            visdom_plot_loss(
                total_losses, loss_windows["total"], "Total Loss Behaviour"
            )
            visdom_plot_loss(
                env_distance_losses,
                loss_windows["dist"],
                "Distributional Distance Loss Behaviour",
            )
            visdom_plot_loss(cinn_losses, loss_windows["cinn"], "CINN Loss Behaviour")
            visdom_plot_loss(l0_losses, loss_windows["l0"], "L0 Loss Behaviour")
            if not use_ground_truth_mask:
                mask = l0_masker_net.final_layer().detach().flatten()
            else:
                mask = l0_mask

            if epoch % 1 == 0:
                with torch.no_grad():
                    for env, env_indices in env_map.items():
                        # plot the distribution of the environment
                        x_range = (
                            torch.as_tensor(
                                np.linspace(
                                    *quantiles_envs[env],
                                    target_data[env_indices].shape[0],
                                )
                            )
                            .unsqueeze(1)
                            .to(dev)
                        )
                        gauss_sample = cinn(
                            x=torch.randn(env_indices.size, 1, device=dev),
                            condition=(complete_data[env_indices] * mask).view(
                                -1, mask.shape[-1]
                            ),
                            rev=True,
                        )
                        viz.line(
                            X=s(x_range),
                            Y=s(
                                cinn(
                                    x=x_range,
                                    condition=(complete_data[env_indices] * mask).view(
                                        -1, mask.shape[-1]
                                    ),
                                    rev=True,
                                )
                            ),
                            win=vis_wins["map"][env],
                            opts=dict(title=f"Forward Map Env {env}"),
                        )
                        viz.histogram(
                            X=gauss_sample,
                            win=vis_wins["density"][env],
                            opts=dict(title=f"Estimated Density Env {env}"),
                        )
        # epoch_pbar.set_description(
        #     str(
        #         sorted(
        #             {
        #                 possible_parents[idx]: round(
        #                     mask.cpu().numpy().flatten()[idx], 2
        #                 )
        #                 for idx in range(len(possible_parents))
        #             }.items(),
        #             key=lambda x: x[0],
        #         )
        #     )
        # )
    if not use_ground_truth_mask:
        mask = l0_masker_net.final_layer().detach().flatten()
    else:
        mask = l0_mask
    results = {
        var: mask
        for (var, mask) in sorted(
            {
                possible_parents[idx]: round(mask.cpu().numpy().flatten()[idx], 2)
                for idx in range(len(possible_parents))
            }.items(),
            key=lambda x: x[0],
        )
    }
    return results


if __name__ == "__main__":

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    np.random.seed(seed)
    viz = visdom.Visdom()
    ###################
    # Data Generation #
    ###################

    for i, (scm_generator, target_var) in enumerate(
        [
            # (build_scm_minimal, "Y"),
            # (build_scm_basic, "Y"),
            # (build_scm_basic_discrete, "Y"),
            (build_scm_medium, "Y"),
            (build_scm_large, "Y"),
            # (partial(simulate, nr_genes=15), "G_12"),
            # (partial(simulate, nr_genes=20), "G_16"),
            # (partial(simulate, nr_genes=25), "G_21"),
            # (partial(simulate, nr_genes=30), "G_29"),
        ]
    ):
        (
            complete_data,
            target_data,
            environments,
            envs_unique,
            env_map,
            scm,
            target_var,
            target_parents,
        ) = generate_data_from_scm(
            scm=scm_generator(seed=seed),
            target_var=target_var,
            sample_size=3000,
            seed=seed,
        )
        possible_parents = sorted(scm.get_variables())
        possible_parents.remove(target_var)
        target_parents_indices = np.array(
            [possible_parents.index(par) for par in target_parents]
        )
        nr_envs = envs_unique.max() + 1
        nr_repetitions = 20
        results = []
        epochs = 100
        use_visdom = True

        for _ in range(nr_repetitions):
            results.append(train(epochs=epochs, use_visdom=use_visdom))

        full_results = {var: [] for var in results[0].keys()}
        for res in results:
            for var, mask in res.items():
                full_results[var].append(mask)

        full_results = {
            var: val
            for var, val in sorted(
                full_results.items(),
                key=lambda x: int(x[0].split("_")[-1]) if "_" in x[0] else x[0],
            )
        }
        statistics = {
            var: {
                f'{func}{", " + ", ".join([f"{k}={v}" for k, v in kwargs.items()]) if kwargs else ""}': eval(
                    f"np.{func}(args, {', '.join([f'{kwarg}={val}' for kwarg, val in kwargs.items()])}).round(3)"
                )
                for func, args, kwargs in zip(
                    ("mean", "min", "max", "var", "quantile", "quantile", "quantile"),
                    [values] * 7,
                    ({}, {}, {}, {}, {"q": 0.25}, {"q": 0.5}, {"q": 0.75}),
                )
            }
            for var, values in full_results.items()
        }
        print("\nLearning outcomes:")
        for var, stat_dict in statistics.items():
            print(var)
            for func_str, value in stat_dict.items():
                print(
                    f"\t{func_str}:", value,
                )
            print(f"\tresults:", full_results[var])
        print("")
        res_folder = "./results"
        if not os.path.isdir(res_folder):
            os.mkdir(res_folder)
        results_df = pd.DataFrame.from_dict(statistics, orient="columns")
        results_df.loc["results_array"] = {
            var: str(values) for (var, values) in full_results.items()
        }
        results_df.to_csv(os.path.join(res_folder, f"./results_iter_{i}.csv"))

        # print(l0_masker_net.final_layer())
