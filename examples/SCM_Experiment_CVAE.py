import itertools
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
from causalpy.neural_networks import cINN, L0InputGate, CVAE
import pandas as pd
import numpy as np
from examples.simulation_linear import simulate
from causalpy.causal_prediction.interventional import ICPredictor
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import wasserstein_distance
from plotly import graph_objs as go


import torch as th
import math
from torch.utils.data import Sampler
from geomloss import SamplesLoss


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
    x, y = normalize_jointly(x, y)
    sort_sq_diff = (torch.sort(x, dim=0)[0] - torch.sort(y, dim=0)[0]).pow(2)
    std_prod = torch.std(x) * torch.std(y)
    return torch.mean(sort_sq_diff) / std_prod


def build_scm_minimal(seed=0):
    cn = SCM(
        assignment_map={
            "X_0": (
                [],
                LinearAssignment(1),
                NoiseGenerator("binomial", n=1, p=0.5, seed=seed),
            ),
            "Y": (
                ["X_0"],
                LinearAssignment(1, 0, 1),
                NoiseGenerator("standard_normal", seed=seed + 4),
            ),
        },
        variable_tex_names={"X_0": "$X_0$"},
    )
    return cn


def build_scm_basic_discrete(seed=0):
    cn = SCM(
        assignment_map={
            "X_0": ([], LinearAssignment(1), DiscreteNoise([0, 1, 2], seed=seed),),
            "X_1": ([], LinearAssignment(1), DiscreteNoise([0, 1, 2], seed=seed + 1),),
            "X_2": ([], LinearAssignment(1), DiscreteNoise([0, 1, 2], seed=seed + 1),),
            "Y": (
                ["X_0", "X_1"],
                LinearAssignment(1, 0.67, 1, -1),
                NoiseGenerator("standard_normal", seed=seed + 4),
            ),
        },
        variable_tex_names={"X_0": "$X_0$", "X_1": "$X_1$", "X_2": "$X_2$"},
    )
    return cn


def build_scm_basic(seed=0):
    cn = SCM(
        assignment_map={
            "X_0": (
                [],
                LinearAssignment(1),
                NoiseGenerator("standard_normal", seed=seed),
            ),
            "X_1": (
                [],
                LinearAssignment(1),
                NoiseGenerator("standard_normal", seed=seed + 1),
            ),
            "Y": (
                ["X_0", "X_1"],
                LinearAssignment(1, 0.67, 1, -1),
                NoiseGenerator("standard_normal", seed=seed + 4),
            ),
        },
        variable_tex_names={"X_0": "$X_0$", "X_1": "$X_1$",},
    )
    return cn


def build_scm_medium(seed=0):
    cn = SCM(
        assignment_map={
            "X_0": (
                [],
                LinearAssignment(1),
                NoiseGenerator("standard_normal", seed=seed),
            ),
            "X_1": (
                ["X_0"],
                LinearAssignment(1, 1, 1),
                NoiseGenerator("standard_normal", seed=seed + 1),
            ),
            "X_2": (
                ["X_0", "X_1"],
                LinearAssignment(1, 1, 0.8, -1.2),
                NoiseGenerator("standard_normal", seed=seed + 2),
            ),
            "X_3": (
                ["X_1", "X_2"],
                LinearAssignment(1, 0, 0.3, 0.4),
                NoiseGenerator("standard_normal", seed=seed + 3),
            ),
            "Y": (
                ["X_3", "X_0"],
                LinearAssignment(1, 0.67, 1, -1),
                NoiseGenerator("standard_normal", seed=seed + 4),
            ),
            "X_4": (
                ["Y"],
                LinearAssignment(1, 1.2, -0.7),
                NoiseGenerator("standard_normal", seed=seed + 5),
            ),
            "X_5": (
                ["X_3", "Y"],
                LinearAssignment(1, 0.5, -0.7, 0.4),
                NoiseGenerator("standard_normal", seed=seed + 6),
            ),
        },
        variable_tex_names={
            "X_0": "$X_0$",
            "X_1": "$X_1$",
            "X_2": "$X_2$",
            "X_3": "$X_3$",
            "X_4": "$X_4$",
            "X_5": "$X_5$",
        },
    )
    return cn


def generate_data_from_scm(target_var=None, countify=False, sample_size=100, seed=None):
    # scm = simulate(nr_genes, 2, seed=seed)
    rng = np.random.default_rng(seed)
    scm = build_scm_basic_discrete(seed)
    # scm = build_scm_minimal(seed)
    variables = sorted(scm.get_variables())
    if target_var is None:
        target_var = rng.choice(variables[len(variables) // 2 :])
    other_variables = sorted(scm.get_variables())
    other_variables.remove(target_var)
    target_parents = list(scm.graph.predecessors(target_var))
    scm.reseed(seed)
    environments = []
    sample_size_per_env = sample_size
    sample = scm.sample(sample_size_per_env)
    sample_data = [sample[variables]]
    environments += [0] * sample_size_per_env
    for parent in other_variables:
        interv_value = rng.choice([-1, 1]) * rng.random(1) * 10
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
                torch.nn.Softplus(beta=1)(torch.as_tensor(data.to_numpy())).numpy()
            ),
            columns=data.columns,
        )
        data += np.random.normal(0, 0.1, size=data.shape)

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

    return data, target_data, environments, envs_unique, env_map, scm, target_var


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


def env_distributions_dist_loss(target_samples_env, nr_cond_dist_plots=0):
    """
    The following recreates np.apply_along_axis - torch doesn't provide it!

    The target Y | X_condition samples are stored in a tensor of semantic shape

      ( condition, l0_monte_carlo_repetition, sample, 1)

    Through torch.unbind over dim 1, we can get exactly those samples pertaining to the
    same condition and compare them with the samples of the same condition of another
    environment.
    """
    # here we choose the visdom window names and nr_cond_dist_plots many conditions,
    # of which we visualize their distribution for every environment. This will give
    # us some visual aid in determining the distributional distances of the environments
    nr_conditions = target_samples_env[0].size(2)
    select_conds_plot = np.random.default_rng(seed).choice(
        list(range(nr_conditions)), replace=False, size=nr_cond_dist_plots
    )

    visdom_win_names = {cond: f"cond_{cond}" for cond in select_conds_plot}
    # unbind the conditions as the dependent dimension and make it the dominant one for iterating
    nr_envs = len(target_samples_env)
    # the list structure will be:
    # condition: tuple (len == nr_samples) of (mask, sample, data_dim) shaped tensor

    unwrapped_target_samples = []
    for samples in target_samples_env:
        unwrapped_target_samples.append(torch.unbind(samples, dim=2))

    # now first plot the conditional distributions
    for condition, env_conditionals in zip(
        select_conds_plot,
        (
            tuple(unwrapped_target_samples[env][cond] for env in range(nr_envs))
            for cond in select_conds_plot
        ),
    ):
        # plot the conditional distribution of this specific condition, to see the visual differences
        fig = go.Figure()
        for i, conditional_dist_data in enumerate(env_conditionals):
            fig.add_trace(
                go.Histogram(
                    x=torch.mean(conditional_dist_data, dim=1)
                    .view(-1)
                    .cpu()
                    .detach()
                    .numpy(),
                    name=f"Env {i}",
                )
            )

        # Overlay both histograms
        fig.update_layout(barmode="overlay")
        # Reduce opacity to see both histograms
        fig.update_traces(opacity=0.5)
        fig.update_layout(
            title=f"C{condition}: {[round(v, 1) for v in masked_batch_data[condition].tolist()]} Y sample",
        )
        viz.plotlyplot(fig, win=visdom_win_names[condition])

    # now we compute the conditional distributions
    loss = torch.zeros(1, device=dev)
    if nr_envs > 0:
        ys_ref_env = unwrapped_target_samples[0]
        for env, ys_other_env in enumerate(unwrapped_target_samples[1:]):
            loss_per_condition = torch.zeros(1, device=dev)
            for condition, (ys_c_ref, ys_c_env_i) in enumerate(
                zip(ys_ref_env, ys_other_env)
            ):
                # unbind over dim 1 (the sample dimension) to get a tuple of samples (tuple length == nr_l0_masks).
                loss_per_conditition_per_mask = []
                for ys_c_env_ref_mask_j, ys_c_env_i_mask_j in zip(
                    *map(partial(torch.unbind, dim=1), (ys_c_ref, ys_c_env_i))
                ):
                    loss_per_conditition_per_mask.append(
                        wasserstein(ys_c_env_ref_mask_j, ys_c_env_i_mask_j)
                    )
                loss_per_condition = (
                    loss_per_condition
                    + torch.stack(loss_per_conditition_per_mask).mean()
                )
            # print("\nEnv 0 vs", i+1, "loss", loss_per_condition.item())
            loss = loss + 1 * loss_per_condition
    return loss


def visdom_plot_loss(losses, loss_win, title="Loss Behaviour"):
    losses = np.array(losses)
    ytickmax = (
        np.mean(losses[losses < np.quantile(losses, 0.95)]) * 2
        if len(losses) > 1
        else None
    )
    viz.line(
        X=np.arange(epoch + 1),
        Y=losses,
        win=loss_win,
        opts=dict(title=title, ytickmin=-1, ytickmax=ytickmax,),
    )


def inn_max_likelihood_loss(gauss_sample, log_grad):
    return (gauss_sample ** 2 / 2 - log_grad).mean()


if __name__ == "__main__":

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    viz = visdom.Visdom()
    seed = 0
    np.random.seed(seed)

    ###################
    # Data Generation #
    ###################

    (
        complete_data,
        target_data,
        environments,
        envs_unique,
        env_map,
        scm,
        target_var,
    ) = generate_data_from_scm(target_var="Y", sample_size=250, seed=seed)
    target_parents = sorted(scm.graph.predecessors(target_var))
    possible_parents = sorted(scm.get_variables())
    possible_parents.remove(target_var)
    target_parents_indices = np.array(
        [possible_parents.index(par) for par in target_parents]
    )

    ####################
    # Model Generation #
    ####################

    dim_condition = complete_data.shape[1]
    cINN_list = [
        cINN(nr_blocks=3, dim=1, nr_layers=30, dim_condition=dim_condition).to(dev)
        for _ in envs_unique
    ]

    l0_masker_net = L0InputGate(
        complete_data.shape[1], monte_carlo_sample_size=1, device=dev
    ).to(dev)
    mask_rep_size = l0_masker_net.mcs_size

    optimizer = torch.optim.Adam(
        [
            {"params": l0_masker_net.parameters(), "lr": 1e-2},
            {"params": itertools.chain(*[c_inn.parameters() for c_inn in cINN_list])},
        ],
        lr=0.001,
        # weight_decay=1e-5,
    )

    ####################
    #  Visdom Windows  #
    ####################

    vis_wins = {"map": [], "density": []}
    loss_windows = dict()
    for _, win_name in zip(range(3), ("total", "dist", "cinn")):
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
    for env, cinn in enumerate(cINN_list):
        t_data = target_data[env_map[env]]
        quantiles_envs[env] = [np.quantile(t_data.cpu(), 0.01)]
        quantiles_envs[env].append(np.quantile(t_data.cpu(), 0.99))
        x = (
            torch.as_tensor(np.linspace(*quantiles_envs[env], complete_data.shape[0]))
            .unsqueeze(1)
            .to(dev)
        )
        y_sample = cinn(
            x=x,
            condition=torch.zeros(complete_data.shape[0], dim_condition).to(dev),
            rev=False,
        )
        vis_wins["map"].append(
            viz.line(X=s(x), Y=s(y_sample), opts=dict(title=f"Forward Map Env {env}"))
        )
        vis_wins["density"].append(
            viz.line(
                X=s(x),
                Y=s(std(y_sample) * torch.exp(cinn.log_jacobian_cache)),
                opts=dict(title=f"Estimated Density Env {env}"),
            )
        )
        vis_wins["ground_truth"] = viz.histogram(
            X=t_data, opts=dict(numbins=20, title=f"Target data histogram Env {env}")
        )

    ##################
    # Model Training #
    ##################
    use_ground_truth_mask = True
    total_losses = []
    env_distance_losses = []
    cinn_losses = []

    nr_total_samples = complete_data.shape[0]
    l0_hyperparam = 0.0005
    epochs = 200
    epoch_pbar = tqdm(range(epochs))
    batch_size = 1000

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
        if not use_ground_truth_mask:
            l0_mask = l0_masker_net.create_gates()
        else:
            l0_mask = torch.cat(
                [
                    torch.ones(mask_rep_size, 1, 2, device=dev),
                    torch.zeros(mask_rep_size, 1, 1, device=dev),
                ],
                dim=2,
            )
        for batch_indices, _ in dataloader:
            optimizer.zero_grad()

            batch_indices = np.sort(batch_indices.numpy())

            masked_batch_data = l0_masker_net(complete_data[batch_indices], l0_mask)
            batch_loss = torch.zeros(1).to(dev)
            target_samples_env = []
            for env_cinn, (env, env_indices) in zip(cINN_list, env_map.items()):
                env_batch_indices = np.intersect1d(batch_indices, env_indices)
                env_batch_size = len(env_batch_indices)

                # we need to repeat the input by the number of monte carlo samples we generate in the
                # l0 masker network, as this is the number of repeated, different maskings of the data.
                xdata = (
                    target_data[env_batch_indices].unsqueeze(1).repeat(mask_rep_size, 1)
                )
                y_sample = env_cinn(
                    x=xdata,
                    condition=l0_masker_net(complete_data[env_batch_indices], l0_mask),
                    rev=False,
                )
                log_grad = env_cinn.log_jacobian_cache
                batch_loss += inn_max_likelihood_loss(y_sample, log_grad)

                # we need to repeat the input by the number of monte carlo samples we generate in the
                # l0 masker network, as this is the number of repeated, different maskings of the data.

                # So here is the catch:
                #  P(Y^i | X_PA = a) = P(Y^j | X_PA = a) is the causal parents condition.
                #  So in order to get the conditional distributions to be comparable
                #  and their deviations measurable, I need to provide the same conditions
                #  for all the samples I draw. Otherwise the theory wouldn't apply! Thus
                #  I need to give all conditions to the environmental networks, even
                #  if most of the data has never been seen in the respective environments.
                with prohibit_model_grad(env_cinn):
                    distrib_sampling_size = 400
                    target_samples = env_cinn(
                        x=torch.randn(
                            distrib_sampling_size * mask_rep_size * batch_size,
                            1,
                            device=dev,
                        ),
                        condition=masked_batch_data.repeat(distrib_sampling_size, 1),
                        rev=True,
                        retain_jacobian_cache=False,
                    ).view(distrib_sampling_size, mask_rep_size, batch_size, 1)

                    target_samples_env.append(target_samples)

            # all environmental distributions of Y should become equal, thus use
            # MMD to measure the distance of their distributions.
            env_distribution_distance = env_distributions_dist_loss(
                target_samples_env, 2
            )

            batch_distr_losses.append(env_distribution_distance.item())
            batch_inn_losses.append(batch_loss.item())

            batch_loss += env_distribution_distance

            visdom_plot_mask(
                l0_masker_net.final_layer().cpu().detach().numpy(),
                possible_parents,
                mask_win,
            )
            # print(loss)
            batch_losses.append(batch_loss.item())
            batch_loss.backward(retain_graph=True)
            optimizer.step()
        # Add the L0 regularization
        l0_loss = l0_hyperparam * l0_masker_net.complexity_loss()
        env_distance_losses.append(np.mean(batch_distr_losses))
        cinn_losses.append(np.mean(batch_inn_losses))
        total_losses.append(np.mean(batch_losses) + l0_loss.item())
        l0_loss.backward()

        visdom_plot_loss(total_losses, loss_windows["total"], "Total Loss Behaviour")
        visdom_plot_loss(
            env_distance_losses,
            loss_windows["dist"],
            "Distributional Distance Loss Behaviour",
        )
        visdom_plot_loss(cinn_losses, loss_windows["cinn"], "CINN Loss Behaviour")
        if not use_ground_truth_mask:
            mask = l0_masker_net.final_layer().detach().flatten()
        else:
            mask = l0_mask

        if epoch % 1 == 0:
            with torch.no_grad():
                for env_cinn, (env, env_indices) in zip(cINN_list, env_map.items()):
                    # plot the distribution of the environment
                    x_range = (
                        torch.as_tensor(
                            np.linspace(
                                *quantiles_envs[env], target_data[env_indices].shape[0]
                            )
                        )
                        .unsqueeze(1)
                        .to(dev)
                    )
                    y_sample = env_cinn(
                        x=torch.randn(env_indices.size * mask_rep_size, 1, device=dev),
                        condition=(complete_data[env_indices] * mask).view(
                            -1, mask.shape[-1]
                        ),
                        rev=True,
                    )
                    viz.line(
                        X=s(x_range),
                        Y=s(
                            env_cinn(
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
                        X=y_sample,
                        win=vis_wins["density"][env],
                        opts=dict(title=f"Estimated Density Env {env}"),
                    )
        epoch_pbar.set_description(
            str(
                sorted(
                    {
                        possible_parents[idx]: round(
                            mask.cpu().numpy().flatten()[idx], 2
                        )
                        for idx in range(len(possible_parents))
                    }.items(),
                    key=lambda x: x[0],
                )
            )
        )

    print(l0_masker_net.final_layer())
