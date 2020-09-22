import itertools
import os
from functools import partial
from typing import Union, Collection, Optional

# import visdom
import torch
from torch.nn import Parameter, Module
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
import math

from tqdm.auto import tqdm

from causalpy import SCM, LinearAssignment, NoiseGenerator, DiscreteNoise
from causalpy.neural_networks import CINN, L0InputGate
import pandas as pd
import numpy as np
from examples.simulation_linear import simulate
from causalpy.causal_prediction.interventional import ICPredictor, AgnosticPredictor
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import wasserstein_distance
from plotly import graph_objs as go
from build_scm_funcs import *
from study_cases import *
from linear_regression_eval import *
import matplotlib.pyplot as plt
import seaborn as sns
import torch as th
import math
from torch.utils.data import Sampler
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    # dev = "cpu"
    np.random.seed(seed)

    sample_sizes = [2 ** (i + 1) for i in range(16)]

    ###################
    # Data Generation #
    ###################
    pref = "single"
    data = {}
    for ss in sample_sizes:
        for i, (scm_generator, target_var, fname) in enumerate(
            [
                # (build_scm_minimal, "Y", f"{pref}_min"),
                # (build_scm_minimal2, "Y", f"{pref}_min2"),
                # (build_scm_basic, "Y", f"{pref}_basic"),
                # (build_scm_basicst, "Y", f"{pref}_basicst"),
                # (build_scm_basic_discrete, "Y", f"{pref}_basic_disc"),
                # (build_scm_exponential, "Y", f"{pref}_exp"),
                # (build_scm_medium, "Y", f"{pref}_medium"),
                # (build_scm_large, "Y", f"{pref}_large"),
                # (build_scm_massive, "Y", f"{pref}_massive"),
                (study_scm, "Y", f"{pref}_study9"),
                # (build_scm_polynomial, "Y", f"{pref}_polynomial"),
                # (partial(simulate, nr_genes=100), "G_12", f"{pref}_sim100"),
                # (partial(simulate, nr_genes=20), "G_16", f"{pref}_sim20"),
                # (partial(simulate, nr_genes=25), "G_21", f"{pref}_sim25"),
                # (partial(simulate, nr_genes=30), "G_29", f"{pref}_sim30"),
            ]
        ):
            (
                complete_data,
                environments,
                scm,
                possible_parents,
                target_parents,
            ) = generate_data_from_scm(
                scm=scm_generator(seed=seed),
                countify=False,
                intervention_reach="markov",
                intervention_style="do",
                target_var=target_var,
                sample_size=ss,
                seed=seed,
            )
            target_parents_indices = np.array(
                [possible_parents.index(par) for par in target_parents]
            )
            complete_data.columns = [f"${val}$" for val in complete_data.columns]
            complete_data["ss"] = ss
            data[ss] = complete_data.loc[np.where(environments == 0)]

    plt.style.use("science")
    fig, axes = plt.subplots(4, 4, figsize=(5, 4), sharey=True, sharex=True)
    vs = [f"${val}$" for val in ["X_1", "X_2", "X_3", "Y"]]
    for i, var in enumerate(vs):
        for j, ss in enumerate([16, 512, 8192, 2 ** 16]):
            ax = axes[i, j]
            sns.distplot(
                data[ss][var],
                ax=ax,
                hist=True,
                axlabel=False,
                kde_kws=dict(linewidth=1),
            )
            if j == 0:
                if var == "$Y$":
                    ylabel = r"$p_" + "{" + f"{var.replace('$', '')}" + " }" r"(y)$"
                    xlabel = "y"
                else:
                    ylabel = r"$p_" + "{" + f"{var.replace('$', '')}" + " }" r"(x)$"
                    xlabel = "x"

                ax.set_ylabel(ylabel)
            if i == 0:
                ax.set_title(f"N = {ss}")
    fig.savefig("./plots/data_sanity_check.pdf")
    plt.show()

    # for dataset in data.values():
    #     dataset.hist(bins=50, figsize=(10, 10), density=True)
    #     plt.show()
