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
from causalpy.causal_prediction.interventional import ICPredictor, AgnosticPredictor
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import wasserstein_distance
from plotly import graph_objs as go
from build_scm_funcs import *


import torch as th
import math
from torch.utils.data import Sampler


if __name__ == "__main__":

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    np.random.seed(seed)

    ###################
    # Data Generation #
    ###################

    for i, (scm_generator, target_var) in enumerate(
        [
            # (build_scm_minimal, "Y"),
            # (build_scm_basic, "Y"),
            # (build_scm_basic_discrete, "Y"),
            # (build_scm_exponential, "Y"),
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
            environments,
            scm,
            possible_parents,
            target_parents,
        ) = generate_data_from_scm(
            scm=scm_generator(seed=seed),
            target_var=target_var,
            sample_size=3000,
            seed=seed,
        )
        target_parents_indices = np.array(
            [possible_parents.index(par) for par in target_parents]
        )
        nr_envs = np.unique(environments).max() + 1
        nr_repetitions = 5
        results = []
        epochs = 100
        use_visdom = True
        for _ in range(nr_repetitions):
            results.append(
                AgnosticPredictor(
                    epochs=epochs, batch_size=10000, visualize_with_visdom=use_visdom
                ).infer(
                    complete_data, environments, target_var,
                )
            )

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
