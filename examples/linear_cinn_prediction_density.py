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
from causalpy.neural_networks import CINN, L0InputGate
import pandas as pd
import numpy as np
from examples.simulation_linear import simulate
from causalpy.causal_prediction.interventional import (
    ICPredictor,
    AgnosticPredictor,
    DensityBasedPredictor,
    MultiAgnosticPredictor,
)
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import wasserstein_distance
from plotly import graph_objs as go
from build_scm_funcs import *
from study_cases import study_scm
from linear_regression_eval import *


import torch as th
import math
from torch.utils.data import Sampler
import warnings

warnings.filterwarnings("ignore")

if __name__ == "__main__":

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    np.random.seed(seed)

    ###################
    # Data Generation #
    ###################
    pref = "single"
    for i, (scm_generator, target_var, fname) in enumerate(
        [
            # (build_scm_minimal, "Y", f"{pref}_min"),
            # (build_scm_minimal2, "Y", f"{pref}_min2"),
            # (build_scm_basic, "Y", f"{pref}_basic"),
            # (build_scm_basic_discrete, "Y", f"{pref}_basic_disc"),
            # (build_scm_exponential, "Y", f"{pref}_exp"),
            # (build_scm_medium, "Y", f"{pref}_medium"),
            # (build_scm_large, "Y", f"{pref}_large"),
            # (build_scm_massive, "Y", f"{pref}_massive"),
            (study_scm, "Y", f"{pref}_study"),
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
            intervention_style="markov",
            target_var=target_var,
            sample_size=2048,
            seed=seed,
        )
        target_parents_indices = np.array(
            [possible_parents.index(par) for par in target_parents]
        )
        nr_envs = np.unique(environments).max() + 1

        nr_runs = 20

        epochs = 1000
        use_visdom = 0

        ap = MultiAgnosticPredictor(
            epochs=epochs,
            batch_size=100000,
            visualize_with_visdom=bool(use_visdom),
            device="cuda:0",
        )
        results_mask, results_loss, res_str = ap.infer(
            complete_data,
            environments,
            target_var,
            nr_runs=nr_runs,
            normalize=True,
            save_results=True,
            results_filename=fname,
        )
        last_losses = [
            {key: results_loss[i][key][-1] for key in results_loss[0].keys()}
            for i in range(len(results_loss))
        ]
        print(res_str)

        # evaluate(
        #     complete_data,
        #     ap,
        #     environments,
        #     ground_truth_assignment=scm[target_var][1][scm.function_key],
        #     x_vars=target_parents,
        #     targ_var=target_var,
        # )
