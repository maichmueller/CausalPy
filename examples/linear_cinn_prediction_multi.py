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
from causalpy.causal_prediction.interventional import ICPredictor, MultiAgnosticPredictor
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import wasserstein_distance
from plotly import graph_objs as go
from build_scm_funcs import *
from linear_regression_eval import *


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
            # (build_scm_large, "Y"),
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
            markovblanket_interv_only=False,
            target_var=target_var,
            sample_size=3000,
            seed=seed,
        )
        target_parents_indices = np.array(
            [possible_parents.index(par) for par in target_parents]
        )
        nr_envs = np.unique(environments).max() + 1
        nr_repetitions = 1
        results = []
        epochs = 300
        use_visdom = 0
        # for _ in range(nr_repetitions)
        ap = MultiAgnosticPredictor(
            epochs=epochs, batch_size=10000, visualize_with_visdom=bool(use_visdom)
        )
        results.append(ap.infer(complete_data, environments, target_var,))
        print(results[-1])

        evaluate(
            complete_data,
            ap,
            environments,
            ground_truth_assignment=scm[target_var][1][scm.function_key],
            x_vars=target_parents,
            targ_var=target_var,
        )
