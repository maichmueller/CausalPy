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
    MultiAgnosticPredictor,
)
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import wasserstein_distance
from plotly import graph_objs as go
from build_scm_funcs import *
from linear_regression_eval import *
from causalpy.datasets import HeinzeData
from causalpy.utils import TempFolder

import torch as th
import math
from torch.utils.data import Sampler
import logging


if __name__ == "__main__":

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    np.random.seed(seed)
    nr_heinze_configs = 100
    for i_heinze_cfg in range(nr_heinze_configs):

        with open(f"./results/heinzeRun_3rd_{i_heinze_cfg}.log", "w") as f:
            data_configurator = HeinzeData(seed=seed + i_heinze_cfg)
            data, target, envs = data_configurator.sample()
            target_parents = list(data_configurator.scm[target][0])
            possible_parents = list(data.columns)
            target_parents_indices = np.array(
                [possible_parents.index(par) for par in target_parents]
            )

            nr_envs = np.unique(envs).max() + 1

            nr_runs = 20

            epochs = 600
            use_visdom = 0

            ap = AgnosticPredictor(
                epochs=epochs,
                batch_size=10000,
                visualize_with_visdom=bool(use_visdom),
                masker_network_params=dict(monte_carlo_sample_size=1),
                device="cuda:0",
            )
            f.write("Config\n")
            interv_1 = {
                f"Intervention 1 variable {var} value": value
                for var, value in zip(*data_configurator.intervention_values[1])
            }
            interv_2 = {
                f"Intervention 1 variable {var} value": value
                for var, value in zip(*data_configurator.intervention_values[2])
            }

            for key, val in itertools.chain(
                data_configurator.config.items(),
                interv_1.items(),
                interv_2.items(),
                ap.hyperparams.items(),
                ap.masker_net_params.items(),
                ap.network_params.items(),
                ap.scheduler_params.items(),
                ap.optimizer_params.items(),
            ):
                s = f"{key}: {val}\n"
                print(s, end="")
                f.write(s)
            s = f"{data_configurator.scm}\n"
            print(s)
            f.write(s)

            results_mask, results_loss, res_str = ap.infer(
                data, envs, target, nr_runs=nr_runs, normalize=True
            )
            s = f"{res_str}\n"
            print(s)
            f.write(s)
            print(res_str)
