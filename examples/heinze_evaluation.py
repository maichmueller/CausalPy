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
from linear_regression_eval import *
from causalpy.datasets import HeinzeData

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

    data_configurator = HeinzeData()
    data, target, envs = data_configurator.sample()
    target_parents = list(data_configurator.scm[target][0])
    possible_parents = list(data.columns)
    target_parents_indices = np.array(
        [possible_parents.index(par) for par in target_parents]
    )
    print("Config")
    for key, val in data_configurator.config.items():
        print(f"{key}:", val)
    print(data_configurator.scm)
    nr_envs = np.unique(envs).max() + 1

    nr_runs = 5

    epochs = 300
    use_visdom = 0

    ap = AgnosticPredictor(
        epochs=epochs, batch_size=10000, visualize_with_visdom=bool(use_visdom)
    )
    results_mask, results_loss = ap.infer(
        data, envs, target, nr_runs=nr_runs, normalize=True
    )
    print(results_mask)

