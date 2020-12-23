import argparse
import itertools
import os
from functools import partial
from typing import Union, Collection, Optional
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import multiprocessing

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
from causalpy.causal_prediction.interventional import (
    ICPredictor,
    AgnosticPredictor,
    MultiAgnosticPredictor,
    DensityBasedPredictor,
)
from sklearn.model_selection import StratifiedShuffleSplit
from scipy.stats import wasserstein_distance
from plotly import graph_objs as go
from build_scm_funcs import *
from study_cases import *
from linear_regression_eval import *


import torch as th
import math
from torch.utils.data import Sampler
import warnings

warnings.filterwarnings("ignore")


def run_scenario(
    AP,
    batch_size,
    runs,
    params,
    epochs,
    sample_size,
    scm_generator,
    fname,
    mcs=1,
    seed=None,
    target="Y",
    device=None,
    **kwargs,
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
        target_var=target,
        sample_size=sample_size,
        seed=seed,
    )
    # target_parents_indices = np.array(
    #     [possible_parents.index(par) for par in target_parents]
    # )
    # nr_envs = np.unique(environments).max() + 1
    nr_runs = runs
    use_visdom = 0

    ap = AP(
        epochs=epochs,
        batch_size=batch_size,
        visualize_with_visdom=bool(use_visdom),
        device=device,
        hyperparams=params,
        masker_network_params=dict(monte_carlo_sample_size=mcs),
    )
    results_mask, results_loss, res_str = ap.infer(
        complete_data,
        environments,
        target,
        nr_runs=nr_runs,
        normalize=True,
        save_results=True,
        results_filename=fname,
        **kwargs,
    )
    print(res_str)


if __name__ == "__main__":

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    # dev = "cpu"
    np.random.seed(seed)
    multiprocessing.set_start_method("spawn")
    man = multiprocessing.Manager()

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", nargs="?", type=str, default="single")
    parser.add_argument("--l0", nargs="?", type=float, default=0.8)
    parser.add_argument("--inn", nargs="?", type=float, default=1)
    parser.add_argument("--inne", nargs="?", type=float, default=1)
    parser.add_argument("--res", nargs="?", type=float, default=2.5)
    parser.add_argument("--ind", nargs="?", type=float, default=50)
    parser.add_argument("--l2", nargs="?", type=float, default=0)
    parser.add_argument("--fname", nargs="?", type=str, default=None)
    parser.add_argument("--samples", nargs="?", type=int, default=1024)
    parser.add_argument("--batch_size", nargs="*", type=int, default=5000)
    parser.add_argument("--runs", nargs="?", type=int, default=20)
    parser.add_argument("--workers", nargs="?", type=int, default=1)
    parser.add_argument("--epochs", nargs="?", type=int, default=1000)
    parser.add_argument("--mcs", nargs="?", type=int, default=1)

    args = parser.parse_args()

    model = args.model
    l0 = args.l0
    l2 = args.l2
    inn = args.inn
    inne = args.inne
    res = args.res
    ind = args.ind
    name = args.fname
    samples = args.samples
    batch_sizes = args.batch_size
    runs = args.runs
    workers = args.workers
    epochs = args.epochs
    mcs = args.mcs
    if name is None:
        name = f"{model}"
    else:
        name = f"{model}_{name}"

    params = dict(
        l0=l0,
        inn=inn,
        inn_e=inne,
        residuals=res,
        independence=ind,
        l2=l2,
        samples=samples,
        runs=runs,
        mcs=mcs,
    )

    for k, v in params.items():
        name += f"_{k}-{v}"

    if model == "single":
        AP = AgnosticPredictor
    elif model == "multi":
        AP = MultiAgnosticPredictor
    elif model == "density":
        AP = DensityBasedPredictor
    else:
        raise ValueError("wrong model")
    ###################
    # Data Generation #
    ###################
    pref = ""

    for i, (scm_generator, target_var, _) in enumerate(
        [
            # (build_scm_minimal, "Y", f"{pref}_min"),
            # (build_scm_minimal2, "Y", f"{pref}_min2"),
            # (build_scm_basic, "Y", f"{pref}_basic"),
            # (build_scm_basic_discrete, "Y", f"{pref}_basic_disc"),
            (build_scm_medium, "Y", f"{pref}_medium"),
            # (build_scm_large, "Y", f"{pref}_large"),
            # (study_scm, "Y", f"_study"),
            # (partial(simulate, nr_genes=100), "G_12", f"{pref}_sim100"),
            # (partial(simulate, nr_genes=20), "G_16", f"{pref}_sim20"),
            # (partial(simulate, nr_genes=25), "G_21", f"{pref}_sim25"),
            # (partial(simulate, nr_genes=30), "G_29", f"{pref}_sim30"),
        ]
    ):
        lock = man.Lock()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = list(
                (
                    executor.submit(
                        run_scenario,
                        AP,
                        batch_size,
                        runs,
                        params,
                        scm_generator=scm_generator,
                        epochs=epochs,
                        sample_size=samples,
                        LOCK=lock,
                        mcs=mcs,
                        device=dev,
                        seed=seed,
                        target=target_var,
                        fname=name + f"_batch_size-{batch_size}",
                    )
                    for batch_size in batch_sizes
                )
            )
            for future in as_completed(futures):
                if isinstance(future.exception(), Exception):
                    raise future.exception()
        # evaluate(
        #     complete_data,
        #     ap,
        #     environments,
        #     ground_truth_assignment=scm[target_var][1][scm.function_key],
        #     x_vars=target_parents,
        #     targ_var=target_var,
        # )
