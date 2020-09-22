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
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed


import torch as th
import math
from torch.utils.data import Sampler
import warnings

warnings.filterwarnings("ignore")


def run_scenario(
    Predictor, sample_size, nr_runs, epochs, scenario, step, device=None, **kwargs,
):
    seed = 0
    np.random.seed(seed)
    # print(coeffs)
    scm = study_scm(seed=seed)
    # standard_sample = scm.sample(sample_size)

    # plt.show()
    (
        data,
        environments,
        scm,
        possible_parents,
        target_parents,
    ) = generate_data_from_scm(scm, target_var="Y", sample_size=sample_size, seed=seed)

    data2 = data.copy()
    data2["env"] = environments
    if not os.path.isdir("./data"):
        os.mkdir("./data")
    data2.to_csv(
        f"./data/{modelclass}_{test_name}_scenario_{scenario}_step_{step}.csv",
        index=False,
    )

    nr_envs = np.unique(environments).max() + 1
    use_visdom = 0

    ap = Predictor(
        epochs=epochs,
        batch_size=min(data.shape[0], 5000),
        visualize_with_visdom=bool(use_visdom),
        masker_network_params=dict(monte_carlo_sample_size=1),
        device=device,
    )

    results_mask, results_loss, res_str = ap.infer(
        data,
        environments,
        "Y",
        nr_runs=nr_runs,
        normalize=True,
        save_results=True,
        results_filename=f"{modelclass}_{test_name}_scenario-{scenario}_step-{step+1}_ss_{sample_size}",
        **kwargs,
    )

    s = f"{res_str}\n"
    return {
        "res_str": s,
        "params": params,
        "sample_size": sample_size,
        "step": step,
        "scenario": scenario,
        "nr_runs": nr_runs,
        "epochs": epochs,
        "scm": scm,
    }


if __name__ == "__main__":

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    # dev = "cpu"
    np.random.seed(seed)

    sample_sizes = [2 ** (i + 1) for i in range(13)]

    ###################
    # Data Generation #
    ###################
    pref = "single"
    for ss in sample_sizes:
        for i, (scm_generator, target_var, fname) in enumerate(
            [
                # (build_scm_minimal, "Y", f"{pref}_min"),
                # (build_scm_minimal2, "Y", f"{pref}_min2"),
                (build_scm_basic, "Y", f"{pref}_basic"),
                # (build_scm_basicst, "Y", f"{pref}_basicst"),
                # (build_scm_basic_discrete, "Y", f"{pref}_basic_disc"),
                # (build_scm_exponential, "Y", f"{pref}_exp"),
                # (build_scm_medium, "Y", f"{pref}_medium"),
                # (build_scm_large, "Y", f"{pref}_large"),
                # (build_scm_massive, "Y", f"{pref}_massive"),
                # (study_scm, "Y", f"{pref}_study9"),
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
            fname += f"_{ss}"
            target_parents_indices = np.array(
                [possible_parents.index(par) for par in target_parents]
            )
            nr_envs = np.unique(environments).max() + 1

            nr_runs = 20

            epochs = 1500
            use_visdom = 0

            lock = man.Lock()
            with ProcessPoolExecutor(max_workers=nr_work) as executor:
                futures = list(
                    (
                        executor.submit(
                            run_scenario,
                            nr_epochs=epochs,
                            nr_runs=nr_runs,
                            epochs=epochs,
                            step=step,
                            sample_size=ss,
                            LOCK=lock,
                            device=dev,
                        )
                        for i, ss in enumerate(sample_sizes)
                    )
                )
                for future in as_completed(futures):
                    if isinstance(future.exception(), Exception):
                        raise future.exception()
            ap = AgnosticPredictor(
                epochs=epochs,
                batch_size=5000,
                visualize_with_visdom=bool(use_visdom),
                device=dev,
                masker_network_params=dict(monte_carlo_sample_size=1),
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
