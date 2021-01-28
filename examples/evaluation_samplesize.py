import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Optional, List, Collection

from causalpy import Assignment
from causalpy.causal_prediction.interventional import (
    AgnosticPredictor,
    MultiAgnosticPredictor,
    DensityBasedPredictor,
)
from examples.study_cases import study_scm, generate_data_from_scm
import numpy as np
import torch
from plotly import graph_objs as go
from time import gmtime, strftime
import os

from tqdm import tqdm
import matplotlib.pyplot as plt


import itertools
from typing import Optional, Callable, Type, Tuple, List, Union

import torch
from causalpy.neural_networks.utils import get_jacobian
from abc import ABC, abstractmethod
import numpy as np
import argparse

from distortedCINN import *


def run_scenario(
    Predictor,
    modelclass,
    params,
    sample_size,
    nr_runs,
    epochs,
    scenario,
    step,
    device=None,
    **kwargs,
):
    seed = 0
    np.random.seed(seed)
    # print(coeffs)
    scm = study_scm(seed=seed)
    # standard_sample = scm.sample(sample_size)

    nets = fc_net(**params, device=device)
    nonlinearities = {
        "X_0": (None, nets["X_0"], None),
        "X_1": (None, nets["X_1"], None),
        "X_2": (None, nets["X_2"], None),
        "X_3": (None, nets["X_3"], None),
        "X_4": (None, nets["X_4"], None),
        "X_5": (None, nets["X_5"], None),
        "X_6": (None, nets["X_6"], None),
        "X_7": (None, nets["X_7"], None),
        "Y": (None, nets["Y"], None),
    }
    if scenario == "parents":
        nonlinearities = {key: nonlinearities[key] for key in ["X_1", "X_2", "X_3"]}
    elif scenario == "children":
        nonlinearities = {key: nonlinearities[key] for key in ["X_4", "X_6"]}
    elif scenario == "target":
        nonlinearities = {key: nonlinearities[key] for key in ["Y"]}
    elif scenario == "all":
        pass
    scm.intervention(nonlinearities)
    scm.clear_intervention_backup()  # make the current intervention the standard
    # scm.plot(alpha=1)
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


def fc_net(
    nr_layers: int, nr_hidden, strength: float, nr_blocks=1, seed=None, device=None
):
    if seed is not None:
        network_params = [
            {
                "nr_layers": nr_layers,
                "nr_blocks": nr_blocks,
                "nr_hidden": nr_hidden,
                "strength": strength,
                "seed": seed,
                "device": device,
            }
            for seed in range(seed + 1, seed + 1 + 10)
        ]
    else:
        network_params = [
            {
                "nr_layers": nr_layers,
                "nr_blocks": nr_blocks,
                "nr_hidden": nr_hidden,
                "strength": strength,
                "seed": seed,
                "device": device,
            }
            for _ in range(10)
        ]
    nets = {
        "X_0": CINNFC(dim_in=1, **(network_params)[0]).to(device),
        "X_1": CINNFC(dim_in=2, **(network_params)[1]).to(device),
        "X_2": CINNFC(dim_in=2, **(network_params)[2]).to(device),
        "X_3": CINNFC(dim_in=1, **(network_params)[3]).to(device),
        "X_4": CINNFC(dim_in=3, **(network_params)[9]).to(device),
        "X_5": CINNFC(dim_in=1, **(network_params)[5]).to(device),
        "X_6": CINNFC(dim_in=3, **(network_params)[6]).to(device),
        "X_7": CINNFC(dim_in=2, **(network_params)[7]).to(device),
        "X_8": CINNFC(dim_in=1, **(network_params)[8]).to(device),
        "Y": CINNFC(dim_in=4, **(network_params)[4]).to(device),
    }
    return nets


def init(l):
    global LOCK
    LOCK = l


test_name = "samplesizetest"
modelclass = None
if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not os.path.isdir("./log"):
        os.mkdir("./log")
    log_fname = f'{test_name}_{strftime("%Y-%m-%d_%H-%M-%S", gmtime())}'
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        filename=f"./log/{log_fname}.log",
    )  # pass explicit filename here
    logger = logging.getLogger()  # get the root logger

    multiprocessing.set_start_method("spawn")
    man = multiprocessing.Manager()
    steps = 13
    sample_size = lambda x: 2 ** (x + 1)
    nr_runs = 20
    epochs = 1500
    results = []
    scenarios = ["linear", "halflinear", "nonlinear"]

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", type=str, nargs="?", help="The model to evaluate",
    )

    parser.add_argument(
        "--workers",
        type=int,
        nargs="?",
        default=2,
        help="The number of multiprocessing workers",
    )

    parser.add_argument(
        "--start", type=int, nargs="?", default=1, help="Step from which to start",
    )

    parser.add_argument(
        "--end",
        type=int,
        nargs="?",
        default=steps + 1,
        help="Step until which to compute",
    )

    parser.add_argument(
        "--scenario",
        type=str,
        nargs="?",
        default=None,
        help="Step from which to start",
    )

    args = parser.parse_args()
    modelclass = args.model
    nr_work = args.workers
    start_step = args.start - 1
    end_step = args.end - 1
    scenario = args.scenario

    if modelclass == "single":
        PredictorClass = AgnosticPredictor
    elif modelclass == "multi":
        PredictorClass = MultiAgnosticPredictor
    elif modelclass == "density":
        PredictorClass = DensityBasedPredictor
    else:
        raise ValueError(
            f"Modelclass {modelclass} not recognized. Use one of 'single', 'multi', or 'density'"
        )

    linearity_settings = {
        key: params
        for key, params in zip(
            scenarios,
            [
                {
                    "nr_layers": 1,
                    "nr_blocks": 10,
                    "nr_hidden": 128,
                    "strength": 0,
                    "seed": 0,
                },
                {
                    "nr_layers": 1,
                    "nr_blocks": 10,
                    "nr_hidden": 128,
                    "strength": 0.5,
                    "seed": 0,
                },
                {
                    "nr_layers": 1,
                    "nr_blocks": 10,
                    "nr_hidden": 128,
                    "strength": 1,
                    "seed": 0,
                },
            ],
        )
    }

    if scenario is not None:
        scenarios = [scenario]

    # we test 4 scenarios:
    # 1. increasing nonlinearity in the parents,
    # 2. increasing nonlinearity in the children,
    # 3. increasing nonlinearity on the target,
    # 4. increasing nonlinearity on all
    for scenario in scenarios:
        params = linearity_settings[scenario]
        lock = man.Lock()
        with ProcessPoolExecutor(max_workers=nr_work) as executor:
            futures = list(
                (
                    executor.submit(
                        run_scenario,
                        PredictorClass,
                        modelclass,
                        params,
                        nr_epochs=epochs,
                        nr_runs=nr_runs,
                        scenario=scenario,
                        epochs=epochs,
                        step=step,
                        sample_size=sample_size(step),
                        LOCK=lock,
                        device=device,
                    )
                    for step in range(start_step, end_step)
                )
            )
            for future in as_completed(futures):
                if isinstance(future.exception(), Exception):
                    raise future.exception()
                lock.acquire()
                results.append(future.result())
                lock.release()

    results = sorted(
        results,
        key=lambda x: -(
            (len(scenarios) - scenarios.index(x["scenario"])) * 1000 - x["step"]
        ),
    )

    for res in results:
        for key, value in res.items():
            logger.info(f"{key}={value}")
