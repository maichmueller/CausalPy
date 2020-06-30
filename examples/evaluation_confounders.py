import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Optional, List, Collection

from causalpy import Assignment, LinearAssignment, NoiseGenerator
from causalpy.causal_prediction.interventional import (
    AgnosticPredictor,
    MultiAgnosticPredictor,
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


def run_scenario(
    Predictor, sample_size, nr_runs, epochs, scenario, step, device=None, **kwargs,
):
    seed = 0
    np.random.seed(seed)
    # print(coeffs)
    scm = study_scm(seed=seed)

    # standard_sample = scm.sample(sample_size)

    if scenario == 1:
        vars = {"X_9": ([], LinearAssignment(1, 0), NoiseGenerator("standard_normal"))}
        scm.add_variables(vars)
        scm.intervention(
            {
                "X_2": (
                    ["X_0", "X_9"],
                    LinearAssignment(1, 0, 1, 1),
                    NoiseGenerator("standard_normal"),
                ),
                "Y": (
                    ["X_1", "X_2", "X_3", "X_9"],
                    LinearAssignment(1, 0, 1, 1, 1, 1),
                    NoiseGenerator("standard_normal"),
                ),
            }
        )
        scm.clear_intervention_backup()
        scm.plot(alpha=1)
        plt.show()
    elif scenario == 2:
        vars = {"X_9": ([], LinearAssignment(1, 0), NoiseGenerator("standard_normal"))}

        scm.add_variables(vars)
        scm.intervention(
            {
                "X_2": (
                    ["X_0", "X_9"],
                    LinearAssignment(1, 0, 1, 1),
                    NoiseGenerator("standard_normal"),
                ),
                "X_1": (
                    ["X_0", "X_9"],
                    LinearAssignment(1, 0, 1, 1),
                    NoiseGenerator("standard_normal"),
                ),
            }
        )
        scm.clear_intervention_backup()
        scm.plot(alpha=1)
        plt.show()
    elif scenario == 3:
        vars = {
            "X_9": ([], LinearAssignment(1, 0), NoiseGenerator("standard_normal")),
            "X_10": (
                ["X_9"],
                LinearAssignment(1, 0, 1),
                NoiseGenerator("standard_normal"),
            ),
        }
        scm.add_variables(vars)
        scm.intervention(
            {
                "Y": (
                    ["X_1", "X_2", "X_3", "X_9"],
                    LinearAssignment(1, 0, 1, 1, 1, 1),
                    NoiseGenerator("standard_normal"),
                ),
            }
        )
        scm.clear_intervention_backup()
        scm.plot(alpha=1)
        plt.show()
    (
        data,
        environments,
        scm,
        possible_parents,
        target_parents,
    ) = generate_data_from_scm(scm, target_var="Y", sample_size=sample_size, seed=seed)

    data.drop(columns=["X_9"], inplace=True)
    possible_parents.remove("X_9")

    data["env"] = environments
    if not os.path.isdir("./data"):
        os.mkdir("./data")
    data.to_csv(f"./data/confounders_scenario_{scenario}_.csv", index=False)
    data.drop(columns=["env"], inplace=True)

    nr_envs = np.unique(environments).max() + 1
    use_visdom = 0

    ap = Predictor(
        epochs=epochs,
        batch_size=10000,
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
        results_filename=f"{test_name}_scenario-{scenario}",
        **kwargs,
    )
    s = f"{res_str}\n"
    return {
        "res_str": s,
        "sample_size": sample_size,
        "step": step,
        "scenario": scenario,
        "nr_runs": nr_runs,
        "epochs": epochs,
        "scm": scm,
    }


def init(l):
    global LOCK
    LOCK = l


test_name = "confoundertest"

if __name__ == "__main__":

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if not os.path.isdir("./log"):
        os.mkdir("./log")
    log_fname = f'{test_name}_{strftime("%Y-%m-%d_%H-%M-%S", gmtime())}'
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        filename=f"./log/{log_fname}.log",
    )  # pass explicit filename here
    logger = logging.getLogger()  # get the root logger

    PredictorClass = AgnosticPredictor
    multiprocessing.set_start_method("spawn")
    man = multiprocessing.Manager()
    steps = 1
    sample_size = 4096
    nr_runs = 30
    epochs = 1000
    results = []
    scenarios = [1, 2, 3]
    # we test 4 scenarios:
    # 1. increasing nonlinearity in the parents,
    # 2. increasing nonlinearity in the children,
    # 3. increasing nonlinearity on the target,
    # 4. increasing nonlinearity on all
    for scenario in scenarios:
        lock = man.Lock()
        with ProcessPoolExecutor(max_workers=min(5, steps)) as executor:
            futures = list(
                (
                    executor.submit(
                        run_scenario,
                        PredictorClass,
                        nr_epochs=epochs,
                        nr_runs=nr_runs,
                        scenario=scenario,
                        epochs=epochs,
                        step=step,
                        sample_size=sample_size,
                        LOCK=lock,
                        device=device,
                    )
                    for step, (layers, strength) in enumerate(
                        zip(range(steps), range(steps))
                    )
                )
            )
            for future in as_completed(futures):
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
