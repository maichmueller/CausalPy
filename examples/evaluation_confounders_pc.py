import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Optional, List, Collection

from causalpy import Assignment, LinearAssignment, NoiseGenerator
from causalpy.causal_prediction import ICPredictor, LinPredictor
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

import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse

import itertools
from typing import Optional, Callable, Type, Tuple, List, Union

import torch
from causalpy.neural_networks.utils import get_jacobian
from abc import ABC, abstractmethod
import numpy as np
import cdt
from cdt import SETTINGS
import networkx as nx

test_name = "confoundertest"
modelclass = "pc"
# modelclass = "gies"


def run_scenario(
    sample_size, scenario, **kwargs,
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
        # scm.plot(alpha=1)
        # plt.show()
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
        # scm.plot(alpha=1)
        # plt.show()
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
        # scm.plot(alpha=1)
        # plt.show()
    (
        data,
        environments,
        scm,
        possible_parents,
        target_parents,
    ) = generate_data_from_scm(scm, target_var="Y", sample_size=sample_size, seed=seed)

    data.drop(columns=["X_9"], inplace=True)
    possible_parents.remove("X_9")

    # data["env"] = environments
    # if not os.path.isdir("./data"):
    #     os.mkdir("./data")
    # data.to_csv(
    #     f"./data/{modelclass}_confounders_scenario_{scenario}_.csv", index=False
    # )
    # data.drop(columns=["env"], inplace=True)
    data["ENV"] = environments
    res = []
    for _ in range(1):
        if modelclass == "pc":
            graph = cdt.causality.graph.PC().create_graph_from_data(data)
        elif modelclass == "gies":
            graph = cdt.causality.graph.GIES().create_graph_from_data(data)
        else:
            raise ValueError("Wrong modelclass")
        parents = set(graph.predecessors("Y"))
        children = set(graph["Y"])
        ps = pd.Series(0, index=sorted(possible_parents, key=lambda x: int(x[2:])))
        for var in possible_parents:
            if var in parents and var not in children:
                ps[var] = 1
        res.append(ps.to_frame())
    res = pd.concat(res, ignore_index=True, axis=1).T
    return res


def init(l):
    global LOCK
    LOCK = l


if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # SETTINGS.rpath = "/usr/bin/Rscript"
    if not os.path.isdir("./log"):
        os.mkdir("./log")
    log_fname = f'{test_name}_{strftime("%Y-%m-%d_%H-%M-%S", gmtime())}'
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        filename=f"./log/{log_fname}.log",
    )  # pass explicit filename here
    logger = logging.getLogger()  # get the root logger

    sample_size = 2048

    scenarios = [1, 2, 3]

    results = {}
    for scenario in scenarios:
        results[f"{modelclass}_{test_name}_scenario-{scenario}"] = run_scenario(
            sample_size, scenario
        )

    for i, (title, res) in zip(scenarios, results.items()):
        print("scenario:", i, ", result:\n", res)
        if os.path.isfile(f"./results/{title}.csv"):
            prev_res = pd.read_csv(f"./results/{title}.csv", index_col=None)
            res = pd.concat([prev_res, res])
        res.to_csv(f"./results/{title}.csv", index=False)
