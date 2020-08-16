import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

import networkx as nx

from causalpy.causal_prediction import LinPredictor
from causalpy.causal_prediction.interventional import (
    AgnosticPredictor,
    MultiAgnosticPredictor,
    DensityBasedPredictor,
)
from examples.study_cases import study_scm, generate_data_from_scm
import numpy as np
import torch

from time import gmtime, strftime
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import cdt
from networkx.drawing.nx_agraph import graphviz_layout


def run_scenario(
    strength, sample_size, reach, scenario, **kwargs,
):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    np.random.seed(seed)
    # print(coeffs)
    scm = study_scm(seed=seed)

    (
        data,
        environments,
        scm,
        possible_parents,
        target_parents,
    ) = generate_data_from_scm(
        scm,
        target_var="Y",
        intervention_style=scenario,
        strength=strength,
        intervention_reach=reach,
        fix_strength=True,
        sample_size=sample_size,
        seed=seed,
    )

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


test_name = "interventionstest"
modelclass = "pc"
# modelclass = "gies"

if __name__ == "__main__":

    # device = "cuda:0" if torch.cuda.is_available() else "cpu"

    if not os.path.isdir("./log"):
        os.mkdir("./log")
    log_fname = f'{test_name}_{strftime("%Y-%m-%d_%H-%M-%S", gmtime())}'
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        filename=f"./log/{log_fname}.log",
    )  # pass explicit filename here
    logger = logging.getLogger()  # get the root logger

    sample_size = 1024
    scenarios = ["do", "meanshift", "scaling"]
    # reach = ["markov", "children", "parents", "parents-and-children"]
    reach = ["children", "parents-and-children"]
    steps = 11

    results = {}
    for scenario in scenarios:
        for r in reach:
            for step, strength in enumerate([i for i in range(0, steps)]):
                results[
                    f"{modelclass}_{test_name}_scenario-{scenario}_reach-{r}_step-{step+1}"
                ] = run_scenario(strength / 2, sample_size, r, scenario)

    for (title, res) in results.items():
        if os.path.isfile(f"./results/{title}.csv"):
            prev_res = pd.read_csv(f"./results/{title}.csv", index_col=None)
            res = pd.concat([prev_res, res])
        res.to_csv(f"./results/{title}.csv", index=False)
