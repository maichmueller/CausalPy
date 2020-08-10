import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

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

from tqdm import tqdm
import matplotlib.pyplot as plt
import argparse
import pandas as pd
import cdt


def run_scenario(dists, sample_size):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    np.random.seed(seed)
    # print(coeffs)
    scm = study_scm(seed=seed, noise_dists=dists)
    # scm.plot()
    # plt.show()
    (
        data,
        environments,
        scm,
        possible_parents,
        target_parents,
    ) = generate_data_from_scm(scm, target_var="Y", sample_size=sample_size, seed=seed)

    data["ENV"] = environments
    res = []
    # if "normal" == dists["dist"]:
    ci_test = "gaussian"
    # else:
    #     ci_test = "rcit"
    for _ in range(1):
        if modelclass == "pc":
            graph = cdt.causality.graph.PC(CItest=ci_test).create_graph_from_data(data)
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


test_name = "noisetest"
# modelclass = "pc"
modelclass = "gies"
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

    steps = None
    sample_size = 1024
    nr_runs = 30
    epochs = 2000
    results = {}
    dists = {
        "normal": ([dict(scale=i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], "numpy",),
        "exponential": (
            [dict(scale=i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
            "numpy",
        ),
        "cauchy": ([dict(scale=i) for i in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]], "scipy",),
    }

    scenarios = ["target", "all"]
    # we test 4 scenarios:
    # 1. increasing nonlinearity in the parents,
    # 2. increasing nonlinearity in the children,
    # 3. increasing nonlinearity on the target,
    # 4. increasing nonlinearity on all

    for scenario in scenarios:
        for dist, package in dists.items():
            params, source = package

            for step, param in enumerate(params):
                param["source"] = source
                name = f"{modelclass}_{test_name}_dist-{dist}_args-{','.join([k + '=' + str(v) for k,v in param.items()])}_scenario-{scenario}_step-{step+1}"
                results[name] = run_scenario(
                    {"dist": dist, "kwargs": param}, sample_size
                )

    for (title, res) in results.items():
        if os.path.isfile(f"./results/{title}.csv"):
            prev_res = pd.read_csv(f"./results/{title}.csv", index_col=None)
            res = pd.concat([prev_res, res])
        res.to_csv(f"./results/{title}.csv", index=False)
