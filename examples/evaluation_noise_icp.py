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


def run_scenario(dists, scenario, sample_size):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    np.random.seed(seed)
    # print(coeffs)
    if scenario == "all":
        noise_dists = {
            "X_0": {"dist": dists["dist"], "kwargs": dists["kwargs"]},
            "X_1": {"dist": dists["dist"], "kwargs": dists["kwargs"]},
            "X_2": {"dist": dists["dist"], "kwargs": dists["kwargs"]},
            "X_3": {"dist": dists["dist"], "kwargs": dists["kwargs"]},
            "X_4": {"dist": dists["dist"], "kwargs": dists["kwargs"]},
            "X_5": {"dist": dists["dist"], "kwargs": dists["kwargs"]},
            "X_6": {"dist": dists["dist"], "kwargs": dists["kwargs"]},
            "X_7": {"dist": dists["dist"], "kwargs": dists["kwargs"]},
            "X_8": {"dist": dists["dist"], "kwargs": dists["kwargs"]},
            "Y": {"dist": dists["dist"], "kwargs": dists["kwargs"]},
        }
    elif scenario == "target":
        noise_dists = {
            "Y": {"dist": dists["dist"], "kwargs": dists["kwargs"]},
        }
    scm = study_scm(seed=seed, noise_dists=noise_dists)
    # scm.plot()
    # plt.show()
    (
        data,
        environments,
        scm,
        possible_parents,
        target_parents,
    ) = generate_data_from_scm(scm, target_var="Y", sample_size=sample_size, seed=seed)

    ap = LinPredictor(filter_variables=False)

    parents, _ = ap.infer(data, "Y", environments)
    ps = pd.Series(0, index=sorted(possible_parents, key=lambda x: int(x[2:])))
    for var in possible_parents:
        if var in parents:
            ps[var] = 1
    return ps


def init(l):
    global LOCK
    LOCK = l


test_name = "noisetest"
modelclass = None
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
                name = f"{test_name}_dist-{dist}_args-{','.join([k + '=' + str(v) for k,v in param.items()])}_scenario-{scenario}_step-{step+1}"
                results[name] = (
                    run_scenario({"dist": dist, "kwargs": param}, scenario, sample_size)
                    .to_frame()
                    .T
                )

    for title, res in results.items():
        print(title, res, sep="\n")
        res.to_csv(f"./results/icp_{title}.csv", index=False)
