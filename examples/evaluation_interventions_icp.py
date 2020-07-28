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
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd


def run_scenario(
    strength, sample_size, scenario, **kwargs,
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
        sample_size=sample_size,
        seed=seed,
    )

    ap = LinPredictor(filter_variables=False)

    parents, _ = ap.infer(data, "Y", environments, **kwargs,)
    ps = pd.Series(0, index=sorted(possible_parents, key=lambda x: int(x[2:])))
    for var in possible_parents:
        if var in parents:
            ps[var] = 1
    return ps


def init(l):
    global LOCK
    LOCK = l


test_name = "interventionstest"
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

    sample_size = 2048
    scenarios = ["do", "meanshift", "scaling"]

    steps = 11

    results = {}
    for scenario in scenarios:
        for step, strength in enumerate([i for i in range(0, steps)]):
            results[f"interventionstest_scenario-{scenario}_step-{step}"] = (
                run_scenario(strength, sample_size, scenario).to_frame().T
            )

    for title, res in results.items():
        print(title, res, sep="\n")
        res.to_csv(f"./results/icp_{title}.csv", index=False)
