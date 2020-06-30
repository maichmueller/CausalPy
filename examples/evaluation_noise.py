import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

from causalpy.causal_prediction.interventional import (
    AgnosticPredictor,
    MultiAgnosticPredictor,
)
from examples.study_cases import study_scm, generate_data_from_scm
import numpy as np
import torch

from time import gmtime, strftime
import os

from tqdm import tqdm
import matplotlib.pyplot as plt


def run_scenario(
    Predictor, dists, sample_size, nr_runs, epochs, scenario, step, **kwargs
):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    np.random.seed(seed)
    # print(coeffs)
    scm = study_scm(seed=seed, noise_dists=dists)
    scm.plot()
    plt.show()
    (
        data,
        environments,
        scm,
        possible_parents,
        target_parents,
    ) = generate_data_from_scm(scm, target_var="Y", sample_size=sample_size, seed=seed)

    nr_envs = np.unique(environments).max() + 1
    use_visdom = 0

    ap = Predictor(
        epochs=epochs,
        batch_size=10000,
        visualize_with_visdom=bool(use_visdom),
        masker_network_params=dict(monte_carlo_sample_size=1),
        device="cuda:0",
    )

    results_mask, results_loss, res_str = ap.infer(
        data,
        environments,
        "Y",
        nr_runs=nr_runs,
        normalize=True,
        save_results=True,
        results_filename=f"{test_name}_scenario-{scenario}_step-{step+1}",
        **kwargs,
    )
    s = f"{res_str}\n"
    return {
        "res_str": s,
        "noise": dists,
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


test_name = "noisetest"

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
    steps = None
    sample_size = 4096
    nr_runs = 30
    epochs = 1000
    results = []
    dists = [
        (
            "normal",
            [
                dict(loc=0, scale=1),
                dict(loc=0, scale=2),
                dict(loc=0, scale=5),
                dict(loc=5, scale=1),
                dict(loc=-5, scale=1),
            ],
            "numpy",
        ),
        (
            "exponential",
            [dict(scale=1), dict(scale=5), dict(scale=10), dict(scale=30),],
            "numpy",
        ),
        (
            "cauchy",
            [
                dict(loc=0, scale=1),
                dict(loc=0, scale=2),
                dict(loc=0, scale=5),
                dict(loc=5, scale=1),
                dict(loc=-5, scale=1),
            ],
            "scipy",
        ),
        (
            "beta",
            [
                dict(a=1, b=1),
                dict(a=0.5, b=1),
                dict(a=0.25, b=1),
                dict(a=1, b=0.5),
                dict(a=1, b=0.25),
            ],
            "numpy",
        ),
    ]
    # we test 4 scenarios:
    # 1. increasing nonlinearity in the parents,
    # 2. increasing nonlinearity in the children,
    # 3. increasing nonlinearity on the target,
    # 4. increasing nonlinearity on all
    scenarios = ["parents", "children", "target", "all"]

    for scenario in scenarios:
        for package in dists:
            dist, params, source = package
            for param in params:
                param["source"] = source
            lock = man.Lock()
            with ProcessPoolExecutor(max_workers=5) as executor:
                futures = list(
                    (
                        executor.submit(
                            run_scenario,
                            PredictorClass,
                            {"dist": dist, "kwargs": params},
                            nr_epochs=epochs,
                            nr_runs=nr_runs,
                            scenario=scenario,
                            epochs=epochs,
                            step=step,
                            sample_size=sample_size,
                            LOCK=lock,
                        )
                        for step, param in enumerate(params)
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
