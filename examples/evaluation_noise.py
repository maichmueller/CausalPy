import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

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


def run_scenario(
    Predictor, modelclass, dists, sample_size, nr_runs, epochs, scenario, step, **kwargs
):
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
        results_filename=f"{modelclass}_{test_name}_dist-{dists['dist']}_args-{','.join([k + '=' + str(v) for k,v in dists['kwargs'].items()])}_scenario-{scenario}_step-{step+1}",
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

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "modelclass",
        metavar="modelclass",
        type=str,
        nargs=1,
        help="The model to evaluate",
    )

    parser.add_argument(
        "nr_workers",
        metavar="nr_workers",
        type=int,
        nargs=1,
        default=5,
        help="The number of multiprocessing workers",
    )

    parser.add_argument(
        "dist",
        metavar="dist",
        type=str,
        nargs=1,
        default="all",
        help="Distribution to compute",
    )

    parser.add_argument(
        "start_step",
        metavar="start_step",
        type=int,
        nargs=1,
        default=0,
        help="Step from which to start",
    )

    parser.add_argument(
        "end_step",
        metavar="end_step",
        type=int,
        nargs=1,
        default=0,
        help="Step until which to compute",
    )

    parser.add_argument(
        "scenario",
        metavar="scenario",
        type=str,
        nargs=1,
        default=None,
        help="Step from which to start",
    )

    args = parser.parse_args()
    modelclass = args.modelclass[0]
    nr_work = args.nr_workers[0]
    dist = args.dist[0]
    start_step = args.start_step[0]
    end_step = args.end_step[0]
    scenario = args.scenario[0]
    if modelclass == "single":
        PredictorClass = AgnosticPredictor
    elif modelclass == "multi":
        PredictorClass = MultiAgnosticPredictor
        nr_work = min(nr_work, 3)
    elif modelclass == "density":
        PredictorClass = DensityBasedPredictor
    else:
        raise ValueError(
            f"Modelclass {modelclass} not recognized. Use one of 'single', 'multi', or 'density'"
        )
    multiprocessing.set_start_method("spawn")
    man = multiprocessing.Manager()
    steps = None
    sample_size = 4096
    nr_runs = 30
    epochs = 1000
    results = []
    dists = {
        "normal": (
            [
                dict(loc=0, scale=1),
                dict(loc=0, scale=2),
                dict(loc=0, scale=5),
                dict(loc=5, scale=1),
                dict(loc=-5, scale=1),
            ],
            "numpy",
        ),
        "exponential": (
            [dict(scale=1), dict(scale=5), dict(scale=10), dict(scale=30)],
            "numpy",
        ),
        "cauchy": (
            [
                dict(loc=0, scale=1),
                dict(loc=0, scale=2),
                dict(loc=0, scale=5),
                dict(loc=5, scale=1),
                dict(loc=-5, scale=1),
            ],
            "scipy",
        ),
        "beta": (
            [
                dict(a=1, b=1),
                dict(a=0.5, b=1),
                dict(a=0.25, b=1),
                dict(a=1, b=0.5),
                dict(a=1, b=0.25),
            ],
            "numpy",
        ),
    }
    if dist != "all":
        dists = {dist: dists[dist]}
        if end_step != 0:
            print(dists)
            dists[dist] = dists[dist][0][start_step:end_step], dists[dist][1]
        else:
            dists[dist] = dists[dist][0][start_step:], dists[dist][1]

    scenarios = ["parents", "children", "target", "all"]
    if scenario is not None:
        scenarios = [scenario]
    # we test 4 scenarios:
    # 1. increasing nonlinearity in the parents,
    # 2. increasing nonlinearity in the children,
    # 3. increasing nonlinearity on the target,
    # 4. increasing nonlinearity on all

    for scenario in scenarios:
        for dist, package in dists.items():
            params, source = package
            for param in params:
                param["source"] = source
            lock = man.Lock()
            with ProcessPoolExecutor(max_workers=nr_work) as executor:
                futures = list(
                    (
                        executor.submit(
                            run_scenario,
                            PredictorClass,
                            modelclass,
                            {"dist": dist, "kwargs": param},
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
