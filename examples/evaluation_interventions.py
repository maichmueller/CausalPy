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
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt


def run_scenario(
    Predictor,
    modelclass,
    strength,
    sample_size,
    nr_runs,
    epochs,
    scenario,
    reach,
    step,
    **kwargs,
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
        intervention_reach=reach,
        strength=strength,
        fix_strength=True,
        sample_size=sample_size,
        seed=seed,
    )

    nr_envs = np.unique(environments).max() + 1
    use_visdom = 0

    ap = Predictor(
        epochs=epochs,
        batch_size=5000,
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
        results_filename=f"{modelclass}_{test_name}_scenario-{scenario}_reach-{reach}_step-{step+1}",
        **kwargs,
    )
    s = f"{res_str}\n"
    return {
        "res_str": s,
        "strength": strength,
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


test_name = "interventionstest"
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

    steps = 11
    sample_size = 1024
    nr_runs = 20
    epochs = 1500
    results = []

    scenarios = ["do", "meanshift", "scaling"]

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

    parser.add_argument(
        "--reach",
        metavar="reach",
        type=str,
        nargs="?",
        default="markov",
        help="Step from which to start",
    )

    args = parser.parse_args()
    modelclass = args.model
    nr_work = args.workers
    start_step = args.start - 1
    end_step = args.end - 1
    scenario = args.scenario
    reach = args.reach

    if scenario is not None:
        scenarios = [scenario]

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
    multiprocessing.set_start_method("spawn")
    man = multiprocessing.Manager()

    for scenario in scenarios:
        lock = man.Lock()
        with ProcessPoolExecutor(max_workers=nr_work) as executor:
            futures = list(
                (
                    executor.submit(
                        run_scenario,
                        PredictorClass,
                        modelclass,
                        strength / 2,
                        nr_epochs=epochs,
                        nr_runs=nr_runs,
                        scenario=scenario,
                        reach=reach,
                        epochs=epochs,
                        step=step,
                        sample_size=sample_size,
                        LOCK=lock,
                    )
                    for step, strength in enumerate(
                        [i for i in range(start_step, end_step)], start_step
                    )
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
