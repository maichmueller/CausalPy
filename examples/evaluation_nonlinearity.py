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
    Predictor, coeffs, sample_size, nr_runs, epochs, scenario, step, **kwargs
):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = 0
    np.random.seed(seed)
    # print(coeffs)
    scm = study_scm(seed=seed, coeffs_by_var=coeffs)
    scm.plot(alpha=1)
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
        results_filename=f"scenario-{scenario}_step-{step+1}",
        **kwargs,
    )
    s = f"{res_str}\n"
    return {
        "res_str": s,
        "coeffs": coeffs,
        "sample_size": sample_size,
        "step": step,
        "scenario": scenario,
        "nr_runs": nr_runs,
        "epochs": epochs,
        "scm": scm,
    }


def return_coeffs_by_var(coeffs, scenario):
    if scenario == "parents":
        coeffs_by_var = {"X_1": [coeffs], "X_2": [coeffs]}
    elif scenario == "children":
        coeffs_by_var = {"X_3": [coeffs], "X_6": [coeffs, coeffs]}
    elif scenario == "target":
        coeffs_by_var = {"Y": [coeffs, coeffs]}
    elif scenario == "all":
        coeffs_by_var = {
            "X_1": [coeffs],
            "X_2": [coeffs],
            "X_3": [coeffs],
            "X_4": [coeffs],
            "X_6": [coeffs, coeffs],
            "Y": [coeffs, coeffs],
        }
    return coeffs_by_var


def init(l):
    global LOCK
    LOCK = l


if __name__ == "__main__":

    if not os.path.isdir("./log"):
        os.mkdir("./log")
    log_fname = f'{strftime("%Y-%m-%d_%H-%M-%S", gmtime())}'
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s",
        filename=f"./log/{log_fname}.log",
    )  # pass explicit filename here
    logger = logging.getLogger()  # get the root logger

    PredictorClass = AgnosticPredictor
    multiprocessing.set_start_method("spawn")
    man = multiprocessing.Manager()
    steps = 40
    sample_size = 4096
    nr_runs = 20
    epochs = 1000
    results = []
    scenarios = ["parents", "children", "target", "all"]
    # we test 4 scenarios:
    # 1. increasing nonlinearity in the parents,
    # 2. increasing nonlinearity in the children,
    # 3. increasing nonlinearity on the target,
    # 4. increasing nonlinearity on all
    for scenario in scenarios:
        lock = man.Lock()
        with ProcessPoolExecutor(max_workers=20) as executor:
            futures = list(
                (
                    executor.submit(
                        run_scenario,
                        PredictorClass,
                        return_coeffs_by_var(coeffs, scenario),
                        nr_epochs=epochs,
                        nr_runs=nr_runs,
                        scenario=scenario,
                        epochs=epochs,
                        step=step,
                        sample_size=sample_size,
                        LOCK=lock,
                    )
                    for step, coeffs in enumerate(
                        zip(
                            [0] * steps,
                            [1] * steps,
                            np.linspace(0.1, 4, steps),
                            np.linspace(-0.01, -0.4, steps),
                            np.linspace(0.001, 0.04, steps),
                        )
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
        logger.info(f"scenario={res['scenario']}")
        logger.info(f"step={res['step']}")
        logger.info(f"epochs={res['epochs']}")
        logger.info(f"nr_runs={res['nr_runs']}")
        logger.info(f"sample_size={res['sample_size']}")
        logger.info(f"coefficients={res['coeffs']}")
        logger.info(f"scm=\n{res['scm']}")
        logger.info(f"outcome=\n{res['res_str']}\n")
