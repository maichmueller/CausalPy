from causalpy.scm import (
    RandomSCM,
    LinearAssignment,
    PolynomialAssignment,
    NoiseGenerator,
)
from causalpy.icp import LiNGAMPredictor
import matplotlib.pyplot as plt
from timeit import timeit
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

if __name__ == "__main__":
    times_per_n = []
    for n in tqdm(range(2, 20)):
        times_per_iter = []
        for i in range(10):
            causal = RandomSCM(
                n,
                assignment_functions=[LinearAssignment, PolynomialAssignment],
                noise_models=[
                    NoiseGenerator("normal", scale=i / 10) for i in range(1, 100)
                ],
            )
            vars = list(causal.get_variables())
            samples = [causal.sample(1000)]
            envs = [0] * 1000
            if "X_0" in causal.graph:
                causal.do_intervention(["X_0"], [10])
                samples.append(causal.sample(1000)[vars])
                envs += [1] * 1000
            if "X_1" in causal.graph:
                causal.do_intervention(["X_1"], [10])
                samples.append(causal.sample(1000)[vars])
                envs += [2] * 1000
            if "X_2" in causal.graph:
                causal.do_intervention(["X_2"], [10])
                samples.append(causal.sample(1000)[vars])
                envs += [3] * 1000
            obs = pd.concat(samples)
            envs = np.array(envs)
            target = np.random.choice(vars)
            times_per_iter.append(
                timeit(
                    lambda: LiNGAMPredictor(filter_variables=False).infer(
                        obs, target, envs
                    ),
                    number=1,
                )
            )
        times_per_n.append(np.mean(times_per_iter))
    print(times_per_n)
    plt.plot(times_per_n)
    plt.show()