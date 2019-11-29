from functools import reduce

from causality import (
    LinearAssignment,
    PolynomialAssignment,
    NoiseGenerator,
    SCM,
    LinICP,
)
import numpy as np
import pandas as pd
from scipy import stats
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt


def build_scm(seed=0):
    cn = SCM(
        assignment_map={
            "X_0": (
                [],
                LinearAssignment(1),
                NoiseGenerator("standard_normal", seed=seed),
            ),
            "X_1": (
                ["X_0"],
                LinearAssignment(1, 1, 2),
                NoiseGenerator("standard_normal", seed=seed),
            ),
            "X_2": (
                ["X_0"],
                LinearAssignment(1, 1, 3),
                NoiseGenerator("standard_normal", seed=seed),
            ),
            "X_3": (
                ["X_1"],
                LinearAssignment(1, 0, 0.3),
                NoiseGenerator("standard_normal", seed=seed),
            ),
            "Y": (
                ["X_3"],
                LinearAssignment(1, 3, 5),
                NoiseGenerator("standard_normal", seed=seed),
            ),
            "X_4": (
                ["Y"],
                LinearAssignment(1, 3, 9),
                NoiseGenerator("standard_normal", seed=seed),
            ),
            "X_5": (
                ["X_3"],
                LinearAssignment(1, 3, -2.7),
                NoiseGenerator("standard_normal", seed=seed),
            ),
        },
        variable_tex_names={
            "X_0": "$X_0$",
            "X_1": "$X_1$",
            "X_2": "$X_2$",
            "X_3": "$X_3$",
            "X_4": "$X_4$",
            "X_5": "$X_5$",
        },
    )
    return cn


def test_linear_icp():
    cn = build_scm()
    cn.plot(alpha=1)
    data_unintervend = cn.sample(100)
    cn.do_intervention(["X_1"], [-4])
    data_intervention_1 = cn.sample(100)
    cn.undo_intervention()
    cn.do_intervention(["X_0"], [5])
    data_intervention_2 = cn.sample(100)
    cn.undo_intervention()
    cn.do_intervention(["X_4"], [5])
    data_intervention_3 = cn.sample(100)

    obs = pd.concat(
        [
            data_unintervend,
            data_intervention_1,
            data_intervention_2,
            data_intervention_3,
        ],
        axis=0,
    ).reset_index(drop=True)

    envs = np.array(
        reduce(
            lambda x, y: x + y,
            [
                [i] * len(data)
                for i, data in zip(
                    range(4),
                    (
                        data_unintervend,
                        data_intervention_1,
                        data_intervention_2,
                        data_intervention_3,
                    ),
                )
            ],
        )
    )
    target = "Y"

    causal_parents, p_vals = LinICP().infer(obs, envs, target, alpha=0.05)

    assert causal_parents == tuple(cn.graph.predecessors(target))
