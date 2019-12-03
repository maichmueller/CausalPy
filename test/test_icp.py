from functools import reduce

from causalpy import (
    LinearAssignment,
    PolynomialAssignment,
    NoiseGenerator,
    SCM,
    LINGAMPredictor,
)
import numpy as np
import pandas as pd
from scipy import stats
from numpy.polynomial.polynomial import Polynomial
import matplotlib.pyplot as plt


def build_scm_simple(seed=0):
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
                NoiseGenerator("standard_normal", seed=seed+1),
            ),
            "X_2": (
                ["X_0"],
                LinearAssignment(1, 1, 3),
                NoiseGenerator("standard_normal", seed=seed+2),
            ),
            "X_3": (
                ["X_1"],
                LinearAssignment(1, 0, 0.3),
                NoiseGenerator("standard_normal", seed=seed+3),
            ),
            "Y": (
                ["X_3"],
                LinearAssignment(1, 3, 5),
                NoiseGenerator("standard_normal", seed=seed+4),
            ),
            "X_4": (
                ["Y"],
                LinearAssignment(1, 3, 9),
                NoiseGenerator("standard_normal", seed=seed+5),
            ),
            "X_5": (
                ["X_3"],
                LinearAssignment(1, 3, -2.7),
                NoiseGenerator("standard_normal", seed=seed+6),
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


def build_scm_medium(seed=0):
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
                NoiseGenerator("standard_normal", seed=seed+1),
            ),
            "X_2": (
                ["X_0", "X_1"],
                LinearAssignment(1, 1, 3, -2),
                NoiseGenerator("standard_normal", seed=seed+2),
            ),
            "X_3": (
                ["X_1", "X_2"],
                LinearAssignment(1, 0, 0.3, 0.4),
                NoiseGenerator("standard_normal", seed=seed+3),
            ),
            "Y": (
                ["X_3", "X_0"],
                LinearAssignment(1, 3, 5, -1),
                NoiseGenerator("standard_normal", seed=seed+4),
            ),
            "X_4": (
                ["Y"],
                LinearAssignment(1, 3, 9),
                NoiseGenerator("standard_normal", seed=seed+5),
            ),
            "X_5": (
                ["X_3", "Y"],
                LinearAssignment(1, 3, -2.7, 4.),
                NoiseGenerator("standard_normal", seed=seed+6),
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


def test_linear_icp_simple():
    cn = build_scm_simple()
    # cn.plot(alpha=1)
    # plt.show()
    cn_vars = list(cn.get_variables())
    data_unintervend = cn.sample(1000)[cn_vars]
    cn.do_intervention(["X_3"], [-4])
    data_intervention_1 = cn.sample(1000)[cn_vars]
    cn.undo_intervention()
    cn.do_intervention(["X_0"], [5])
    data_intervention_2 = cn.sample(1000)[cn_vars]
    cn.undo_intervention()
    cn.do_intervention(["X_4"], [5])
    data_intervention_3 = cn.sample(1000)[cn_vars]

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

    causal_parents, p_vals = LINGAMPredictor(alpha=0.01).infer(obs, envs, target)

    assert sorted(causal_parents) == sorted(cn.graph.predecessors(target))


def test_linear_icp_medium():
    cn = build_scm_medium()
    # cn.plot(alpha=1)
    # plt.show()
    cn_vars = list(cn.get_variables())
    data_unintervend = cn.sample(1000)[cn_vars]
    cn.do_intervention(["X_3"], [-4])
    data_intervention_1 = cn.sample(1000)[cn_vars]
    cn.undo_intervention()
    cn.do_intervention(["X_0"], [5])
    data_intervention_2 = cn.sample(1000)[cn_vars]
    cn.undo_intervention()
    cn.do_intervention(["X_1"], [-5])
    data_intervention_3 = cn.sample(1000)[cn_vars]
    cn.undo_intervention()
    cn.do_intervention(["X_2"], [3.89])
    data_intervention_4 = cn.sample(1000)[cn_vars]
    cn.undo_intervention()
    cn.do_intervention(["X_4"], [5])
    data_intervention_5 = cn.sample(1000)[cn_vars]

    obs = pd.concat(
        [
            data_unintervend,
            data_intervention_1,
            data_intervention_2,
            data_intervention_3,
            data_intervention_4,
            data_intervention_5,
        ],
        axis=0,
    ).reset_index(drop=True)

    envs = np.array(
        reduce(
            lambda x, y: x + y,
            [
                [i] * len(data)
                for i, data in zip(
                    range(6),
                    (
                        data_unintervend,
                        data_intervention_1,
                        data_intervention_2,
                        data_intervention_3,
                        data_intervention_4,
                        data_intervention_5,
                    ),
                )
            ],
        )
    )
    target = "Y"

    causal_parents, p_vals = LINGAMPredictor(alpha=0.1, residual_test="normal", log_level="DEBUG").infer(obs, envs, target)

    assert sorted(causal_parents) == sorted(cn.graph.predecessors(target))
