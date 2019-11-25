import unittest
from unittest import TestCase

from causality import LinearAssignment, PolynomialAssignment, NoiseGenerator, SCM, LinICP
import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import Polynomial


def build_scm():
    cn = SCM(
        assignment_map={
            "X_0": (
                [],
                LinearAssignment(1),
                NoiseGenerator("standard_normal", seed=0),
            ),
            "X_1": (
                ["X_0"],
                LinearAssignment(1, 1, 2),
                NoiseGenerator("standard_normal", seed=0),
            ),
            "X_2": (
                ["X_0", "X_1"],
                LinearAssignment(1, 1, 3, 2),
                NoiseGenerator("standard_normal", seed=0),
            ),
            "X_3": (
                ["X_1", "X_2"],
                LinearAssignment(1, 0, 0.3, 0.7),
                NoiseGenerator("standard_normal", seed=0),
            ),
            "Y": (
                ["X_0", "X_2", "X_3"],
                LinearAssignment(1, 3, -1, -2, 5),
                NoiseGenerator("standard_normal", seed=0),
            ),
        },
        variable_tex_names={
            "X_0": "$X_0$",
            "X_1": "$X_1$",
            "X_2": "$X_2$",
            "X_3": "$X_3$",
        },
    )
    return cn


class TestICP(TestCase):
    def test_linear_icp(self):
        cn = build_scm()

        data_unintervend = cn.sample(100)
        cn.do_intervention(["X_1"], [2])
        data_intervention_1 = cn.sample(100)
        cn.soft_intervention(["X_2"], [NoiseGenerator("standard_t", df=1, seed=0)])
        data_intervention_2 = cn.sample(100)

        obs = pd.concat([data_unintervend, data_intervention_1, data_intervention_2], axis=0)
        envs = np.array([0] * 100 + [1] * 100 * [2] * 100)
        target = "Y"

        causal_parents = LinICP().infer(obs, envs, target, alpha=0.05)

