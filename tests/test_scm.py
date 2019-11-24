import unittest
from unittest import TestCase

from causality import LinearAssignment, PolynomialAssignment, NoiseGenerator, SCM
import numpy as np
import pandas as pd
from numpy.polynomial.polynomial import Polynomial


class TestAssignments(TestCase):
    def test_linear_assignment(self):
        func = LinearAssignment(1, 0, 1, 2, 3)
        input_args = np.array([1, 2]), np.array([2, 4]), np.array([3, 6])

        self.assertTrue(
            (func(np.array([1, 1]), *input_args) == np.array([15, 29])).all()
        )

    def test_scm(self):
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
                    PolynomialAssignment([0, 1], [0, 1, .5], [0, 0, 4]),
                    NoiseGenerator("standard_normal", seed=0),
                ),
                "Y": (
                    ["X_0", "X_2"],
                    PolynomialAssignment([0, 1], [0, 0, 1.5], [0, 2]),
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

        nodes_in_graph = list(cn.graph.nodes)
        self.assertEqual(nodes_in_graph, ["X_0", "X_1", "X_2", "X_3", "Y"])

        def noise(n):
            return np.random.default_rng(0).standard_normal(n)

        scm_sample = cn.sample(10)
        sample = np.empty((10, 5), dtype=scm_sample.values.dtype)
        sample[:, 0] = noise(10)
        sample[:, 1] = 1 + noise(10) + 2 * sample[:, 0]
        sample[:, 2] = 1 + noise(10) + 3 * sample[:, 0] + 2 * sample[:, 1]
        sample[:, 3] = noise(10) + Polynomial([0, 1, .5])(sample[:, 1]) + Polynomial([0, 0, 4])(sample[:, 2])
        sample[:, 4] = noise(10) + Polynomial([0, 0, 1.5])(sample[:, 0]) + Polynomial([0, 2])(sample[:, 2])
        sample = pd.DataFrame(sample, columns=nodes_in_graph)
        sample_order_scm = list(cn._causal_iterator())
        sample = sample[sample_order_scm]
        # floating point inaccuracy needs to be accounted for
        self.assertTrue((sample - scm_sample).abs().values.sum() < 10e-10)
        del sample




if __name__ == "__main__":
    unittest.main()
