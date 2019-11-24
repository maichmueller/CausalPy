import unittest
from unittest import TestCase

from scm import LinearAssignment, PolynomialAssignment, NoiseGenerator, SCM
import numpy as np


class TestAssignments(TestCase):
    def test_linear_assignment(self):
        func = LinearAssignment(1, 0, 1, 2, 3)
        input_args = np.array([1, 2]), np.array([2, 4]), np.array([3, 6])

        self.assertTrue(
            (func(np.array([1, 1]), *input_args) == np.array([15, 29])).all()
        )

    def test_scm_build(self):
        cn = SCM(
            assignment_map={
                "X_0": (
                    [],
                    LinearAssignment(1),
                    NoiseGenerator("negative_binomial", n=30, p=0.5),
                ),
                "X_1": (
                    ["X_0"],
                    LinearAssignment(1, 1, 2),
                    NoiseGenerator("standard_normal"),
                ),
                "X_2": (
                    ["X_0", "X_1"],
                    LinearAssignment(1, 1, 3, 2),
                    NoiseGenerator("standard_normal"),
                ),
                "X_3": (
                    ["X_1", "X_2"],
                    PolynomialAssignment([0, 1], [0, 1, .5], [0, 0, 4]),
                    NoiseGenerator("standard_normal"),
                ),
                "Y": (
                    ["X_0", "X_2"],
                    PolynomialAssignment([0, 1], [0, 0, 1.5], [0, 2]),
                    NoiseGenerator("standard_normal"),
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

        sample = cn.sample(10)



if __name__ == "__main__":
    unittest.main()
