import unittest
from unittest import TestCase

from scm import LinearAssignment, PolynomialAssignment, NoiseGenerator, SCM
import numpy as np


class TestAssignments(TestCase):
    def test_linear_assignment(self):
        func = LinearAssignment(1, 0, 1, 2, 3)
        input_args = np.array([1, 2]), np.array([2, 4]), np.array([3, 6])

        self.assertTrue((func(np.array([1, 1]), *input_args) == np.array([15, 29])).all())

    def test_scm_build(self):
        cn = SCM(
            assignment_map={"X_0": ([],
                                    LinearAssignment(1),
                                    NoiseGenerator("negative_binomial", n=30, p=0.5)),
                            "X_1": (["X_0"],
                                    LinearAssignment(1, 1, 2),
                                    NoiseGenerator("negative_binomial", n=30, p=0.5)),
                            "X_2": (["X_0", "X_1"],
                                    LinearAssignment(1, 1, 3, 2),
                                    NoiseGenerator("negative_binomial", n=30, p=0.5)),
                            "X_3": (["X_1", "X_2"],
                                    PolynomialAssignment([0, 1], [0, 1, 1 / 2], [0, 0, 4]),
                                    NoiseGenerator("negative_binomial", n=30, p=0.5)),
                            "Y": (["X_0", "X_2"],
                                  PolynomialAssignment([0, 1], [0, 0, 1.5], [0, 1]),
                                  NoiseGenerator("negative_binomial", n=30, p=0.5))},
            variable_tex_names={"X_0": "$X_0$", "X_1": "$X_1$", "X_2": "$X_2$", "X_3": "$X_3$"}
        )
        cn.plot()
        sample = cn.sample(100000)
        print(sample)
        print(sample.mean(axis=0))
        print(cn)


if __name__ == '__main__':
    unittest.main()
