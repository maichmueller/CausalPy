import unittest
from unittest import TestCase
from Assignments import LinearFunc
import numpy as np


class TestAssignments(TestCase):
    def test_linear_func(self):
        func = LinearFunc(1, 0, 1, 2, 3)
        input_args = np.array([1, 2]), np.array([2, 4]), np.array([3, 6])

        self.assertTrue((func(np.array([1, 1]), *input_args) == np.array([15, 29])).all())


if __name__ == '__main__':
    unittest.main()