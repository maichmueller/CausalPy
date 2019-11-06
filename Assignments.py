import numpy as np
from numpy.polynomial import polynomial
from typing import List, Union, Dict


class LinearFunc:
    def __init__(self, noise_coeff, offset=0, *coefficients):
        self.noise_coeff = noise_coeff
        self.offset = offset
        self.coefficients = np.asarray(coefficients) if len(coefficients) > 0 else np.array([])

    def __call__(self, noise, *args):
        return self.offset + self.noise_coeff * noise + self.coefficients @ args

    def __str__(self):
        variables = []
        rep = f"{round(self.offset, 2)} + {round(self.noise_coeff, 2)} N"
        for i, c in enumerate(self.coefficients):
            if c != 0:
                variables.append(f"X_{i}")
                rep += f" + {round(c, 2)} X_{i}"
        prefix = f"f(N, " + ", ".join(variables) + ") = "
        return prefix + rep


class Polynomial:
    def __init__(self,
                 coefficients_list: List[List[float]]
                 ):
        self.polynomials = []
        if len(coefficients_list) > 0:
            for coeffs in coefficients_list:
                self.polynomials.append(polynomial.Polynomial(coeffs))
        self.polynomials = np.asarray(self.polynomials)

    def __call__(self, *args):
        assert (len(args) == len(self.polynomials))
        # args[0] is assumed to be the noise
        return sum([poly(args) for poly in self.polynomials])
