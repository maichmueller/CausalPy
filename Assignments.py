import numpy as np
from numpy.polynomial import polynomial
from typing import List, Union, Dict


class LinearFunc:
    def __init__(self, *coefficients):
        self.offset = coefficients[0]
        self.coefficients = np.asarray(coefficients)[1:] if len(coefficients) > 1 else np.array([])

    def __call__(self, *args):
        return self.offset + self.coefficients @ args


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
        return sum([poly(args) for poly in self.polynomials])


