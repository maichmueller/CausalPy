from .base import BaseAssignment

import numpy as np
from numpy.polynomial import polynomial
from typing import List


class PolynomialAssignment(BaseAssignment):
    def __init__(self, *coefficients_list: List[float]):
        polynomials = []
        if len(coefficients_list) > 0:
            for coefficients in coefficients_list:
                polynomials.append(polynomial.Polynomial(coefficients))
        self.polynomials: np.ndarray = np.asarray(polynomials)

    def __call__(self, *args):
        assert len(args) == len(self.polynomials)
        # args[0] is assumed to be the noise
        return sum((poly(arg) for poly, arg in zip(self.polynomials, args)))

    def __len__(self):
        return len(self.polynomials)

    def function_str(self, variable_names=None):
        assignment = []
        for poly, var in zip(self.polynomials, variable_names):
            coeffs = poly.coef
            this_assign = []
            for deg, c in enumerate(coeffs):
                if c != 0:
                    this_assign.append(
                        f"{round(c, 2)} {var}{f'**{deg}' if deg != 1 else ''}"
                    )
            assignment.append(" + ".join(this_assign))
        return " + ".join(assignment)
