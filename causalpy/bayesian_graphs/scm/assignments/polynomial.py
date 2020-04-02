from .assignments import Assignment

import numpy as np
from numpy.polynomial import polynomial
from typing import List, Collection, Union, Optional


class PolynomialAssignment(Assignment):
    r"""
    A polynomial assignment function of the form:

    .. math:: f(X_S, N) = c_{noise} * N + offset + \sum_{i \in S} {\sum^{p_i}}_{k = 0} c_{ik} * X_i^k

    """

    def __init__(self, *coefficients_list: Collection[Union[int, float]]):
        super().__init__()
        polynomials = []
        if len(coefficients_list) > 0:
            for coefficients in coefficients_list:
                polynomials.append(polynomial.Polynomial(coefficients))
        self.polynomials = polynomials

    def __call__(self, noise, *args, **kwargs):
        args = tuple([noise, *self.parse_call_input(*args, **kwargs)])
        assert len(args) == len(self.polynomials)
        # args[0] is assumed to be the noise
        return sum((poly(arg) for poly, arg in zip(self.polynomials, args)))

    def __len__(self):
        return len(self.polynomials)

    def function_str(self, variable_names: Optional[Collection[str]] = None):
        rep = ""
        assignment = []
        const = 0
        for poly, var in zip(self.polynomials, variable_names):
            for deg, c in enumerate(poly.coef):
                if deg > 0:
                    break
                const += c
        for poly, var in zip(self.polynomials, variable_names):
            coeffs = poly.coef
            this_assign = []
            for deg, c in enumerate(coeffs):
                if deg == 0:
                    continue
                if c != 0:
                    this_assign.append(
                        f"{round(c, 2)} {var}{f'**{deg}' if deg != 1 else ''}"
                    )
            assignment.append(" + ".join(this_assign))
        if const != 0:
            rep += str(const)
        rep += " + ".join(assignment)
        return rep

    @staticmethod
    def random_factory(nr_variables, seed=None):
        rs = np.random.default_rng(seed)
        coeffs = []
        for n in range(nr_variables + 1):
            deg = rs.integers(1, 6)  # allows degrees d <= 5
            coeffs.append(np.random.normal(loc=0, scale=0.1, size=deg))
        return PolynomialAssignment(*coeffs)
