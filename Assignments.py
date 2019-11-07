import numpy as np
from numpy.polynomial import polynomial
from typing import List, Union, Dict
from abc import ABC, abstractmethod


class BaseAssignment(ABC):
    @abstractmethod
    def __call__(self, noise, *args, **kwargs):
        raise NotImplementedError("Assignment subclasses must provide '__call__' method.")

    @abstractmethod
    def to_str(self, *args, **kwargs):
        raise NotImplementedError("Assignment subclasses must provide 'to_str' method.")

    def __str__(self):
        return self.to_str()


class LinearAssignment(BaseAssignment):
    def __init__(self, noise_factor, offset=0, *coefficients):
        self.noise_factor = noise_factor
        self.offset = offset
        self.coefficients = np.asarray(coefficients) if len(coefficients) > 0 else np.array([])

    def __call__(self, noise, *args, **kwargs):
        return self.offset + self.noise_factor * noise + self.coefficients @ args

    def to_str(self, variable_names=None):
        if variable_names is None:
            variable_names = [f"x_{i}" for i in range(len(self.coefficients))]
        variable_names = ["N"] + variable_names
        rep = f"{round(self.offset, 2)} + {round(self.noise_factor, 2)} N"
        for i, c in enumerate(self.coefficients):
            if c != 0:
                rep += f" + {round(c, 2)} {variable_names[i+1]}"
        prefix = f"f({', '.join(variable_names)}) = "
        return prefix + rep


class PolynomialAssignment(BaseAssignment):
    def __init__(self,
                 *coefficients_list: List[float]
                 ):
        self.polynomials = []
        if len(coefficients_list) > 0:
            for coefficients in coefficients_list:
                self.polynomials.append(polynomial.Polynomial(coefficients))
        self.polynomials = np.asarray(self.polynomials)

    def __call__(self, *args):
        assert (len(args) == len(self.polynomials))
        # args[0] is assumed to be the noise
        return sum((poly(arg) for poly, arg in zip(self.polynomials, args)))

    def to_str(self, variable_names=None):
        if variable_names is None:
            variable_names = [f"x_{i}" for i in range(len(self.polynomials))]
        variable_names = ["N"] + variable_names
        assignment = []
        for poly, var in zip(self.polynomials, variable_names):
            coeffs = poly.coef
            this_assign = []
            for deg, c in enumerate(coeffs):
                if c != 0:
                    this_assign.append(f"{round(c,2)} {var}{'**'+str(deg) if deg != 1 else ''}")
            assignment.append(" + ".join(this_assign))
        prefix = f"f({', '.join(variable_names)}) = "
        return prefix + " + ".join(assignment)

