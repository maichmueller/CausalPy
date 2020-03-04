from .assignments import Assignment

import numpy as np


class PeriodicAssignment(Assignment):
    r"""
    The Linear Assignment function of the form:
        f(X_S, N) = \sin a_i * X_i + noise_coeff * N
    """
    def __init__(self, noise_factor, offset=0, *coefficients):
        super().__init__()
        self.noise_factor = noise_factor
        self.offset = offset
        self.coefficients = (
            np.asarray(coefficients) if len(coefficients) > 0 else np.array([])
        )

    def __call__(self, noise, *args, **kwargs):
        args = self.parse_call_input(*args, **kwargs)
        return self.offset + self.noise_factor * noise + self.coefficients @ args

    def __len__(self):
        return 1 + len(self.coefficients)

    def function_str(self, variable_names=None):
        rep = f"{f'{round(self.offset, 2)} + ' if self.offset != 0 else ''}{round(self.noise_factor, 2)} N"
        for i, c in enumerate(self.coefficients):
            if c != 0:
                rep += f" + {round(c, 2)} {variable_names[i + 1]}"
        return rep

    @staticmethod
    def random_factory(nr_variables, seed=None):
        rs = np.random.default_rng(seed)
        offset = rs.normal()
        coeffs = rs.normal(loc=0, scale=1, size=nr_variables)
        return LinearAssignment(1, offset, *coeffs)
