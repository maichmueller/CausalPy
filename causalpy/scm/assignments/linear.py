from .assignmentbase import BaseAssignment

import numpy as np


class LinearAssignment(BaseAssignment):
    def __init__(self, noise_factor, offset=0, *coefficients):
        self.noise_factor = noise_factor
        self.offset = offset
        self.coefficients = (
            np.asarray(coefficients) if len(coefficients) > 0 else np.array([])
        )

    def __call__(self, noise, *args):
        return self.offset + self.noise_factor * noise + self.coefficients @ args

    def __len__(self):
        return 1 + len(self.coefficients)

    def function_str(self, variable_names=None):
        rep = f"{f'{round(self.offset, 2)} + ' if self.offset != 0 else ''}{round(self.noise_factor, 2)} N"
        for i, c in enumerate(self.coefficients):
            if c != 0:
                rep += f" + {round(c, 2)} {variable_names[i + 1]}"
        return rep
