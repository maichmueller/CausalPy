from causalpy.bayesian_graphs.scm.assignments import Assignment
import numpy as np


class MaxAssignment(Assignment):
    r"""
    The Linear Assignment function of the form:
        f(X_S, N) = offset + \sum_{i \in S} a_i * X_i + noise_coeff * N
    """

    def __init__(self, coefficient, noise_factor=0):
        super().__init__()
        self.coefficient = coefficient
        self.noise_factor = noise_factor

    def __call__(self, noise, *args, **kwargs):
        args = self.parse_call_input(*args, **kwargs)
        return np.max(self.coefficient * args[0], 0) + self.noise_factor * noise

    def __len__(self):
        return 1

    def function_str(self, variable_names=None):
        coeff_s = (
            str(self.coefficient) if self.coefficient >= 0 else f"({self.coefficient})"
        )
        return f"max(0, {coeff_s} * {variable_names[0]})"


class IdentityAssignment(Assignment):
    r"""
    The Linear Assignment function of the form:
        f(X_S, N) = offset + \sum_{i \in S} a_i * X_i + noise_coeff * N
    """

    def __init__(self, coefficient, noise_factor=0):
        super().__init__()
        self.coefficient = coefficient
        self.noise_factor = noise_factor

    def __call__(self, noise, *args, **kwargs):
        args = self.parse_call_input(*args, **kwargs)
        return self.coefficient * args[0] + self.noise_factor * noise

    def __len__(self):
        return 1

    def function_str(self, variable_names=None):
        coeff_s = (
            str(self.coefficient) if self.coefficient >= 0 else f"({self.coefficient})"
        )
        return f"{coeff_s} * {variable_names[0]}"


class SignSqrtAssignment(Assignment):
    r"""
    The Linear Assignment function of the form:
        f(X_S, N) = offset + \sum_{i \in S} a_i * X_i + noise_coeff * N
    """

    def __init__(self, coefficient, noise_factor=0):
        super().__init__()
        self.coefficient = coefficient
        self.noise_factor = noise_factor

    def __call__(self, noise, *args, **kwargs):
        args = self.parse_call_input(*args, **kwargs)
        return (
            self.coefficient * np.sign(args[0]) * np.sqrt(np.abs(args[0]))
            + self.noise_factor * noise
        )

    def __len__(self):
        return 1

    def function_str(self, variable_names=None):
        arg = f"{self.coefficient} * {variable_names[0]}"
        return f"sign({arg}) * sqrt(|{arg}|)"


class SinAssignment(Assignment):
    r"""
    The Linear Assignment function of the form:
        f(X_S, N) = offset + \sum_{i \in S} a_i * X_i + noise_coeff * N
    """

    def __init__(self, coefficient, noise_factor=0):
        super().__init__()
        self.coefficient = coefficient
        self.noise_factor = noise_factor

    def __call__(self, noise, *args, **kwargs):
        args = self.parse_call_input(*args, **kwargs)
        return (
            self.coefficient * np.sin(2 * np.pi * args[0]) + self.noise_factor * noise
        )

    def __len__(self):
        return 1

    def function_str(self, variable_names=None):
        coeff_s = (
            str(self.coefficient) if self.coefficient >= 0 else f"({self.coefficient})"
        )
        arg = f"{coeff_s} * {variable_names[0]}"
        return f"sin(2*pi*{arg})"
