from .assignments import Assignment, Assignment

from scipy.special import expit
from typing import Callable
import types
import numpy as np


class LinkerAssignment(Assignment):
    def __init__(self, linker_func: Callable, assignment_func: Assignment):
        super().__init__()
        self.linker = linker_func
        self.assign_func = assignment_func

    def __call__(self, *args, **kwargs):
        return self.linker(self.assign_func(*args, **kwargs))

    def __len__(self):
        return len(self.assign_func)

    def function_str(self, variable_names=None):
        linker_name = (
            type(self.linker).__name__
            if not isinstance(self.linker, types.FunctionType)
            else self.linker.__name__
        )
        return f"{linker_name}({self.assign_func.function_str(variable_names)})"


def scaler(func, x=1, y=1):
    def scaled(data_in, **kwargs):
        return y * func(data_in / x, **kwargs)

    return scaled


def sigmoid(x, *args, **kwargs):
    return expit(x, *args, **kwargs)


def tanh(x, *args, **kwargs):
    return (x / (1 + np.abs(x)) + 1) / 2


def identity(x):
    return x
