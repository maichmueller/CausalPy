from .assignments import Assignment, Assignment

from scipy.special import expit
from typing import Callable
import types
import numpy as np


class LinkerAssignment(Assignment):
    def __init__(self, linker_func: Callable, assignment_func: Assignment):
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


def sigmoid(x, *args, **kwargs):
    return expit(x, *args, **kwargs)


def identity(x):
    return x


class Scale:
    def __init__(self, scale):
        self.scale = np.array([scale])

    def __call__(self, *args, **kwargs):
        return self.scale @ args

