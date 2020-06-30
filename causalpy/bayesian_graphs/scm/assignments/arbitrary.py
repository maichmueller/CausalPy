from .assignments import Assignment

from scipy.special import expit
from typing import Callable
import types
import numpy as np
import torch


class ArbitraryAssignment(Assignment):
    def __init__(self, func: Callable, nr_vars: int):
        super().__init__()
        self.func = func
        self.nr_vars = nr_vars

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __len__(self):
        return self.nr_vars

    def function_str(self, variable_names=None):
        func_name = (
            type(self.func).__name__
            if not isinstance(self.func, types.FunctionType)
            else self.func.__name__
        )
        return f"{func_name}({variable_names}))"
