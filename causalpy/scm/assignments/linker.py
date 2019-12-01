from .base import BaseAssignment, Assignment

import numpy as np
from typing import Callable


class LinkerAssignment(BaseAssignment):
    def __init__(self, linker_func: Callable, assignment_func: Assignment):
        self.linker = linker_func
        self.assign_func = assignment_func

    def __call__(self, *args, **kwargs):
        return self.linker(self.assign_func(*args, **kwargs))

    def __len__(self):
        return len(self.assign_func)

    def function_str(self, variable_names=None):
        return f"{self.linker.__name__}({self.assign_func.str(variable_names)})"
