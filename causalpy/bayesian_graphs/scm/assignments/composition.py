from typing import Tuple, Union, Optional, Collection

from .assignments import Assignment

import numpy as np


class CompositionAssignment(Assignment):
    r"""
    The Linear Assignment function of the form:
        f(X_S, N) = offset + \sum_{i \in S} a_i * X_i + noise_coeff * N
    """

    def __init__(self, *assignment_funcs: Assignment):
        super().__init__()
        self.assignment_root_node = CompositionNode(assignment_funcs)
        self.nr_assignments = len(assignment_funcs)

    def __call__(self, noise: Union[float, np.ndarray], *args, **kwargs):
        args = self.parse_call_input(*args, **kwargs)
        self.assignment_root_node(noise, *args)
        return

    def __len__(self):
        return len(self.assignment_root_node)

    def function_str(self, variable_names=None):
        return self.assignment_root_node.function_str(variable_names=variable_names)

    # @staticmethod
    # def random_factory(nr_variables: int, seed=None):
    #     rs = np.random.default_rng(seed)
    #     offset = rs.normal()
    #     coeffs = rs.normal(loc=0, scale=1, size=nr_variables)
    #     # return LinearAssignment(1, offset, *coeffs)


class CompositionNode:
    def __init__(self, assignment_funcs: Tuple[Assignment]):
        self.assignment = assignment_funcs[0]
        remaining_assignments = assignment_funcs[1:]
        if len(remaining_assignments) > 0:
            self.next_node = CompositionNode(remaining_assignments)
        else:
            self.next_node = None

    def __call__(self, noise, *args):
        if self.next_node is not None:
            return self.assignment(self.next_node(noise, *args))
        else:
            return self.assignment(noise, *args)

    def __len__(self):
        if self.next_node is not None:
            return 1 + len(self.next_node)
        else:
            return 1

    def function_str(self, variable_names: Optional[Collection[str]] = None):
        if self.next_node is not None:
            rep = self.assignment.function_str(
                variable_names=[
                    self.next_node.function_str(variable_names=variable_names)
                ]
            )
        else:
            rep = self.assignment.function_str(variable_names=variable_names)
        return rep
