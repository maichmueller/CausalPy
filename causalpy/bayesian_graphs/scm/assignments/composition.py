from typing import Tuple, Union, Optional, Collection

from .assignments import Assignment

import numpy as np


class CompositionAssignment(Assignment):
    r"""
    The Linear Assignment function of the form:
        f(X_S, N) = offset + \sum_{i \in S} a_i * X_i + noise_coeff * N
    """

    def __init__(self, pass_down_noise: bool, *assignment_funcs: Assignment):
        super().__init__()
        # All the nodes need to pass down the noise in either case, but the root node
        # decides whether the passed down noise is the real noise (pass_down_noise == True)
        # or just dummy 0 noise.
        # TODO: this brings up the question whether the noise wouldn't need to be handled differently
        #  overall, as 0 noise could also change the data evaluation as not always f(0) = 0.
        self.assignment_root_node = CompositionNode(True, assignment_funcs)
        self.assignment_root_node.pass_down_noise = pass_down_noise
        self.nr_assignments = len(assignment_funcs)
        self.noise_at_end = pass_down_noise

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
    def __init__(self, pass_down_noise: bool, assignment_funcs: Tuple[Assignment]):
        self.pass_down_noise = pass_down_noise
        self.assignment = assignment_funcs[0]
        remaining_assignments = assignment_funcs[1:]
        if len(remaining_assignments) > 0:
            self.next_node = CompositionNode(pass_down_noise, remaining_assignments)
        else:
            self.next_node = None

    def __call__(self, noise, *args):
        if self.next_node is not None:
            if self.pass_down_noise:
                return self.assignment(self.next_node(noise, *args))
            else:
                return self.assignment(noise, self.next_node(0, *args))
        else:
            return self.assignment(noise, *args)

    def __len__(self):
        if self.next_node is not None:
            return 1 + len(self.next_node)
        else:
            return 1

    def set_noise_passing(self, pass_down_noise: bool):
        self.pass_down_noise = pass_down_noise

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
