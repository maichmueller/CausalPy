import itertools
from typing import Optional, Callable, Type, Tuple, List, Union, Collection

import torch

from causalpy import Assignment, NoiseGenerator, SCM
from causalpy.neural_networks import CINN
from causalpy.neural_networks.utils import get_jacobian
from abc import ABC, abstractmethod
import numpy as np


class CINNFC(torch.nn.Module, Assignment):
    def __init__(
        self,
        dim_in: int,
        nr_layers: int = 0,
        nr_blocks: int = 1,
        strength: float = 0.0,
        device=None,
        **kwargs,
    ):
        super().__init__()
        Assignment.__init__(self)
        self.nr_layers = nr_layers
        self.dim_in = dim_in - 1
        self.strength = strength
        self.device = device
        # -1 bc additive noise
        self.cinn = CINN(
            dim=dim_in - 1,
            dim_condition=0,
            nr_layers=nr_layers,
            nr_blocks=nr_blocks,
            device="cuda",
        )

    def forward(self, *args, **kwargs):
        with torch.no_grad():
            noise = (
                torch.from_numpy(np.array(list(args), dtype=np.float32).T)
                .to(self.device)
                .reshape(-1, 1)
            )
            variables = (
                torch.from_numpy(np.array(list(kwargs.values()), dtype=np.float32).T)
                .squeeze(0)
                .to(self.device)
                .reshape(noise.shape[0], -1)
            )

            x = variables
            #             print(noise.shape)
            if self.dim_in > 0:
                base = torch.cat((noise, variables), dim=1).to(self.device)
            else:
                base = noise

            #             print("Baseshape", base.shape)

            if self.dim_in == 0:
                nonlin_part = noise
            else:
                ev = self.cinn(x)[:, 0].reshape(-1, 1)
                #                 print("EV", ev.shape)
                #                 print("NOise", noise.shape)
                nonlin_part = ev + noise

            #             print("nonlin", nonlin_part.shape)

            base_sum = torch.sum(base, dim=1).view(-1, 1)

            x = nonlin_part * self.strength + (1 - self.strength) * base_sum

            return x.cpu().numpy().reshape(-1)

    def __len__(self):
        raise 2

    def function_str(self, variable_names: Optional[Collection[str]] = None):
        return f"CINN({', '.join(variable_names)})"


def fc_net(
    dim_in: int,
    nr_layers: int,
    nr_hidden,
    strength: float,
    nr_blocks=1,
    seed=None,
    device=None,
):
    network_params = {
        "nr_layers": nr_layers,
        "nr_blocks": nr_blocks,
        "nr_hidden": nr_hidden,
        "strength": strength,
        "seed": seed,
        "device": device,
    }
    return CINNFC(dim_in=dim_in, **network_params).to(device)


def random_graphs(
    nr_variables: int = None,
    seed: int = None,
    p_con: float = 0.5,
    p_con_down: Callable = None,
    device=None,
):
    rng = np.random.default_rng(seed=seed)
    if nr_variables is None:
        nr_variables = rng.integers(2, 15)
    elif nr_variables < 2:
        raise ValueError("At least 2 variables needed to build a graph.")

    if p_con_down is None:

        def p_con_down(this_level, other_level):
            if this_level == other_level:
                return rng.random()
            else:
                return 1 / (other_level - this_level)

    variables = list(range(nr_variables))
    vars_copy = list(range(nr_variables))
    target = rng.choice(variables)

    parents = {v: [] for v in variables}
    var_names = (
        [f"X_{i}" for i in variables[0:target]]
        + ["Y"]
        + [f"X_{i - 1}" for i in variables[target + 1 :]]
    )

    nr_levels = rng.integers(1, nr_variables)
    var_to_level = dict()
    level_to_vars = dict()
    for level in range(nr_levels):
        if len(vars_copy) > 0:
            vars_in_level = rng.choice(np.arange(len(vars_copy)), size=1, replace=False)
            vars_in_level = vars_copy[: int(vars_in_level)]
            level_to_vars[level] = vars_in_level
            for v in vars_in_level:
                var_to_level[v] = level
                vars_copy.remove(v)

    for level in range(nr_levels):
        vars_in_level = level_to_vars[level]

        for next_level in range(level, nr_levels):
            for i, v in enumerate(vars_in_level):
                vars_in_next_level = level_to_vars[next_level]
                if level == next_level:
                    vars_in_next_level = vars_in_next_level[i + 1 :]

                for other_v in vars_in_next_level:
                    p = p_con * p_con_down(level, next_level)
                    is_child = rng.choice([0, 1], p=[1 - p, p])
                    if is_child:
                        parents[other_v].append(var_names[v])

    assignment_map = {}

    for var, name in zip(variables, var_names):

        assignment_map[name] = (
            parents[var],
            fc_net(
                dim_in=len(parents[var]) + 1,  # +1 because of noise variable
                **{
                    "nr_layers": 1,
                    "nr_blocks": rng.integers(10, 11),
                    "nr_hidden": 128,
                    "strength": rng.random() * (1 - 0.8) + 0.8,
                    "device": device,
                },
            ),
            NoiseGenerator("normal", scale=rng.random() * 5),
        )

    return SCM(assignment_map=assignment_map)
