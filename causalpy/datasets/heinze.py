import copy
from typing import Optional, Collection, Any, Dict, Tuple

from causalpy.bayesian_graphs.scm import (
    SCM,
    NoiseGenerator,
    Assignment,
    IdentityAssignment,
    MaxAssignment,
    SignSqrtAssignment,
    SinAssignment,
)
import networkx as nx
import pandas as pd
import numpy as np


class SumAssignment(Assignment):
    def __init__(self, *assignments, offset: float = 0.0):
        super().__init__()
        self.assignment = assignments
        self.offset = offset
        self.coefficients = np.ones(len(assignments))

    def __call__(self, noise, *args, **kwargs):
        args = self.parse_call_input(*args, **kwargs)
        return noise + self.coefficients @ args + self.offset

    def __len__(self):
        return len(self.coefficients)

    def function_str(self, variable_names: Optional[Collection[str]] = None):
        rep = "N"
        var_strs = [
            f"{assignment.function_str([var])}"
            for assignment, var in zip(self.assignment, variable_names[1:])
        ]
        if var_strs:
            rep += f" + {' + '.join(var_strs)}"
        return rep


class ProductAssignment(Assignment):
    def __init__(self, *assignments, offset: float = 0.0):
        super().__init__()
        self.assignment = assignments
        self.offset = offset
        self.coefficients = np.ones(len(assignments))

    def __call__(self, noise, *args, **kwargs):
        args = self.parse_call_input(*args, **kwargs)
        out = noise + self.offset
        if args:
            out += np.prod(args, axis=0)
        return out

    def __len__(self):
        return len(self.coefficients)

    def function_str(self, variable_names: Optional[Collection[str]] = None):
        rep = "N"
        var_strs = [
            f"{assignment.function_str([var])}"
            for assignment, var in zip(self.assignment, variable_names[1:])
        ]
        if var_strs:
            rep += f" + {' * '.join(var_strs)}"
        return rep


class HeinzeData:

    _possible_values_ = dict(
        sample_size=[100, 200, 500, 2000, 5000],
        target=[f"X_{i}" for i in range(6)],
        noise_df=[2, 3, 5, 10, 20, 50, 100],
        multiplicative=[True, False],
        shift=[True, False],
        meanshift=[0, 0.1, 0.2, 0.5, 1, 2, 5, 10],
        strength=[0, 0.1, 0.2, 0.5, 1, 2, 5, 10],
        mechanism=[
            IdentityAssignment,
            MaxAssignment,
            SignSqrtAssignment,
            SinAssignment,
        ],
        interventions=["all", "rand", "close"],
    )

    def __init__(
        self, config: Optional[Dict[str, Any]] = None, seed: Optional[int] = None
    ):
        self.seed = seed
        self.rng = np.random.default_rng(self.seed)
        self.config = (
            self.draw_config() if config is None else self.verify_config(config)
        )
        self.scm = self.get_scm()
        self.intervention_values = dict()

    def verify_config(self, config: Dict[str, Any]):
        for key in config.keys():
            if config[key] not in self._possible_values_[key]:
                raise ValueError(
                    f"Value '{config[key]}' of key '{key}' not within range of allowed values."
                )
        return config

    def draw_config(self):
        rng = np.random.default_rng(self.seed)
        poss_vals = self._possible_values_
        config = dict()

        # uniform draws
        for param in self._possible_values_.keys():
            config[param] = rng.choice(poss_vals[param])
        return config

    def get_scm(self, noise_seed: Optional[int] = None):
        config = self.config
        assignment_map = dict()

        def get_seed(i):
            if noise_seed is not None:
                return noise_seed + i
            return None

        binary_op_assignment = (
            ProductAssignment if config["multiplicative"] else SumAssignment
        )
        mechanism = config["mechanism"]
        coeffs = dict()
        coeffs[(0, 1)] = 1
        coeffs[(0, 2)] = 1
        coeffs[(1, 2)] = -1
        coeffs[(2, 3)] = -1
        coeffs[(3, 5)] = -1
        coeffs[(2, 5)] = 1
        coeffs[(4, 5)] = 1
        df = config["noise_df"]
        assignment_map["X_0"] = (
            [],
            binary_op_assignment(),
            NoiseGenerator("standard_t", df=df, seed=get_seed(0)),
        )
        assignment_map["X_1"] = (
            ["X_0"],
            binary_op_assignment(mechanism(coefficient=coeffs[0, 1])),
            NoiseGenerator("standard_t", df=df, seed=get_seed(1)),
        )
        assignment_map["X_2"] = (
            ["X_0", "X_1"],
            binary_op_assignment(
                mechanism(coefficient=coeffs[0, 2]), mechanism(coefficient=coeffs[1, 2])
            ),
            NoiseGenerator("standard_t", df=df, seed=get_seed(2)),
        )
        assignment_map["X_3"] = (
            ["X_2"],
            binary_op_assignment(mechanism(coefficient=coeffs[2, 3])),
            NoiseGenerator("standard_t", df=df, seed=get_seed(3)),
        )
        assignment_map["X_4"] = (
            [],
            binary_op_assignment(),
            NoiseGenerator("standard_t", df=df, seed=get_seed(4)),
        )
        assignment_map["X_5"] = (
            ["X_3", "X_2", "X_4"],
            binary_op_assignment(
                mechanism(coefficient=coeffs[3, 5]),
                mechanism(coefficient=coeffs[2, 5]),
                mechanism(coefficient=coeffs[4, 5]),
            ),
            NoiseGenerator("standard_t", df=df, seed=get_seed(5)),
        )
        return SCM(assignment_map)

    def set_intervention_values(self, intervention_number: int = 1):
        try:
            return self.intervention_values[intervention_number]
        except KeyError:

            config = self.config
            interv_setting = config["interventions"]
            target = config["target"]
            meanshift = config["meanshift"]
            scale = config["strength"]

            if interv_setting == "all":
                variables = [var for var in self.scm.graph.nodes if var != target]
                values = (
                    self.rng.standard_t(size=len(variables), df=config["noise_df"])
                    * scale
                    + meanshift
                )
            elif interv_setting == "rand":
                parents = list(self.scm[target][0])
                descendants = list(
                    nx.algorithms.dag.descendants(self.scm.graph, target)
                )

                parent = [self.rng.choice(parents)] if parents else []
                descendant = [self.rng.choice(descendants)] if descendants else []
                variables = parent + descendant
                values = (
                    self.rng.standard_t(size=len(variables), df=config["noise_df"])
                    * scale
                    + meanshift
                )
            else:
                parents = list(self.scm[target][0])
                children = list(self.scm.graph.successors(target))
                parent = [self.rng.choice(parents)] if parents else []
                child = [self.rng.choice(children)] if children else []
                variables = parent + child
                values = (
                    self.rng.standard_t(size=len(variables), df=config["noise_df"])
                    * scale
                    + meanshift
                )
            self.intervention_values[intervention_number] = variables, values

    def set_intervention(self, intervention_number: int):
        variables, values = self.intervention_values[intervention_number]
        if self.config["shift"]:
            new_assignments = {
                var: {"assignment": copy.deepcopy(self.scm[var][1]["assignment"])}
                for var in variables
            }
            for (var, items), value in zip(new_assignments.items(), values):
                items["assignment"].coefficient = value
            self.scm.intervention(interventions=new_assignments)
        else:
            self.scm.do_intervention(variables, values)

    def sample(self) -> Tuple[pd.DataFrame, str, np.ndarray]:
        self.set_intervention_values(1)
        self.set_intervention_values(2)

        sample_size = self.config["sample_size"]
        obs = [self.scm.sample(sample_size)]
        envs = [0] * sample_size
        vars = sorted(self.scm.get_variables())
        for i in range(1, 3):
            self.set_intervention(i)
            obs.append(self.scm.sample(sample_size)[vars])
            self.scm.undo_intervention()
            envs += [i] * sample_size
        obs = pd.concat(obs, sort=True)
        envs = np.array(envs)
        return obs, self.config["target"], envs
