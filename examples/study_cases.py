from typing import Callable

from causalpy import (
    SCM,
    LinearAssignment,
    NoiseGenerator,
    DiscreteNoise,
    PolynomialAssignment,
    LinkerAssignment,
)
import pandas as pd
import numpy as np
import torch


def study_scm(seed=0, coeffs_by_var=None, noise_dists=None):
    coeffs = {
        "X_0": [],
        "X_1": [[0, 1]],
        "X_2": [[0, 1]],
        "X_3": [],
        "X_4": [[0, 1], [0, 1]],
        "X_5": [],
        "X_6": [[0, 1], [0, 1]],
        "X_7": [[0, 1]],
        "X_8": [],
        "Y": [[0, 1], [0, 1], [0, 1]],
    }
    if coeffs_by_var is not None:
        coeffs.update(coeffs_by_var)
    noises = {
        "X_0": {"dist": "standard_normal", "kwargs": {}},
        "X_1": {"dist": "standard_normal", "kwargs": {}},
        "X_2": {"dist": "standard_normal", "kwargs": {}},
        "X_3": {"dist": "standard_normal", "kwargs": {}},
        "X_4": {"dist": "standard_normal", "kwargs": {}},
        "X_5": {"dist": "standard_normal", "kwargs": {}},
        "X_6": {"dist": "standard_normal", "kwargs": {}},
        "X_7": {"dist": "standard_normal", "kwargs": {}},
        "X_8": {"dist": "standard_normal", "kwargs": {}},
        "Y": {"dist": "standard_normal", "kwargs": {}},
    }
    if noise_dists is not None:
        noises.update(noise_dists)
    cn = SCM(
        assignment_map={
            "X_0": (  # ancestor of target
                [],
                PolynomialAssignment([0, 1], *coeffs["X_0"]),
                NoiseGenerator(
                    noises["X_0"]["dist"], **noises["X_0"]["kwargs"], seed=seed + 1
                ),
            ),
            "X_1": (  # parent of target
                ["X_0"],
                PolynomialAssignment([0, 1], *coeffs["X_1"]),
                NoiseGenerator(
                    noises["X_1"]["dist"], **noises["X_1"]["kwargs"], seed=seed + 2
                ),
            ),
            "X_2": (  # parent of target
                ["X_0"],
                PolynomialAssignment([0, 1], *coeffs["X_2"]),
                NoiseGenerator(
                    noises["X_2"]["dist"], **noises["X_2"]["kwargs"], seed=seed + 3
                ),
            ),
            "X_3": (  # child of target
                [],
                PolynomialAssignment([0, 1], *coeffs["X_3"]),
                NoiseGenerator(
                    noises["X_3"]["dist"], **noises["X_3"]["kwargs"], seed=seed + 5
                ),
            ),
            "Y": (  # target
                ["X_1", "X_2", "X_3"],
                PolynomialAssignment([0, 1], *coeffs["Y"]),
                NoiseGenerator(
                    noises["Y"]["dist"], **noises["Y"]["kwargs"], seed=seed + 4
                ),
            ),
            "X_4": (  # descendant of target
                ["Y", "X_3"],
                PolynomialAssignment([0, 1], *coeffs["X_4"]),
                NoiseGenerator(
                    noises["X_4"]["dist"], **noises["X_4"]["kwargs"], seed=seed + 6
                ),
            ),
            "X_5": (  # spouse of target
                [],
                PolynomialAssignment([0, 1], *coeffs["X_5"]),
                NoiseGenerator(
                    noises["X_5"]["dist"], **noises["X_5"]["kwargs"], seed=seed + 7
                ),
            ),
            "X_6": (  # child of target
                ["Y", "X_5"],
                PolynomialAssignment([0, 1], *coeffs["X_6"]),
                NoiseGenerator(
                    noises["X_6"]["dist"], **noises["X_6"]["kwargs"], seed=seed + 8
                ),
            ),
            "X_7": (  # descendant of target
                ["X_4"],
                PolynomialAssignment([0, 1], *coeffs["X_7"]),
                NoiseGenerator(
                    noises["X_7"]["dist"], **noises["X_7"]["kwargs"], seed=seed + 7
                ),
            ),
            "X_8": (  # dummy
                [],
                PolynomialAssignment([0, 1], *coeffs["X_8"]),
                NoiseGenerator(
                    noises["X_8"]["dist"], **noises["X_8"]["kwargs"], seed=seed + 7
                ),
            ),
        },
        variable_tex_names={
            "X_0": "$X_0$",
            "X_1": "$X_1$",
            "X_2": "$X_2$",
            "X_3": "$X_3$",
            "X_4": "$X_4$",
            "X_5": "$X_5$",
            "X_6": "$X_6$",
            "X_7": "$X_7$",
        },
    )
    return cn


def generate_data_from_scm(
    scm,
    target_var=None,
    intervention_reach="markov",
    intervention_style="do",
    strength=5,
    fix_strength=False,
    countify=False,
    sample_size=100,
    seed=None,
):
    # scm = simulate(nr_genes, 2, seed=seed)
    rng = np.random.default_rng(seed)
    # scm = build_scm_minimal(seed)
    variables = sorted(scm.get_variables())
    if target_var is None:
        target_var = rng.choice(variables[len(variables) // 2 :])
    other_variables = sorted(scm.get_variables())
    other_variables.remove(target_var)
    target_parents = sorted(scm.graph.predecessors(target_var))

    possible_parents = sorted(scm.get_variables())
    possible_parents.remove(target_var)
    if seed is not None:
        scm.reseed(seed)
    environments = []
    sample_size_per_env = sample_size
    sample = scm.sample(sample_size_per_env)
    sample_data = [sample[variables]]
    environments += [0] * sample_size_per_env
    if intervention_reach == "markov":
        interv_variables = set(target_parents)
        for child in scm.graph.successors(target_var):
            child_pars = set(scm.graph.predecessors(child))
            child_pars = child_pars.union([child])
            child_pars.remove(target_var)
            interv_variables = interv_variables.union(child_pars)

    elif intervention_reach == "parents":
        interv_variables = set(target_parents)

    elif intervention_reach == "children":
        interv_variables = set([])
        for child in scm.graph.successors(target_var):
            child_pars = set(scm.graph.predecessors(child))
            child_pars = child_pars.union([child])
            child_pars.remove(target_var)
            interv_variables = interv_variables.union(child_pars)
    else:
        interv_variables = other_variables

    # perform interventions on selected variables
    for var in interv_variables:
        if fix_strength:
            interv_value = rng.choice([-1, 1]) * strength
        else:
            interv_value = rng.choice([-1, 1]) * rng.random(1) * strength

        if intervention_style == "do":

            scm.do_intervention([var], [interv_value])
            print(
                f"Environment {environments[-1] + 1}: Do-Intervention on variable {var} for value {interv_value}."
            )

        elif intervention_style == "meanshift":
            old_assignment = scm[var][1][scm.function_key]
            new_assignment = LinkerAssignment(
                lambda x: x + interv_value, old_assignment
            )
            scm.intervention({var: (None, new_assignment, None)})
            print(
                f"Environment {environments[-1] + 1}: Shift-Intervention on variable {var} for value {interv_value}."
            )

        elif intervention_style == "scaling":
            old_assignment = scm[var][1][scm.function_key]
            new_assignment = LinkerAssignment(
                lambda x: x * interv_value, old_assignment
            )
            scm.intervention({var: (None, new_assignment, None)})
            print(
                f"Environment {environments[-1] + 1}: Scale-Intervention on variable {var} for value {interv_value}."
            )
        else:
            raise ValueError(
                f"intervention style '{intervention_style}' not recognized "
            )

        sample_data.append(scm.sample(sample_size_per_env))
        environments += [environments[-1] + 1] * sample_size_per_env
        scm.undo_intervention()
    data = pd.concat(sample_data, sort=True)[variables].reset_index(drop=True)

    if countify:
        data = pd.DataFrame(
            np.random.poisson(
                torch.nn.Softplus(beta=1)(torch.as_tensor(data.to_numpy())).numpy()
            ),
            columns=data.columns,
        )
        # data += np.random.normal(0, 0.1, size=data.shape)

    environments = np.array(environments)
    print(scm)
    print("Target Variable:", target_var)
    print("Actual Parents:", ", ".join(target_parents))
    print("Candidate Parents:", ", ".join(possible_parents))

    return (data, environments, scm, possible_parents, target_parents)
