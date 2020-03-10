from causalpy import (
    SCM,
    LinearAssignment,
    NoiseGenerator,
    DiscreteNoise,
    PolynomialAssignment,
)
import pandas as pd
import numpy as np
import torch


def build_scm_minimal(seed=0):
    cn = SCM(
        assignment_map={
            "X_0": (
                [],
                LinearAssignment(1),
                NoiseGenerator("binomial", n=1, p=0.5, seed=seed),
            ),
            "Y": (
                ["X_0"],
                LinearAssignment(1, 0, 1),
                NoiseGenerator("standard_normal", seed=seed + 4),
            ),
        },
        variable_tex_names={"X_0": "$X_0$"},
    )
    return cn


def build_scm_basic_discrete(seed=0):
    cn = SCM(
        assignment_map={
            "X_0": ([], LinearAssignment(1), DiscreteNoise([0, 1], seed=seed),),
            "X_1": ([], LinearAssignment(1), DiscreteNoise([0, 1], seed=seed + 1),),
            "X_2": ([], LinearAssignment(1), DiscreteNoise([0, 1, 2], seed=seed + 1),),
            "Y": (
                ["X_0", "X_1"],
                LinearAssignment(1, 0.0, 1, -1),
                NoiseGenerator("standard_normal", seed=seed + 4),
            ),
        },
        variable_tex_names={"X_0": "$X_0$", "X_1": "$X_1$", "X_2": "$X_2$"},
    )
    return cn


def build_scm_basic(seed=0):
    cn = SCM(
        assignment_map={
            "X_0": (
                [],
                LinearAssignment(1),
                NoiseGenerator("standard_normal", seed=seed),
            ),
            "X_1": (
                [],
                LinearAssignment(1),
                NoiseGenerator("standard_normal", seed=seed + 1),
            ),
            "X_2": (
                [],
                LinearAssignment(1),
                NoiseGenerator("standard_normal", seed=seed + 1),
            ),
            "Y": (
                ["X_0", "X_1"],
                LinearAssignment(1, 0.0, 1, -1),
                NoiseGenerator("standard_normal", seed=seed + 4),
            ),
        },
        variable_tex_names={"X_0": "$X_0$", "X_1": "$X_1$"},
    )
    return cn


def build_scm_linandpoly(seed=0):
    cn = SCM(
        assignment_map={
            "X_0": (
                [],
                LinearAssignment(1),
                NoiseGenerator("standard_normal", seed=seed),
            ),
            "X_1": (
                ["X_0"],
                LinearAssignment(1, 1, 2),
                NoiseGenerator("standard_normal", seed=seed + 1),
            ),
            "X_2": (
                ["X_0", "X_1"],
                LinearAssignment(1, 1, 0.5, 2),
                NoiseGenerator("standard_normal", seed=seed + 2),
            ),
            "X_3": (
                ["X_1", "X_2"],
                PolynomialAssignment([0, 1], [0, 1, 0.5], [0, 0, 2]),
                NoiseGenerator("standard_normal", seed=seed + 3),
            ),
            "Y": (
                ["X_0", "X_2"],
                PolynomialAssignment([0, 1], [0, 0, 1.5], [0, 2]),
                NoiseGenerator("standard_normal", seed=seed + 4),
            ),
        },
        variable_tex_names={
            "X_0": "$X_0$",
            "X_1": "$X_1$",
            "X_2": "$X_2$",
            "X_3": "$X_3$",
        },
    )
    return cn


def build_scm_exponential(seed=0):
    cn = SCM(
        assignment_map={
            "X_0": ([], LinearAssignment(1), NoiseGenerator("exponential", seed=seed),),
            "X_1": (
                ["X_0"],
                LinearAssignment(1, 1, 2),
                NoiseGenerator("exponential", scale=0.4, seed=seed + 1),
            ),
            "X_2": (
                ["X_0", "X_1"],
                LinearAssignment(1, 1, 0.5, 2),
                NoiseGenerator("exponential", scale=0.4, seed=seed + 2),
            ),
            "X_3": (
                ["X_1", "X_2"],
                PolynomialAssignment([0, 1], [0, 1, 0.5], [0, 0, 2]),
                NoiseGenerator("standard_normal", seed=seed + 3),
            ),
            "Y": (
                ["X_0", "X_2"],
                PolynomialAssignment([0, 1], [0, 0, 1.5], [0, 2]),
                NoiseGenerator("exponential", scale=0.1, seed=seed + 4),
            ),
        },
        variable_tex_names={
            "X_0": "$X_0$",
            "X_1": "$X_1$",
            "X_2": "$X_2$",
            "X_3": "$X_3$",
        },
    )
    return cn


def build_scm_medium(seed=0):
    scale = .5
    cn = SCM(
        assignment_map={
            "X_0": (
                [],
                LinearAssignment(1),
                NoiseGenerator("normal", loc=0, scale=scale, seed=seed),
            ),
            "X_1": (
                ["X_0"],
                LinearAssignment(1, 1, 1),
                NoiseGenerator("normal", loc=0, scale=scale, seed=seed + 1),
            ),
            "X_2": (
                ["X_0"],
                LinearAssignment(1, 1, 0.8),
                NoiseGenerator("normal", loc=0, scale=scale, seed=seed + 2),
            ),
            "X_3": (
                ["X_1", "X_2"],
                LinearAssignment(1, 0, 0.3, 0.4),
                NoiseGenerator("normal", loc=0, scale=scale, seed=seed + 3),
            ),
            "Y": (
                ["X_3", "X_0"],
                LinearAssignment(1, 0.6, 1, -1),
                NoiseGenerator("normal", loc=0, scale=scale, seed=seed + 4),
            ),
            "X_4": (
                ["Y"],
                LinearAssignment(1, 1.2, -0.7),
                NoiseGenerator("normal", loc=0, scale=scale, seed=seed + 5),
            ),
            "X_5": (
                ["X_3", "Y"],
                LinearAssignment(1, 0.5, -0.7, 0.4),
                NoiseGenerator("normal", loc=0, scale=scale, seed=seed + 6),
            ),
        },
        variable_tex_names={
            "X_0": "$X_0$",
            "X_1": "$X_1$",
            "X_2": "$X_2$",
            "X_3": "$X_3$",
            "X_4": "$X_4$",
            "X_5": "$X_5$",
        },
    )
    return cn


def build_scm_large(seed=0):
    scale = .5
    cn = SCM(
        assignment_map={
            "X_0": (
                [],
                LinearAssignment(1),
                NoiseGenerator("normal", loc=0, scale=scale, seed=seed)
            ),
            "X_1": (
                ["X_0"],
                LinearAssignment(1, 1, 1),
                NoiseGenerator("normal", loc=0, scale=scale, seed=seed + 1),
            ),
            "X_2": (
                ["X_0"],
                LinearAssignment(1, 1, 0.8),
                NoiseGenerator("normal", loc=0, scale=scale, seed=seed + 2),
            ),
            "X_3": (
                ["X_1", "X_2"],
                LinearAssignment(1, 0, 0.3, 0.4),
                NoiseGenerator("normal", loc=0, scale=scale, seed=seed + 3),
            ),
            "Y": (
                ["X_3", "X_0"],
                LinearAssignment(1, 0.67, 1, -1),
                NoiseGenerator("normal", loc=0, scale=scale, seed=seed + 4),
            ),
            "X_4": (
                ["Y"],
                LinearAssignment(1, 1.2, -0.7),
                NoiseGenerator("normal", loc=0, scale=scale, seed=seed + 5),
            ),
            "X_5": (
                ["X_3", "Y"],
                LinearAssignment(1, 0.5, -0.7, 0.4),
                NoiseGenerator("normal", loc=0, scale=scale, seed=seed + 6),
            ),
            "X_6": (
                ["X_5", "X_4"],
                LinearAssignment(1, 0.5, -1.7, 1.4),
                NoiseGenerator("normal", loc=0, scale=scale, seed=seed + 7),
            ),
            "X_7": (
                [],
                LinearAssignment(1, 0.5),
                NoiseGenerator("normal", loc=0, scale=scale, seed=seed + 8),
            ),
            "X_8": (
                ["Y"],
                LinearAssignment(1, 0.5, 0.1),
                NoiseGenerator("normal", loc=0, scale=scale, seed=seed + 9),
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
            "X_8": "$X_8$",
        },
    )
    return cn


def generate_data_from_scm(
    scm,
    target_var=None,
    markovblanket_interv_only=True,
    countify=False,
    sample_size=100,
    seed=None,
):
    # scm = simulate(nr_genes, 2, seed=seed)
    rng = np.random.default_rng(seed + 10)
    # scm = build_scm_minimal(seed)
    variables = sorted(scm.get_variables())
    if target_var is None:
        target_var = rng.choice(variables[len(variables) // 2 :])
    other_variables = sorted(scm.get_variables())
    other_variables.remove(target_var)
    target_parents = sorted(scm.graph.predecessors(target_var))

    possible_parents = sorted(scm.get_variables())
    possible_parents.remove(target_var)

    scm.reseed(seed)
    environments = []
    sample_size_per_env = sample_size
    sample = scm.sample(sample_size_per_env)
    sample_data = [sample[variables]]
    environments += [0] * sample_size_per_env
    if markovblanket_interv_only:
        interv_variables = set(target_parents)
        for child in scm.graph.successors(target_var):
            child_pars = set(scm.graph.predecessors(child))
            child_pars = child_pars.union([child])
            child_pars.remove(target_var)
            interv_variables = interv_variables.union(child_pars)
    else:
        interv_variables = other_variables

    # perform interventions on selected variables
    for parent in interv_variables:
        interv_value = rng.choice([-1, 1]) * rng.random(1) * 3
        # interv_value = 0
        scm.do_intervention([parent], [interv_value])
        print(
            f"Environment {environments[-1] + 1}: Intervention on variable {parent} for value {interv_value}."
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
