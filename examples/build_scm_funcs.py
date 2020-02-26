
from causalpy import SCM, LinearAssignment, NoiseGenerator, DiscreteNoise, PolynomialAssignment


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
                NoiseGenerator("standard_normal", seed=seed+1),
            ),
            "X_2": (
                ["X_0", "X_1"],
                LinearAssignment(1, 1, .5, 2),
                NoiseGenerator("standard_normal", seed=seed+2),
            ),
            "X_3": (
                ["X_1", "X_2"],
                PolynomialAssignment([0, 1], [0, 1, 0.5], [0, 0, 2]),
                NoiseGenerator("standard_normal", seed=seed+3),
            ),
            "Y": (
                ["X_0", "X_2"],
                PolynomialAssignment([0, 1], [0, 0, 1.5], [0, 2]),
                NoiseGenerator("standard_normal", seed=seed+4),
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
            "X_0": (
                [],
                LinearAssignment(1),
                NoiseGenerator("exponential", seed=seed),
            ),
            "X_1": (
                ["X_0"],
                LinearAssignment(1, 1, 2),
                NoiseGenerator("exponential", scale=.4, seed=seed+1),
            ),
            "X_2": (
                ["X_0", "X_1"],
                LinearAssignment(1, 1, .5, 2),
                NoiseGenerator("exponential", scale=.4, seed=seed+2),
            ),
            "X_3": (
                ["X_1", "X_2"],
                PolynomialAssignment([0, 1], [0, 1, 0.5], [0, 0, 2]),
                NoiseGenerator("standard_normal", seed=seed+3),
            ),
            "Y": (
                ["X_0", "X_2"],
                PolynomialAssignment([0, 1], [0, 0, 1.5], [0, 2]),
                NoiseGenerator("exponential", scale=.1, seed=seed+4),
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
    cn = SCM(
        assignment_map={
            "X_0": (
                [],
                LinearAssignment(1),
                NoiseGenerator("standard_normal", seed=seed),
            ),
            "X_1": (
                ["X_0"],
                LinearAssignment(1, 1, 1),
                NoiseGenerator("standard_normal", seed=seed + 1),
            ),
            "X_2": (
                ["X_0", "X_1"],
                LinearAssignment(1, 1, 0.8, -1.2),
                NoiseGenerator("standard_normal", seed=seed + 2),
            ),
            "X_3": (
                ["X_1", "X_2"],
                LinearAssignment(1, 0, 0.3, 0.4),
                NoiseGenerator("standard_normal", seed=seed + 3),
            ),
            "Y": (
                ["X_3", "X_0"],
                LinearAssignment(1, 0.67, 1, -1),
                NoiseGenerator("standard_normal", seed=seed + 4),
            ),
            "X_4": (
                ["Y"],
                LinearAssignment(1, 1.2, -0.7),
                NoiseGenerator("standard_normal", seed=seed + 5),
            ),
            "X_5": (
                ["X_3", "Y"],
                LinearAssignment(1, 0.5, -0.7, 0.4),
                NoiseGenerator("standard_normal", seed=seed + 6),
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
    cn = SCM(
        assignment_map={
            "X_0": (
                [],
                LinearAssignment(1),
                NoiseGenerator("standard_normal", seed=seed),
            ),
            "X_1": (
                ["X_0"],
                LinearAssignment(1, 1, 1),
                NoiseGenerator("standard_normal", seed=seed + 1),
            ),
            "X_2": (
                ["X_0", "X_1"],
                LinearAssignment(1, 1, 0.8, -1.2),
                NoiseGenerator("standard_normal", seed=seed + 2),
            ),
            "X_3": (
                ["X_1", "X_2"],
                LinearAssignment(1, 0, 0.3, 0.4),
                NoiseGenerator("standard_normal", seed=seed + 3),
            ),
            "Y": (
                ["X_3", "X_0"],
                LinearAssignment(1, 0.67, 1, -1),
                NoiseGenerator("standard_normal", seed=seed + 4),
            ),
            "X_4": (
                ["Y"],
                LinearAssignment(1, 1.2, -0.7),
                NoiseGenerator("standard_normal", seed=seed + 5),
            ),
            "X_5": (
                ["X_3", "Y"],
                LinearAssignment(1, 0.5, -0.7, 0.4),
                NoiseGenerator("standard_normal", seed=seed + 6),
            ),
            "X_6": (
                ["X_5", "X_4"],
                LinearAssignment(1, 0.5, -1.7, 1.4),
                NoiseGenerator("standard_normal", seed=seed + 7),
            ),
            "X_7": (
                [],
                LinearAssignment(1, 0.5),
                NoiseGenerator("standard_normal", seed=seed + 8),
            ),
            "X_8": (
                ["Y"],
                LinearAssignment(1, 0.5, 0.1),
                NoiseGenerator("standard_normal", seed=seed + 9),
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