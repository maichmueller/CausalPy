from scmodels import SCM
from sympy.stats import Binomial, Normal, FiniteRV


def build_scm_minimal(seed=0):
    cn = SCM(
        {"X_0": ("N", Binomial("N", n=1, p=0.5)), "Y": ("N + X_0", Normal("N", 0, 1))},
        variable_tex_names={"X_0": "$X_0$"},
        seed=seed,
    )
    return cn


def build_scm_minimal2(seed=0):
    cn = SCM(
        {"X_0": ("N", Binomial("N", n=1, p=0.5)), "Y": ("N", Normal("N", 0, 1))},
        variable_tex_names={"X_0": "$X_0$"},
        seed=seed,
    )
    return cn


def build_scm_basic_discrete(seed=0):
    cn = SCM(
        {
            "X_0": ("N", Binomial("N", n=1, p=0.5)),
            "X_1": ("N", FiniteRV("N", density={0: 0.5, 1: 0.5})),
            "X_2": ("N", FiniteRV("N", density={0: 1 / 3, 1: 1 / 3, 2: 1 / 3})),
            "Y": ("N + X_0 - X_1", Normal("N", 0, 1)),
        },
        variable_tex_names={"X_0": "$X_0$", "X_1": "$X_1$", "X_2": "$X_2$"},
        seed=seed,
    )
    return cn


def build_scm_basic(seed=0):
    cn = SCM(
        {
            "X_0": ("N", Normal("N", 0, 1),),
            "X_1": ("N", Normal("N", 0, 1),),
            "X_2": ("N", Normal("N", 0, 1),),
            "Y": ("N + X_0 - X_1", Normal("N", 0, 1)),
        },
        variable_tex_names={"X_0": "$X_0$", "X_1": "$X_1$", "X_2": "$X_2$"},
        seed=seed,
    )
    return cn


def build_scm_basicst(seed=0):
    cn = SCM(
        {
            "X_0": ("N", Normal("N", 0, 1),),
            "X_1": ("N", Normal("N", 0, 1),),
            "Y": ("N + X_0 - X_1", Normal("N", 0, 1),),
        },
        variable_tex_names={"X_0": "$X_0$", "X_1": "$X_1$"},
        seed=seed,
    )
    return cn


def build_scm_medium(seed=0):
    cn = SCM(
        {
            "X_0": ("N", Normal("N", 0, 1),),
            "X_1": ("1 + N + X_0", Normal("N", 0, 1)),
            "X_2": ("1 + N + 0.8 * X_0", Normal("N", 0, 1)),
            "X_3": ("1 + N + 0.3 * X_1 + 0.4 * X_2", Normal("N", 0, 1)),
            "Y": (".6 + N + X_3 - X_0", Normal("N", 0, 1)),
            "X_4": ("1.2 + N - 0.7 * Y", Normal("N", 0, 1)),
            "X_5": (".5 + N - 0.7 * X_3 + 0.4 * Y", Normal("N", 0, 1)),
        },
        variable_tex_names={
            "X_0": "$X_0$",
            "X_1": "$X_1$",
            "X_2": "$X_2$",
            "X_3": "$X_3$",
            "X_4": "$X_4$",
            "X_5": "$X_5$",
        },
        seed=seed,
    )
    return cn


def build_scm_large(seed=0):
    cn = SCM(
        {
            "X_0": ("N", Normal("N", 0, 1),),
            "X_1": ("1 + N + X_0", Normal("N", 0, 1)),
            "X_2": ("1 + N + 0.8 * X_0", Normal("N", 0, 1)),
            "X_3": ("1 + N + 0.3 * X_1 + 0.4 * X_2", Normal("N", 0, 1)),
            "Y": (".6 + N + X_3 - X_0", Normal("N", 0, 1)),
            "X_4": ("1.2 + N - 0.7 * Y", Normal("N", 0, 1)),
            "X_5": (".5 + N - 0.7 * X_3 + 0.4 * Y", Normal("N", 0, 1)),
            "X_6": (".5 + N - 1.7 * X_5 + 1.4 * X_4", Normal("N", 0, 1)),
            "X_7": (".5 + N", Normal("N", 0, 1)),
            "X_8": (".5 + N + 0.1 * Y", Normal("N", 0, 1)),
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
        seed=seed,
    )
    return cn
