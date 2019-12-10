from typing import List
from causalpy.bayesian_graphs import SCM, LinearAssignment, NoiseGenerator
import pandas as pd


def print_results(
    target_variable: str,
    estimated_parents: List,
    actual_parents: List,
    as_dataframe=True,
    verbose: bool = False,
):
    actual_parents = sorted(actual_parents)
    n_iters = len(estimated_parents)
    success_rate = (
        sum((sorted(estimated_parents[i]) == actual_parents) for i in range(n_iters))
        / n_iters
    )
    partial_rate = (
        sum(
            (
                bool(estimated_parents[i])
                and all(
                    estimated_parents[i][k] in actual_parents
                    for k in range(len(estimated_parents[i]))
                )
            )
            for i in range(n_iters)
        )
        / n_iters
    )
    wrong_rate = (
        sum(
            bool(estimated_parents[i])
            and any(
                estimated_parents[i][k] not in actual_parents
                for k in range(len(estimated_parents[i]))
            )
            for i in range(n_iters)
        )
        / n_iters
    )
    empty_rate = sum((not bool(estimated_parents[i])) for i in range(n_iters)) / n_iters
    if as_dataframe:
        return pd.DataFrame.from_dict(
            {
                "Correct": [f"{100 * success_rate:.1f}%"],
                "PartialCorrect": [f"{100 * partial_rate:.1f}%"],
                "Wrong": [f"{100 * wrong_rate:.1f}%"],
                "Empty": [f"{100 * empty_rate:.1f}%"],
            },
            orient="columns",
        )
    print("Target:", target_variable)
    print("Actual parents:", sorted(actual_parents))
    print(f"Correct estimates: {100 * success_rate:.1f}%")
    print(f"Partial estimates: {100 * partial_rate:.1f}%")
    print(f"Wrong estimates: {100 * wrong_rate:.1f}%")
    print(f"Empty estimates: {100 * empty_rate:.1f}%")
    if verbose:
        print("Predicted parents:")
        print(
            "\n".join(
                (
                    f"\tIter {i}: \t {sorted(estimated_parents[i])}"
                    for i in range(n_iters)
                )
            )
        )


def build_simple_scm_0():
    assignment_map = {
        "X": ([], LinearAssignment(1, 1), NoiseGenerator("normal", scale=1)),
        "Y": (["X"], LinearAssignment(1, 0, 2), NoiseGenerator("normal", scale=2),),
    }
    scm = SCM(assignment_map)
    return scm


def build_simple_scm_1():
    assignment_map = {
        "X_0": ([], LinearAssignment(1, 0), NoiseGenerator("normal", scale=0.3)),
        "X_1": (["X_0"], LinearAssignment(1, 1, 1), NoiseGenerator("normal", scale=1)),
        "Y": (["X_1"], LinearAssignment(1, 0, 2), NoiseGenerator("normal", scale=2),),
    }
    scm = SCM(assignment_map)
    return scm


def build_simple_scm_2():
    assignment_map = {
        "X_0": ([], LinearAssignment(1, 0), NoiseGenerator("normal", scale=0.3)),
        "X_1": (["X_0"], LinearAssignment(1, 1, 1), NoiseGenerator("normal", scale=1)),
        "Y": (
            ["X_0", "X_1"],
            LinearAssignment(1, 0, 2, -1),
            NoiseGenerator("normal", scale=2),
        ),
    }
    scm = SCM(assignment_map)
    return scm


def build_simple_scm_3():
    assignment_map = {
        "X_0": ([], LinearAssignment(1, 0), NoiseGenerator("normal", scale=0.3)),
        "X_1": (["X_0"], LinearAssignment(1, 1, 1), NoiseGenerator("normal", scale=1)),
        "Y": (
            ["X_0", "X_1"],
            LinearAssignment(1, 0, 2, -1),
            NoiseGenerator("normal", scale=2),
        ),
    }
    scm = SCM(assignment_map)
    return scm


def build_simple_scm_4():
    assignment_map = {
        "X_0": ([], LinearAssignment(1, 0), NoiseGenerator("normal", scale=0.3)),
        "X_1": ([], LinearAssignment(1), NoiseGenerator("normal", scale=1)),
        "X_2": (
            ["X_0", "X_1"],
            LinearAssignment(1, 1, 1, 1),
            NoiseGenerator("normal", scale=0.4),
        ),
        "X_3": (
            ["X_0", "X_1"],
            LinearAssignment(1, 1, -1, 0.1),
            NoiseGenerator("normal", scale=0.1),
        ),
        "X_4": (
            ["X_0", "X_1"],
            LinearAssignment(1, 1, -0.5, 0.5),
            NoiseGenerator("normal", scale=0.2),
        ),
        "Y": (
            ["X_0", "X_1"],
            LinearAssignment(1, 0, 2, -1),
            NoiseGenerator("normal", scale=0.2),
        ),
    }
    scm = SCM(assignment_map)
    return scm


def build_simple_scm_5():
    assignment_map = {
        "X_0": ([], LinearAssignment(1, 0), NoiseGenerator("skewnorm", scale=0.3, a=0.4, source="scipy")),
        "X_1": ([], LinearAssignment(1), NoiseGenerator("skewnorm", scale=0.3, a=0.4, source="scipy")),
        "X_2": (
            ["X_0", "X_1"],
            LinearAssignment(1, 1, 1, 1),
            NoiseGenerator("normal", scale=0.4),
        ),
        "X_3": (
            ["X_0", "X_1"],
            LinearAssignment(1, 1, -1, 0.1),
            NoiseGenerator("normal", scale=0.1),
        ),
        "X_4": (
            ["X_0", "X_1"],
            LinearAssignment(1, 1, -0.5, 0.5),
            NoiseGenerator("normal", scale=0.2),
        ),
        "Y": (
            ["X_0", "X_1"],
            LinearAssignment(1, 0, 2, -1),
            NoiseGenerator("normal", scale=0.2),
        ),
    }
    scm = SCM(assignment_map)
    return scm


def build_simple_scm_6():
    assignment_map = {
        "X_0": ([], LinearAssignment(1, 0), NoiseGenerator("skewnorm", scale=0.3, a=0.4, source="scipy")),
        "X_1": ([], LinearAssignment(1), NoiseGenerator("skewnorm", scale=0.3, a=-0.4, source="scipy")),
        "X_2": (
            ["X_0", "X_1"],
            LinearAssignment(1, 1, 1, 1),
            NoiseGenerator("normal", scale=0.4),
        ),
        "X_3": (
            ["X_0", "X_1"],
            LinearAssignment(1, 1, -1, 0.1),
            NoiseGenerator("normal", scale=0.1),
        ),
        "X_4": (
            ["X_0", "X_1"],
            LinearAssignment(1, 1, -0.5, 0.5),
            NoiseGenerator("normal", scale=0.2),
        ),
        "Y": (
            ["X_0", "X_1"],
            LinearAssignment(1, 0, 2, -1),
            NoiseGenerator("normal", scale=0.2),
        ),
    }
    scm = SCM(assignment_map)
    return scm


def build_simple_scm_7():
    assignment_map = {
        "X_0": ([], LinearAssignment(1, 0), NoiseGenerator("skewnorm", scale=0.3, a=0.4, source="scipy")),
        "X_1": ([], LinearAssignment(1), NoiseGenerator("skewnorm", scale=0.3, a=0.4, source="scipy")),
        "X_2": (
            ["X_0", "X_1"],
            LinearAssignment(1, 1, 1, 1),
            NoiseGenerator("skewnorm", scale=0.3, a=0.4, source="scipy")
        ),
        "X_3": (
            ["X_0", "X_1"],
            LinearAssignment(1, 1, -1, 0.1),
            NoiseGenerator("skewnorm", scale=0.3, a=0.4, source="scipy")
        ),
        "X_4": (
            ["X_0", "X_1"],
            LinearAssignment(1, 1, -0.5, 0.5),
            NoiseGenerator("skewnorm", scale=0.3, a=0.4, source="scipy")
        ),
        "Y": (
            ["X_0", "X_1"],
            LinearAssignment(1, 0, 2, -1),
            NoiseGenerator("skewnorm", scale=0.3, a=0.4, source="scipy")
        ),
    }
    scm = SCM(assignment_map)
    return scm


def build_simple_scm_8():
    assignment_map = {
        "X_0": ([], LinearAssignment(1, 0), NoiseGenerator("skewnorm", scale=0.3, a=0.4, source="scipy")),
        "X_1": ([], LinearAssignment(1), NoiseGenerator("skewnorm", scale=0.3, a=0.4, source="scipy")),
        "X_2": (
            ["X_0", "X_1"],
            LinearAssignment(1, 1, 1, 1),
            NoiseGenerator("skewnorm", scale=0.3, a=0.9, source="scipy")
        ),
        "X_3": (
            ["X_0", "X_1"],
            LinearAssignment(1, 1, -1, 0.1),
            NoiseGenerator("skewnorm", scale=0.3, a=-0.2, source="scipy")
        ),
        "X_4": (
            ["X_0", "X_1"],
            LinearAssignment(1, 1, -0.5, 0.5),
            NoiseGenerator("skewnorm", scale=0.3, a=0.1, source="scipy")
        ),
        "Y": (
            ["X_0", "X_1"],
            LinearAssignment(1, 0, 2, -1),
            NoiseGenerator("skewnorm", scale=0.3, a=-0.8, source="scipy")
        ),
    }
    scm = SCM(assignment_map)
    return scm


def build_simple_scm_9():
    assignment_map = {
        "X_0": ([], LinearAssignment(1, 0), NoiseGenerator("lognormal", sigma=0.3, source="numpy")),
        "X_1": ([], LinearAssignment(1), NoiseGenerator("lognormal", sigma=0.3, source="numpy")),
        "X_2": (
            ["X_0", "X_1"],
            LinearAssignment(1, 1, 1, 1),
            NoiseGenerator("normal", scale=0.8, source="numpy")
        ),
        "X_3": (
            ["X_0", "X_1"],
            LinearAssignment(1, 1, -1, 0.1),
            NoiseGenerator("gamma", shape=1, scale=1, source="numpy")
        ),
        "X_4": (
            ["X_0", "X_1"],
            LinearAssignment(1, 1, -0.5, 0.5),
            NoiseGenerator("skewnorm", scale=0.3, a=0.1, source="scipy")
        ),
        "Y": (
            ["X_0", "X_1"],
            LinearAssignment(1, 0, 2, -1),
            NoiseGenerator("skewnorm", scale=0.3, a=-0.8, source="scipy")
        ),
    }
    scm = SCM(assignment_map)
    return scm


def build_simple_scm_10():
    assignment_map = {
        "X_0": ([], LinearAssignment(1, 0), NoiseGenerator("lognormal", sigma=5, source="numpy")),
        "X_1": ([], LinearAssignment(1), NoiseGenerator("lognormal", sigma=5, source="numpy")),
        "X_2": (
            ["X_0", "X_1"],
            LinearAssignment(1, 1, 1, 1),
            NoiseGenerator("normal", scale=2, source="numpy")
        ),
        "X_3": (
            ["X_0", "X_1", "X_2"],
            LinearAssignment(1, 1, -1, 0.7, 2),
            NoiseGenerator("gamma", shape=2, scale=2, source="numpy")
        ),
        "X_4": (
            ["X_0", "X_1"],
            LinearAssignment(1, 1, -0.5, 0.5),
            NoiseGenerator("skewnorm", scale=0.3, a=0.1, source="scipy")
        ),
        "Y": (
            ["X_0", "X_1", "X_3"],
            LinearAssignment(1, 0, 2, -1, 1),
            NoiseGenerator("skewnorm", scale=0.3, a=-0.8, source="scipy")
        ),
    }
    scm = SCM(assignment_map)
    return scm


def build_simple_scm_11():
    assignment_map = {
        "X_0": ([], LinearAssignment(1, 0), NoiseGenerator("lognormal", sigma=5, source="numpy")),
        "X_1": ([], LinearAssignment(1), NoiseGenerator("lognormal", sigma=5, source="numpy")),
        "X_2": (
            ["X_0", "X_1"],
            LinearAssignment(1, 1, 1, 1),
            NoiseGenerator("normal", scale=2, source="numpy")
        ),
        "X_3": (
            ["X_0", "X_1", "X_2"],
            LinearAssignment(1, 1, -1, 0.7, 2),
            NoiseGenerator("gamma", shape=2, scale=2, source="numpy")
        ),
        "X_4": (
            ["X_0", "X_1"],
            LinearAssignment(1, 1, -0.5, 0.5),
            NoiseGenerator("skewnorm", scale=0.3, a=0.1, source="scipy")
        ),
        "Y": (
            ["X_0", "X_1", "X_3"],
            LinearAssignment(1, 0, 2, -1, 1),
            NoiseGenerator("skewnorm", scale=8, a=-0.8, source="scipy")
        ),
    }
    scm = SCM(assignment_map)
    return scm
