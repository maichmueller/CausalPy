import re

import numpy as np
import itertools
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd
import os
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)
from scipy.interpolate import interp1d
import matplotlib.patches as mpatches
from cycler import cycler
import proplot
import math


def comp_avg_metric(data, metric=accuracy_score, round_after=False):
    gt = ground_truth

    mean = data.mean(axis=0)
    quantiles = data.quantile([0.25, 0.5, 0.75], axis=0)

    if round_after:
        mean = mean.round()
        quantiles = quantiles.round()
    #     print(quantiles)
    metric_mean = metric(gt, mean)
    metric_quantiles = dict()
    for name, quantile in quantiles.iterrows():
        metric_quantiles[name] = metric(gt, quantile)

    return {"mean": metric_mean, "quantiles": metric_quantiles}


def comp_metric_avg(data, metric=accuracy_score, round_after=False):

    gt = named_ground_truth
    vals = []

    for i, row in data.iterrows():
        #         print(row, gt)
        vals.append(pd.Series(metric(gt, row)))
    vals_df = pd.concat(vals)
    metric_mean = vals_df.mean(axis=0)
    metric_quantiles = vals_df.quantile([0.25, 0.5, 0.75])

    return {"mean": metric_mean, "quantiles": metric_quantiles}


def cap(df, high=1, low=0):
    return np.max(
        [np.min([df, np.ones_like(df) * 1], axis=0), np.zeros_like(df) + low], axis=0
    )


if __name__ == "__main__":
    vars = ["X_0", "X_1", "X_2", "X_3", "X_4", "X_5"]
    ground_truth = [1, 0, 0, 1, 0, 0]
    named_ground_truth = pd.Series({name: val for name, val in zip(vars, ground_truth)})

    experiment = 0
    results_single = dict()

    res_folder = "/home/michael/Desktop/resultsss2"
    ss = [
        str(i)
        for i in [
            1000,
            2000,
            3000,
            4000,
            5000,
            6000,
            7000,
            8000,
            9000,
            10000,
            11000,
            12000,
            13000,
            14000,
            15000,
            # 16384,
        ]
    ]
    for filename in sorted(os.listdir(res_folder)):

        if any(
            [
                "l0-0.8" in filename
                and (
                    f"samples-{s}_runs-100_mcs-1_batch_size-5000" in filename
                    or f"samples-{s}." in filename
                )
                for s in ss
            ]
        ):
            print(filename)
            results_single[os.path.splitext(filename)[0]] = pd.read_csv(
                os.path.join(res_folder, filename), index_col=None
            )

    model_names = ["Single"]
    # model_names = ["Single", "Density"]
    long_metric_names = {
        "acc-avg": "Accuracy",
        "avg-acc": r"$\overline{M}$ Accuracy",
        "prec-avg": "Precision-AVG",
        "rec-avg": "Recall",
        "aucpr-avg": "AUCPR",
        "avg-aucpr": r"$\overline{M}$ AUCPR",
    }
    metrics_names = ["acc-avg", "aucpr-avg", "avg-acc", "avg-aucpr"]
    metric_to_func = {
        "acc-avg": accuracy_score,
        "avg-acc": accuracy_score,
        "prec": precision_score,
        "rec": recall_score,
        "aucpr-avg": average_precision_score,
        "avg-aucpr": average_precision_score,
    }
    metrics = {m: {} for m in metrics_names}
    bs_p = re.compile(r"(?<=samples-)\d+")
    for name, df in itertools.chain(results_single.items()):

        rounded_df = pd.DataFrame(
            np.ones_like(df.to_numpy(), dtype=int), columns=df.columns
        )
        rounded_df[df < 0.5] = 0

        eval_func = None
        data_df = df
        round_after = False
        ss = int(re.search(bs_p, name).group())
        for metric in metrics_names:

            if "-avg" in metric:
                eval_func = comp_metric_avg
                if "acc" in metric:
                    data_df = rounded_df
            elif "avg-" in metric:
                eval_func = comp_avg_metric
                if "acc" in metric:
                    round_after = True

            metrics[metric][ss] = eval_func(
                data_df, metric_to_func[metric], round_after=round_after
            )
    zero_estim = pd.DataFrame(np.zeros(df.shape))
    zero_estim_metric = {m: {} for m in metrics_names}
    for metric in metrics_names:

        round_after = False
        if "-avg" in metric:
            eval_func = comp_metric_avg
        elif "avg-" in metric:
            eval_func = comp_avg_metric
            if "acc" in metric:
                round_after = True

        zero_estim_metric[metric] = eval_func(
            zero_estim, metric_to_func[metric], round_after=round_after
        )

    colors = [
        f"#{c}" for c in ["076AB0", "CB152B", "058C42", "ED872D", "5C5C5C", "792359"]
    ]
    # plt.style.use("science")
    # fig, ax = plt.subplots(2, 1, figsize=(5.8, 4), sharex=True, sharey=True)
    # x = np.array(list(metrics["avg-acc"].keys()))
    # sort = np.argsort(x)
    # x = x[sort]
    #
    # acc_metric = "avg-acc"
    # aucpr_metric = "avg-aucpr"
    #
    # yacc = np.array([metrics[acc_metric][ss]["mean"] for ss in x])
    # yaucpr = np.array([metrics[aucpr_metric][ss]["mean"] for ss in x])
    # xnew = np.linspace(np.min(x), np.max(x), 1000, endpoint=True)
    # yacc_interp = cap(interp1d(x, yacc, "cubic")(xnew))
    # yaucpr_interp = cap(interp1d(x, yaucpr, "cubic")(xnew))
    #
    # quantile_25_acc = np.array([metrics[acc_metric][ss]["quantiles"][0.25] for ss in x])
    # quantile_50_acc = np.array([metrics[acc_metric][ss]["quantiles"][0.5] for ss in x])
    # quantile_75_acc = np.array([metrics[acc_metric][ss]["quantiles"][0.75] for ss in x])
    #
    # quantile_25_aucpr = np.array(
    #     [metrics[aucpr_metric][ss]["quantiles"][0.25] for ss in x]
    # )
    # quantile_50_aucpr = np.array(
    #     [metrics[aucpr_metric][ss]["quantiles"][0.5] for ss in x]
    # )
    # quantile_75_aucpr = np.array(
    #     [metrics[aucpr_metric][ss]["quantiles"][0.75] for ss in x]
    # )
    #
    # yacc_25_interp = cap(interp1d(x, quantile_25_acc, "cubic")(xnew))
    # yacc_75_interp = cap(interp1d(x, quantile_75_acc, "cubic")(xnew))
    # yaucpr_25_interp = cap(interp1d(x, quantile_25_aucpr, "cubic")(xnew))
    # yaucpr_75_interp = cap(interp1d(x, quantile_75_aucpr, "cubic")(xnew))
    #
    # ax[0].plot(
    #     xnew, yacc_interp, "-", color=colors[0], label=r"$\overline{M}$ Accuracy"
    # )
    # # ax[0].fill_between(
    # #     xnew, cap(yacc_25_interp), cap(yacc_75_interp), alpha=0.3, facecolor=colors[0]
    # # )
    # # ax[1].fill_between(
    # #     xnew,
    # #     cap(yaucpr_25_interp),
    # #     cap(yaucpr_75_interp),
    # #     alpha=0.3,
    # #     facecolor=colors[1],
    # # )
    # ax[0].plot(
    #     x, yacc, "o", color=colors[0], markersize=1.5,
    # )
    #
    # # ax[0].plot(x, yacc, "-", color=colors[0], label=r" Accuracy")
    # ax[1].plot(
    #     x, yaucpr, "o", color=colors[1], markersize=1.5,
    # )
    # ax[1].plot(xnew, yaucpr_interp, "-", color=colors[1])
    # # ax[1].plot(x, yaucpr, "-", color=colors[1], label=r"$\overline{M}$ AUCPR")
    #
    # plt.legend()
    # ax[1].set_xlabel("Sample size")
    # ax[0].set_ylabel("$\overline{M}$ Accuracy")
    # ax[1].set_ylabel("$\overline{M}$ AUCPR")
    # ax[0].set_title("Sample size experiment on reduced graph")
    # fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # # ax.set_xscale("log", base=2)
    # # ax.set_ylim(0, 1)
    # fig.savefig("plots/extra_test_eval.pdf")
    # plt.show

    plt.style.use("science")
    fig, ax = plt.subplots(1, 1, figsize=(5.8, 3.5), sharex=True, sharey=True)
    x = np.array(list(metrics["avg-acc"].keys()))
    sort = np.argsort(x)
    x = x[sort]

    acc_metric = "avg-acc"
    aucpr_metric = "avg-aucpr"

    yacc = np.array([metrics[acc_metric][ss]["mean"] for ss in x])
    yaucpr = np.array([metrics[aucpr_metric][ss]["mean"] for ss in x])
    xnew = np.linspace(np.min(x), np.max(x), 1000, endpoint=True)
    yacc_interp = cap(interp1d(x, yacc, "cubic")(xnew))
    yaucpr_interp = cap(interp1d(x, yaucpr, "cubic")(xnew))

    quantile_25_acc = np.array([metrics[acc_metric][ss]["quantiles"][0.25] for ss in x])
    quantile_50_acc = np.array([metrics[acc_metric][ss]["quantiles"][0.5] for ss in x])
    quantile_75_acc = np.array([metrics[acc_metric][ss]["quantiles"][0.75] for ss in x])

    quantile_25_aucpr = np.array(
        [metrics[aucpr_metric][ss]["quantiles"][0.25] for ss in x]
    )
    quantile_50_aucpr = np.array(
        [metrics[aucpr_metric][ss]["quantiles"][0.5] for ss in x]
    )
    quantile_75_aucpr = np.array(
        [metrics[aucpr_metric][ss]["quantiles"][0.75] for ss in x]
    )

    yacc_25_interp = cap(interp1d(x, quantile_25_acc, "cubic")(xnew))
    yacc_75_interp = cap(interp1d(x, quantile_75_acc, "cubic")(xnew))
    yaucpr_25_interp = cap(interp1d(x, quantile_25_aucpr, "cubic")(xnew))
    yaucpr_75_interp = cap(interp1d(x, quantile_75_aucpr, "cubic")(xnew))

    ax.plot(xnew, yacc_interp, "-", color=colors[0], label=r"$\overline{M}$ Accuracy")
    # ax[0].fill_between(
    #     xnew, cap(yacc_25_interp), cap(yacc_75_interp), alpha=0.3, facecolor=colors[0]
    # )
    # ax[1].fill_between(
    #     xnew,
    #     cap(yaucpr_25_interp),
    #     cap(yaucpr_75_interp),
    #     alpha=0.3,
    #     facecolor=colors[1],
    # )
    ax.plot(
        x, yacc, "o", color=colors[0], markersize=1.5,
    )

    ax.hlines(
        zero_estim_metric[acc_metric]["mean"],
        x.min(),
        x.max(),
        linestyles="dashed",
        linewidth=0.5,
        color=colors[0],
        label="Zero-Estimate Accuracy",
    )
    # ax[0].plot(x, yacc, "-", color=colors[0], label=r" Accuracy")
    ax.plot(
        x, yaucpr, "o", color=colors[1], markersize=1.5,
    )
    ax.plot(
        xnew,
        cap(yaucpr_interp, low=zero_estim_metric[aucpr_metric]["mean"]),
        "-",
        color=colors[1],
        label=r"$\overline{M}$ AUCPR",
    )
    # ax[1].plot(x, yaucpr, "-", color=colors[1], label=r"$\overline{M}$ AUCPR")
    ax.hlines(
        zero_estim_metric[aucpr_metric]["mean"],
        x.min(),
        x.max(),
        linestyles="dashed",
        linewidth=0.5,
        color=colors[1],
        label="Zero-Estimate AUCPR",
    )
    plt.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, -0.45),
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    ax.set_xlabel("Sample size")
    ax.set_ylabel("$\overline{M}$ Scores")
    ax.set_title("Sample size control experiment on the reduced graph for Single-CAREN")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    # ax.set_xscale("log", base=2)
    # ax.set_ylim(0, 1)
    fig.savefig("plots/extra_test_eval.pdf")
    plt.show()
