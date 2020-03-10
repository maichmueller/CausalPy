import sklearn.linear_model
import numpy as np
import torch
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
import sklearn.linear_model
from causalpy.neural_networks.utils import StratifiedSampler


def regression_analytically(x, y):
    x_T = x.transpose()
    beta = np.linalg.inv(x_T @ x) @ x_T @ y
    y_hat = x @ beta
    residuals = y - y_hat
    return y_hat, residuals, beta


def regression_sklearn(x, y):
    lr = sklearn.linear_model.LinearRegression(fit_intercept=True).fit(x, y)
    y_hat = lr.predict(x)
    residuals = y - y_hat
    coeffs = lr.coef_.copy()
    coeffs[0] = lr.intercept_
    return lr, y_hat, residuals, coeffs


def evaluate(
    complete_data,
    ap,
    environments,
    x_vars,
    targ_var,
    ground_truth_assignment=None,
    plot=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    if len(x_vars) > 2:
        raise ValueError("Needs to be less than 3 predictors.")
    y = complete_data[targ_var]
    complete_data = complete_data.drop(columns=targ_var)
    y = y.values.reshape(-1, 1)

    y_hat = (
        ap.predict(torch.as_tensor(complete_data.values).float().to(device))
        .detach()
        .cpu()
        .numpy()
    )
    x_with_dummy = sklearn.preprocessing.add_dummy_feature(complete_data[x_vars].values)
    y_reg_hat, residuals_hat, beta_hat = regression_analytically(x_with_dummy, y_hat)
    y_reg, residuals, beta = regression_analytically(x_with_dummy, y)
    loss_y_hat = np.power(residuals_hat, 2).sum()
    loss_y = np.power(residuals, 2).sum()

    print("Regression on observed y:")
    print(
        targ_var,
        "=",
        beta[0],
        "+",
        " + ".join(
            [f"{beta.flatten()[i+1]} * {x_vars[i]}" for i in range(len(x_vars))]
        ),
    )
    print("L2 loss:", loss_y)

    print("Regression on cINN estimated y:")
    print(
        targ_var,
        "=",
        beta_hat[0],
        "+",
        " + ".join(
            [f"{beta_hat.flatten()[i+1]} * {x_vars[i]}" for i in range(len(x_vars))]
        ),
    )
    print("L2 loss:", loss_y_hat)

    residuals = residuals.flatten()
    residuals_hat = residuals_hat.flatten()
    y_reg = y_reg.flatten()
    y_reg_hat = y_reg_hat.flatten()

    # plotting
    if plot:
        residuals, residuals_hat = scatter_plot_estimates(
            complete_data,
            environments,
            ground_truth_assignment,
            residuals,
            residuals_hat,
            x_vars,
            y_reg,
            y_reg_hat,
        )
    return (residuals_hat, beta_hat), (residuals, beta), y_hat


def evaluate_convergence(
    complete_data,
    ap,
    environments,
    x_vars,
    targ_var,
    mask=None,
    ground_truth_assignment=None,
    plot=True,
    device="cuda" if torch.cuda.is_available() else "cpu",
):
    if len(x_vars) != 2:
        raise ValueError("Needs to be less than 3 predictors.")
    y = complete_data[targ_var]
    complete_data = complete_data.drop(columns=targ_var)
    y = y.values.reshape(-1, 1)

    y_hat = (
        ap.predict(torch.as_tensor(complete_data.values).float().to(device), mask=mask)
        .detach()
        .cpu()
        .numpy()
    )
    x_with_dummy = sklearn.preprocessing.add_dummy_feature(complete_data[x_vars].values)
    y_reg_hat, residuals_hat, beta_hat = regression_analytically(x_with_dummy, y_hat)
    y_reg, residuals, beta = regression_analytically(x_with_dummy, y)
    loss_y_hat = np.power(residuals_hat, 2).sum()
    loss_y = np.power(residuals, 2).sum()

    print("Regression on observed y:")
    print(
        targ_var,
        "=",
        beta[0],
        "+",
        " + ".join(
            [f"{beta.flatten()[i+1]} * {x_vars[i]}" for i in range(len(x_vars))]
        ),
    )
    print("L2 loss:", loss_y)

    print("Regression on cINN estimated y:")
    print(
        targ_var,
        "=",
        beta_hat[0],
        "+",
        " + ".join(
            [f"{beta_hat.flatten()[i+1]} * {x_vars[i]}" for i in range(len(x_vars))]
        ),
    )
    print("L2 loss:", loss_y_hat)

    residuals = residuals.flatten()
    residuals_hat = residuals_hat.flatten()
    y_reg = y_reg.flatten()
    y_reg_hat = y_reg_hat.flatten()

    # plotting
    if plot:
        residuals, residuals_hat = scatter_plot_estimates(
            complete_data,
            environments,
            ground_truth_assignment,
            residuals,
            residuals_hat,
            x_vars,
            y_reg,
            y_reg_hat,
        )
    return (residuals_hat, beta_hat), (residuals, beta), y_hat


def scatter_plot_estimates(
    complete_data,
    environments,
    ground_truth_assignment,
    residuals,
    residuals_hat,
    x_vars,
    y_reg,
    y_reg_hat,
):
    matplotlib.use("TkAgg")
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    plotting_sampler = StratifiedSampler(
        data_source=np.arange(len(complete_data)),
        class_vector=torch.as_tensor(environments),
        batch_size=1000,
    )
    samp_ind = plotting_sampler.gen_sample_array()
    if ground_truth_assignment is not None:
        y_nonoise = ground_truth_assignment(
            0, **{var: complete_data[var] for var in x_vars}
        )[samp_ind]
    y_reg = y_reg[samp_ind]
    y_reg_hat = y_reg_hat[samp_ind]
    residuals = residuals[samp_ind]
    residuals_hat = residuals_hat[samp_ind]
    data = complete_data.loc[samp_ind]
    if ground_truth_assignment:
        color_normal = np.abs(y_nonoise - y_reg)
        color_approx = np.abs(y_nonoise - y_reg_hat)
    else:
        color_normal = np.abs(residuals - y_reg)
        color_approx = np.abs(residuals_hat - y_reg_hat)
    ax.scatter(
        data[x_vars[0]], data[x_vars[1]], y_reg, c=color_normal, s=1.5, cmap="Greens",
    )
    # cmap = plt.get_cmap("Greens")
    # cs = np.abs(residuals - y_reg)
    # norm = matplotlib.colors.Normalize(vmin=cs.min(), vmax=cs.max())
    # sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    # cs = sm.to_rgba(cs)
    # ax.plot_surface(
    #     complete_data[x_vars[0]],
    #     complete_data[x_vars[1]],
    #     y_reg,
    #     c="green",
    # )
    ax.scatter(
        data[x_vars[0]], data[x_vars[1]], y_reg_hat, c=color_approx, s=1.5, cmap="Reds",
    )
    # cmap = plt.get_cmap("Reds")
    # cs = np.abs(residuals_hat - y_reg_hat)
    # norm = matplotlib.colors.Normalize(vmin=cs.min(), vmax=cs.max())
    # sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    # cs = sm.to_rgba(cs)
    # ax.plot_surface(
    #     complete_data[x_vars[0]],
    #     complete_data[x_vars[1]],
    #     y_reg_hat,
    #     c="red",
    # )
    ax.set_xlabel(x_vars[0])
    ax.set_ylabel(x_vars[1])
    ax.set_zlabel("Y")
    plt.show()
    return residuals, residuals_hat
