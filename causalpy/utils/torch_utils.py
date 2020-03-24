from typing import Optional, Collection

import numpy as np


class VisdomLinePlotter(object):
    """Plots to Visdom"""

    def __init__(self, visdom_obj, env_name="main"):
        self.viz = visdom_obj
        self.env = env_name
        self.plots = {}

    def plot(self, var_name, split_name, title_name, x, y):
        if var_name not in self.plots:
            self.plots[var_name] = self.viz.line(
                X=np.array([x, x]),
                Y=np.array([y, y]),
                env=self.env,
                opts=dict(
                    legend=[split_name],
                    title=title_name,
                    xlabel="Epochs",
                    ylabel=var_name,
                ),
            )
        else:
            self.viz.line(
                X=np.array([x]),
                Y=np.array([y]),
                env=self.env,
                win=self.plots[var_name],
                name=split_name,
                update="append",
            )

    def plot_loss(
        self,
        loss_arr: Collection[float],
        loss_win: Optional[str] = None,
        title="Loss Behaviour",
    ):
        loss_arr = np.asarray(loss_arr)
        if not len(loss_arr):
            ytickmax = np.quantile(loss_arr, 0.99)
            ytickmax = ytickmax + 0.5 * abs(ytickmax)
            ytickmin = np.quantile(loss_arr, 0)
            ytickmin = ytickmin - 0.5 * abs(ytickmin)
        else:
            ytickmin = ytickmax = None
        self.viz.line(
            X=np.arange(len(loss_arr)),
            Y=loss_arr,
            win=loss_win,
            opts=dict(
                title=title, ytickmin=ytickmin, ytickmax=ytickmax, xlabel="Epoch"
            ),
        )
