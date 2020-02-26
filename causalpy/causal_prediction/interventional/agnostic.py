import os
from collections import namedtuple
from typing import Union, Optional, Callable, List, Dict, Type

import numpy as np
import torch
import pandas as pd
import networkx as nx
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .icpbase import ICPredictor
import warnings
from causalpy.neural_networks import (
    AgnosticModel,
    Linear3D,
    MatrixSampler,
    L0Mask,
    L0InputGate,
    cINN,
)

Hyperparams = namedtuple("Hyperparams", "inn env l0")


class AgnosticPredictor(ICPredictor):
    def __init__(
        self,
        network: Optional[torch.nn.Module] = None,
        network_params: Optional[Dict] = None,
        masker_network: Optional[torch.nn.Module] = None,
        masker_network_params: Optional[Dict] = None,
        epochs: int = 100,
        batch_size: int = 1024,
        mask_monte_carlo_size=1,
        optimizer: Type[torch.optim.optimizer] = None,
        optimizer_params: Optional[Dict] = None,
        scheduler: Optional[_LRScheduler] = None,
        hyperparams: Optional[Hyperparams] = None,
        visualize_with_visdom: bool = False,
        log_level: bool = True,
    ):
        super().__init__(log_level=log_level)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.epochs = epochs
        self.hyperparams = (
            Hyperparams(l0=0.2, env=5, inn=2) if hyperparams is None else hyperparams
        )

        self.network = network
        self.network_params = (
            network_params
            if network_params is not None
            else dict(nr_blocks=3, dim=1, nr_layers=30)
        )

        self.optimizer = optimizer
        self.optimizer_params = (
            optimizer_params
            if optimizer_params is not None
            else dict(lr=1e-3, masker_lr=1e-2)
        )

        self.masker_net = masker_network
        self.masker_net_params = (
            masker_network_params
            if masker_network_params is not None
            else dict(monte_carlo_size=mask_monte_carlo_size, device=self.device)
        )

        self.scheduler = scheduler if scheduler is not None else None

        if visualize_with_visdom:
            try:
                import visdom
                self.viz = visdom.Visdom()
            except ImportError as e:
                warnings.warn(f"Package Visdom required for training visualizationm, but not found! "
                              f"Continuing without visualization.\n"
                              f"Exact error message was:\n{print(e)}")

    def _set_scheduler(self, force: bool = False):
        if self.scheduler is None or force:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer(**self.optimizer_params), step_size=50, gamma=0.9
            )

    def _set_optimizer(self, force: bool = False):
        if self.optimizer is None or force:
            torch.optim.Adam(
                [
                    {
                        "params": self.masker_net.parameters(),
                        "lr": self.optimizer_params.pop("masker_lr", 1e-2),
                    },
                    {"params": self.network.parameters()},
                ],
                lr=self.optimizer_params.pop("lr", 1e-3),
                **self.optimizer_params
            )

    def infer(
        self,
        obs: Union[pd.DataFrame, np.ndarray],
        envs: np.ndarray,
        target_variable: Union[int, str],
        alpha: float = 0.05,
        *args,
        **kwargs
        # skeleton=None,
        # train=5000, test=1000,
        # batch_size=-1, lr_gen=.001,
        # lr_disc=.01, lambda1=0.001, lambda2=0.0000001, nh=None, dnh=None,
        # verbose=True, losstype="fgan",
        # dagstart=0, dagloss=False,
        # dagpenalization=0.05, dagpenalization_increase=0.0,
        # linear=False, hlayers=2, idx=0):
    ):
        obs, target, environments = self.preprocess_input(obs, target_variable, envs)
        list_nodes = self.index_to_varname.index

        obs = obs.astype("float32")
        rows, cols = obs.shape()
        obs_tensor = torch.from_numpy(obs).to(self.device)

        candidates_masker = L0InputGate(dim_input=self.p)
        cINNs: List[
            torch.nn.Module
        ] = []  # the networks which render each environment distribution.
        for env in environments:
            cINNs.append()
        optimizer = torch.optim.Adam(
            conditional_net.parameters(), lr=kwargs.pop("lr", 0.001)
        )
