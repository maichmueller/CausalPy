import os
from collections import namedtuple
from typing import Union, Optional, Callable, List

import numpy as np
import torch
import pandas as pd
import networkx as nx
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from tqdm import tqdm

from .icpbase import ICPredictor
from causalpy.neural_networks import AgnosticModel, Linear3D, MatrixSampler, L0Mask, L0InputGate


Hyperparams = namedtuple("Hyperparams", "alpha beta gamma")


class AgnosticPredictor(ICPredictor):
    def __init__(
            self,
            network: Optional[torch.nn.Module] = None,
            epochs: int = 100,
            batch_size: int = 1024,
            loss_transform_res_to_par: str = "sum",
            loss_transform_res_to_res: str = "sum",
            compare_residuals_pairwise: bool = True,
            residual_equality_measure: Union[str, Callable] = "mmd",
            variable_independence_metric: str = "hsic",
            optimizer: Optional[Optimizer] = None,
            scheduler: Optional[_LRScheduler] = None,
            hyperparams: Optional[Hyperparams] = None,
            log_level: bool = True,
    ):
        super().__init__(log_level=log_level)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = batch_size
        self.epochs = epochs
        self.network = network if network is not None else AgnosticModel()
        self.hyperparams = (
            Hyperparams(alpha=1, beta=1, gamma=1)
            if hyperparams is None
            else hyperparams
        )

        self.optimizer = (
            torch.optim.Adam(network.parameters(), lr=1e-3)
            if optimizer is None
            else optimizer
        )

        if scheduler is None:
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=50, gamma=0.9
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

        obs = obs.astype('float32')
        rows, cols = obs.shape()
        obs_tensor = torch.from_numpy(obs).to(self.device)

        candidates_masker = L0InputGate(dim_input=self.p)
        cINNs: List[torch.nn.Module] = []  # the networks which render each environment distribution.
        for env in environments:
            cINNs.append()
        optimizer = torch.optim.Adam(conditional_net.parameters(), lr=kwargs.pop("lr", 0.001))



