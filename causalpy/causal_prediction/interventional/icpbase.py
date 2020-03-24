import logging
from typing import *

import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
import os


class ICPredictor(ABC):
    def __init__(self, log_level: Optional[Union[str, int]] = False):
        self.n: int = -1
        self.p: int = -1
        self.target_name: Hashable = -1
        self.index_to_varname: pd.Series = pd.Series([])
        self.varname_to_index: pd.Series = pd.Series([])
        self.variables: np.ndarray = np.array([])
        self.env_start_end: Dict[int, Tuple[int, int]] = dict()

        self.log_level = log_level.upper() if isinstance(log_level, str) else log_level
        logging.basicConfig(
            format="%(asctime)s - %(message)s", datefmt="%d-%b-%y %H:%M:%S"
        )
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

    @abstractmethod
    def infer(
        self,
        obs: Union[pd.DataFrame, np.ndarray],
        envs: np.ndarray,
        target_variable: Union[int, str],
        *args,
        **kwargs,
    ):
        raise NotImplementedError

    def preprocess_input(
        self,
        obs: Union[pd.DataFrame, np.ndarray],
        target_name: Hashable,
        envs: Union[List, Tuple, np.ndarray],
        normalize: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:

        self.n, self.p = obs.shape[0], obs.shape[1] - 1
        assert (
            len(envs) == self.n
        ), f"Number of observation samples ({len(envs)}) and number of environment labels ({self.n}) have to be equal."

        self.target_name = target_name

        if normalize:
            mean = obs.mean(0)
            std = obs.std(0)
            obs = np.divide(np.subtract(obs, mean[np.newaxis, :]), std[np.newaxis, :])

        obs = pd.DataFrame(obs)  # force to be a DataFrame
        obs["ENV"] = envs
        obs = obs.sort_values(by="ENV")
        envs = obs["ENV"]
        obs.drop(columns=["ENV"], inplace=True)

        self.variables = obs.columns.values
        target = obs[target_name].to_numpy().flatten()
        obs.drop(columns=[target_name], inplace=True)
        self.index_to_varname = pd.Series(obs.columns, index=range(self.p))
        self.varname_to_index = pd.Series(range(self.p), index=obs.columns)
        obs = obs.to_numpy()

        # dict of env -> env indices.
        environments = {env: np.where(envs == env)[0] for env in np.unique(envs)}
        for env, indices in environments.items():
            self.env_start_end[env] = indices[0], indices[-1]

        return obs, target, environments

    def get_parent_candidates(self):
        return self.index_to_varname.values
