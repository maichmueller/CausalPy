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

        self.log_level = log_level.upper() if isinstance(log_level, str) else log_level
        logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%y %H:%M:%S')
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
            target_variable: Hashable,
            envs: Union[List, Tuple, np.ndarray],
            normalize: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        self.n, self.p = obs.shape[0], obs.shape[1] - 1
        assert (
                len(envs) == self.n
        ), f"Number of observation samples ({len(envs)}) and number of environment labels ({self.n}) have to be equal."

        self.target_name = target_variable

        if normalize:
            obs = np.subtract(obs, obs.mean(1)[:, np.newaxis])
            obs = np.divide(obs, obs.std(1)[:, np.newaxis])

        if isinstance(obs, pd.DataFrame):
            self.variables = obs.columns.values
            target = obs[target_variable].to_numpy().flatten()
            obs = obs.drop(columns=[target_variable])
            self.index_to_varname = pd.Series(obs.columns, index=range(self.p))
            self.varname_to_index = pd.Series(range(self.p), index=obs.columns)
            obs = obs.to_numpy()

        elif isinstance(obs, np.ndarray):
            self.index_to_varname = pd.Series(np.arange(self.p))
            self.variables = self.index_to_varname.values
            target = obs[:, target_variable].flatten()
            obs = np.delete(obs, target_variable, axis=1)
            itv = self.index_to_varname
            target_index = itv[itv == target_variable].index
            self.varname_to_index = (itv.loc[target_index + 1 :] - 1).drop(
                index=target_variable
            )
        else:
            raise ValueError(
                "Observations have to be passed as a pandas DataFrame or numpy ndarray."
            )

        environments = {env: np.where(envs == env)[0] for env in np.unique(envs)}

        return obs, target, environments

    def get_parent_candidates(self):
        return self.index_to_varname.values

