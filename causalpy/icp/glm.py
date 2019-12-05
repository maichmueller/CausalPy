from . import *
import itertools as it
from copy import deepcopy
from typing import *

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.linear_model
from statsmodels.api import GLM, families
from tqdm.auto import tqdm


class GLMPredictor(LINGAMPredictor):

    def __init__(
        self,
        glm_family: families.Family,
        alpha: float = 0.05,
        ignored_subsets: Optional[Set] = None,
        nr_parents_limit: Optional[int] = None,
        **kwargs,
    ):
        """

        Parameters
        ----------
        alpha
        fit_intercept
        residual_test
        filter_variables
        filter_method
        ignored_subsets : (optional) Set
            Specific subsets of variables, that are not to be checked for causal parents.
            The method will still check the variables occurring in these subsets in other combinations!
        nr_parents_limit : (optional) int
            The upper limiting number of causal parents to consider. If not given, the method will search in all
            possible, not excluded subsets.
        """
        super().__init__(filter_variables=False, **kwargs)
        self.glm_family = glm_family
        self.ignored_subsets: Optional[Set] = ignored_subsets
        self.nr_parents_limit: Optional[int] = nr_parents_limit

        self.alpha: float = alpha
        self.accepted_sets: Set = set()

    def _phi_estimator(self, target: np.ndarray, mu_pred: np.ndarray, n_envs: int, n_obs: int):
        n = len(target)
        phi_hat = 1 / (n - n_envs * (n_obs + 1)) * np.sum((target - mu_pred) ** 2 / self.glm_family.variance(mu_pred))
        return phi_hat

    def _test_plausible_parents(
        self,
        obs: np.ndarray,
        target: np.ndarray,
        envs: Dict,
        s: Union[np.ndarray, List, Tuple],
    ):
        if not len(s):
            obs_S = np.ones((self.n, 1))
        else:
            obs_S = sklearn.preprocessing.add_dummy_feature(obs[:, s])
        glm_reg = GLM(target, obs_S, family=self.glm_family).fit()
        mu_pred = glm_reg.predict(obs_S)
        deviance_total = glm_reg.deviance
        phi_S = self._phi_estimator(target, mu_pred, len(envs), len(obs_S))
        deviances_env = dict()
        for env, env_indices in envs.items():
            deviances_env[env] = GLM(target[env_indices], obs_S[env_indices], family=self.glm_family).fit().deviance
        a = (obs_S.shape[1] + 1) * (len(envs) - 1)
        b = obs_S.shape[0] * (len(envs) - 1) * (obs_S.shape[1] + 1)

        test_score = 1 / a * deviance_total - sum(deviances_env.values()) / phi_S
        p_value = scipy.stats.f.cdf(test_score, a, b)

        return p_value
