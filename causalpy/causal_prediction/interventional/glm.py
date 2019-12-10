from causalpy.causal_prediction.interventional import *
import itertools as it
from copy import deepcopy
from typing import *

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.linear_model
from statsmodels.api import GLM, families
from tqdm.auto import tqdm


class GLMPredictor(LinPredictor):
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

    def _phi_estimator(
        self, target: np.ndarray, mu_pred: np.ndarray, n_envs: int, n_features: int
    ):
        n = len(target)
        phi_hat = np.sum(
            (target - mu_pred) ** 2 / self.glm_family.variance(mu_pred)
        ) / (n - n_envs * (n_features + 1))
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

        n_envs = len(envs)

        glm_reg = GLM(target, obs_S, family=self.glm_family).fit()
        deviance_total = glm_reg.deviance

        mu_pred = np.empty(obs.shape[0])
        deviances_env = dict()
        for env, env_indices in envs.items():
            env_data = obs_S[env_indices]
            env_fit = GLM(
                target[env_indices], env_data, family=self.glm_family
            ).fit()
            deviances_env[env] = env_fit.deviance
            mu_pred[env_indices] = env_fit.predict(env_data)

        phi_S = self._phi_estimator(target, mu_pred, n_envs, obs_S.shape[1])
        a = (obs_S.shape[1] + 1) * (n_envs - 1)
        b = obs_S.shape[0] - n_envs * (obs_S.shape[1] + 1)

        test_score = (deviance_total - sum(deviances_env.values())) / phi_S / a
        p_value = 2 * min(
            scipy.stats.f.cdf(test_score, a, b), scipy.stats.f.sf(test_score, a, b)
        )

        return p_value
