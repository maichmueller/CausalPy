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

    def phi_estimator(self, target: np.ndarray, mu_pred: np.ndarray, n_envs: int, n_obs: int):
        n = len(target)
        phi_hat = 1 / (n - n_envs * (n_obs + 1)) * np.sum((target - mu_pred) ** 2 / self.glm_family.variance(mu_pred))
        self.glm_family.variance

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
        lr.fit(obs_S, target)
        residuals = target - lr.predict(obs_S)
        p_value = 1
        # the paper suggests to test the residual of data in an environment e against the
        # the residuals of the data not in e.
        # TODO: find out, whether this isn't equivalent to the slightly faster method of testing the residuals of
        # TODO: each environment e against environment e + 1.
        for env in envs:
            env_indices = envs[env]
            p_value_update = self.residuals_test(
                residuals[env_indices],
                residuals[np.logical_not(env_indices)],
                test=self.residual_test,
            )
            p_value = min(p_value, p_value_update)
        return p_value * len(envs)  # Bonferroni correction for p value
