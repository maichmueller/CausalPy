import itertools as it
from collections import namedtuple
from copy import deepcopy
from typing import *

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.linear_model
from abc import ABC, abstractmethod


class ICP(ABC):
    @abstractmethod
    def infer(
            self,
            obs: Union[pd.DataFrame, np.ndarray],
            envs: np.ndarray,
            target: Union[int, str],
            alpha: float = 0.05,
            *args,
            **kwargs,
    ):
        pass


class LinICP(ICP):
    def __init__(
            self, fit_intercept: bool = True,
            noise_assumption: str = None,
            filter_variables: bool = True,
            filter_method: str = "lasso_sqrt"
    ):

        self.fit_intercept = fit_intercept
        self.noise_assumption = noise_assumption
        self.filter_variables = filter_variables
        self.filter_method = filter_method

        self.n, self.p = None, None
        self.p_filtered = None
        self.alpha = None
        self.target_name = None
        self.index_to_varname = None
        self.varname_to_index = None
        self.variables = None
        self.target = None
        self.obs = None
        self.environments = None

    @staticmethod
    def pre_filter_candidates(obs, target, nr_iter=100):
        _, filtered, _ = sklearn.linear_model.lars_path(
            obs, target, method="lasso", max_iter=nr_iter, return_path=False
        )
        return filtered

    def infer(
            self,
            obs: Union[pd.DataFrame, np.ndarray],
            envs: np.ndarray,
            target: Union[int, str],
            alpha: float = 0.05,
            prefilter_variables=True,
            ignored_subsets: Set = None,
            nr_parents_limit: int = None,
            **kwargs,
    ):
        r"""
        Perform Linear Invariant Causal Prediction (ICP) on the data provided on object construction.
        This method assumes the data stems from a linear gaussian SCM, i.e.
         - all assignment functions are linear in their parents and noise variables,
         - the residuals follow a gaussian distribution.

        For details refer to [1]_.

        Parameters
        ----------
        alpha : float
            Significance level of the test. P(\hat{S} \subset S^*) \gte 1-`alpha`
        obs : (n, p) ndarray or DataFrame
            The data of all environment observations from the variables of interest.
        envs : (n,) ndarray
            Array of environment indices for the observation dataset.
        target : int or ``obs`` DataFrame column accessor
            The target variable to perform causal parent identification on.
        ignored_subsets : (optional) Set
            Specific subsets of variables, that are not to be checked for causal parents.
            The method will still check the variables occurring in these subsets in other combinations!
        nr_parents_limit : (optional) int
            The upper limiting number of causal parents to consider. If not given, the method will search in all
            possible, not excluded subsets.

        Returns
        -------
        tuple
            The identified causal parent set, \hat{S}, as tuple of variable names.

        References
        ----------
        [1] J. Peters, P. Bühlmann, N. Meinshausen:
        Causal inference using invariant prediction: identification and confidence intervals, arXiv:1501.01332,
        Journal of the Royal Statistical Society, Series B (with discussion) 78(5):947-1012, 2016.
        """
        self.n, self.p = obs.shape[0], obs.shape[1] - 1
        assert (
            len(envs) == self.n,
            "Number of observation samples and number of environment labels have to be equal.",
        )
        self.alpha = alpha
        self.target_name = target

        if isinstance(obs, pd.DataFrame):
            self.target = obs[target].to_numpy().flatten()
            self.obs = obs.drop(columns=[target]).to_numpy()
            self.index_to_varname = pd.Series(obs.columns.drop(target), index=range(self.p))
            self.varname_to_index = pd.Series(range(self.p), index=obs.columns.drop(target))
            self.variables = obs.columns.values

        elif isinstance(obs, np.ndarray):
            self.index_to_varname = pd.Series(np.arange(self.p))
            self.variables = self.index_to_varname.values
            self.target = obs[:, target].flatten()
            self.obs = np.delete(obs, target, axis=1)
            self.varname_to_index = (self.index_to_varname.loc[target + 1:] - 1).drop(index=target)
        else:
            raise ValueError(
                "Observations have to be either a pandas DataFrame or numpy ndarray."
            )

        if prefilter_variables:
            pre_filtered_vars = np.sort(
                self.pre_filter_candidates(
                    self.obs, self.target, kwargs.pop("nr_iter", 100)
                )
            )
            self.obs = self.obs[:, pre_filtered_vars]
            preds = self.obs.shape[1]
            self.index_to_varname = self.index_to_varname.reindex(
                pre_filtered_vars
            ).reset_index(drop=True)
            self.varname_to_index = pd.Series(
                self.index_to_varname.index, index=self.index_to_varname.values
            )
        else:
            preds = self.p

        self.environments = {env: envs == env for env in np.unique(envs)}

        subset_iterator = self._subset_iterator(
            elements=self.index_to_varname.values,
            rejected_subsets=ignored_subsets,
            nr_parents_limit=nr_parents_limit,
        )
        p_values_per_elem = np.zeros(preds)

        subset, finished = next(subset_iterator)
        while not finished:
            subset_indices = self.varname_to_index[list(subset)].values
            p_value = self._test_plausible_parents(subset_indices)
            if subset:
                # this if condition excludes the test case of the empty set
                p_values_per_elem[subset_indices] = np.maximum(
                    p_values_per_elem[subset_indices], p_value
                )
            subset, finished = subset_iterator.send(p_value <= self.alpha)
        # once the routine has finished the variable subset will hold the latest estimate of the causal parents

        p_values_per_elem = pd.Series(
            1 if not subset else p_values_per_elem,
            index=[self.index_to_varname[i] for i in np.arange(preds)],
        )  # add variable names information to the data
        return subset, p_values_per_elem

    @staticmethod
    def test_residuals(sample1, sample2, test="normal", **levene_kwargs):
        """
        Test for the equality of the distribution of input 1 versus 2.
        This test is supposed to be performed when the samples are assumed to follow a gaussian distribution, which
        is fully defined by mean and variance.

        This method performs a t-test for equal mean and Levene's test for equal variance.

        Parameters
        ----------
        sample1 : (n, p) np.ndarray,
            The data pertaining to the first gaussian.
        sample2 : (n, p) np.ndarray,
            The data pertaining to the second gaussian.
        levene_kwargs: kwargs,
            Arguments passable to the scipy.stats.levene method. More information can be found in [1]_

        Returns
        -------
        tuple
            A 2-tuple of the test p-values of the equal mean test and the equal variance test.

        References
        ----------
        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html
        [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
        """
        if test == "ranks":
            p_equal_mean = scipy.stats.ttest_ind(sample1, sample2, equal_var=False).pvalue
            #  the levene test is apparently more robust than the standard F-test given the sensitivity of the F-test
            # to non-normality of the data.
            p_equal_var = scipy.stats.levene(sample1, sample2, **levene_kwargs).pvalue
            p_val = 2 * min(p_equal_mean, p_equal_var)
        elif isinstance(test, Callable):
            p_val = test(sample1, sample2)
        elif "normal":
            # TODO: compute myself the statistics
            p_equal_mean = scipy.stats.ttest_ind(sample1, sample2, equal_var=False).pvalue
            p_equal_var = scipy.stats.levene(sample1, sample2, **levene_kwargs).pvalue
        return

    def _test_plausible_parents(self, s: Union[np.ndarray, List, Tuple]):
        if not len(s):
            obs_S = np.ones((self.n, 1))
        else:
            obs_S = sklearn.preprocessing.add_dummy_feature(self.obs[:, s])
        lr = sklearn.linear_model.LinearRegression(fit_intercept=True)
        lr.fit(obs_S, self.target)
        residuals = lr.predict(obs_S) - self.target
        p_value = 1
        # the paper suggests to test the residual of data in an environment e against the
        # the residuals of the data not in e.
        # TODO: find out, whether this isn't equivalent to the slightly faster method of testing the residuals of
        # TODO: each environment e against environment e + 1.
        for env in self.environments:
            env_indices = self.environments[env]
            p_value_update = self.test_residuals(
                residuals[env_indices], residuals[np.logical_not(env_indices)]
            )
            p_value = min(p_value, p_value_update)
        return p_value * len(self.environments)   # Bonferroni correction for p value

    @staticmethod
    def _subset_iterator(
            elements: Union[int, Collection, Set],
            candidates: Set[Tuple] = None,
            rejected_subsets: Set[Tuple] = None,
            nr_parents_limit: int = None,
    ):
        if isinstance(elements, int):
            elements = set(range(elements))
        elif isinstance(elements, Collection):
            elements = set(elements)
        else:
            raise ValueError(
                f"Expected 'elements' to be a sequence of identifiers. "
                f"Got {type(elements)} instead."
            )

        if rejected_subsets is None:
            rejected_subsets = set()
        if candidates is None:
            if not isinstance(elements, Set):
                candidates = set(elements)
            else:
                candidates = deepcopy(elements)
        else:
            candidates = set(candidates)

        if nr_parents_limit is None:
            nr_parents_limit = len(elements)

        for subset in it.chain.from_iterable(
                it.combinations(elements, len_subset)
                for len_subset in range(nr_parents_limit + 1)
        ):
            subset = candidates.intersection(subset)
            # converting to tuple here is necessary as sets are mutable, thus not hashable
            subset_tup = tuple(subset)
            if subset_tup in rejected_subsets:
                # the intersected subset has already been checked, moving on
                continue

            rejected = yield (subset_tup, False)

            if not rejected:
                candidates = subset
                if not candidates:
                    # empty candidates set means we have found no causal parent as there were at least two accepted
                    # subsets with empty intersection.
                    yield tuple(), True

            else:
                rejected_subsets.add(subset_tup)

        yield tuple(candidates), True
