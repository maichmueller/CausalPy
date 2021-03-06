from .icpbase import ICPredictor

import itertools as it
from copy import deepcopy
from typing import *

import numpy as np
import pandas as pd
import scipy.stats
import sklearn.linear_model


class LinPredictor(ICPredictor):
    def __init__(
        self,
        alpha: float = 0.05,
        fit_intercept: bool = True,
        residual_test: str = "ranks",
        filter_variables: bool = True,
        filter_method: str = "lasso_sqrt",
        use_sklearn: bool = False,
        ignored_subsets: Optional[Set] = None,
        nr_parents_limit: Optional[int] = None,
        **kwargs,
    ):
        """
        Predictor class for Linear Acyclic Models.

        This predictor allows to estimate the causal parents of a target variable given interventional data.
        Full identifiability of the target variable is theoretically guaranteed only when certain conditions are
        fulfilled. For a list of these, check reference [1]_.

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

        References
        ----------
        [1] J. Peters, P. Bühlmann, N. Meinshausen:
        Causal inference using invariant prediction: identification and confidence intervals, arXiv:1501.01332,
        Journal of the Royal Statistical Society, Series B (with discussion) 78(5):947-1012, 2016.
        """
        super().__init__(**kwargs)
        self.fit_intercept = fit_intercept
        self.residual_test = residual_test
        self.filter_variables = filter_variables
        self.filter_method = filter_method
        self.ignored_subsets: Optional[Set] = ignored_subsets
        self.nr_parents_limit: Optional[int] = nr_parents_limit

        self.regression_function = (
            self._regression_sklearn if use_sklearn else self._regression_analytically
        )

        self.alpha: float = alpha
        self.accepted_sets: Set = set()

    def filter_candidates(self, obs, target, nr_iter=100):
        if self.filter_method == "lasso_sqrt":
            _, filtered, _ = sklearn.linear_model.lars_path(
                obs, target, method="lasso", max_iter=nr_iter, return_path=False
            )
        else:
            raise NotImplementedError(
                f"Method for filtering type '{self.filter_method}' not yet implemented."
            )
        return filtered

    def infer(
        self,
        obs: Union[pd.DataFrame, np.ndarray],
        target_variable: Union[int, str],
        envs: Optional[np.ndarray] = None,
        alpha: Optional[float] = None,
        *args,
        **kwargs,
    ):
        r"""
        Perform Linear Invariant Causal Prediction (LinICP).

        Parameters
        ----------
        obs : (n, p) ndarray or DataFrame
            The data of all environment observations from the variables of interest.
        target_variable : int or ``obs`` DataFrame column accessor
            The target variable to perform causal parent identification on.
        envs : (optional) (n,) ndarray
            Array of environment indices for the observation dataset. If not provided, defaults to all
            observations stemming from the same environment.
        alpha : (optional) float
            Significance level of the test. P(\hat{S} \subset S^*) \gte 1-`alpha`

        Returns
        -------
        tuple
            The identified causal parent set, \hat{S}, as tuple of variable names.
        """
        if alpha is None:
            alpha = self.alpha

        if envs is None:
            envs = np.zeros(len(obs))

        obs, target, environments = self.preprocess_input(obs, target_variable, envs)
        if self.filter_variables:
            pre_filtered_vars = np.sort(
                self.filter_candidates(obs, target, kwargs.pop("nr_iter", 100))
            )
            obs = obs[:, pre_filtered_vars]
            p = obs.shape[1]

            self.index_to_varname = self.index_to_varname.reindex(
                pre_filtered_vars
            ).reset_index(drop=True)

            self.varname_to_index = pd.Series(
                self.index_to_varname.index, index=self.index_to_varname.values
            )
        else:
            p = self.p

        subset_iterator = self._subset_iterator(
            elements=self.index_to_varname.values,
            rejected_subsets=self.ignored_subsets,
            nr_parents_limit=self.nr_parents_limit,
        )
        p_values_per_elem = np.zeros(p)

        subset, finished = next(subset_iterator)
        while not finished:
            subset_indices = self.varname_to_index[list(subset)].values
            p_value = self._test_plausible_parents(
                obs, target, environments, subset_indices
            )
            if subset:
                # this if condition excludes the test case of the empty set
                p_values_per_elem[subset_indices] = np.maximum(
                    p_values_per_elem[subset_indices], p_value
                )
            rejected = p_value <= alpha
            if not rejected:
                self.accepted_sets.add(subset)
                self.logger.debug(
                    f"Subset: {subset}\t p-value: {p_value} --> accepted."
                )
            else:
                self.logger.debug(
                    f"Subset: {subset}\t p-value: {p_value} --> rejected."
                )

            subset, finished = subset_iterator.send(rejected)

        # the subset variable will hold the latest best estimation of the parent set of the target.
        if self.accepted_sets:
            parents = subset
        else:
            parents = tuple()
        self.logger.debug(f"Final parents set {parents}")

        p_values_per_elem = pd.Series(
            1 if not parents else p_values_per_elem,
            index=[self.index_to_varname[i] for i in np.arange(p)],
        )  # add variable names information to the data
        return parents, p_values_per_elem

    @staticmethod
    def _regression_sklearn(x, y):
        lr = sklearn.linear_model.LinearRegression(fit_intercept=True).fit(x, y)
        residuals = y - lr.predict(x)
        coeffs = lr.coef_.copy()
        coeffs[0] = lr.intercept_
        return residuals, coeffs

    @staticmethod
    def _regression_analytically(x, y):
        x_T = x.transpose()
        beta = np.linalg.inv(x_T @ x) @ x_T @ y
        residuals = y - x @ beta
        return residuals, beta

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
        # opting here to allow for an analytical regression calculation speeds up the computation significantly.
        # This might come at the cost of some numerical stability, which might be provided by sklearn, but not by a
        # direct, analytical beta calculation.
        residuals, beta = self.regression_function(obs_S, target)
        self.logger.debug(
            f"Subset: {tuple(self.index_to_varname[i] for i in s)}\t regression coefficients: {beta}"
        )
        p_value = 1
        # the paper suggests to test the residual of data in an environment e against the
        # the residuals of the data not in e.
        # TODO: find out, whether this isn't equivalent to the slightly faster method of testing the residuals of
        # TODO: each environment e against environment e + 1.
        all_indices = np.arange(self.n)
        for env, env_indices in envs.items():
            p_value_update = self.residuals_test(
                residuals[env_indices],
                residuals[np.delete(all_indices, env_indices)],
                test=self.residual_test,
            )
            p_value = min(p_value, p_value_update)
        return p_value * len(envs)  # Bonferroni correction for p value

    @staticmethod
    def residuals_test(sample1, sample2, test="normal", **kwargs):
        """
        Test for the equality of the distribution of input 1 versus 2.

        This method performs a different test depending on the ``test`` parameter. See the parameter documentation for
        more information.

        Parameters
        ----------
        sample1 : (n, p) np.ndarray,
            The data pertaining to the first gaussian.
        sample2 : (n, p) np.ndarray,
            The data pertaining to the second gaussian.
        test : "normal", "ranks", "ks" or Callable,
            For 'normal' a t-test on the mean and an F-test on the variance are performed.
            For 'ranks' the F-test is switched with Levene's test on equal variance.
            For 'ks' a Kolmogorov-Smirnov test on equal distribution is applied.
            For a Callable, the method is called with 'test(sample1, sample2)' and assumed to return a p-value.
        kwargs: kwargs,
            Arguments passable to the scipy.stats.levene, Kolomogorov-Smirnov or callable test method, depending on
            which has been chosen. More information can be found in [1]_ and [2]_.

        Returns
        -------
        tuple
            A 2-tuple of the test p-values of the equal mean test and the equal variance test.

        References
        ----------
        [1] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.levene.html
        [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html
        [3] https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ttest_ind.html
        """
        if test == "normal":
            n1 = len(sample1)
            n2 = len(sample2)
            mean1 = np.mean(sample1)
            mean2 = np.mean(sample2)
            # ddof = 0 as per numpy doc: "ddof = 0 provides the ML estimation of variance for normally distributed
            # variables"
            std1 = np.std(sample1, ddof=0)
            std2 = np.std(sample2, ddof=0)
            p_equal_mean = scipy.stats.ttest_ind_from_stats(
                mean1, std1, n1, mean2, std2, n2, equal_var=False
            ).pvalue
            f_test = (std1 ** 2) / (std2 ** 2)
            p_equal_var = 2 * min(
                scipy.stats.f.cdf(f_test, n1 - 1, n2 - 1),
                scipy.stats.f.sf(f_test, n1 - 1, n2 - 1),
            )
            p_val = 2 * min(p_equal_mean, p_equal_var)

        elif test == "ks":
            p_val = scipy.stats.ks_2samp(sample1, sample2, **kwargs).pvalue

        elif test == "ranks":
            p_equal_mean = scipy.stats.ttest_ind(
                sample1, sample2, equal_var=False
            ).pvalue
            #  the Levene test is apparently more robust than the standard F-test given the sensitivity of the F-test
            # to non-normality of the data.
            p_equal_var = scipy.stats.levene(sample1, sample2, **kwargs).pvalue
            p_val = 2 * min(p_equal_mean, p_equal_var)

        elif isinstance(test, Callable):
            p_val = test(sample1, sample2, **kwargs)

        else:
            raise ValueError(f"Test input parameter '{test}' not supported.")

        return p_val

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

            def previously_rejected(s):
                return False

        else:
            rejected_subsets = set(rejected_subsets)

            def previously_rejected(s):
                return s in rejected_subsets

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
            if previously_rejected(subset):
                # the intersected subset has already been checked, moving on
                continue

            rejected = yield subset, False

            if not rejected:
                candidates = candidates.intersection(subset)
                if not candidates:
                    # empty candidates set means we have found no causal parent as there were at least two accepted
                    # subsets with empty intersection.
                    yield tuple(), True

        yield tuple(candidates), True
