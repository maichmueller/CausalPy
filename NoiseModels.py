import numpy as np
from numpy.random import Generator, PCG64
from functools import partial


class NoiseGenerator:
    """
    A simple feed forward convenience class to generate different numpy provided distributions to the user.
    Selectable distributions are specified by the numpy docs:

    -- Beta distribution
        - beta(a, b[, size])
    -- Binomial distribution
        - binomial(n, p[, size])
    -- chi-square distribution
        - chisquare(df[, size])
    -- Dirichlet distribution
        - dirichlet(alpha[, size])
    -- exponential distribution
        - exponential([scale, size])
    -- F distribution
        - f(dfnum, dfden[, size])
    -- Gamma distribution
        - gamma(shape[, scale, size])
    -- geometric distribution
        - geometric(p[, size])
    -- Gumbel distribution
        - gumbel([loc, scale, size])
    -- Hypergeometric distribution
        - hypergeometric(ngood, nbad, nsample[, size])
    -- Laplace or double exponential distribution with specified location (or mean) and scale (decay)
        - laplace([loc, scale, size])
    -- logistic distribution
        - logistic([loc, scale, size])
    -- log-normal distribution
        - lognormal([mean, sigma, size])
    -- logarithmic series distribution
        - logseries(p[, size])
    -- multinomial distribution
        - multinomial(n, pvals[, size])
    -- multivariate normal distribution
        - multivariate_normal(mean, cov[, size, …])
    -- negative binomial distribution
        - negative_binomial(n, p[, size])
    -- noncentral chi-square distribution
        - noncentral_chisquare(df, nonc[, size])
    -- noncentral F distribution
        - noncentral_f(dfnum, dfden, nonc[, size])
    -- normal (Gaussian) distribution
        - normal([loc, scale, size])
    -- Pareto II or Lomax distribution with specified shape
        - pareto(a[, size])
    -- Poisson distribution
        - poisson([lam, size])
    -- [0, 1] from a power distribution with positive exponent a - 1
        - power(a[, size])
    -- Rayleigh distribution
        - rayleigh([scale, size])
    -- standard Cauchy distribution with mode = 0
        - standard_cauchy([size])
    -- standard exponential distribution
        - standard_exponential([size, dtype, method, out])
    -- standard Gamma distribution
        - standard_gamma(shape[, size, dtype, out])
    -- standard Normal distribution (mean=0, stdev=1)
        - standard_normal([size, dtype, out])
    -- standard Student’s t distribution with df degrees of freedom
        - standard_t(df[, size])
    -- triangular distribution over the interval [left, right]
        - triangular(left, mode, right[, size])
    -- uniform distribution
        - uniform([low, high, size])
    -- von Mises distribution
        - vonmises(mu, kappa[, size])
    -- Wald, or inverse Gaussian, distribution
        - wald(mean, scale[, size])
    -- Weibull distribution
        - weibull(a[, size])
    -- Zipf distribution
        - zipf(a[, size])
    """

    def __init__(self, distribution: str = "", **distribution_kwargs):
        rg = Generator(PCG64())
        self.distribution = partial(eval(f"rg.{distribution}"), **distribution_kwargs)

    def __call__(self, size=1):
        return self.distribution(size=size)
