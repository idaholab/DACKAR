# Copyright 2024, Battelle Energy Alliance, LLC  ALL RIGHTS RESERVED
"""
Created on July, 2025

@author: wangc, mandd
"""

# Code modified from https://github.com/emanuele/kernel_two_sample_test
import numpy as np
from sys import stdout
from sklearn.metrics import pairwise_kernels

import logging
logger = logging.getLogger(__name__)


def MMD2u(K, m, n):
    """
      Method designed to perform MMD^2_u unbiased statistic using U-statistics.

      Args:
        K: np.array, 2-D matrix
        m: int, size of first vector
        n: int, size of second vector
      Return:
        val: float, MMD^2_u unbiased statistic using U-statistics
    """
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]
    val = 1.0 / (m * (m - 1.0)) * (Kx.sum() - Kx.diagonal().sum()) + \
          1.0 / (n * (n - 1.0)) * (Ky.sum() - Ky.diagonal().sum()) - \
          2.0 / (m * n) * Kxy.sum()
    return val

def MMD2b(K, m, n):
    """
      Method designed to perform MMD^2 biased statistics using V-statistics

      Args:
        K: np.array, 2-D matrix
        m: int, size of first vector
        n: int, size of second vector
      Return:
        out: float, MMD^2 biased statistics using V-statistics
    """
    Kx = K[:m, :m]
    Ky = K[m:, m:]
    Kxy = K[:m, m:]
    out = 1.0/(m**2) * Kx.sum() + 1.0/(n**2)*Ky.sum() - 2.0/(m*n)*Kxy.sum()
    return out

def MMD2u_UCB(K, m, alpha=0.05):
    """
      Method designed to calculate the uniform convergence bound for MMD2u
      
      Args:
        K: np.array, 2-D matrix
        m: int, sample size
        alpha: float, acceptance value for hypothesis testing
      Return:
        ucb: float, uniform convergence bound for MMD2u
    """
    assert alpha > 0 and alpha < 1, f'alpha should be within [0,1], but got {alpha}'
    Kxy = K[:m, m:]
    maxKxy = np.max((0,np.max(Kxy)))
    ucb = 4.0*maxKxy/np.sqrt(m) * np.sqrt(np.log(1.0/alpha))
    return ucb


def MMD2b_UCB(K, m, alpha=0.05):
    """
      Method designed to calculate the uniform convergence bound for MMD2b
      
      Args:
        K: np.array, 2-D matrix
        m: int, sample size
        alpha: float, acceptance value for hypothesis testing
      Return:
        ucb: float, uniform convergence bound for MMD2b
    """
    assert alpha > 0 and alpha < 1, f'alpha should be within [0,1], but got {alpha}'
    Kxy = K[:m, m:]
    maxKxy = np.max((0,np.max(Kxy)))
    ucb = np.sqrt(2.0*maxKxy/m) * (1.0+np.sqrt(2.0*np.log(1.0/alpha)))
    return ucb


def compute_null_distribution(K, m, n, iterations=1000, verbose=False,
                              random_state=None, marker_interval=500):
    """
      Method designed to calculate the bootstrap null-distribution of MMD2u.
      
      Args:
        K: np.array, 2-D matrix
        m: int, size of first vector
        n: int, size of second vector
        iterations: int, number of bootstrap iterations
        verbose: bool, flag to provide calculation details
        random_state: np class, numpy random number generator class
        marker_interval: int, interval where calculation details are displayed
      Return:
        mmd2u_null: np.array, null-distribution of MMD2u
    """

    if type(random_state) == type(np.random.RandomState()):
        rng = random_state
    else:
        rng = np.random.RandomState(random_state)

    mmd2u_null = np.zeros(iterations)
    for i in range(iterations):
        if verbose and (i % marker_interval) == 0:
            logger.info(i),
            stdout.flush()
        idx = rng.permutation(m+n)
        K_i = K[idx, idx[:, None]]
        mmd2u_null[i] = MMD2u(K_i, m, n)
    return mmd2u_null


def compute_null_distribution_given_permutations(K, m, n, permutation,
                                                 iterations=None):
    """
      Method designed to calculate the bootstrap null-distribution of MMD2u given predefined permutations.
      
      Args:
        K: np.array, 2-D matrix
        m: int, size of first vector
        n: int, size of second vector
        permutation: np.array, array of permutations
        iterations: int, number of bootstrap iterations
      Return:
        mmd2u_null: np.array, null-distribution of MMD2u given predefined permutations
    """
    if iterations is None:
        iterations = len(permutation)

    mmd2u_null = np.zeros(iterations)
    for i in range(iterations):
        idx = permutation[i]
        K_i = K[idx, idx[:, None]]
        mmd2u_null[i] = MMD2u(K_i, m, n)

    return mmd2u_null


def kernel_two_sample_test(X, Y, kernel_function='rbf', iterations=2000,
                           verbose=False, random_state=None, alpha=0.05, thin=None, **kwargs):
    """
      Method designed to calculate MMD^2_u, its null distribution and the p-value of the kernel two-sample test.

      Args:
        X: np.array, first vector
        Y: np.array, second vector
        kernel_function: string, type of kenerl function. Valid values are: additive_chi2, chi2, 
                                 linear, poly, polynomial, rbf, laplacian, sigmoid, cosine
        iterations: int, number of iterations
        verbose: bool, flag to provide calculation details
        random_state: np class, numpy random number generator class
        alpha: float, acceptance value for hypothesis testing
        thin: int, sample size for thinning calculation
        **kwargs: dict, dictionary of parameteres that are passed to pairwise_kernels() as kernel parameters. 
                        E.g. if kernel_two_sample_test(..., kernel_function='rbf', gamma=0.1), then this will 
                        result in getting the kernel through kernel_function(metric='rbf', gamma=0.1).
      Return:
        mmd2u: float, MMD^2_u unbiased statistic using U-statistics
        mmd2u_null: np.array, null-distribution of MMD2u
        p_value: float, calculated p-value for hypothesis testing
    """

    if len(X.shape) == 1:
        X = X.reshape((-1,1))
    if len(Y.shape) == 1:
        Y = Y.reshape((-1,1))
    if thin is not None:
        try:
            thin = int(thin)
        except ValueError:
            logger.error(f'Try to perform thin calculation, but get non-integer value for "thin: {thin}"')
            thin = None
    if thin is not None:
        X = X[0::thin, :]
        Y = Y[0::thin, :]
    m = len(X)
    n = len(Y)
    XY = np.vstack([X, Y])
    K = pairwise_kernels(XY, metric=kernel_function, **kwargs)
    mmd2u = MMD2u(K, m, n)

    if verbose:
        logger.info("MMD^2_u = %s" % mmd2u)
        logger.info("Computing the null distribution.")
    ucb = MMD2u_UCB(K, m, alpha=0.05)
    if verbose:
        logger.info(f"UCB bound for MMD^2_u is {ucb}")
    mmd2u_null = compute_null_distribution(K, m, n, iterations,
                                           verbose=verbose,
                                           random_state=random_state)
    p_value = max(1.0/iterations, (mmd2u_null > mmd2u).sum() /
                  float(iterations))
    if verbose:
        logger.info("p-value ~= %s \t (resolution : %s)" % (p_value, 1.0/iterations))
        if p_value < alpha:
            logger.info(f"Deviaition from null hyothesis is statistically significant at test level {alpha}, reject null hyothesis, \n \
                         i.e., the two samples are from different distributions!")

    return mmd2u, mmd2u_null, p_value

def chebyshevTesting(X, Y, kernel_function='rbf', iterations=2000,
                           verbose=False, random_state=None, alpha=0.01, **kwargs):
    """
      Method designed to perform Chebyshev testing using Chebyshev's inequality

      Args:
        X: np.array, first vector
        Y: np.array, second vector
        kernel_function: string, type of kernel function
        iterations: int, number of bootstrap iterations
        verbose: bool, flag to provide calculation details
        random_state: np class, numpy random number generator class
        alpha: float, acceptance value for hypothesis testing
        **kwargs: dict, dictionary of parameteres that are passed to pairwise_kernels() as kernel parameters.
      Return:
        accept: bool, outcome of Chebyshev testing          
    """
    accept = False
    mmd2u, mmd2u_null, p_value = kernel_two_sample_test(X, Y, kernel_function=kernel_function, iterations=iterations,
                           verbose=verbose, random_state=random_state, alpha=alpha, **kwargs)
    accept = chebyshevTesting_precomputed_mmd(mmd2u, mmd2u_null, alpha)
    return accept

def chebyshevTesting_precomputed_mmd(mmd2u, mmd2u_null, alpha=0.01):
    """
      Method designed to perform MMD Chebyshev testing 

      Args:
        mmd2u: float, MMD^2_u unbiased statistic using U-statistics
        mmd2u_null: np.array, null-distribution of MMD2u
        alpha: float, acceptance value for hypothesis testing
      Return:
        accept: bool, outcome of Chebyshev testing 
    """
    accept = False
    mu = mmd2u_null.mean()
    sigma = np.std(mmd2u_null)
    nu = 1.0/(np.sqrt(alpha))
    diff = abs(mmd2u-mu)
    if diff < nu * sigma:
        accept = True
        logger.info(f"Chebyshev Inequality holds: {diff} < {nu*sigma}")
    else:
        accept = False
        logger.info(f"Chebyshev Inequality does not hold: {diff} > {nu*sigma}")
    return accept
