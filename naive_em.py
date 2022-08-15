"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    post = []
    ll = 0
    for x in X:
        mu, var, p = mixture
        var = np.reshape(var, (-1,1))
        pdf = 1/np.sqrt(2*np.pi*var) * np.exp(-1/(2*var) * (x-mu)**2)
        multi_norm = np.prod(pdf, axis=1)
        weighted_norm = p * multi_norm
        ll += np.log(np.sum(weighted_norm))
        post.append(weighted_norm/np.sum(weighted_norm))
    return np.array(post), ll




def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, K = post.shape
    d = X.shape[1]
    p_j = post.sum(axis=0)
    mu_hat = np.dot(post.T, X) / p_j.reshape((-1, 1))
    sigma2 = []
    for j in range(K):
        ss = np.sum((X - mu_hat[j])**2, axis=1)
        sigma2.append(np.sum(post[:,j] * ss) / p_j[j] / d)
    return GaussianMixture(mu_hat, np.array(sigma2), p_j / n)


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    old_ll = None
    new_ll = None
    while (old_ll is None or new_ll - old_ll > 1e-6 * abs(new_ll)):
        old_ll = new_ll
        post, new_ll = estep(X, mixture)
        # print(new_ll)
        mixture = mstep(X, post)

    return mixture, post, new_ll
