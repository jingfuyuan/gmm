"""Mixture model for matrix completion"""
import atexit
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
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
        K, d = mu.shape
        mask = np.broadcast_to((x==0), (K, d))
        var = np.reshape(var, (-1,1))
        # pdf = 1/np.sqrt(2*np.pi*var) * np.exp(-1/(2*var) * (x-mu)**2)
        # log_pdf = np.log(pdf + 1e-16)
        log_pdf = np.log(1/np.sqrt(2*np.pi*var)) - 1/(2*var) * (x-mu)**2
        log_pdf[mask] = 0
        sum_log_pdf = log_pdf.sum(axis=1)
        max_log_pdf = np.max(sum_log_pdf)
        ll += logsumexp(sum_log_pdf, b=p)
        adj_norm = p * np.exp(sum_log_pdf - max_log_pdf)
        post.append(adj_norm/np.sum(adj_norm))
    return np.array(post), ll



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    n, K = post.shape
    d = X.shape[1]
    p_j = post.sum(axis=0)
    mu, var, p = mixture
    mu_hat = []
    sigma2 = []
    for j in range(K):
        mu_i = []
        ss = 0
        cu = 0
        for i in range(d):
            mask = (X[:,i] != 0)
            muhat_ji = np.sum(post[mask,j] * X[mask,i]) / np.sum(post[mask, j])
            if np.sum(post[mask, j]) >= 1:
                mu_i.append(muhat_ji)
            else:
                mu_i.append(mu[j,i])
        mu_i = np.array(mu_i)
        mu_hat.append(mu_i)

        for u in range(n):
            mask = (X[u] != 0)
            ss += post[u, j]* np.sum((X[u, mask] - mu_i[mask])**2)
            cu += len(X[u,mask]) * post[u, j]
        var_j = ss/cu if ss/cu >= min_variance else min_variance
        sigma2.append(var_j)
    return GaussianMixture(np.array(mu_hat), np.array(sigma2), p_j / n)



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
        mixture = mstep(X, post, mixture)

    return mixture, post, new_ll


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    X1 = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X1[i,j] = X[i, j]

    mu, var, p = mixture
    K, d = mu.shape
    for i, x in enumerate(X):
        mask = np.broadcast_to((x==0), (K, d))
        var = np.reshape(var, (-1,1))
        log_pdf = np.log(1/np.sqrt(2*np.pi*var)) - 1/(2*var) * (x-mu)**2
        log_pdf[mask] = 0
        sum_log_pdf = log_pdf.sum(axis=1)
        max_log_pdf = np.max(sum_log_pdf)
        adj_norm = p * np.exp(sum_log_pdf - max_log_pdf)
        post = adj_norm/np.sum(adj_norm)
        # k = np.argmax(sum_log_pdf)

        mask = (x==0)
        X1[i, mask] = (post.reshape((-1,1)) * mu).sum(axis=0)[mask]
    
    return X1
