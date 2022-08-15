from typing import final
import numpy as np
import kmeans
import common
import naive_em
import em
import naive_em
import pandas as pd
import matplotlib.pyplot as plt

from test import X_gold

X_t = np.loadtxt("toy_data.txt")
X = np.loadtxt("netflix_incomplete.txt")

# TODO: Your code here
# Run the Kmeans algorithm with different Ks and seeds.
def kmeans_cost(X):
    Ks = [1,2,3,4]
    seeds = [0, 1, 2, 3, 4]
    costs = []
    for k in Ks:
        c = []
        for s in seeds:
            mixture, post = common.init(X, k, s)
            k_mixture, k_post, cost = kmeans.run(X, mixture, post)
            print(f"Cost for K={k} and seed={s} is {cost}")
            c.append(cost)
        costs.append(c)
    return pd.DataFrame(costs, index=Ks, columns=seeds)

print(kmeans_cost(X_t))

# Run the naive EM algorithm with different Ks and seeds,  and calculate the BIC
def em_bic(X):
    Ks = [1,2,3,4]
    seeds = [0, 1, 2, 3, 4]
    bics = []
    for k in Ks:
        c = []
        for s in seeds:
            mixture, post = common.init(X, k, s)
            mixture, post, ll = naive_em.run(X, mixture, post)
            b = common.bic(X, mixture, ll)
            print(f"BIC for K={k} and seed={s} is {b}")
            c.append(b)
        bics.append(c)

    return pd.DataFrame(bics, index=Ks, columns=seeds)

print(em_bic(X_t))

# plt.scatter(X[:,0], X[:,1])
# plt.show()

def plot_kmeans_em(X, K, seed):
    mixture, post = common.init(X, K = K, seed = seed)
    mixture_k, post_k, cost = kmeans.run(X, mixture, post)
    mixture_em, post_em, ll = naive_em.run(X, mixture, post)
    print(mixture_k)
    print(mixture_em)

    plt.subplot(1,2,1)
    plt.scatter(X[:,0], X[:,1], c=post_k.argmax(axis=1))
    plt.subplot(1,2,2)
    plt.scatter(X[:,0], X[:,1], c=post_em.argmax(axis=1))
    plt.show()

# plot_kmeans_em(X_t, 3, 3)

# Run the EM algorithm on partialy observed matrix
def run_em(X):
    Ks = [1, 12]
    seeds = [0, 1, 2, 3, 4]
    lls = []
    for k in Ks:
        c = []
        for s in seeds:
            mixture, post = common.init(X, k, s)
            mixture, post, ll = em.run(X, mixture, post)
            print(f"Log likelihood for K={k} and seed={s} is {ll}")
            c.append(ll)
        lls.append(c)

    return pd.DataFrame(lls, index=Ks, columns=seeds)

# print(run_em(X))

# fill missing values in the matrix and calculate the rmse
X_gold = np.loadtxt("netflix_complete.txt")

mixture, post = common.init(X, 12, 1)
mixture, post, ll = em.run(X, mixture, post)
X_pred = em.fill_matrix(X, mixture)
print(common.rmse(X_gold, X_pred))