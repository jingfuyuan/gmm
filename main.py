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

# X = np.loadtxt("toy_data.txt")
X = np.loadtxt("netflix_incomplete.txt")
# TODO: Your code here
# Ks = [1,2,3,4]
# seeds = [0, 1, 2, 3, 4]
# costs = []
# for k in Ks:
#     c = []
#     for s in seeds:
#         mixture, post = common.init(X, k, s)
#         k_mixture, k_post, cost = kmeans.run(X, mixture, post)
#         print(f"Cost for K={k} and seed={s} is {cost}")
#         c.append(cost)
#     costs.append(c)

# df = pd.DataFrame(costs, index=Ks, columns=seeds)
# print(df)

# mixture, post = common.init(X, 3, 0)
# naive_em.run(X, mixture, post)

# Ks = [1,2,3,4]
# seeds = [0, 1, 2, 3, 4]
# bics = []
# for k in Ks:
#     c = []
#     for s in seeds:
#         mixture, post = common.init(X, k, s)
#         mixture, post, ll = naive_em.run(X, mixture, post)
#         b = common.bic(X, mixture, ll)
#         print(f"BIC for K={k} and seed={s} is {b}")
#         c.append(b)
#     bics.append(c)

# df = pd.DataFrame(bics, index=Ks, columns=seeds)
# print(df)

# plt.scatter(X[:,0], X[:,1])
# plt.show()

# mixture, post = common.init(X, K = 3, seed = 3)
# mixture_k, post_k, cost = kmeans.run(X, mixture, post)
# mixture_em, post_em, ll = naive_em.run(X, mixture, post)
# print(mixture_k)
# print(mixture_em)

# plt.subplot(1,2,1)
# plt.scatter(X[:,0], X[:,1], c=post_k.argmax(axis=1))
# plt.subplot(1,2,2)
# plt.scatter(X[:,0], X[:,1], c=post_em.argmax(axis=1))
# plt.show()

print(X.shape)

# Ks = [1, 12]
# seeds = [0, 1, 2, 3, 4]
# lls = []
# for k in Ks:
#     c = []
#     for s in seeds:
#         mixture, post = common.init(X, k, s)
#         mixture, post, ll = em.run(X, mixture, post)
#         print(f"Log likelihood for K={k} and seed={s} is {ll}")
#         c.append(ll)
#     lls.append(c)

# df = pd.DataFrame(lls, index=Ks, columns=seeds)
# print(df)

X_gold = np.loadtxt("netflix_complete.txt")

mixture, post = common.init(X, 12, 1)
mixture, post, ll = em.run(X, mixture, post)
X_pred = em.fill_matrix(X, mixture)
print(common.rmse(X_gold, X_pred))