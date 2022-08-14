import numpy as np
import em
import common

X = np.loadtxt("test_incomplete.txt")
X_gold = np.loadtxt("test_complete.txt")

K = 4
n, d = X.shape
seed = 0

# TODO: Your code here

mixture, post = common.init(X, K, seed)
# print(mixture)
# post, ll = em.estep(X,mixture)
# mixture = em.mstep(X, post, mixture)
# print("After first M-step")
# print(mixture)
mixture, p, ll = em.run(X, mixture, post)
print(em.fill_matrix(X, mixture))