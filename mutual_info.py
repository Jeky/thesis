from __future__ import print_function
import numpy as np
from numpy import log2

def biclass_mutual_info(X, y):
    classes = np.unique(y)

    x1i = np.argwhere(y == classes[0])
    x1i = np.squeeze(np.asarray(x1i))
    X1 = X[x1i]

    x2i = np.argwhere(y == classes[1])
    x2i = np.squeeze(np.asarray(x2i))
    X2 = X[x2i]
    clsCount = [X1.shape[0], X2.shape[0]]

    count = np.ones((2, X.shape[1]), np.int32)
    X1[X1 != 0] = 1
    X2[X2 != 0] = 1

    count[0] += np.sum(X1, axis=0)[0]
    count[1] += np.sum(X2, axis=0)[0]

    scores = np.zeros(X.shape[1], np.float)
    N = y.shape[0]
    
    scores = count[0] / N * (log2(N) + log2(count[0])   - log2(count[0] + count[1]) - log2(count[0] + clsCount[0] + 2 - count[0])) +\
             count[1] / N * (log2(N) + log2(count[1]) - log2(count[0] + count[1]) - log2(count[1] + clsCount[1] + 2 - count[1])) +\
             (clsCount[0] + 2 - count[0]) / N * (log2(N) + log2(clsCount[0] + 2 - count[0]) - log2(clsCount[0] + 2 - count[0] + clsCount[1] + 2 - count[1]) - log2(count[0] + clsCount[0] + 2 - count[0])) +\
             (clsCount[1] + 2 - count[1]) / N * (log2(N) + log2(clsCount[1] + 2 - count[1]) - log2(clsCount[0] + 2 - count[0] + clsCount[1] + 2 - count[1]) - log2(count[1] + clsCount[1] + 2 - count[1]))

    return scores


if __name__ == '__main__':
    X = np.array([[0, 1, 0], [0, 2, 2], [1, 2, 0], [1, 0, 2],])

    y = np.array([0, 0, 1, 1])
    print(biclass_mutual_info(X, y))
