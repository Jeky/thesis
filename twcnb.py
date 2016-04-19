import numpy as np
from data import *
from scipy.sparse import csr_matrix as spmatrix
import utils
from sklearn.cross_validation import KFold
from sklearn.metrics import *
from numpy import log2
import sys
from multiprocessing import Pool
from collections import deque

def loadSparseDS():
    return loadDataset(PATH + 'unigram_sp.dat')


def trainNB(S, N):
    # N* = count of users
    # V = count of features
    N1, V = S.shape
    N2, V = N.shape

    prior = np.array([log2(1.0*N1/(N1+N2)), log2(1.0*N2/(N1+N2))])

    F1 = S.sum()
    F2 = N.sum()

    # normalize by length
    cpS = spmatrix(S)
    cpN = spmatrix(N)
    cpS.data = cpS.data ** 2
    cpN.data = cpN.data ** 2
    lenS = 1 / np.power(cpS.sum(axis=1), 0.5)
    lenN = 1 / np.power(cpN.sum(axis=1), 0.5)
    lenS = np.nan_to_num(lenS)
    lenN = np.nan_to_num(lenN)
    diagLenS = spmatrix((N1, N1))
    diagLenS.setdiag(lenS)
    diagLenN = spmatrix((N2, N2))
    diagLenN.setdiag(lenN)

    S = diagLenS * S
    N = diagLenN * N

    probS = S.sum(axis=0)
    probN = N.sum(axis=0)

    probS = log2((probS + 1) / (F1 + V))
    probN = log2((probN + 1) / (F2 + V))

    prob = np.concatenate((probS, probN), axis=0).transpose()

    return prior, prob


def test(prior, prob, cases):
    return prior + cases * prob


def evaluateWork(arg):
    print 'worker starts'
    trainS, trainN, testS, testN = arg

    cm = np.zeros((2,2))

    prior, prob = trainNB(trainS, trainN)

    scoreS = test(prior, prob, testS)
    scoreN = test(prior, prob, testN)

    scoreS = scoreS[:,0] - scoreS[:,1]
    scoreN = scoreN[:,0] - scoreN[:,1]

    cm[0][0] = scoreS[np.where(scoreS > 0)].shape[1]
    cm[0][1] = scoreS[np.where(scoreS < 0)].shape[1]
    cm[1][0] = scoreN[np.where(scoreN > 0)].shape[1]
    cm[1][1] = scoreN[np.where(scoreN < 0)].shape[1]

    randomSCount = scoreS[np.where(scoreS == 0)].shape[1]
    cm[0][1] += randomSCount / 2
    cm[0][0] += randomSCount - randomSCount / 2

    randomNCount = scoreN[np.where(scoreN == 0)].shape[1]
    cm[1][0] += randomNCount / 2
    cm[1][1] += randomNCount - randomNCount / 2

    return cm


def evaluate(ds, cv = 10):
    U, V = ds.instances.shape

    S = ds.instances[:U/2]
    N = ds.instances[U/2:]

    cvIndexes = np.array_split(np.array(list(range(U/2))), cv)

    parameters = []
    for i, index in enumerate(cvIndexes):
        trainIndex = np.hstack(tuple([item for j, item in enumerate(cvIndexes) if j != i]))

        trainS = S[trainIndex,:]
        trainN = N[trainIndex,:]

        testS = S[index, :]
        testN = N[index, :]

        parameters.append((trainS, trainN, testS, testN))

    cmList = []
    for i in range(10):
        cm = evaluateWork(parameters[i])
        cmList.append(cm)

    # pool = Pool()
    # cmsList = pool.map(evaluateWork, parameters)
    # pool.close()
    # pool.join()

    return cmList



def precision(cm):
    return float(cm[0][0]) / (cm[0][0] + cm[1][0])


def recall(cm):
    return float(cm[0][0]) / (cm[0][0] + cm[0][1])


def accuracy(cm):
    return float(cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])


def f1(cm):
    return float(cm[0][0]) * 2 / (cm[0][0] * 2 + cm[1][0] + cm[0][1])


if __name__ == '__main__':
    ds = loadSparseDS()
    cms = evaluate(ds)
    totalCM = reduce(np.add, cms)
    print totalCM
    print precision(totalCM)
    print recall(totalCM)
    print accuracy(totalCM)
    print f1(totalCM)