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

def convertDSMatrix():
    ds = loadDataset(DOC_DATAEST)
    print 'collect vocabulary'
    V = set()
    instances = []
    for i in ds.instances:
        tokenDict = {}

        for l in i.split('\n'):
            tokens = l.split()
            for i in range(len(tokens)):
                t = tokens[i]
                if t not in tokenDict:
                    tokenDict[t] = 0

                tokenDict[t] += 1
                V.add(t)

        instances.append(tokenDict)

    print 'len(V) =', len(V)
    vocabularyIndex = {t:i for i, t in enumerate(V)}
    utils.output('vocabulary_index.txt', [(f,i) for f,i in vocabularyIndex.iteritems()])

    row = []
    col = []
    data = []

    for i, instance in enumerate(instances):
        if i % 100 == 0:
            print i

        for f, c in instance.iteritems():
            j = vocabularyIndex[f]

            row.append(i)
            col.append(j)
            data.append(1.0 * c)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)


    m = spmatrix((data, (row, col)))

    spDS = Dataset()
    spDS.labels = np.array(ds.labels)
    spDS.users = ds.users
    spDS.instances = m
    spDS.vocabularyIndex = vocabularyIndex

    spDS.save(PATH + 'unigram_sp.dat')


def loadSparseDS():
    return loadDataset(PATH + 'unigram_sp.dat')


def loadFeatures(fname, ds):
    print 'loading features'
    if fname.endswith('wapmi.txt'):
        features = utils.load(fname, [str, float])
    elif fname.endswith('badwords-related.txt'):
        with open(fname) as fin:
            features = [l.strip() for l in fin.xreadlines()]
            return np.array([ds.vocabularyIndex[f] for f in features])
    else:
        features = utils.load(fname, [str, int, int, float])

    return np.array([ds.vocabularyIndex[f[0]] for f in features])


def trainNB(S, N):
    # N* = count of users
    # V = count of features
    N1, V = S.shape
    N2, V = N.shape

    prior = np.array([log2(1.0*N1/(N1+N2)), log2(1.0*N2/(N1+N2))])

    F1 = S.sum()
    F2 = N.sum()

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
    trainS, trainN, testS, testN, features, sizes = arg
    cms = {}

    for size in sizes:
        print size
        cm = np.zeros((2,2))

        prior, prob = trainNB(trainS[:, features[:size]], trainN[:, features[:size]])

        scoreS = test(prior, prob, testS[:, features[:size]])
        scoreN = test(prior, prob, testN[:, features[:size]])

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

        cms[size] = cm

    return cms


def evaluate(ds, features, cv = 10):
    U, V = ds.instances.shape

    S = ds.instances[:U/2]
    N = ds.instances[U/2:]

    sizes = [#10, 10**2, 10**3, 10**4, 10**5, 10**6,
            #10**6 + 5*10**5, 2*10**6, V]
            V]

    cvIndexes = np.array_split(np.array(list(range(U/2))), cv)

    parameters = []
    for i, index in enumerate(cvIndexes):
        trainIndex = np.hstack(tuple([item for j, item in enumerate(cvIndexes) if j != i]))

        trainS = S[trainIndex,:]
        trainN = N[trainIndex,:]

        testS = S[index, :]
        testN = N[index, :]

        parameters.append((trainS, trainN, testS, testN, features, sizes))
        break

    pool = Pool()
    cmsList = pool.map(evaluateWork, parameters)
    pool.close()
    pool.join()

    return cmsList



def precision(cm):
    return float(cm[0][0]) / (cm[0][0] + cm[1][0])


def recall(cm):
    return float(cm[0][0]) / (cm[0][0] + cm[0][1])


def accuracy(cm):
    return float(cm[0][0] + cm[1][1]) / (cm[0][0] + cm[0][1] + cm[1][0] + cm[1][1])


def f1(cm):
    return float(cm[0][0]) * 2 / (cm[0][0] * 2 + cm[1][0] + cm[0][1])


def save(cms, fname):
    sizes = list(cms[0].keys())
    sizes.sort()

    merged = []

    for s in sizes:
        cm = reduce(np.add, [cm[s] for cm in cms])
        merged.append([int(s), 
            int(cm[0][0]), 
            int(cm[0][1]), 
            int(cm[1][0]), 
            int(cm[1][1]), 
            precision(cm), recall(cm), accuracy(cm), f1(cm)])

    utils.output(fname, merged)



if __name__ == '__main__':
    # ds = loadSparseDS()

    # fout = PATH + 'badwords-results.txt'
    # features = loadFeatures('badwords-related.txt', ds)
    # cms = evaluate(ds, features)
    # # save(cms, fout)
    # print cms


    convertDSMatrix()    