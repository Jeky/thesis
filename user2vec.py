from data import *
import globe
from gensim.models import word2vec
import random
import numpy as np


def loadWord2vec(dim):
    globe.getLogger().info('Start Loading word vector file')

    with open(PATH + '/tweets-%d.bin.txt' % dim) as fin:
        words = {}
        count, dim = fin.readline().strip().split()
        dim = int(dim)

        for i, l in enumerate(fin.xreadlines()):
            if i % 100000 == 0 and i != 0:
                globe.getLogger().info('read %d lines' % i)

            vector = l.strip().split()
            words[vector[0]] = np.array([float(v) for v in vector[1:]])

    return words


def user2vec(words, d, normalized = True, discardMissing = True):
    globe.getLogger().info('Converting Users to Vectors (NORMALIZED = %r, DISCARD_MISSING_WORD = %r)' % (normalized, discardMissing))
    missingWords = {}
    dim = words[words.keys()[0]].shape[0]

    vectorInstances = []
    for i, instance in enumerate(d.instances):
        if i % 1000 == 0 and i != 0:
            globe.getLogger().info('processed %d instance' % i)

        vector = np.zeros(dim)
        wCount = 0
        for w in instance.split():
            if w in words:
                wCount += 1
                vector += words[w]
            else:
                if not discardMissing:
                    if w not in missingWords:
                        missingWords[w] = np.random.rand(dim)

                    vector += missingWords[w]

        if normalized:
            if wCount != 0:
                vector /= wCount

        vectorInstances.append(vector)

    # remove zero vectors
    vecDataset = Dataset()
    globe.getLogger().info('Removing zero vectors')
    zeroCount = 0
    for i, vector in enumerate(vectorInstances):
        if np.count_nonzero(vector) == 0:
            globe.getLogger().info('User %d is empty' % d.users[i])
            zeroCount += 1
        else:
            vecDataset.users.append(d.users[i])
            vecDataset.labels.append(d.labels[i])
            vecDataset.instances.append(vector)

    globe.getLogger().info('Total Found: %d' % zeroCount)
    if normalized:
        if discardMissing:
            fout = USER_VECTOR_NORM_DISMW_DATASET
        else:
            fout = USER_VECTOR_NORM_DATASET
    else:
        if discardMissing:
            fout = USER_VECTOR_DISMW_DATASET
        else:
            fout = USER_VECTOR_DATASET

    vecDataset.save(fout)

def main():
    words = loadWord2vec(1000)
    d = loadDataset(DOC_DATAEST)
    for n in [True, False]:
        for dm in [True, False]:
            user2vec(words, d, n, dm)
