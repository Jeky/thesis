from data import *
from sklearn import manifold
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from math import fabs

print 'loading'
dataset = loadDataset(PATH + 'paragraph-vectors/paragraph-vector-s1000.obj')
print 'compute similarity'
m = np.vstack(tuple(dataset.instances))
distanceM = pairwise_distances(m, metric = 'cosine', n_jobs = 12)
distanceM = np.abs(distanceM)
print distanceM.shape

tsne = manifold.TSNE(n_components = 2, metric = 'precomputed', random_state = 42, verbose = 5)
points = tsne.fit_transform(distanceM)

with open(PATH + 'paragraph-vectors/pv-1000.tsne.dat', 'w') as fout:
    for p in points:
        fout.write('\t'.join([str(i) for i in p]) + '\n')
 