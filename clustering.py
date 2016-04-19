from data import *
from sklearn.cluster import KMeans


def clustering(dataset, n):
    kmeans = KMeans(n_clusters=n, random_state=42)
    return kmeans.fit_predict(dataset.instances)


for name in [USER_VECTOR_NORM_DISMW_DATASET,
             #USER_VECTOR_NORM_DATASET,
             #USER_VECTOR_DISMW_DATASET,
             #USER_VECTOR_DATASET
             ]:
    dataset = loadDataset(name)
    with open('test.txt', 'w') as fout:
        predicted = clustering(dataset, 10)
        for p in predicted:
            fout.write('%d\n' % p)

    del dataset
