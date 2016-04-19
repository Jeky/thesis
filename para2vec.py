from gensim.models.doc2vec import *
from data import *
import globe

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import *
import sys
from mutual_info import biclass_mutual_info
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def loadParaDataset():
    globe.getLogger().info('Loading Dataset')
    dataset = loadDataset(DOC_DATAEST)
    paraDataset = Dataset()
    paraDataset.users = dataset.users
    paraDataset.labels = dataset.labels

    for i, instance in enumerate(dataset.instances):
        if i % 100 == 0:
            globe.getLogger().info('Processed %d instances' % i)
        paraDataset.instances.append(LabeledSentence(words = instance.split(), tags = [u'T_%d' % i]))

    dataset.save(PRE_PARA_DOC)

    return paraDataset


def train(dataset, dim):
    globe.getLogger().info('Training Paragraph Vectors (Dimension = %d)' % dim)
    model = Doc2Vec(size = dim, window = 8, workers = 8, alpha=0.025, min_alpha=0.025, min_count = 2)

    model.build_vocab(dataset.instances)

    for epoch in range(10):
        globe.getLogger().info('Training %d time' % epoch)
        model.train(dataset.instances)
        model.alpha -= 0.002 # decrease the learning rate
        model.min_alpha = model.alpha # fix the learning rate, no deca
        model.train(dataset.instances)

    model.save(PARA_MODEL)
    return model


def test(model, dim, dataset, out):
    instances = [model.docvecs[u'T_%d' % i] for i in range(len(dataset.instances))]

    d = Dataset()
    zeroCount = 0
    for i, ins in enumerate(instances):
        if np.isfinite(ins).all():
            d.labels.append(dataset.labels[i])
            d.instances.append(ins)
            d.users.append(dataset.users[i])
        else:
            zeroCount += 1

    d.save(PATH + 'paragraph-vector-s%d.obj' % dim)

    globe.getLogger().info('Zero Count: %d' % zeroCount)
    '''
    names = ["Nearest Neighbors",
            "Decision Tree",
            "Random Forest", "AdaBoost", "Naive Bayes", "Linear Discriminant Analysis",
            "Quadratic Discriminant Analysis"]
    # names = ["Linear SVM", "RBF SVM"]
    classifiers = [
       KNeighborsClassifier(10),
       DecisionTreeClassifier(max_depth=5),
       RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
       AdaBoostClassifier(),
       GaussianNB(),
       LinearDiscriminantAnalysis(),
       QuadraticDiscriminantAnalysis()]

    for clf, name in zip(classifiers, names):
        globe.getLogger().info('Testing Classifier: %s', name)
        out.write(name + '\n')
        scores = cross_validation.cross_val_predict(clf, instances, dataset.labels, cv = 10, verbose = 0)

        cm = confusion_matrix(dataset.labels, scores)
        out.write('%d\t%d\t%d\t%d\t%.10f\t%.10f\t%.10f\t%.10f\n' % (
                    cm[0][0], cm[0][1], cm[1][0], cm[1][1],
                    precision_score(dataset.labels, scores),
                    recall_score(dataset.labels, scores),
                    accuracy_score(dataset.labels, scores),
                    f1_score(dataset.labels, scores)
                    ))
     '''

if __name__ == '__main__':
    dims = [300, 600, 1000]
    dataset = loadParaDataset()
    with open(PATH + '/para-result.txt', 'w') as out:
        for dim in dims:
            model = train(dataset, dim)
            test(model, dim, dataset, out)
