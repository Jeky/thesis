from data import *
import globe
from sklearn.naive_bayes import *
import sys
import cv
from sklearn import cross_validation
from sklearn.metrics import *

def testClassifiers(dataset, out):
    names = ['MultinomialNB', 'BernoulliNB']

    classifiers = [
        MultinomialNB,
        BernoulliNB
    ]

    for clf, name in zip(classifiers, names):
        globe.getLogger().info('Testing Classifier: %s', name)
        out.write('Testing Classifier: %s\n' % name)
        cv.crossValidate(dataset, clf(), out)
        scores = cross_validation.cross_val_predict(clf(), dataset.instances, dataset.labels, cv = 10, verbose = 0)
        out.write('Total:\n'+ str(confusion_matrix(dataset.labels, scores)))




if __name__ == '__main__':
    with open(TEXT_CLS_OUTPUT, 'w') as out:
        for fname in [#TOKEN_COUNT_DATASET,
                      TOKEN_NORM_COUNT_DATASET,
                      #BIGRAM_TOKEN_COUNT_DATASET,
                      BIGRAM_TOKEN_NORM_COUNT_DATASET]:
            dataset = loadDataset(fname)
            testClassifiers(dataset, out)
            del dataset

