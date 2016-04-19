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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis


def testClassifiers(dataset, out):
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
        QuadraticDiscriminantAnalysis(),
        #SVC(kernel = 'linear', cache_size = 1500),
        #SVC(kernel = 'rbf', cache_size = 1500)]
    ]
    for clf, name in zip(classifiers, names):
        globe.getLogger().info('Testing Classifier: %s', name)
        out.write(name + '\n')
        scores = cross_validation.cross_val_predict(clf, dataset.instances, dataset.labels, cv = 10, verbose = 0)

        cm = confusion_matrix(dataset.labels, scores)
        out.write('%d\t%d\t%d\t%d\t%.10f\t%.10f\t%.10f\t%.10f\n' % (
                    cm[0][0], cm[0][1], cm[1][0], cm[1][1],
                    precision_score(dataset.labels, scores),
                    recall_score(dataset.labels, scores),
                    accuracy_score(dataset.labels, scores),
                    f1_score(dataset.labels, scores)
                    ))


import user2vec
if __name__ == '__main__':
    # user2vec.main()
    with open(sys.argv[1], 'w') as out:
        for name in [
            '/home/jeky/Projects/data/paragraph-vectors/sample-finalparagraph-vector-s1000.obj',
            '/home/jeky/Projects/data/paragraph-vectors/sample-finalparagraph-vector-s300.obj',
            '/home/jeky/Projects/data/paragraph-vectors/sample-finalparagraph-vector-s600.obj',
            '/home/jeky/Projects/data/pretained-300-user-vectors/user-vectors-discard-missing.obj',
            '/home/jeky/Projects/data/pretained-300-user-vectors/user-vectors-normalized-discard-missing.obj',
            '/home/jeky/Projects/data/pretained-300-user-vectors/user-vectors-normalized.obj',
            '/home/jeky/Projects/data/pretained-300-user-vectors/user-vectors.obj',
            '/home/jeky/Projects/data/tweet-1000-user-vectors/user-vectors-discard-missing.obj',
            '/home/jeky/Projects/data/tweet-1000-user-vectors/user-vectors-normalized-discard-missing.obj',
            '/home/jeky/Projects/data/tweet-1000-user-vectors/user-vectors-normalized.obj',
            '/home/jeky/Projects/data/tweet-1000-user-vectors/user-vectors.obj',
            '/home/jeky/Projects/data/tweet-300-user-vectors/user-vectors-discard-missing.obj',
            '/home/jeky/Projects/data/tweet-300-user-vectors/user-vectors-normalized-discard-missing.obj',
            '/home/jeky/Projects/data/tweet-300-user-vectors/user-vectors-normalized.obj',
            '/home/jeky/Projects/data/tweet-300-user-vectors/user-vectors.obj',
            '/home/jeky/Projects/data/tweet-600-user-vectors/user-vectors-discard-missing.obj',
            '/home/jeky/Projects/data/tweet-600-user-vectors/user-vectors-normalized-discard-missing.obj',
            '/home/jeky/Projects/data/tweet-600-user-vectors/user-vectors-normalized.obj',
            '/home/jeky/Projects/data/tweet-600-user-vectors/user-vectors.obj'
        ]:
            out.write('Testing ' + name + '\n')
            dataset = loadDataset(name)
            testClassifiers(dataset, out)
            del dataset

