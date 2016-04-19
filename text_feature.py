from data import *
import globe
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.naive_bayes import *
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from sklearn import cross_validation
from sklearn.metrics import *
from mutual_info import biclass_mutual_info
from collections import Counter
import text_cls

def testChi2(d):
    globe.getLogger().info('length of vocabulary = %d', len(d.vocabulary))

    c, p = chi2(d.instances, d.labels)

    scores = [(d.vocabulary[i], sf) for i, sf in enumerate(c)]
    scores.sort(key = lambda i: -i[1])
    with open(CHI_OUTPUT, 'w') as fout:
        for s in scores:
            fout.write('%s\t%.10f\n' % s)

    size = 1
    while 10 ** size < len(d.vocabulary):
        clf = MultinomialNB()
        X = SelectKBest(chi2, k= 10 ** size).fit_transform(d.instances, d.labels)
        scores = cross_validation.cross_val_predict(clf, X, d.labels, cv = 10, verbose = 0)
        globe.getLogger().info('10^%d\t%.6f\t%.6f', size, accuracy_score(d.labels, scores), f1_score(d.labels, scores))
        globe.getLogger().info(confusion_matrix(d.labels, scores))
        size += 1

    clf = MultinomialNB()
    scores = cross_validation.cross_val_predict(clf, d.instances, d.labels, cv = 10, verbose = 0)
    globe.getLogger().info('%.1e\t%.6f\t%.6f', len(d.vocabulary), accuracy_score(d.labels, scores), f1_score(d.labels, scores))


def testMI(d):
    globe.getLogger().info('length of vocabulary = %d', len(d.vocabulary))

    # print d.instances.shape
    c = biclass_mutual_info(d.instances, d.labels)
    scores = [(d.vocabulary[i], sf) for i, sf in enumerate(c)]
    scores.sort(key = lambda i: -i[1])
    with open(MI_OUTPUT, 'w') as fout:
        for s in scores:
            fout.write('%s\t%.10f\n' % s)

    size = 1
    while 10 ** size < len(d.vocabulary):
        clf = MultinomialNB()
        X = SelectKBest(biclass_mutual_info, k= 10 ** size).fit_transform(d.instances, d.labels)
        scores = cross_validation.cross_val_predict(clf, X, d.labels, cv = 10, verbose = 0)
        globe.getLogger().info('10^%d\t%.6f\t%.6f', size, accuracy_score(d.labels, scores), f1_score(d.labels, scores))
        globe.getLogger().info(confusion_matrix(d.labels, scores))
        size += 1

    clf = MultinomialNB()
    scores = cross_validation.cross_val_predict(clf, d.instances, d.labels, cv = 10, verbose = 0)
    globe.getLogger().info('%.1e\t%.6f\t%.6f', len(d.vocabulary), accuracy_score(d.labels, scores), f1_score(d.labels, scores))


def countDF(splitter, outputFilename):
    dataset = loadDataset(DOC_DATAEST)
    sGram = Counter()
    nsGram = Counter()

    i = 0
    for label, instance in zip(dataset.labels, dataset.instances):
        if i % 100 == 0 and i != 0:
            globe.getLogger().info('processed %d instances' % i)

        grams = splitter(instance)
        for g in grams:
            if label == SUSPENDED_LABEL:
                sGram[g] += 1
            else:
                nsGram[g] += 1

        i += 1

    globe.getLogger().info('Len(sGram) = %d, Len(nsGram) = %d' % (len(sGram), len(nsGram)))
    features = {}

    for g, c in sGram.items():
        if g not in features:
            features[g] = [c, 0, 0.0]
        else:
            features[g][0] = c

    for g, c in nsGram.items():
        if g not in features:
            features[g] = [0, c, 0.0]
        else:
            features[g][1] = c

    globe.getLogger().info('Sorting Grams by DF')
    features = [(k, v[0], v[1], float(v[0] + 1) / (v[1] + 1)) for k, v in features.items()]
    features.sort(key = lambda item : -item[-1])

    globe.getLogger().info('Saving Result')
    with open(outputFilename, 'w') as fout:
        for f in features:
            fout.write('%s\t%d\t%d\t%.10f\n' % f)


def unigramDF():
    globe.getLogger().info('Counting Unigram')

    def splitter(instance):
        return set(instance.split())

    countDF(splitter, UNIGRAM_DF_OUTPUT)


def bigramDF():
    globe.getLogger().info('Counting Bigram')

    def splitter(instance):
        bigrams = set()
        for l in instance.split('\n'):
            words = l.split()
            for i in range(len(words) - 1):
                bigrams.add(words[i] + '_' + words[i + 1])

        return bigrams

    countDF(splitter, BIGRAM_DF_OUTPUT)


# def trigramDF():
#     globe.getLogger().info('Counting Trigram')

#     def splitter(instance):
#         bigrams = set()
#         for l in instance.split('\n'):
#             words = l.split()
#             for i in range(len(words) - 2):
#                 bigrams.add(words[i] + '_' + words[i + 1] + '_' + words[i + 2])

#         return bigrams

#     countDF(splitter, TRIGRAM_DF_OUTPUT)


def loadDFFeatures(featurePath):
    globe.getLogger().info('Loading Features from %s' % featurePath)
    with open(featurePath) as fin:
        features = []#{}
        for i, l in enumerate(fin.xreadlines()):
            if i % 100000 == 0 and i != 0:
                globe.getLogger().info('loaded %d features' % i)
            gram, df1, df2, score = l.strip().split()
            #features[gram] = [int(df1), int(df2), float(score)]
            features.append(gram)

    return features


def evaluateDF(dataset, features, top):
    globe.getLogger().info('Evaluating with feature count = %d' % top)
    # filter dataset
    filteredDS = Dataset()
    def addFeature(ins, fins, f):
        if f in ins:
            fins[f] = ins[f]
        else:
            fins[f] = 0

    globe.getLogger().info('Filtering Dataset')
    count = 0
    for uid, label, instance in zip(dataset.users, dataset.labels, dataset.instances):
        if count % 100 == 0 and count != 0:
            globe.getLogger().info('processed %d instances' % count)

        filteredInstance = {}
        for i in range(top / 2):
            addFeature(instance, filteredInstance, features[i])
            addFeature(instance, filteredInstance, features[-(i+1)])

        filteredDS.users.append(uid)
        filteredDS.labels.append(label)
        filteredDS.instances.append(filteredInstance)

        count += 1

    filteredDS.instances = DictVectorizer().fit_transform(filteredDS.instances)

    # evaluate
    text_cls.testClassifiers(filteredDS)



def testDFClassification():
    datasets = [UNIGRAM_DICT_DATASET, BIGRAM_DICT_DATASET]
    features = [UNIGRAM_DF_OUTPUT, BIGRAM_DF_OUTPUT]

    for datasetPath, featurePath in zip(datasets, features):
        d = loadDataset(datasetPath)
        f = loadDFFeatures(featurePath)
        n = 10
        while n < len(f):
            evaluateDF(d, f, n)
            n *= 10



if __name__ == '__main__':
    # testDFClassification()
    for fname in [TOKEN_NORM_COUNT_DATASET]: #,
                  #BIGRAM_TOKEN_NORM_COUNT_DATASET]:
        dataset = loadDataset(fname)
        testMI(dataset)
        #testChi2(dataset)
        del dataset




