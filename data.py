import os.path
import globe
import pickle
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
import numpy as np
from collections import Counter

PATH = '/home/jeky/Projects/data/'
ORIGINAL_DATASET = os.path.join(PATH, 'dataset.txt')
DOC_DATAEST = os.path.join(PATH, 'dataset.obj')
TOKEN_COUNT_DATASET = os.path.join(PATH, 'token-dataset.obj')
TOKEN_NORM_COUNT_DATASET = os.path.join(PATH, 'token-norm-dataset.obj')
BIGRAM_TOKEN_COUNT_DATASET = os.path.join(PATH, 'bigram-token-dataset.obj')
BIGRAM_TOKEN_NORM_COUNT_DATASET = os.path.join(PATH, 'bigram-token-norm-dataset.obj')

USER_VECTOR_DISMW_DATASET = os.path.join(PATH, 'user-vectors-discard-missing.obj')
USER_VECTOR_NORM_DISMW_DATASET = os.path.join(PATH, 'user-vectors-normalized-discard-missing.obj')
USER_VECTOR_NORM_DATASET = os.path.join(PATH, 'user-vectors-normalized.obj')
USER_VECTOR_DATASET = os.path.join(PATH, 'user-vectors.obj')

PRETRAIN_USER_VECTOR_DISMW_DATASET = os.path.join(PATH , 'pretained-300-user-vectors', 'user-vectors-discard-missing.obj')
PRETRAIN_USER_VECTOR_NORM_DISMW_DATASET = os.path.join(PATH , 'pretained-300-user-vectors', 'user-vectors-normalized-discard-missing.obj')
PRETRAIN_USER_VECTOR_NORM_DATASET = os.path.join(PATH , 'pretained-300-user-vectors', 'user-vectors-normalized.obj')
PRETRAIN_USER_VECTOR_DATASET = os.path.join(PATH , 'pretained-300-user-vectors', 'user-vectors.obj')

USER_VECTOR_TEST_RESULT = os.path.join(PATH, 'user-vectors-cls-result.txt')

UNIGRAM_DICT_DATASET = os.path.join(PATH, 'unigram-dict-dataset.obj')
BIGRAM_DICT_DATASET = os.path.join(PATH, 'bigram-dict-dataset.obj')

WORD_VECTOR_BIN = os.path.join(PATH, 'GoogleNews-vectors-negative300.bin.gz')
WORD_VECTOR = os.path.join(PATH, 'GoogleNews-vectors-negative300.bin.txt')
TWITTER_WORD_VECTOR = os.path.join(PATH, 'tweets-300.bin.txt')

CHI_OUTPUT = os.path.join(PATH, 'chi.txt')
MI_OUTPUT = os.path.join(PATH, 'mi.txt')
UNIGRAM_DF_OUTPUT = os.path.join(PATH, 'unigram-df.txt')
BIGRAM_DF_OUTPUT = os.path.join(PATH, 'bigram-df.txt')
TRIGRAM_DF_OUTPUT = os.path.join(PATH, 'trigram-df.txt')

TEXT_CLS_OUTPUT = os.path.join(PATH, 'text-cls-result.txt')
FEATURE_TEST_RESULT = os.path.join(PATH, 'feature-cls-result.txt')

PRE_PARA_DOC = os.path.join(PATH, 'pre-para.obj')
PARA_MODEL = os.path.join(PATH, 'para-model.obj')
PARA_DATASET = os.path.join(PATH, 'para-dataset.obj')
PARA_S300_DATASET = os.path.join(PATH, 'para-s300-w8-dataset.obj')

SUSPENDED_PREFIX = 'suspended'
NON_SUSPENDED_PREFIX = 'non-suspended'

SUSPENDED_LABEL = 1
NON_SUSPENDED_LABEL = 0

SUSPENDED_IDS = os.path.join(PATH, SUSPENDED_PREFIX + '-ids.txt')
NON_SUSPENDED_IDS = os.path.join(PATH, NON_SUSPENDED_PREFIX + '-ids.txt')

SUSPENDED_TWEETS_FOLDER = os.path.join(PATH, SUSPENDED_PREFIX + '-tweets')
NON_SUSPENDED_TWEETS_FOLDER = os.path.join(PATH, NON_SUSPENDED_PREFIX + '-tweets')


class Dataset(object):

    def __init__(self):
        self.users = []
        self.labels = []
        self.instances = []


    def read(self, fname):
        '''
        Read dataset from original dataset file.

        Format:
        !ID \t IS_SUSPENDED
        TWEET1
        TWEET2
        ...
        !ID \t IS_SUSPENDED
        ...

        '''

        users = []

        u = {}
        with open(fname) as fin:
            globe.getLogger().info('Start reading file: %s', fname)
            for i, l in enumerate(fin.xreadlines()):
                l = l.strip()

                if i != 0 and i % 10000 == 0:
                    globe.getLogger().info('Read %d lines', i)

                if l != '':
                    if l[0] == '!':
                        if 'id' in u:
                            users.append(u)
                            u = {}

                        uid, isSuspended = l.split()
                        u['id'] = int(uid[1:])
                        u['suspended'] = int(isSuspended)
                        u['tweets'] = []
                    else:
                        u['tweets'].append(l)

        users.append(u)

        for u in users:
            self.users.append(u['id'])
            self.labels.append(u['suspended'])
            self.instances.append('\n'.join(u['tweets']))


    def save(self, fname):
        globe.getLogger().info('Save Dataset to %s', fname)
        with open(fname, 'wb') as fout:
            pickle.dump(self, fout)


def loadDataset(fname):
    globe.getLogger().info('Load Dataset from %s', fname)
    with open(fname) as fin:
        return pickle.load(fin)


def initNGramDatasets():
    dataset = loadDataset(DOC_DATAEST)
    NAMES = [UNIGRAM_DICT_DATASET, BIGRAM_DICT_DATASET]
    for gramCount in range(2):
        d = Dataset()
        n = gramCount + 1
        for uid, label, instance in zip(dataset.users, dataset.labels, dataset.instances):
            gramInstance = Counter()
            for l in instance.split('\n'):
                words = l.split()
                for i in range(len(words) - n):
                    gram = '_'.join(words[i:i+n])
                    gramInstance[gram] += 1

            d.users.append(uid)
            d.labels.append(label)
            d.instances.append(gramInstance)

        d.save(NAMES[gramCount])


def initDataset():
    d = Dataset()
    d.read(ORIGINAL_DATASET)
    d.save(DOC_DATAEST)

    d1 = Dataset()
    d1.users = d.users
    d1.labels = np.array(d.labels)

    count_vect = CountVectorizer(token_pattern=r'\S+')
    X_train_counts = count_vect.fit_transform(d.instances)
    d1.instances = X_train_counts
    d1.save(TOKEN_COUNT_DATASET)
    d1.vocabulary = {v:k for k, v in count_vect.vocabulary_.items()}

    # tfidf_transformer = TfidfTransformer()
    # d1.instances = tfidf_transformer.fit_transform(X_train_counts)
    # d1.save(TOKEN_NORM_COUNT_DATASET)

    count_vect = CountVectorizer(ngram_range=(2, 2), token_pattern=r'\S+')
    X_train_counts = count_vect.fit_transform(d.instances)
    d1.instances = X_train_counts
    d1.vocabulary = {v:k for k, v in count_vect.vocabulary_.items()}
    d1.save(BIGRAM_TOKEN_COUNT_DATASET)

    # tfidf_transformer = TfidfTransformer()
    # d1.instances = tfidf_transformer.fit_transform(X_train_counts)
    # d1.save(BIGRAM_TOKEN_NORM_COUNT_DATASET)

if __name__ == '__main__':
    globe.getLogger().info('Initialize Dataset')
    initNGramDatasets()
