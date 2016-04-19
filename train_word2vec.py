from data import *
from gensim.models.word2vec import *
import utils
from time import gmtime, strftime
def train():
    # load dataset
    ds = loadDataset(DOC_DATAEST)
    sentences = []
    for instance in ds.instances:
        sentences += [l.strip().split() for l in instance.split('\n')]

    # train word2vec
    model = Word2Vec(sentences, size = 2, min_count = 0, workers = 8)
    # model.save(PATH + 'tweets-300.word2vec.model')
    # model.save_word2vec_format(PATH + 'tweets-300.word2vec.bin.txt')
    return model


def output():
    with open(PATH + 'tweets-2.word2vec.bin.txt') as fin:
        with open(PATH + 'tweets-2.word2vec.tokens', 'w') as ftoken:
            with open(PATH + 'tweets-2.word2vec.vectors', 'w') as fv:
                # skip top line
                fin.readline()
                for l in fin.xreadlines():
                    data = l.strip().split()
                    ftoken.write(data[0] + '\n')
                    fv.write('\t'.join(data[1:]) + '\n')


if __name__ == '__main__':
    # model = train()
    # model.save_word2vec_format(PATH + 'tweets-2.word2vec.bin.txt')
    # output()
    print 'loading'
    model = Word2Vec.load(PATH + 'tweets-300.word2vec.model')

    print 'loading'
    with open(PATH + 'tweets-300.word2vec.tokens') as fin:
        v = set([l.strip() for l in fin.xreadlines()])

    print 'loading'
    with open('/home/jeky/Dropbox/thesis/data/badwords.txt') as fin:
        badwords = [l.strip().split('\t')[0] for l in fin.xreadlines() if l.strip().split('\t')[0] in v]

    print len(badwords)

    dis = {w:[0, w] for w in v}
    for i, w in enumerate(badwords):
        if i % 10 == 0:
            print strftime("%Y-%m-%d %H:%M:%S", gmtime())
            print i

        wordDis = model.most_similar(positive = [w], topn = len(v))
        for t, s in wordDis:
            if s > dis[t][0]:
                dis[t] = [s, w]


    dis = [[w] + s for w, s in dis.iteritems()]
    utils.output(PATH + 'hidden_badwords.txt', dis)