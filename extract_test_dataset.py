from data import *
from collections import Counter

ds = loadDataset(DOC_DATAEST)

s = 1134
n = 1152

def writeUser(out, u, l, doc):
    out.write('!%s\t%d\n' % (u, l))
    counter = Counter(doc.strip().split())
    for k, v in counter.most_common():
        out.write('%s\t%d\n' % (k, v))


with open('mnb_test_1.txt', 'w') as out:
    for i, doc in enumerate(ds.instances):
        u = ds.users[i]
        l = ds.labels[i]
        
        if l == SUSPENDED_LABEL:
            if s > 0:
                writeUser(out, u, l, doc)
            s -= 1
        else:
            if n > 0:
                writeUser(out, u, l, doc)
            n -= 1