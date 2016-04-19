from data import *

ds = loadDataset(DOC_DATAEST)
tokenCounter = {}
for label, user in zip(ds.labels, ds.instances):
    tokens = user.split()
    for t in tokens:
        if t not in tokenCounter:
            tokenCounter[t] = [0, 0] # SUPSENDED, NON_SUSPENDE

        if label == SUSPENDED_LABEL:
            tokenCounter[t][0] += 1
        else:
            tokenCounter[t][1] += 1

tokenlist = [[k] + v for k, v in tokenCounter.items()]
tokenlist.sort(key = lambda i: -i[1])

with open(PATH + 'tokens.txt', 'w') as fkout:
    with open(PATH + 'count.txt', 'w') as fvout:
        for t in tokenlist:
            fkout.write('%s\t%d\t%d\n' % tuple(t))
            fvout.write('%d\t%d\n' % (t[1], t[2]))