import utils

hidden = utils.load('../data/hidden_badwords.txt', [str, float, str])

users = utils.load('/home/jeky/Dropbox/thesis/data/users.txt', [str, int, int])
users = {t[0] : t[1:] for t in users}

tokens = utils.load('/home/jeky/Dropbox/thesis/data/tokens.txt', [str, int, int])
tokens = {t[0] : t[1:] for t in tokens}

badwords = utils.load('/home/jeky/Dropbox/thesis/data/badwords.txt', [str] + [int] * 4)
badwords = set([w[0] for w in badwords])

merged = [[w[0], w[1], w[1] * tokens[w[2]][0], w[2]] + tokens[w[0]] + users[w[0]] for w in hidden if w[0] not in badwords]

merged.sort(key = lambda i: -i[2])

utils.output('top10000_hidden_badwords.txt', merged[:10000])