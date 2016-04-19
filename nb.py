#!/usr/bin/env python

import utils
from math import log
from pprint import pprint


class MNB:

    def __init__(self):
        self.prior = [0.0, 0.0]
        self.prob = {}


    def reset(self):
        self.prior = [0.0, 0.0]
        self.prob = {}


    def train(self, X, y):
        F1 = sum([f[1] for f in freq])
        F2 = sum([f[2] for f in freq])
        V  = len(freq)

        for f in freq:
            feature = f[0]
            prob[feature] = [log(float(f[1] + 1) / (F1 + V)), \
                             log(float(f[2] + 1) / (F2 + V))]


    def classify(self, user):
        pass

