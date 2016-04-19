#!/usr/bin/env python
import numpy as np

def load(fname, types):
    '''
    Load data from fname and convert data with types.

    @arg fname: the name of file which contains data (each line should be split by '\t')
    @arg types: the type list of each data in one line

    @return data list
    '''

    data = []
    with open(fname) as fin:
        for l in fin.xreadlines():
            raw = l.strip().split('\t')

            if len(raw) != len(types):
                raise 'The length of types does not match the length of raw data'

            data.append([t(d) for d, t in zip(raw, types)])

    return data


def output(fname, data):
    '''
    Output data into fname.

    @arg fname: the name of output file
    @arg data: data list
    '''
    tplDict = {
        int  : '%d',
        float: '%.10f',
        str  : '%s',
        np.float64 : '%.10f',
        np.int64 : '%d'
    }

    tpl = '\t'.join([tplDict[type(d)] for d in data[0]]) + '\n'

    with open(fname, 'w') as fout:
        for d in data:
            fout.write(tpl % tuple(d))