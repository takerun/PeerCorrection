# coding: utf-8

import os,sys
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-f', '--npz-file', help='show inner values in the file', type=str)
args = parser.parse_args()

if args.npz_file:
    npz_name = args.npz_file
    npz = np.load(npz_name)
    X = npz['arr_0']
    Y = npz['arr_1']
    print(Y)
    print(X)
    print('min: {0}, params: {1}'.format(Y.min(), X[Y.argmin()]))
