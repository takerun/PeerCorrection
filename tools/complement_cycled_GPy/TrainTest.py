#coding: utf-8

import os,sys
import numpy as np
import pandas as pd


def foldDataset(dataFrame, fold_num):
    dNum = len(dataFrame)
    quotient = dNum/fold_num
    remainder = dNum%fold_num
    folds = [quotient]*fold_num
    for i in xrange(remainder):
        folds[i] = folds[i]+1
    blocks = list()
    for i in xrange(fold_num):
        bufDF = dataFrame.iloc[sum(folds[0:i]):sum(folds[0:i+1])]
        blocks.append(bufDF)
    return blocks

def splitDataset523(dataFrame):
    blocks = np.array_split(dataFrame, 10)
    b5 = pd.concat(blocks[:5], ignore_index=True)
    b2 = pd.concat(blocks[5:7], ignore_index=True)
    b3 = pd.concat(blocks[7:], ignore_index=True)
    return b5,b2,b3
