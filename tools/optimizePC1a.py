# conding:utf-8

import os,sys
import time,datetime
import GPy
import GPyOpt
import numpy as np
import pandas as pd
from TrainTest import *
from PC1a import PC1a

# path information
train_fold_num = 5
tune_trial_num = 100
snapshot_path = '../result/snapshot/ability_estimation'
snapshot_form = 'PC1a-hyper5-EI5000w500-fold{0}-{1}'
groundtruth_csv = '../dataset/open_peer_review_v2/peer_review/translated_groundtruth.csv'
peer_review_csv = '../dataset/open_peer_review_v2/peer_review/peer_review_forPG3.csv'

# functions
def timeStamp():
    # return time as str
    todaydetail = datetime.datetime.today()
    return todaydetail.strftime("%Y%m%d%H%M%S")

def saveGPyOpt(myBopt,fold_num):
    filename = snapshot_form.format(fold_num,timeStamp())
    path = os.path.join(snapshot_path, filename)
    np.savez_compressed(path, myBopt.X, myBopt.Y)
    print("Save: {}".format(path))

# preprocess train&test dataset
gDF = pd.read_csv(groundtruth_csv)
rDF = pd.read_csv(peer_review_csv)
np.random.seed(12345678)
shuffled_rDF = rDF.take(np.random.permutation(len(rDF))).reset_index(drop=True)
for i in xrange(train_fold_num):
    fold_id = i
    rBlocks = foldDataset(shuffled_rDF, train_fold_num)
    trainDF = rBlocks.pop(fold_id).reset_index(drop=True)
    testDF = pd.concat(rBlocks).reset_index(drop=True)

    # model setting
    trainModel = PC1a(gDF, trainDF)

    # training
    bounds = [
        {'name': 'mu0', 'type': 'continuous', 'domain': (-2,2)},
        {'name': 'gamma0', 'type': 'continuous', 'domain': (1.0e-2,1.0e2)},
        {'name': 'eta0', 'type': 'continuous', 'domain': (1.0e-2,1.0e2)},
        {'name': 'kappa0', 'type': 'continuous', 'domain': (1.0e-2,1.0e2)},
        {'name': 'kappa1', 'type': 'continuous', 'domain': (1.0e-2,1.0e2)}
    ]
    myBopt = GPyOpt.methods.BayesianOptimization(f=trainModel.FunctionforGPyOpt,domain=bounds,
            initial_design_numdata=tune_trial_num,acquisition_type='LCB')
    saveGPyOpt(myBopt,fold_id)
    print(myBopt.x_opt)
    print(myBopt.fx_opt)

    # testing
    optX = myBopt.x_opt
    testModel = PC1a(gDF,testDF)
    cor = testModel.corrcoefWithTruth(optX)
    with open(os.path.join(snapshot_path, 'resultPC1a.txt'),'a') as f:
        f.write('##PC1a,split{}##\n'.format(i))
        f.write('x_opt: {}\n'.format(myBopt.x_opt))
        f.write('fx_opt: {}\n'.format(myBopt.fx_opt))
        f.write('test corrcoef: {}\n'.format(cor))
