# conding:utf-8

import os,sys
import time,datetime
import GPy
import GPyOpt
import numpy as np
import pandas as pd
from TrainTest import *
from PG4GM import PG4GM

# path information
trial_num = 5
tune_trial_num = 100
snapshot_path = '../result/snapshot/complement'
snapshot_form = 'PG4-hyper4-EI5000w500-split{0}-{1}'
peer_review_csv = '../dataset/open_peer_review/peer_review/peer_review_forPG3_suffled.csv'

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

# load dataset
rDF = pd.read_csv(peer_review_csv)
np.random.seed(12345678)
shuffled_pattern = [np.random.permutation(len(rDF)) for i in xrange(trial_num)]
for i in xrange(trial_num):
    # preprocess train&val&test dataset
    shuffled_rDF = rDF.take(shuffled_pattern[i]).reset_index(drop=True)
    trainDF,valDF,testDF = splitDataset523(shuffled_rDF)

    # model setting
    pg4gm = PG4GM(trainDF, valDF, testDF)

    # training
    bounds = [
        {'name': 'mu0', 'type': 'continuous', 'domain': (0,4)},
        {'name': 'gamma0', 'type': 'continuous', 'domain': (1.0e-2,1.0e2)},
        {'name': 'beta0', 'type': 'continuous', 'domain': (1.0e-2,1.0e2)},
        {'name': 'eta0', 'type': 'continuous', 'domain': (1.0e-2,1.0e2)}
    ]
    myBopt = GPyOpt.methods.BayesianOptimization(f=pg4gm.FunctionforGPyOpt,domain=bounds,
            initial_design_numdata=tune_trial_num,acquisition_type='LCB')
    saveGPyOpt(myBopt,i)

    # testing
    optX = myBopt.x_opt
    rmse = pg4gm.rmseForTest(optX)
    with open(os.path.join(snapshot_path, 'result.txt'),'a') as f:
        f.write('##PG4GM,split{}##\n'.format(i))
        f.write('x_opt: {}\n'.format(myBopt.x_opt))
        f.write('fx_opt: {}\n'.format(myBopt.fx_opt))
        f.write('test RMSE: {}\n'.format(rmse))
