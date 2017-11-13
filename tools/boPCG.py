# conding:utf-8

import os,sys
import time,datetime
import argparse
import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from TrainTest import *

parser = argparse.ArgumentParser()
parser.add_argument('-m','--model', \
        action='store', \
        nargs=None, \
        const=None, \
        default=None, \
        type=str, \
        choices=None, \
        help='Model name which you\'d like to run.', \
        metavar=None)

parser.add_argument('-init','--init-npz', \
        action='store', \
        nargs=None, \
        const=None, \
        default=None, \
        type=str, \
        choices=None, \
        help='npz file which you\'d like to set.', \
        metavar=None)

# config
bo_num_iter = 0
bo_init_points = 100
snapshot_path = '../result/snapshot/ability_estimation_bo'
groundtruth_csv = '../dataset/open_peer_review_v2/peer_review/translated_groundtruth.csv'
peer_review_csv = '../dataset/open_peer_review_v2/peer_review/peer_review_forPG3.csv'

# functions
def timeStamp():
    # return time as str
    todaydetail = datetime.datetime.today()
    return todaydetail.strftime("%Y%m%d%H%M%S")

def saveBO(bo,fold_num):
    filename = snapshot_form.format(fold_num,timeStamp())
    path = os.path.join(snapshot_path, filename)
    npzSave(path,bo)
    print("Save: {}".format(path))

if __name__ == '__main__':
    # preprocess train&test dataset
    gDF = pd.read_csv(groundtruth_csv)
    rDF = pd.read_csv(peer_review_csv)

    #argparse
    args = parser.parse_args()
    if args.model == 'PG1':
        from PG1 import PG1
        snapshot_form = 'PG1-hyper5-RAM5000w500-{0}-{1}'
        trainModel = PG1(gDF, rDF)
    elif args.model == 'PG3':
        from PG3 import PG3
        snapshot_form = 'PG3-hyper5-RAM5000w500-{0}-{1}'
        trainModel = PG3(gDF, rDF)
    elif args.model == 'PG4':
        from PG4 import PG4
        snapshot_form = 'PG4-hyper4-RAM5000w500-{0}-{1}'
        trainModel = PG4(gDF, rDF)
    elif args.model == 'PG5':
        from PG5 import PG5
        snapshot_form = 'PG5-hyper5-RAM5000w500-{0}-{1}'
        trainModel = PG5(gDF, rDF)
    elif args.model == 'PC1':
        from PC1 import PC1
        snapshot_form = 'PC2-hyper4-RAM5000w500-{0}-{1}'
        trainModel = PC1(gDF, rDF)
    elif args.model == 'PC2':
        from PC2 import PC2
        snapshot_form = 'PC2-hyper4-RAM5000w500-{0}-{1}'
        trainModel = PC2(gDF, rDF)
        def evaluate_BO(mu0,gamma0,eta0,kappa0):
            cor = trainModel.corrcoefWithTruth([mu0,gamma0,eta0,kappa0])
            return cor
        pbounds = {'mu0': (0.0, 4.0),
                   'gamma0': (1.0, 3.0e1),
                   'eta0': (1.0, 1.0e1),
                   'kappa0': (1.0, 1.0e2),
                   }
        def npzSave(path,bo):
            keys = bo.keys
            X = bo.X.transpose(1,0)
            Y = bo.Y
            np.savez(path,
                    mu0=X[keys.index('mu0')], gamma0=X[keys.index('gamma0')],
                    eta0=X[keys.index('eta0')], kappa0=X[keys.index('kappa0')],
                    target=Y)
        name_result_file = 'resultPC2.txt'
        label = 'PC2'
    elif args.model == 'PCG1':
        from PCG1 import PCG1
        snapshot_form = 'PCG1-hyper6-RAM5000w500-{0}-{1}'
        trainModel = PCG1(gDF, rDF)
        def evaluate_BO(mu0,gamma0,alpha0,beta0,eta0,kappa0):
            cor = trainModel.corrcoefWithTruth([mu0,gamma0,alpha0,beta0,eta0,kappa0])
            return cor
        pbounds = {'mu0': (0.0, 4.0),
                   'gamma0': (1.0, 5.0),
                   'alpha0': (1.0, 1.0e2),
                   'beta0': (1.0, 1.0e2),
                   'eta0': (1.0, 5.0),
                   'kappa0': (1.0, 1.0e2),
                   }
        def npzSave(path,bo):
            keys = bo.keys
            X = bo.X.transpose(1,0)
            Y = bo.Y
            np.savez(path,
                    mu0=X[keys.index('mu0')], gamma0=X[keys.index('gamma0')],
                    alpha0=X[keys.index('alpha0')], beta0=X[keys.index('beta0')],
                    eta0=X[keys.index('eta0')], kappa0=X[keys.index('kappa0')],
                    target=Y)
        name_result_file = 'resultPCG1.txt'
        label = 'PCG1'
    elif args.model == 'PCG3':
        from PCG3 import PCG3
        snapshot_form = 'PCG3-hyper6-RAM5000w500-{0}-{1}'
        trainModel = PCG3(gDF, rDF)
    elif args.model == 'PCG4':
        from PCG4 import PCG4
        snapshot_form = 'PCG4-hyper5-RAM5000w500-{0}-{1}'
        trainModel = PCG4(gDF, rDF)
    elif args.model == 'PCG5':
        from PCG5 import PCG5
        snapshot_form = 'PCG5-hyper6-RAM5000w500-{0}-{1}'
        trainModel = PCG5(gDF, rDF)
    elif args.model == 'PG1PC2':
        from PG1PC2 import PG1PC2
        snapshot_form = 'PG1PC2-hyper6-RAM5000w500-{0}-{1}'
        trainModel = PG1PC2(gDF, rDF)
    elif args.model == 'PG3PC2':
        from PG3PC2 import PG3PC2
        snapshot_form = 'PG3PC2-hyper6-RAM5000w500-{0}-{1}'
        trainModel = PG3PC2(gDF, rDF)
    elif args.model == 'PG4PC2':
        from PG4PC2 import PG4PC2
        snapshot_form = 'PG4PC2-hyper5-RAM5000w500-{0}-{1}'
        trainModel = PG4PC2(gDF, rDF)
    elif args.model == 'PG5PC2':
        from PG5PC2 import PG5PC2
        snapshot_form = 'PG5PC2-hyper6-RAM5000w500-{0}-{1}'
        trainModel = PG5PC2(gDF, rDF)
    else:
        print('Warning: set model name')
        sys.exit()

    # training
    BO = BayesianOptimization(evaluate_BO, pbounds)
    if args.init_npz:
        npzfile = np.load(args.init_npz)
        BO.initialize(dict(npzfile))
    BO.maximize(init_points=bo_init_points, n_iter=bo_num_iter)
    saveBO(BO,'full')

    with open(os.path.join(snapshot_path, name_result_file),'a') as f:
        f.write('##{},full##\n'.format(label))
        f.write('keys: {}\n'.format(BO.keys))
        f.write('opt X: {}\n'.format(BO.X[BO.Y.argmax()]))
        f.write('max cor: {}\n'.format(BO.Y.max()))
