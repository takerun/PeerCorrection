# conding:utf-8

import os,sys
import pickle
import copy
import argparse
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

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

parser.add_argument('-met','--metric', \
        action='store', \
        nargs=None, \
        const=None, \
        default=None, \
        type=str, \
        choices=None, \
        help='Metric option which you\'d like to set.', \
        metavar=None)

# config
num_folds = 5
groundtruth = '../dataset/open_peer_review_v3/peer_review/translated_groundtruth.csv'

if __name__ == '__main__':
    #argparse
    args = parser.parse_args()

    #set true ability
    gDF = pd.read_csv(groundtruth)
    true_ability = gDF['grade'].get_values()

    #load df_result(table of trial model)
    path_result = '../result/inferred_parameter/{}/models.csv'.format(args.model)
    df_result = pd.read_csv(path_result)

    #load estimated_abilities(estimated ability per trial)
    dir_result = os.path.dirname(path_result)
    estimated_abilities = np.empty((0,len(gDF)))
    for fname in df_result['file']:
        with open(os.path.join(dir_result,fname),'rb') as f:
            est = pickle.load(f)
        add =  np.expand_dims(est['ability'],axis=0)
        estimated_abilities = np.append(estimated_abilities,add,axis=0)

    #generate random user_id and fold that ids
    np.random.seed(12345678)
    permu =np.random.permutation(len(true_ability))
    idx_inFold = np.array_split(permu, num_folds)
    print('-------- result {} --------'.format(args.model))

    if args.metric == 'cor':
        statistic_test = np.empty(0)
        for loop in xrange(num_folds):
            buf_list = copy.copy(idx_inFold)
            idx_train = buf_list.pop(loop)
            idx_test = np.concatenate(buf_list)
            #search best train model
            true_train = true_ability[idx_train]
            estimations_train = estimated_abilities[:,idx_train]
            corrcoefs_train = np.array([np.corrcoef(true_train, estimated_ability)[0,1] for estimated_ability in estimations_train])
            id_best_model = corrcoefs_train.argmax()
            #test the best model
            true_test = true_ability[idx_test]
            estimation_test = estimated_abilities[id_best_model,idx_test]
            corrcoef_test = np.corrcoef(true_test, estimation_test)[0,1]
            print('test corrcoef:{0}, best train corrcoef:{1}, best model:{2}'.format(corrcoef_test,corrcoefs_train.max(), id_best_model))
            #accumulate test results
            statistic_test = np.append(statistic_test,corrcoef_test)
        print('mean:{0}, std:{1}'.format(statistic_test.mean(),statistic_test.std()))
    elif args.metric == '':
        sys.exit()
