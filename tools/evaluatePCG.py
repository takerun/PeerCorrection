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
NUM_FOLDS = 5
GROUNDTRUTH = '../dataset/open_peer_review_v3/peer_review/translated_groundtruth.csv'

# functions
def evaluateEstimationInFold(func_metric,name_metric,num_folds,true_scores,arr_estimated_scores):
    #generate random permutation and fold that index
    np.random.seed(12345678)
    permu =np.random.permutation(len(true_scores))
    idx_inFold = np.array_split(permu, num_folds)
    statistic_test = np.empty(0)
    for loop in xrange(num_folds):
        buf_list = copy.copy(idx_inFold)
        idx_train = buf_list.pop(loop)
        idx_test = np.concatenate(buf_list)
        #train
        true_train = true_scores[idx_train]
        arr_estimated_train = arr_estimated_scores[:,idx_train]
        evalues_train = np.array([func_metric(true_train, estimated_train) for estimated_train in arr_estimated_train])
        id_best_model = evalues_train.argmax()
        #test
        true_test = true_scores[idx_test]
        estimated_test_best = arr_estimated_scores[id_best_model,idx_test]
        evalue_test = func_metric(true_test, estimated_test_best)
        print('test {0}:{1}, best train {0}:{2}, best model:{3}'.format(name_metric,evalue_test,evalues_train.max(), id_best_model))
        #accumulate
        statistic_test = np.append(statistic_test,evalue_test)
    print('mean:{0}, std:{1}'.format(statistic_test.mean(),statistic_test.std()))


if __name__ == '__main__':
    #argparse
    args = parser.parse_args()

    #load true ability
    gDF = pd.read_csv(GROUNDTRUTH)
    true_ability = gDF['grade'].get_values()

    #load estimated abilities
    path_result = '../result/inferred_parameter/{}/models.csv'.format(args.model)
    df_result = pd.read_csv(path_result)
    dir_result = os.path.dirname(path_result)
    estimated_abilities = np.empty((0,len(gDF)))
    for fname in df_result['file']:
        with open(os.path.join(dir_result,fname),'rb') as f:
            est = pickle.load(f)
        add =  np.expand_dims(est['ability'],axis=0)
        estimated_abilities = np.append(estimated_abilities,add,axis=0)

    if args.metric == 'cor':
        func_metric = lambda true,estimated: np.corrcoef(true, estimated)[0,1]
        name_metric = 'corrcoef'
    elif args.metric == 'rcor':
        sys.exit()
    else:
        print('Error: set metrics [cor|rcor|prec]')

    num_folds = NUM_FOLDS
    true_scores = true_ability
    arr_estimated_scores = estimated_abilities
    print('-------- result {} --------'.format(args.model))
    evaluateEstimationInFold(func_metric,name_metric,num_folds,true_scores,arr_estimated_scores)
