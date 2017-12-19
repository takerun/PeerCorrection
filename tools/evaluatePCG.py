# conding:utf-8

import os,sys
import pickle
import copy
import argparse
import numpy as np
from scipy import stats
from sklearn import metrics
import Bio.Cluster
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from ranking import RankingMeasures

# parser settings
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
parser.add_argument('-tune','--tune-metric', \
        action='store', \
        nargs=None, \
        const=None, \
        default='cor', \
        type=str, \
        choices=None, \
        help='Tune metric option which you\'d like to set.', \
        metavar=None)
parser.add_argument('-test','--test-metric', \
        action='store', \
        nargs=None, \
        const=None, \
        default='cor', \
        type=str, \
        choices=None, \
        help='Test metric option which you\'d like to set.', \
        metavar=None)
parser.add_argument('-k','--top-k', \
        action='store', \
        nargs=None, \
        const=None, \
        default=10, \
        type=int, \
        choices=None, \
        help='Number of top-k value which you\'d like to set.', \
        metavar=None)
parser.add_argument('-thre','--threshold', \
        action='store', \
        nargs=None, \
        const=None, \
        default=4, \
        type=int, \
        choices=None, \
        help='Number of threshold value which you\'d like to set.', \
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

def evaluateEstimationInFold2(tune_metric,name_tune_metric,test_metric,name_test_metric,num_folds,true_scores,arr_estimated_scores):
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
        evalues_train = np.array([tune_metric(true_train, estimated_train) for estimated_train in arr_estimated_train])
        id_best_model = evalues_train.argmax()
        #test
        true_test = true_scores[idx_test]
        estimated_test_best = arr_estimated_scores[id_best_model,idx_test]
        evalue_test = test_metric(true_test, estimated_test_best)
        print('test {0}:{1}, best train {2}:{3}, best model:{4}'.format(name_test_metric,evalue_test,name_tune_metric,evalues_train.max(), id_best_model))
        #accumulate
        statistic_test = np.append(statistic_test,evalue_test)
    print('mean:{0}, std:{1}'.format(statistic_test.mean(),statistic_test.std()))

corrcoef = lambda true,estimated: np.corrcoef(true, estimated)[0,1]
kendalltau = lambda true,estimated: stats.kendalltau(true, estimated)[0]
spearmanrho = lambda true,estimated: 1-Bio.Cluster.distancematrix((true,estimated), dist="s")[1][0]

def precisionAtK(true,estimated,top_k,threshold):
    top_ranker_ture = np.array((true >= threshold))
    id_top_k = estimated.argsort()[::-1][:top_k]
    TP = top_ranker_ture[id_top_k].sum()
    return TP/float(top_k)

def auc(true,estimated,threshold):
    fpr, tpr, thresholds = metrics.roc_curve(true >= threshold, estimated, pos_label=1)
    return metrics.auc(fpr, tpr)

def nDCG(true,estimated,top_k):
    rm = RankingMeasures(estimated, true)
    return rm.nDCG(k=top_k)

# main
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

    # set tuning metric
    if args.tune_metric == 'cor':
        tune_metric = corrcoef
        name_tune_metric = 'corrcoef'
    elif args.tune_metric == 'ktau':
        tune_metric = kendalltau
        name_tune_metric = 'kendalltau'
    elif args.tune_metric == 'srho':
        tune_metric = spearmanrho
        name_tune_metric = 'spearmanrho'
    elif args.tune_metric == 'preck':
        top_k = args.top_k
        threshold = args.threshold
        tune_metric = lambda true,estimated: precisionAtK(true,estimated,top_k,threshold)
        name_tune_metric = 'precision@{}'.format(top_k)
    elif args.tune_metric == 'auc':
        threshold = args.threshold
        tune_metric = lambda true,estimated: auc(true,estimated,threshold)
        name_tune_metric = 'auc'
    elif args.tune_metric == 'ndcg':
        top_k = args.top_k
        tune_metric = lambda true,estimated: nDCG(true,estimated,top_k)
        name_tune_metric = 'nDCG@{}'.format(top_k)
    else:
        print('Error: set tune metrics [cor|ktau|srho|preck|auc|ndcg]')
        sys.exit()

    # set test metric
    if args.test_metric == 'cor':
        test_metric = corrcoef
        name_test_metric = 'corrcoef'
    elif args.test_metric == 'ktau':
        test_metric = kendalltau
        name_test_metric = 'kendalltau'
    elif args.test_metric == 'srho':
        test_metric = spearmanrho
        name_test_metric = 'spearmanrho'
    elif args.test_metric == 'preck':
        top_k = args.top_k
        threshold = args.threshold
        test_metric = lambda true,estimated: precisionAtK(true,estimated,top_k,threshold)
        name_test_metric = 'precision@{}'.format(top_k)
    elif args.test_metric == 'auc':
        threshold = args.threshold
        test_metric = lambda true,estimated: auc(true,estimated,threshold)
        name_test_metric = 'auc'
    elif args.test_metric == 'ndcg':
        top_k = args.top_k
        test_metric = lambda true,estimated: nDCG(true,estimated,top_k)
        name_test_metric = 'nDCG@{}'.format(top_k)
    else:
        print('Error: set test metrics [cor|ktau|srho|preck|auc|ndcg]')
        sys.exit()


    num_folds = NUM_FOLDS
    true_scores = true_ability
    arr_estimated_scores = estimated_abilities
    print('-------- result {} --------'.format(args.model))
    evaluateEstimationInFold2(tune_metric,name_tune_metric,test_metric,name_test_metric,num_folds,true_scores,arr_estimated_scores)
