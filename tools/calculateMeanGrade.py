# coding: utf-8

import os,sys
import copy
import numpy as np
from scipy import stats
import Bio.Cluster
from sklearn import metrics
import pandas as pd
from ranking import RankingMeasures

# config
NUM_FOLDS = 5

# path information
groundtruth_csv = '../dataset/open_peer_review_v3/peer_review/translated_groundtruth.csv'
peer_review_csv = '../dataset/open_peer_review_v3/peer_review/peer_review_forPG3.csv'

# load dataset
gDF = pd.read_csv(groundtruth_csv)
rDF = pd.read_csv(peer_review_csv)

# calculate true ability
true_ability = gDF['grade'].get_values()

# calculate mean grades
mean_grades = rDF[['receiver_id','value']].groupby(['receiver_id']).mean()['value'].get_values()
mean_corrected = rDF[['receiver_id','corrected']].groupby(['receiver_id']).mean()['corrected'].get_values()
mean_diff = rDF[['receiver_id','diff']].groupby(['receiver_id']).mean()['diff'].get_values()

# generate random permutation and fold that index
np.random.seed(12345678)
permu =np.random.permutation(len(true_ability))
idx_inFold = np.array_split(permu, NUM_FOLDS)

# set metric
corrcoef = lambda true,estimated: np.corrcoef(true, estimated)[0,1]
kendalltau = lambda true,estimated: stats.kendalltau(true, estimated)[0]
spearmanrho = lambda true,estimated: 1-Bio.Cluster.distancematrix((true,estimated), dist="s")[1][0]
def precisionAtK(true,estimated,top_k,threshold):
    top_ranker_ture = np.array((true >= threshold))
    id_top_k = estimated.argsort()[::-1][:top_k]
    TP = top_ranker_ture[id_top_k].sum()
    return TP/float(top_k)
precAt10_4 = lambda true,estimated: precisionAtK(true,estimated,10,4)
precAt5_4 = lambda true,estimated: precisionAtK(true,estimated,5,4)
precAt3_4 = lambda true,estimated: precisionAtK(true,estimated,3,4)
precAt10_34 = lambda true,estimated: precisionAtK(true,estimated,10,3)
precAt5_34 = lambda true,estimated: precisionAtK(true,estimated,5,3)
def auc(true,estimated,threshold):
    fpr, tpr, thresholds = metrics.roc_curve(true >= threshold, estimated, pos_label=1)
    return metrics.auc(fpr, tpr)
auc4 = lambda true,estimated: auc(true,estimated,4)
auc34 = lambda true,estimated: auc(true,estimated,3)
auc234 = lambda true,estimated: auc(true,estimated,2)
auc1234 = lambda true,estimated: auc(true,estimated,1)
def nDCG(true,estimated,top_k):
    rm = RankingMeasures(estimated, true)
    return rm.nDCG(k=top_k)
ndcgAt10 = lambda true,estimated: nDCG(true,estimated,10)
ndcgAt5 =  lambda true,estimated: nDCG(true,estimated,5)
func_metric = precAt5_34

statistic_test = np.empty(0)
for loop in xrange(NUM_FOLDS):
    buf_list = copy.copy(idx_inFold)
    idx_train = buf_list.pop(loop)
    idx_test = np.concatenate(buf_list)
    #test
    true_test = true_ability[idx_test]
    mean_grade_test = mean_grades[idx_test]
    corrcoef_test = func_metric(true_test, mean_grade_test)
    statistic_test = np.append(statistic_test,corrcoef_test)
print('-------- result mean grade --------')
print('mean:{0}, std:{1}'.format(statistic_test.mean(),statistic_test.std()))

statistic_test = np.empty(0)
for loop in xrange(NUM_FOLDS):
    buf_list = copy.copy(idx_inFold)
    idx_train = buf_list.pop(loop)
    idx_test = np.concatenate(buf_list)
    #test
    true_test = true_ability[idx_test]
    mean_corrected_test = mean_corrected[idx_test]
    corrcoef_test = func_metric(true_test, -mean_corrected_test)
    statistic_test = np.append(statistic_test,corrcoef_test)
print('-------- result mean corrected --------')
print('mean:{0}, std:{1}'.format(statistic_test.mean(),statistic_test.std()))

statistic_test = np.empty(0)
for loop in xrange(NUM_FOLDS):
    buf_list = copy.copy(idx_inFold)
    idx_train = buf_list.pop(loop)
    idx_test = np.concatenate(buf_list)
    #test
    true_test = true_ability[idx_test]
    mean_diff_test = mean_diff[idx_test]
    corrcoef_test = func_metric(true_test, -mean_diff_test)
    statistic_test = np.append(statistic_test,corrcoef_test)
print('-------- result mean diff --------')
print('mean:{0}, std:{1}'.format(statistic_test.mean(),statistic_test.std()))
