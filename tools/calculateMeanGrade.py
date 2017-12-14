# coding: utf-8

import os,sys
import copy
import numpy as np
import pandas as pd

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
func_metric = lambda true,estimated: np.corrcoef(true, estimated)[0,1]

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
    corrcoef_test = func_metric(true_test, mean_corrected_test)
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
    corrcoef_test = func_metric(true_test, mean_diff_test)
    statistic_test = np.append(statistic_test,corrcoef_test)
print('-------- result mean diff --------')
print('mean:{0}, std:{1}'.format(statistic_test.mean(),statistic_test.std()))
