# coding: utf-8

import os
import sys
import numpy as np
import pandas as pd

# path information
groundtruth_csv = '../dataset/open_peer_review/peer_review/translated_groundtruth.csv'
peer_review_csv = '../dataset/open_peer_review/peer_review/peer_review_forPG3.csv'

# load dataset
gDF = pd.read_csv(groundtruth_csv)
rDF = pd.read_csv(peer_review_csv)

# calculate mean grade
meanGradeDF = rDF[['receiver_id','value']].groupby(['receiver_id']).mean()

# interpolate missing value
# レビューを受けていないユーザは平均値に割りあてる
userNum = gDF['grade'].get_values().shape[0]
avgValue = meanGradeDF['value'].mean()
meanGrade = np.array([avgValue]*userNum, dtype=np.float64)
for i in xrange(userNum):
    if i in meanGradeDF.index:
        meanGrade[i] = meanGradeDF['value'][i]

# corrcoef
truth = gDF['grade'].get_values()
print('correlation coefficient: {}'.format(np.corrcoef(truth,meanGrade)[0,1]))
