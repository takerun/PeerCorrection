# coding: utf-8

import os,sys
import numpy as np
import pandas as pd
import pystan

# path information
PG5PC2_path = '../models/PG5PC2.stan'

class PG5PC2:
    def __init__(self, groundtruth_DataFrame, review_DataFrame):
        '''
            正解とレビューのdataFrameを受け取る。
            stan用の入力データを作成、stanDataに格納。
        '''
        self.gDF = groundtruth_DataFrame
        self.rDF = review_DataFrame
        self.grade = self.gDF['grade'].get_values()
        self.sender = self.rDF['sender_id'].get_values().astype(np.int64)
        self.receiver = self.rDF['receiver_id'].get_values().astype(np.int64)
        self.value = self.rDF['value'].get_values().astype(np.int64)
        self.diff = self.rDF['diff'].get_values().astype(np.int64)
        self.reviewNum = self.sender.shape[0]
        self.userNum = self.grade.shape[0]
        self.reviewerNum = self.userNum
        self.stanData = {
            'N': self.reviewNum, 'uNum':self.userNum, 'vNum':self.reviewerNum,
            'sender': self.sender, 'receiver':self.receiver, 'value':self.value,
            'diff':self.diff, 'hyper':[1]*6
            }

    def fit(self, hyper_list, iteration=5000, chains=4, warmup=500, n_jobs=-1, algorithm='NUTS'):
        '''
            ハイパーパラメータを受け取り、数を確認。
            stanによるパラメータ推定を実行、結果をstan_fitに格納。
        '''
        if len(hyper_list) != 6:
            print('NError: {} given. Set 6 hyper parameters.'.format(len(hyper_list)))
            sys.exit(1)
        self.stanData['hyper'] = hyper_list
        stan_fit = pystan.stan(file=PG5PC2_path, data=self.stanData, algorithm=algorithm,
                n_jobs=n_jobs, iter=iteration, chains=chains, warmup=warmup, refresh=0)
        return stan_fit

    def corrcoefWithTruth(self,hyper_list):
        '''
            self.fitを実行し、stan実行
            正解と推定の相関係数を計算
        '''
        stan_fit = self.fit(hyper_list)
        eap_value = stan_fit.summary()['summary'][:,0]
        estimated = eap_value[0:self.userNum]
        cor = np.corrcoef(self.grade, estimated)[0,1]
        return cor

    def FunctionforGPyOpt(self,x):
        '''
            ベイズ最適化のための関数
            ハイパーパラメータを入力して、正解との相関を返す
        '''
        x0,x1,x2,x3,x4,x5 = x[0,0],x[0,1],x[0,2],x[0,3],x[0,4],x[0,5]
        corrcoef = self.corrcoefWithTruth([x0,x1,x2,x3,x4,x5])
        return -corrcoef
