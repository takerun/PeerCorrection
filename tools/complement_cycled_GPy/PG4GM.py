# coding: utf-8

import os,sys
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
import pandas as pd
import pystan

# path information
PG4_path = '../models/PG4.stan'

# prefix
USER_NUM = 438

# functions
def splitDataFrame(dataFrame):
    '''
        訓練データから各カラムを切り離す。
    '''
    sender = dataFrame['sender_id'].get_values().astype(np.int64)
    receiver = dataFrame['receiver_id'].get_values().astype(np.int64)
    value = dataFrame['value'].get_values().astype(np.int64)
    return sender, receiver, value

# class
class PG4GM:
    def __init__(self, train_review_DF, val_review_DF, test_review_DF):
        '''
            学習、ハイパラ調整用検証、テストのためのレビューcsvを受け取る。
            stan用の入力データを作成、trainDataに格納。
        '''
        self.trDF = train_review_DF
        self.valDF = val_review_DF
        self.tsDF = test_review_DF
        self.sender, self.receiver, self.value = splitDataFrame(self.trDF)
        self.reviewNum = self.sender.shape[0]
        self.userNum = USER_NUM
        self.reviewerNum = self.userNum
        self.trainData = {
            'N': self.reviewNum, 'uNum':self.userNum, 'vNum':self.reviewerNum,
            'sender': self.sender, 'receiver':self.receiver, 'value':self.value,
            'hyper':[1]*4
            }
        self.senderVal, self.receiverVal, self.valueVal = splitDataFrame(self.valDF)
        self.senderTest, self.receiverTest, self.valueTest = splitDataFrame(self.tsDF)

    def fit(self, hyper_list, iteration=5000, chains=4, warmup=500, n_jobs=-1, algorithm='NUTS'):
        '''
            ハイパーパラメータを受け取り、数を確認。
            stanによるパラメータ推定を実行、結果をstan_fitに格納。
        '''
        if len(hyper_list) != 4:
            print('NError: {} given. Set 4 hyper parameters.'.format(len(hyper_list)))
            sys.exit(1)
        self.trainData['hyper'] = hyper_list
        stan_fit = pystan.stan(file=PG4_path, data=self.trainData, algorithm=algorithm,
                n_jobs=n_jobs, iter=iteration, chains=chains, warmup=warmup)
        return stan_fit

    def generateValue(self, sender, receiver, ability, reliability, bias):
        '''
            今回の実装では、穴埋め値を生成せず、正規分布の平均を返すこととする
        '''
        return ability[receiver] + bias[sender]

    def rmseForVal(self,hyper_list):
        '''
            self.fitを実行し、stan実行
            穴埋めの正解と推定のRMSEを計算
        '''
        stan_fit = self.fit(hyper_list)
        eap_value = stan_fit.summary()['summary'][:,0]
        split = self.userNum
        ability = eap_value[0:split]
        reliability = eap_value[split:split*2]
        bias = eap_value[split*2:split*3]
        valueEst = self.generateValue(
                self.senderVal, self.receiverVal,ability,reliability,bias)
        rmse = sqrt(mean_squared_error(self.valueVal, valueEst))
        return rmse

    def rmseForTest(self,hyper_list):
        '''
            self.fitを実行し、stan実行
            穴埋めの正解と推定のRMSEを計算
        '''
        stan_fit = self.fit(hyper_list)
        eap_value = stan_fit.summary()['summary'][:,0]
        split = self.userNum
        ability = eap_value[0:split]
        reliability = eap_value[split:split*2]
        bias = eap_value[split*2:split*3]
        valueEst = self.generateValue(
                self.senderTest, self.receiverTest,ability,reliability,bias)
        rmse = sqrt(mean_squared_error(self.valueTest, valueEst))
        return rmse

    def FunctionforGPyOpt(self,x):
        '''
            ベイズ最適化のための関数
            ハイパーパラメータを入力して、正解とのRMSEを返す
            最小化したい値を返す
        '''
        x0,x1,x2,x3 = x[0,0],x[0,1],x[0,2],x[0,3]
        rmse = self.rmseForVal([x0,x1,x2,x3])
        return rmse
