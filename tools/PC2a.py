# coding: utf-8

import os,sys
import time,datetime
import pickle
import numpy as np
import pandas as pd
import pystan

RECORD = True

# path information
PC2_path = '../models/PC2a.stan'
save_param_dir = '../result/inferred_parameter/PC2a'

# function
def recordPC2aInfo(hyper_list, model_file, csv_file):
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file)
    else:
        df = pd.DataFrame(columns=['file','mu0','gamma0','eta0','kappa0'])
    add_srs = pd.Series([model_file]+hyper_list, index=df.columns)
    df = df.append(add_srs, ignore_index=True)
    df.to_csv(csv_file, index=False)

def saveStanExtract(var, pkl_file):
    with open(pkl_file,'wb') as f:
        pickle.dump(var,f)

def timeStamp():
    # return time as str
    todaydetail = datetime.datetime.today()
    return todaydetail.strftime("%Y%m%d%H%M%S")

class PC2a:
    def __init__(self, groundtruth_DataFrame, review_DataFrame):
        '''
            正解とレビューのdataFrameを受け取る。
            stan用の入力データを作成、stanDataに格納。
        '''
        self.gDF = groundtruth_DataFrame
        self.rDF = review_DataFrame
        self.grade = self.gDF['grade'].get_values()
        # data
        self.receiver = self.rDF['receiver_id'].get_values().astype(np.int64)
        self.sender_origin = self.rDF['sender_id'].get_values().astype(np.int64)
        self.diff = self.rDF['diff'].get_values().astype(np.int64)
        # total number
        self.reviewNum = len(self.receiver)
        self.userNum = len(set(self.receiver))
        set_sender = set(self.sender_origin)
        self.reviewerNum = len(set_sender)
        # transform sender_id
        self.sender_origin_id = list(set_sender)
        self.sender_origin_id.sort()
        self.sender = np.copy(self.sender_origin)
        for i,origin in enumerate(self.sender_origin_id):
            self.sender[self.sender == origin] = i
        # for stan
        self.stanData = {
            'N':self.reviewNum, 'uNum':self.userNum, 'vNum':self.reviewerNum,
            'sender':self.sender, 'receiver':self.receiver, 'diff':self.diff,
            'senderOrigin':self.sender_origin_id, 'hyper':[1]*4
            }
        self.stanmodel = pystan.StanModel(file=PC2a_path)

    def fit(self, hyper_list, iteration=5000, chains=4, warmup=500, n_jobs=-1, algorithm='NUTS'):
        '''
            ハイパーパラメータを受け取り、数を確認。
            stanによるパラメータ推定を実行、結果をstan_fitに格納。
        '''
        if len(hyper_list) != 4:
            print('NError: {} given. Set 4 hyper parameters.'.format(len(hyper_list)))
            sys.exit(1)
        self.stanData['hyper'] = hyper_list
        stan_fit = self.stanmodel.sampling(data=self.stanData, algorithm=algorithm,
                n_jobs=n_jobs, iter=iteration, chains=chains, warmup=warmup, refresh=0)
        return stan_fit

    def corrcoefWithTruth(self,hyper_list):
        '''
            self.fitを実行し、stan実行
            推定パラメータを保存
            その後、正解と推定の相関係数を計算
        '''
        stan_fit = self.fit(hyper_list)
        ext = stan_fit.extract()
        inferred_ability = ext['ability'].mean(axis=0)
        if RECORD == True:
            inferred_bias = ext['bias'].mean(axis=0)
            inferred_noise = ext['noise'].mean(axis=0)
            parameters = {'ability':inferred_ability, 'bias':inferred_bias, 'noise':inferred_noise}
            # save
            pkl_name = 'PC2a-{}.pkl'.format(timeStamp())
            saveStanExtract(parameters, os.path.join(save_param_dir,pkl_name))
            # record table
            recordPC2aInfo(hyper_list,pkl_name,os.path.join(save_param_dir,'models.csv'))
        # calculate corrcoef
        cor = np.corrcoef(self.grade, inferred_ability)[0,1]
        return cor
