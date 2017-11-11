# coding: utf-8

import os,time
import numpy as np
import pandas as pd
import pystan
import pickle

# path infomation
dataset_path = '../dataset'
model_path = '../models'
result_path = '../result'

if __name__ == '__main__':
    gdf = pd.read_csv(os.path.join(dataset_path, 'peer_review/translated_groundtruth.csv'))
    grade_list = gdf['grade'].get_values()
    user_num = grade_list.shape[0]

    rdf = pd.read_csv(os.path.join(dataset_path, 'peer_review/peer_review_dataset.csv'))
    sender = rdf['sender_id'].get_values()
    receiver = rdf['receiver_id'].get_values()
    corrected = rdf['corrected'].get_values()

    N = sender.shape[0]
    uNum = grade_list.shape[0]
    vNum = uNum
    sender = sender.astype(np.int64)
    receiver = receiver.astype(np.int64)
    # correctedの0と1反転処理を挟む
    corrected = (~corrected.astype(np.bool)).astype(np.int64)
    hyper = [1,1,10, 0.5]

    stan_data = {'N': N, 'uNum':uNum, 'vNum':vNum, 'sender': sender, 'receiver':receiver, 'corrected':corrected, 'hyper':hyper}

    start = time.time()
    fit = pystan.stan(file='../models/PC1.stan', data=stan_data, algorithm='NUTS', n_jobs=-1, iter=3000, chains=4, warmup=200)
    elapsed_time = time.time() - start

    print('Sampling finished. ELAPSED TIME: {}sec'.format(elapsed_time))
