# coding:utf-8

import sys,os
import copy
import collections
import numpy as np

# path info
eval_dir = '../result/inferred_parameter/evaluations'

# 2017/12/19
metrics_test = ['precisionAt3_4','precisionAt5_4','precisionAt5_34','precisionAt10_34','nDCGAt5','nDCGAt10']
num_fold = [5]
metrics_tune = ['Corrcoef','Kendalltau','Spearmanrho']
models = ['mean grade','mean corrected','mean diff',
            'PC1','PC2','PG1','PG3','PG4','PG5',
            'PCG1','PCG3','PCG4','PCG5']

def getLine(lines,word):
    rest = copy.deepcopy(lines)
    line = rest.pop(0)
    while line.find(word) == -1:
        line = rest.pop(0)
        if not rest: break
    return line, rest

def findMeanStd(lines,model):
    rest = copy.deepcopy(lines)
    marker, rest = getLine(rest,'result')
    if marker.find(model) == -1:
        print('Warning: mean&std may not match the result of {}↓'.format(model))
    result, rest = getLine(rest,'mean')
    mean, std = result.strip().split(',')
    return rest, float(mean.split(':')[1]), float(std.split(':')[1])

if __name__ == '__main__':
    # make table of file list
    tables = collections.defaultdict(list)
    for metric_tune in metrics_tune:
        for metric_test in metrics_test:
            for num in num_fold:
                tables[metric_tune].append('{0}_{1}fold_tuned{2}.txt'.format(metric_test,num,metric_tune))

    # check existence of file
    for metric_tune in metrics_tune:
        print(metric_tune,[os.path.exists(os.path.join(eval_dir,fname)) for fname in tables[metric_tune]])

    # make [tune metric] list
    for metric_tune in metrics_tune:
        # make [test metric]*[model]*[mean,std] matrix for results
        results = np.empty((0,len(models),2))
        for fname in tables[metric_tune]:
            path = os.path.join(eval_dir,fname)
            with open(path,'r') as f:
                lines = f.readlines()
            results_inFile = np.empty((0,2))
            for model in models:
                lines, mean, std = findMeanStd(lines,model)
                result_model = np.array([[mean,std]])
                results_inFile = np.append(results_inFile,result_model,axis=0)
            results = np.append(results,[results_inFile],axis=0)
        # transpose [model]*[test metric]*[mean,std] matrix for results
        results = results.transpose(1,0,2)
        print('---- tune:{0} ----'.format(metric_tune))
        for i,results_model in enumerate(results):
            print '{}'.format(models[i]),
            for j,results_mean_std in enumerate(results_model):
                print '& ${0:.3f}\\pm{1:.3f}$'.format(round(results_mean_std[0],3),round(results_mean_std[1],3)),
                #print('{0}: mean={1}, std={2}'.format(metrics_test[j],results_mean_std[0],results_mean_std[1]))
            print('\\\\')
