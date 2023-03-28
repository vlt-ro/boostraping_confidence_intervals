"""
@author Valentina Roquemen-Echeverri
@update 03-28-2023
"""
# Libraries
import numpy as np
from sklearn.metrics import confusion_matrix
import random
import pandas as pd
np.random.seed(0)

"""
@brief Calculate metrics
@param y_true: array-like of shape (n_samples,) Ground truth (correct) target values.
@param y_pred: array-like of shape (n_samples,) Estimated targets as returned by a classifier.
@return sensitivity, specificity, likelihood ratio +, likelihood ratio -
"""
def calculate_metrics(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred,labels=[0,1]).ravel()

    sensitivity = tp / (tp + fn)
    specificity = 1 - fp / (fp + tn)

    if specificity!=1:
        LR_pos = sensitivity/(1-specificity)
    else:
        LR_pos = np.nan

    if specificity != 0 :
        LR_neg = (1-sensitivity)/specificity
    else:
        LR_neg = np.nan

    return sensitivity, specificity,LR_pos,LR_neg

"""
@brief 
@param y_true: array-like of shape (n_samples,) - Ground truth (correct) target values.
@param y_pred: array-like of shape (n_samples,) - Estimated targets as returned by a classifier.
@param n_resamples: int, default: 1000 - The number of resamples performed to form the bootstrap distribution of the statistic.
@param alpha: float, default: 0.95 - The confidence level of the confidence interval.
@return 
"""
def bootstraping_confidence_intervals(y_true, y_pred,n_resamples = 1000,alpha = 0.95):

    data = np.transpose(np.array([y_true,y_pred]))

    metrics_bootstrap = []
    for i in range(n_resamples):
        random_data = np.array(random.choices(data.tolist(), k = len(data)))
        metrics_bootstrap.append(np.array(calculate_metrics(random_data[:,0],random_data[:,1])))

    metrics_bootstrap = np.array(metrics_bootstrap)

    results = pd.DataFrame(index = ['Sensitivity','Specificity','LRpos','LRneg'], columns=['Point Estimate','Lower C.I.','Upper C.I.'])
    results['Point Estimate'] = np.array(calculate_metrics(y_true,y_pred))

    for i,index in enumerate(results.index):

        p = ((1.0-alpha)/2.0) * 100
        lower = np.percentile(metrics_bootstrap[:,i], p)
        results.loc[index,'Lower C.I.'] = lower

        p = (alpha+((1.0-alpha)/2.0)) * 100
        upper = np.percentile(metrics_bootstrap[:,i], p)
        results.loc[index,'Upper C.I.'] = upper

    return results