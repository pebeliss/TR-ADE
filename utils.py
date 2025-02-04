import random
import numpy as np
import pandas as pd
import tensorflow as tf
import os
from scipy import stats
from scipy.stats import skew

def get_data_types(data, cat_threshold=10):
    data_types = []
    for col in data.columns:
        # row = [None]*4
        n_uniq = data[col].nunique(dropna=True)
        if n_uniq == 2:
            row = ['bin', n_uniq,None,None]
        elif n_uniq > 2 and n_uniq <= cat_threshold:
            print(f'{col} seems to be categorical and will be removed. Please onehot encode or add categorical likelihood handling.')
            row = ['cat', n_uniq,None,None]
        elif n_uniq > cat_threshold and data[col].dropna().min() >= 0:
            col_data = data[col].dropna()
            skewness = abs(skew(col_data))
            best_fit, diff = AIC_comparison(col_data)
            if best_fit == 'norm' and skewness > 1:
                best_fit = 'lognorm'
            elif best_fit == 'lognorm' and skewness <= 1:
                best_fit = 'norm'
            row = [best_fit, 1, round(diff, 1), round(skewness, 2)]
        elif n_uniq > cat_threshold and data[col].dropna().min() < 0:
            skewness = skew(col_data)
            row = ['norm', 1,None,round(skewness, 2)]
        else:
            print(f'{col} not within defined distributions.')
            print(n_uniq)
        data_types.append(row)
    data_types = pd.DataFrame(data_types)
    data_types.columns = ['type', 'dim', 'AIC_difference', 'skewness']
    data_types.index = data.columns
    return data_types

def AIC_comparison(input_data):
    '''Calculate Akaike Information Criterion'''
    eps = 1e-8
    data = input_data.dropna()
    # Fit normal & lognormal distributions
    normal_params = stats.norm.fit(data)
    lognormal_params = stats.lognorm.fit(data + eps)
    # AIC of normal
    normal_log_likelihood = np.sum(stats.norm.logpdf(data, *normal_params))
    normal_aic = 2 * len(normal_params) - 2 * normal_log_likelihood
    # AIC of lognormal
    lognormal_log_likelihood = np.sum(stats.lognorm.logpdf(data, *lognormal_params))
    lognormal_aic = 2 * len(lognormal_params) - 2 * lognormal_log_likelihood
    best_fit = "norm" if normal_aic < lognormal_aic else "lognorm"
    difference = abs(normal_aic - lognormal_aic)
    return best_fit, difference