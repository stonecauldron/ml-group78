# -*- coding: utf-8 -*-
"""some functions to preprocess the data"""
import numpy as np

def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]
    
    return x

def binarize_invalid_data(col, invalid_value=-999):
    """Transform features with -999 values into a binary categorisation"""
    result = np.zeros((col.shape))
    
    invalid_value_indices = col == invalid_value
    result[invalid_value_indices] = -1
    result[~invalid_value_indices] = 1
    return result

def binarize_positive_negative(col):
    """Binary categorisation of negative and positive values"""
    result = np.zeros((col.shape))
    
    positive_indices = col >= 0
    result[positive_indices] = 1
    result[~positive_indices] = -1
    return result

def feature_scaling(col, min_range=-1, max_range=1):
    """Scale the values of a feature to the range min_range to max_range"""
    min_val = np.min(col)
    max_val = np.max(col)
    
    return min_range + ((col - min_val) * (max_range - min_range))/(max_val - min_val)

def standardize_col(col):
    """Standardize a feature vector"""
    return (col - np.mean(col)) / np.std(col)

def generate_binary_features(col):
    """Generate binary features from a discrete feature"""
    discrete_cols = []
    
    for i in range(int(col.max() + 1)):
        discrete_indices = col == i
    
        discrete_col = np.zeros(col.shape)
        discrete_col[discrete_indices] = 1
        discrete_col[~discrete_indices] = -1
    
        discrete_cols.append(discrete_col)
    return discrete_cols

def preprocess_inputs(tX):
    """Preprocess the input data"""
    (N, D) = tX.shape
    preprocessed_tX = []

    for d in range(D):
        col = tX[:,d]

        # binarize invalid values
        if d == 0 or d == 4 or d == 5 or d == 6 or d == 12 or d == 23 or d == 24 or d == 25 or d == 26 or d == 27 or d == 28:
            col = binarize_invalid_data(col)
            preprocessed_tX.append(col)

        # binarize feature 11
        elif d == 11:
            col = binarize_positive_negative(col)
            preprocessed_tX.append(col)

        # scale uniform features to the -1, 1 range
        elif d == 14 or d == 15 or d == 17 or d == 18 or d == 20:
            col = feature_scaling(col)
            preprocessed_tX.append(col)

        # standardize the feature that have an exponential family distribution
        elif d == 1 or d == 2 or d == 3 or d == 7 or d == 8 or d == 9 or d == 10 or d == 11 or d == 13 or d == 16 or d == 19 or d == 21 or d == 29:
            col = standardize_col(col)
            col = feature_scaling(col)
            preprocessed_tX.append(col)

        # convert discrete feature into multiple binary features
        elif d == 22:
            cols = generate_binary_features(col)
            for col in cols:
                preprocessed_tX.append(col)

        else:
            preprocessed_tX.append(col)
    
    result = np.asarray(preprocessed_tX).T
    return np.hstack((np.ones((result.shape[0],1)), result))