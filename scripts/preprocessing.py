# -*- coding: utf-8 -*-
"""some functions to preprocess the data"""
import numpy as np



"""0. Preprocessing Method Used in run.py"""
##Separates the total set into three groups based on the jet number [0,1,2&3]
##Returns the processed groups and the indices they correspond to in order to match the indices in the classification vector
def separate_data(X, jet_col_nb=22):
    """Separate data based on the number of jets"""
    
    # feature scale uniform distributions
    uniform_cols = [14, 15, 17, 18, 20]
    for col in uniform_cols:
        pass
        #X[:,col] = feature_scaling(X[:,col])
        
    # binary features
    bin_cols = [11]
    for col in bin_cols:
        pass
        #X[:,col] = binarize_positive_negative(X[:,col])
        
    # apply log to exponential distributions
    exponential_cols = [13]
    for col in exponential_cols:
        pass
        #X[:,col] = np.log(X[:,col])
    
    # find the indices of each discrete category
    invalid_cols_0 = [4, 5, 6, 9, 12, 22, 23, 24, 25, 26, 27, 28, 29]
    invalid_cols_1 = [4, 5, 6, 9, 12, 22, 26, 27, 28]
    invalid_cols_2 = [9,22]
    #invalid_cols_3 = [22]
    
    indices_0 = X[:,jet_col_nb] == 0
    indices_1 = X[:,jet_col_nb] == 1
    indices_2 = X[:,jet_col_nb] == 2
    indices_3 = X[:,jet_col_nb] == 3
    
    X_0 = X[indices_0]
    X_1 = X[indices_1]
    X_2 = X[np.logical_or(indices_2, indices_3)]
    #X_3 = X[indices_3]
     
    X_0 = np.delete(X_0, invalid_cols_0, axis=1)
    X_1 = np.delete(X_1, invalid_cols_1, axis=1)
    X_2 = np.delete(X_2, invalid_cols_2, axis=1)
    #X_3 = np.delete(X_3, invalid_cols_3, axis=1)
    
    X_0 = median_replace(X_0)
    X_1 = median_replace(X_1)
    X_2 = median_replace(X_2)
    
    X_1= feature_scaling_matrix(X_1, min_range=-1, max_range=1)
    X_2= feature_scaling_matrix(X_2, min_range=-1, max_range=1)
    
    X_0 = standardize(X_0)
    X_1 = standardize(X_1)
    X_2 = standardize(X_2)
    #X_3 = standardize(X_3)
    
    return X_0, X_1, X_2, indices_0, indices_1, np.logical_or(indices_2, indices_3)


##Pre-processing functions tried

""" 1. Column Based Operations"""   

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
    

def binarize_magnitude(col,threshold=2):
    """Binarizes a column based on whether the magnitude is greater than or less than a threshold value"""
    result = np.zeros((col.shape))
    pos=abs(col)>=threshold
    result[pos]=1
    result[~pos]=-1
    return result

def eliminate_outliers(col_x,thershold):
    mean= col_x[col_x <= thershold].mean()
    col_x[col_x > thershold] = mean
    return col_x

def clean_up_invalid_values_mean(col_x, invalid_value=-999):
    col_x
    mean = col_x[col_x != -999].mean()
    col_x[col_x == -999] = mean
    return col_x

def clean_up_invalid_values_median(col_x, invalid_value=-999):
    col_x
    median = np.median(col_x[col_x != -999])
    col_x[col_x == -999] = median
    return col_x

def clean_up_invalid_values_mean_special(col_x):
    """Replaces specified invalid values by the mean"""
    col=col_x[col_x != -999]
    col=col[col != 0]
    mean = col.mean()
    col_x[col_x == -999] 
    col_x[col_x==0] = mean
    return col_x
    
def feature_scaling(col, min_range=-1, max_range=1):
    """Scale the values of a feature to the range min_range to max_range"""
    min_val = np.min(col)
    max_val = np.max(col)
    
    if max_val - min_val == 0:
        return col
    
    return min_range + ((col - min_val) * (max_range - min_range))/(max_val - min_val)
    

def standardize_col(col):
    """Standardize a feature vector"""
    return (col - np.mean(col)) / np.std(col)

""" 2. Simplified Matrix Based Operations"""
def standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]
    tx = np.hstack((np.ones((x.shape[0],1)), x))
    return tx
    
def mean_replace(X):
    X0=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        X0[:,i]=clean_up_invalid_values_mean(X[:,i])
    return X0

def median_replace(X):
    X0=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        X0[:,i]=clean_up_invalid_values_median(X[:,i])
    return X0

def feature_scaling_matrix(X, min_range=-1, max_range=1):
    """Scale the values of a matrix to the range min_range to max_range"""
    for col in range(X.shape[1]):
        X[:,col] = feature_scaling(X[:,col])
    return X

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
    

"""3-Correlation"""

##Inputs:   Matrix X, Vector y, and threshold value below which vectors are deeemed to be uncorrelated
##Returns:  uncorrelated_features: Vector whose entries correspond to the columns of X whose correlation with y exceeds the specified threshold 
def determine_uncorrelated_features(X,y,threshold=0.005):
    """Returns the column index of the uncorrelated features within a specified threshold"""
    V=compute_correlation(X,y)
    uncorrelated=(abs(V[::])< threshold)
    uncorrelated_features=np.where(uncorrelated==True)
    return uncorrelated_features
    
##Inputs  Matrix X, Vector y, and threshold value below which vectors are deeemed to be uncorrelated
##Returns New_X: Updated Matrix without uncorrelated Columns
def remove_uncorrelated_features(X,y,threshold=0.005):
    """Calls determine_uncorrelated_features to determine and eliminate the uncorrelated columns"""
    delete=determine_uncorrelated_features(X,y)
    New_X=np.delete(X,delete[0],1)
    return New_X    
    
    
 
"""4- Group Based Preprocessing""" 
    
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
        elif d == 1 or d == 2 or d == 3 or d == 7 or d == 8 or d == 10 or d == 11 or d == 13 or d == 16 or d == 19 or d == 21:
            col = standardize_col(col)
            preprocessed_tX.append(col)

        else:
            preprocessed_tX.append(col)
    
    result = np.asarray(preprocessed_tX).T
    return np.hstack((np.ones((result.shape[0],1)), result))
 
def special_preprocess(tX):
    
    (N, D) = tX.shape
    preprocessed_tX = []
   
    for d in range(D):
        col = tX[:,d]
     #binarize columns 0,4,25,27
        if d==0 or d==4 or d==25 or d==27:
        
            col = binarize_invalid_data(col)
            preprocessed_tX.append(col)
        
    # binarize feature 11
        #elif d == 11:
            #col = binarize_positive_negative(col)
            #preprocessed_tX.append(col)  
            
        elif d==15:
            col=binarize_magnitude(col)
            col=feature_scaling(col)
            col= standardize(col)
            preprocessed_tX.append(col)
    
    # convert discrete feature into multiple binary features
        elif d == 22:
            cols = generate_binary_features(col)
            for col in cols:
                col=feature_scaling(col)
                col= standardize(col)
                preprocessed_tX.append(col)
    
    #eliminate column 9 since highly correlated with 13,16,19,21,23,26,29, and eliminate columns 14 and 18       
        #elif d==9 or d==14 or d==18:
        elif d==9: 
            pass
        
        elif d==17:
            col=abs(col)
            col=feature_scaling(col)
            col= standardize(col)
            preprocessed_tX.append(col) 
        
        else:
            col=clean_up_invalid_values_mean(col)
            col=feature_scaling(col)
            col= standardize(col)           
            preprocessed_tX.append(col) 
    
    return  np.asarray(preprocessed_tX).T
    



