
import numpy as np
import matplotlib.pyplot as plt

def nb_invalid_data(col_x, invalid_value=-999):
    return col_x[col_x == invalid_value].shape[0]
    

def clean_up_invalid_values_mean(col_x, invalid_value=-999):
    col_x
    mean = col_x[col_x != -999].mean()
    col_x[col_x == -999] = mean
    return col_x

def mean_replace(X):
    X0=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        X0[:,i]=clean_up_invalid_values_mean(X[:,i])
    return X0

def clean_up_invalid_values_median(col_x, invalid_value=-999):
    col_x
    col=col_x[col_x != -999]
    median=np.median(col)
    col_x[col_x == -999] = median
    return col_x

def median_replace(X):
    X0=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        X0[:,i]=clean_up_invalid_values_median(X[:,i])
    return X0


def feature_hist(X,col_x, bins=50, title=''):
    plt.hist(col_x, bins=bins)
    plt.title(title)
    plt.ylim([0,  X.shape[0]])
    plt.grid(True)
    
def feature_plot(X):
    nb_cols_subplot = 2
    nb_rows_subplot = np.ceil(X.shape[1]/nb_cols_subplot)
    plt.figure(figsize=(10,50))
    for i in range(X.shape[1]):
        plt.subplot(nb_rows_subplot, nb_cols_subplot, i + 1)
        feature_hist(X,X[:,i], bins=50, title='feature ' + str(i + 1))
plt.show()

def observe_feature_classification(X,y):
    nb_cols_subplot = 2
    nb_rows_subplot = np.ceil(X.shape[1] /nb_cols_subplot)
    plt.figure(figsize=(10,50))
    for i in range(X.shape[1]):
        plt.subplot(nb_rows_subplot, nb_cols_subplot, i + 1)
        plt.ylim(-1.5,1.5)
        plt.title('feature ' + str(i + 1))
        plt.plot(X[0:250000:1,i],y[0:250000:1],'*b')
plt.show()


def plot_number_invalid_values(X):
    invalid_values=np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        invalid_values[i] = nb_invalid_data(X[:,i])  

    plt.bar(range(30),invalid_values, width=0.5)
    plt.ylim([0, X.shape[0]])
    plt.title("Number of invalid values")
    plt.xlabel("feature index")
    plt.ylabel("total number")
    plt.minorticks_on()
    plt.grid(True)
plt.show()




###############Standardizations####################
#col=22
def change_discrete(tX,col):
    vector=tX[::,col]
    X=np.delete(tX,col,1)
    mat=np.zeros((tX.shape[0],int(vector.max() + 1)))
    for i in (range(len(vector))):
        j=int(vector[i])
        mat[i,j]=1  
    X=np.concatenate((X,mat),1)
    return X

def regular_standardize(x, mean_x=None, std_x=None):
    """Standardize the original data set."""
    if mean_x is None:
        mean_x = np.mean(x, axis=0)
    x = x - mean_x
    if std_x is None:
        std_x = np.std(x, axis=0)
    x[:, std_x>0] = x[:, std_x>0] / std_x[std_x>0]
    
    return x

def compute_correlation(X,y):
    corr=np.zeros(X.shape[1])
    for i in (range(X.shape[1])):
        corr[i]=np.corrcoef(X[::,i],y)[0,1]
    return corr


def normalize_data(X):
    maximum=np.max(X,axis=0)
    minimum=np.min(X,axis=0)
    return (X-minimum)/(maximum-minimum)
    
    
    
def correlation_plot(Xin,y,title=''):
    V=compute_correlation(Xin,y)
    plt.figure(figsize=(10,10))
    plt.title(title)
    plt.yticks(np.linspace(-0.7,0.7,14))
    plt.xticks(np.linspace(0,Xin.shape[1],Xin.shape[1]+1))
    plt.xlim(0,Xin.shape[1]+1)
    plt.ylim(-0.7,0.7)
    plt.minorticks_on()
    plt.grid(True)
    plt.xlabel("feature index")
    plt.ylabel("Correlation_coefficient")
    X=np.linspace(0,Xin.shape[1],Xin.shape[1])
    Y1=V
    plt.bar(X,Y1, facecolor='#ff9999', edgecolor='white')
plt.show()

def determine_uncorrelated_features(X,y,threshold=0.005):
    V=compute_correlation(X,y)
    uncorrelated=(abs(V[::])< threshold)
    uncorrelated_features=np.where(uncorrelated==True)
    return uncorrelated_features

def remove_uncorrelated_features(X,y,threshold=0.005):
    delete=determine_uncorrelated_features(X,y)
    New_X=np.delete(X,delete[0],1)
    return New_X

def remove_columns(X,V):
     return np.delete(X,V,1)


  
    
    #####################   PEDRO ##############
def binarize_invalid_data(col, invalid_value=-999):
    """Transform features with -999 values into a binary categorisation"""
    result = np.zeros((col.shape))
    
    invalid_value_indices = col == invalid_value
    result[invalid_value_indices] = 0
    result[~invalid_value_indices] = 1
    return result

def binarize_positive_negative(col):
    """Binary categorisation of negative and positive values"""
    result = np.zeros((col.shape))
    
    positive_indices = col >= 0
    result[positive_indices] = 1
    result[~positive_indices] = 0
    return result

def feature_scaling(col, min_range=-1, max_range=1):
    """Scale the values of a feature to the range min_range to max_range"""
    min_val = np.min(col)
    max_val = np.max(col)
    
    return min_range + ((col - min_val) * (max_range - min_range))/(max_val - min_val)

def standardize(col):
    """Standardize a feature vector"""
    return (col - np.mean(col)) / np.std(col)


def generate_binary_features(col):
    """Generate binary features from a discrete feature"""
    discrete_cols = []
    
    for i in range(int(col.max() + 1)):
        discrete_indices = col == i
    
        discrete_col = np.zeros(col.shape)
        discrete_col[discrete_indices] = 1
        discrete_col[~discrete_indices] = 0
    
        discrete_cols.append(discrete_col)
    return discrete_cols
    
    
def pedro_preprocess(tX):
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
            col = standardize(col)
            preprocessed_tX.append(col)
    
    # convert discrete feature into multiple binary features
        elif d == 22:
            cols = generate_binary_features(col)
            for col in cols:
                preprocessed_tX.append(col)
            
        else:
            preprocessed_tX.append(col)
            np.asarray(preprocessed_tX).T.shape   
        
    return  np.asarray(preprocessed_tX).T
    
    
    