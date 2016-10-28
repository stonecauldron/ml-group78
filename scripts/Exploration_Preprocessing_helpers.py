###This script contains all the helper functions used for data exploration and preprocessing



import numpy as np
import matplotlib.pyplot as plt


############################################### Data Exploration Functions ##########################################33
def feature_hist(X,col_x, bins=50, title=''):
    """Displays the data distribution """
    plt.hist(col_x, bins=bins)
    plt.title(title)
    plt.ylim([0,  X.shape[0]])
    plt.grid(True)
    
def feature_plot(X):
    """Displays the data values """
    nb_cols_subplot = 2
    nb_rows_subplot = np.ceil(X.shape[1]/nb_cols_subplot)
    plt.figure(figsize=(10,50))
    for i in range(X.shape[1]):
        plt.subplot(nb_rows_subplot, nb_cols_subplot, i + 1)
        feature_hist(X,X[:,i], bins=50, title='feature ' + str(i + 1))
plt.show()



def observe_feature_classification(X,y):
    """Plots the classification versus each feature """
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
    """Plots the number of -999 values within each feature"""
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
    plt.savefig("invalid_Values")
plt.show()

def plot_number_zeros(X):
    """Plots the number of -999 values within each feature"""
    invalid_values=np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        invalid_values[i] = nb_invalid_data(X[:,i],0)  
    plt.bar(range(30),invalid_values, width=0.5)
    plt.ylim([0, X.shape[0]])
    plt.title("Number of invalid values")
    plt.xlabel("feature index")
    plt.ylabel("total number")
    plt.minorticks_on()
    plt.grid(True)
plt.show()

def plot_number_particle(y,col):
    """Plots the number of -999 values within each feature"""
    number=np.zeros(4)
    count=np.zeros(4)
    number,count=nb_particles(y,col,4)
    a=np.linspace(0,3,4)
    plt.ylim([0, len(y)/5])
    plt.xlim([0,col.max()+1])
    plt.xlabel("Discrete_value")
    plt.ylabel("total number of particles")
    plt.minorticks_on()
    plt.grid(True)
    plt.bar(a, number, width=0.25,color='blue')
    plt.show()
    plt.ylim([0, len(y)+10])
    plt.xlabel("Discrete_value")
    plt.ylabel("total number of rows")
    plt.bar(a, count, width=0.25, color='red')
    plt.show()
    return number,count
   


def nb_invalid_data(col_x, invalid_value=-999):
    """Helper function to determine the number of invalid values for each feature"""
    return col_x[col_x == invalid_value].shape[0]

def nb_particles(y,col_x,discrete_vals):
    """Helper function to determine the number of particles for each discrete value"""
    count=np.zeros(int(discrete_vals))
    vec=np.zeros(int(discrete_vals))
    for i in range (len(y)):
        count[int(col_x[i])]=count[int(col_x[i])]+1
        if y[i]==1:
            vec[int(col_x[i])]=vec[int(col_x[i])]+1
    return vec,count
    

def compute_correlation(X,y):
    """Computes the Pearson product-moment correlation coefficients and returns the only the required coefficient"""
    corr=np.zeros(X.shape[1])
    for i in (range(X.shape[1])):
        corr[i]=np.corrcoef(X[::,i],y)[0,1]
    return corr

def vector_correlation(x,y):
    corr=np.corrcoef(x,y)[0,1]
    return corr

def correlation_plot(Xin,y,title=''):
    """Plots the correlation of a vector y with each column of a matrix X"""
    V=compute_correlation(Xin,y)
    plt.figure(figsize=(10,10))
    plt.title(title)
    plt.yticks(np.linspace(-1,1,20))
    plt.xticks(np.linspace(0,Xin.shape[1],Xin.shape[1]+1))
    plt.xlim(0,Xin.shape[1]+1)
    plt.ylim(-1,1)
    plt.minorticks_on()
    plt.grid(True)
    plt.xlabel("feature index")
    plt.ylabel("Correlation_coefficient")
    X=np.linspace(0,Xin.shape[1],Xin.shape[1])
    Y1=V
    plt.bar(X,Y1, facecolor='#ff9999', edgecolor='white')
plt.show()

def determine_uncorrelated_features(X,y,threshold=0.005):
    """Returns the column index of the uncorrelated features within a specified threshold"""
    V=compute_correlation(X,y)
    uncorrelated=(abs(V[::])< threshold)
    uncorrelated_features=np.where(uncorrelated==True)
    return uncorrelated_features

def remove_uncorrelated_features(X,y,threshold=0.005):
    """Calls determine_uncorrelated_features to determine and eliminate the uncorrelated columns"""
    delete=determine_uncorrelated_features(X,y)
    New_X=np.delete(X,delete[0],1)
    return New_X

####################################################### Data Pre-Processing Functions ##############################################

""" 1. Column Based Operations"""

def clean_up_invalid_values_mean(col_x,invalid_value=-999):
    """Replaces specified invalid values by the mean"""
    mean = col_x[col_x != -999].mean()
    col_x[col_x == -999] = mean
    return col_x

def clean_up_invalid_values_mean_special(col_x):
    """Replaces specified invalid values by the mean"""
    col=col_x[col_x != -999]
    col=col[col != 0]
    mean = col.mean()
    col_x[col_x == -999] 
    col_x[col_x==0] = mean
    return col_x

def eliminate_outliers(col_x,thershold):
    mean= col_x[col_x <= thershold].mean()
    col_x[col_x > thershold] = mean
    return col_x
    

def clean_up_invalid_values_median(col_x, invalid_value=-999):
    """Replaces specified invalid values by the median"""
    col=col_x[col_x != -999]
    median=np.median(col)
    col_x[col_x == -999] = median
    return col_x
    
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
    positive_indices = col > 0
    result[positive_indices] = 1
    result[~positive_indices] =-1
    return result

def binarize_magnitude(col,threshold=2):
    """Binarizes a column based on whether the magnitude is greater than or less than a threshold value"""
    result = np.zeros((col.shape))
    pos=abs(col)>=threshold
    result[pos]=1
    result[~pos]=-1
    return result
   

def feature21(col):
    result = np.zeros((col.shape))
    t=abs(col)
    median=np.median(t)
    pos=abs(col)>=median
    result[pos]=1
    result[~pos]=0
    return result

def feature8(col):
    result = np.zeros((col.shape))
    pos=abs(col)>=120
    result[pos]=0
    result[~pos]=1
    return result

def feature26(col_x):
    result = np.zeros((col_x.shape))
    median=np.median(col_x[col_x !=-999])
    pos=(abs(col_x)<median)
    result[pos]=0
    result[~pos]=1
    return result

def feature11(col): 
    result = np.zeros((col.shape))
    t=abs(col)
    median=np.median(t)
    pos=abs(col)>=median
    result[pos]=1
    result[~pos]=0
    return result
    
    
    
def feature_scaling(col, min_range=0, max_range=1):
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
        discrete_col[~discrete_indices] = -1
    
        discrete_cols.append(discrete_col)
    return discrete_cols

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi=np.matrix
    phi=np.zeros((len(x), degree))
    for i in range (0, degree):
        phi[:,i]=x**(i+1)
    return phi



""" 2. Simplified Matrix Based Operations"""

def mean_replace(X):
    """Replaces all invalid values in a mtarix by the column's mean value"""
    X0=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        X0[:,i]=clean_up_invalid_values_mean(X[:,i])
    return X0

def median_replace(X):
    """Replaces all invalid values in a mtarix by the column's median value"""
    X0=np.zeros((X.shape[0],X.shape[1]))
    for i in range(X.shape[1]):
        X0[:,i]=clean_up_invalid_values_median(X[:,i])
    return X0

def change_discrete(tX,col):
    """Replaces a single discrete column by multiple columns each reresenting some feature"""
    vector=tX[::,col]
    X=np.delete(tX,col,1)
    mat=np.zeros((tX.shape[0],int(vector.max() + 1)))
    for i in (range(len(vector))):
        j=int(vector[i])
        mat[i,j]=1  
    X=np.concatenate((X,mat),1)
    print(X.shape)
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

def normalize_data(X):
    """Normalizes the data between 0 and 1"""
    maximum=np.max(X,axis=0)
    minimum=np.min(X,axis=0)
    return (X-minimum)/(maximum-minimum)
   

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
            
        elif d==15 or d==18:
            col=binarize_magnitude(col)
    
    # scale uniform features to the -1, 1 range
        elif d == 14 or d == 17 or d == 20:
            col = feature_scaling(col)
            preprocessed_tX.append(col)
    
    # standardize the feature that have an exponential family distribution
        elif d == 1 or d == 2 or d == 3 or d == 7 or d == 8 or d == 9 or d == 10 or d == 11 or d == 13 or d == 16 or d == 19 or d == 21 or d == 29:
            col = standardize(col)
            col=feature_scaling(col)
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

def nadeen_preprocess(tX):
    
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
    
    
    #col = binarize_invalid_data()
    #preprocessed_tX.append(col)
    
    
    return  np.asarray(preprocessed_tX).T
    
def calculate_accuracy(y,tx,w):
    count=0;
    y_pred=tx.dot(w)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    for i in range(len(y)):
        if y[i]==y_pred[i]:
            count=count+1
    accuracy=count/len(y)
    return accuracy    
    