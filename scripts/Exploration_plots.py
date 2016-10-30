
###This script contains all the functions used to generate data exploration plots
import numpy as np
import matplotlib.pyplot as plt






"Plots the histogram for each feature"
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
        feature_hist(X,X[:,i], bins=50, title='feature ' + str(i))
    plt.savefig("Figures/Features_Plot")    
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
    plt.bar(range(30),invalid_values, width=0.5,color='red')
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
   
def correlation_plot_all(Xin,title=''):
    """Plots the correlation of a vector y with each column of a matrix X"""
    nb_cols_subplot = 2
    nb_rows_subplot = np.ceil(Xin.shape[1]/nb_cols_subplot)
    plt.figure(figsize=(20,100))
    for i in range(Xin.shape[1]):
        plt.subplot(nb_rows_subplot, nb_cols_subplot, i + 1)
        V=compute_correlation(Xin,Xin[:,i])
        plt.yticks(np.linspace(-1,1,20))
        plt.xticks(np.linspace(0,Xin.shape[1],Xin.shape[1]+1))
        plt.xlim(0,Xin.shape[1]+1)
        plt.ylim(-1,1)
        plt.minorticks_on()
        #plt.grid(True)
        plt.xlabel("Feature index")
        plt.ylabel("Correlation_coefficients_Feature"+str(i))
        X=np.linspace(0,Xin.shape[1],Xin.shape[1])
        Y1=V
        plt.bar(X,Y1, facecolor='#ff9999', edgecolor='white')
plt.show()

#######################################################Plot Helpers###################################

#Returns :       vec: Vector with each entry corresponding to the number of Higgs Boson Particle for a jet number group
#                count: Vector that counts the number of samples for each jet number
def nb_particles(y,col_x,discrete_vals):
    """Helper function to determine the number of particles for each discrete value"""
    count=np.zeros(int(discrete_vals))
    vec=np.zeros(int(discrete_vals))
    for i in range (len(y)):
        count[int(col_x[i])]=count[int(col_x[i])]+1
        if y[i]==1:
            vec[int(col_x[i])]=vec[int(col_x[i])]+1
    return vec,count
    
##Inputs:       Matrix X, Vector y
##Returns:      Corr: Vector Representing the Pearson Correlation between each column in X and vector y
def compute_correlation(X,y):
    """Computes the Pearson product-moment correlation coefficients and returns the only the required coefficient"""
    corr=np.zeros(X.shape[1])
    for i in (range(X.shape[1])):
        corr[i]=np.corrcoef(X[::,i],y)[0,1]
    return corr

##Inputs:       Vectors x,y_pred
##Returns       Corr: Scalar value representing the Pearson Correlation between the two vectors
def vector_correlation(x,y):
    corr=np.corrcoef(x,y)[0,1]
    return corr
  

   
    


 
    