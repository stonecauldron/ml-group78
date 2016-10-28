import numpy as np
import matplotlib.pyplot as plt
from Final_ToolBox import *
from Exploration_Preprocessing_helpers import *

def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def calculate_accuracy_logistic_regression(y,tx,w):
    """Calculates the accuracy of the prediction"""
    count=0
    a=tx.dot(w)
    for i in range( len(a)):
        a[i]=sigmoid(a[i])
    a[np.where(a < 0.5)] = 0
    a[np.where(a >= 0.5)] = 1
    for i in range(len(y)):
        if y[i]==a[i]:
            count=count+1
    accuracy=count/len(y)
    return accuracy 

def calculate_accuracy_linear_regression(y,tx,w):
    """Calculates the accuracy of the prediction"""
    count=0
    y_pred=tx.dot(w)
    y_pred[np.where(y_pred <= 0)] = -1
    y_pred[np.where(y_pred > 0)] = 1
    for i in range(len(y)):
        if y[i]==y_pred[i]:
            count=count+1
    accuracy=count/len(y)
    return accuracy 

def cross_validation_linear_gradient_descent(y, x, k_indices, k, gamma,max_iters):
    """Perform k fold cross validation on given data set using ridge regression"""
    test_y=y[k_indices[k]]
    test_x=x[k_indices[k]]
    train_y=y
    train_x=x
    train_x=np.delete(train_x,k_indices[k],axis=0)
    train_y=np.delete(train_y,k_indices[k])
    weight=least_squares_GD(y, x, gamma, max_iters)
    loss_tr=calculate_loss_mse(train_y, train_x, weight)
    loss_te=calculate_loss_mse(test_y,test_x, weight)
    accuracy=calculate_accuracy_linear_regression(test_y,test_x,weight)
    return weight,loss_tr, loss_te,accuracy



def cross_validation_ridge_regression(y, x, k_indices, k, lambda_):
    """Perform k fold cross validation on given data set using ridge regression"""
    test_y=y[k_indices[k]]
    test_x=x[k_indices[k]]
    train_y=y
    train_x=x
    train_x=np.delete(train_x,k_indices[k],axis=0)
    train_y=np.delete(train_y,k_indices[k])
    weight=ridge_regression(train_y,train_x, lambda_)
    loss_tr=calculate_loss_mse(train_y, train_x, weight)
    loss_te=calculate_loss_mse(test_y,test_x, weight)
    accuracy=calculate_accuracy_linear_regression(test_y,test_x,weight)
    return weight,loss_tr, loss_te,accuracy

def cross_validation_logistic_descent(y,x, k_indices, k, gamma,max_iters):
    """Perform k fold cross validation on given data set using ridge regression"""
    test_y=y[k_indices[k]]
    test_x=x[k_indices[k]]
    train_y=y
    train_x=x
    train_x=np.delete(train_x,k_indices[k],axis=0)
    train_y=np.delete(train_y,k_indices[k])
    weight=logistic_regression(y, x, gamma, max_iters)
    loss_tr=calculate_loss_log_likelihood(train_y, train_x, weight)
    loss_te=calculate_loss_log_likelihood(test_y,test_x, weight)
    accuracy=calculate_accuracy_logistic_regression(test_y,test_x,weight)
    return weight,loss_tr, loss_te,accuracy

def cross_validation_regularized_logistic_descent(y,x, k_indices, k,lambda_, gamma,max_iters):
    """Perform k fold cross validation on given data set using ridge regression"""
    test_y=y[k_indices[k]]
    test_x=x[k_indices[k]]
    train_y=y
    train_x=x
    train_x=np.delete(train_x,k_indices[k],axis=0)
    train_y=np.delete(train_y,k_indices[k])
    weight=regularized_logistic_regression(y, x, lambda_, gamma, max_iters)
    loss_tr=calculate_loss_log_likelihood(train_y, train_x, weight)
    loss_te=calculate_loss_log_likelihood(test_y,test_x, weight)
    accuracy=calculate_accuracy_logistic_regression(test_y,test_x,weight)
    return weight,loss_tr, loss_te,accuracy


def cross_validation_demo_linear_gradient_descent(y,x,k_fold,gamma,max_iters,seed=250):
    
    k_indices = build_k_indices(y, k_fold, seed)
    loss_tr=np.zeros(k_fold)
    loss_te=np.zeros(k_fold) 
    accuracy=np.zeros(k_fold) 
    weights=[]
    w_initial = np.zeros((x.shape[1]))
        
    for k in range (k_fold):
        w,loss_tr[k],loss_te[k],accuracy[k]=cross_validation_linear_gradient_descent(y, x, k_indices, k, gamma,max_iters)
        weights.append(w)
       
    mse_tr=np.mean(loss_tr)
    mse_te=np.mean(loss_te)
    mse_var_tr=np.var(loss_tr)
    mse_var_te=np.var(loss_te)
    accuracy_means=np.mean(accuracy)
   
    print("MSEtr={r},MSEte={rt}, MSEvartr={mvar}, MSEvarte={mvarte}, Accuracy={acc}".format(r=mse_tr,rt= mse_te,mvar= mse_var_tr , mvarte= mse_var_te, acc=accuracy_means))
    return weights


def cross_validation_demo_ridge_regression(y,x,k_fold,lamb,seed=250):
    
    k_indices = build_k_indices(y, k_fold, seed)

    loss_tr=np.zeros(k_fold)
    loss_te=np.zeros(k_fold) 
    accuracy=np.zeros(k_fold) 
    mse_tr = np.zeros(len(lamb))
    mse_te = np.zeros(len(lamb))
    mse_var_tr=np.zeros(len(lamb))
    mse_var_te=np.zeros(len(lamb))
    accuracy_means=np.zeros(len(lamb))
    weights=[]

    w_initial = np.zeros((x.shape[1]))


    for ind, lam in enumerate(lamb):
        for k in range (k_fold):
            w,loss_tr[k],loss_te[k],accuracy[k]=cross_validation_ridge_regression(y,x, k_indices,k,lam)
            weights.append(w)
       
        mse_tr[ind]=np.mean(loss_tr)
        mse_te[ind]=np.mean(loss_te)
        mse_var_tr[ind]=np.var(loss_tr)
        mse_var_te[ind]=np.var(loss_te)
        accuracy_means[ind]=np.mean(accuracy)
   
    print("MSEtr={r},MSEte={rt}, MSEvartr={mvar}, MSEvarte={mvarte}, Accuracy={acc}".format(r=mse_tr,rt= mse_te,mvar= mse_var_tr , mvarte= mse_var_te, acc=accuracy_means))
    return weights

def cross_validation_demo_logistic_descent(y,x,k_fold,gamma,max_iters,seed=250):
    
    k_indices = build_k_indices(y, k_fold, seed)
    loss_tr=np.zeros(k_fold)
    loss_te=np.zeros(k_fold) 
    accuracy=np.zeros(k_fold) 
    weights=[]
    w_initial = np.zeros((x.shape[1]))
        
    for k in range (k_fold):
        w,loss_tr[k],loss_te[k],accuracy[k]=cross_validation_logistic_descent(y, x, k_indices, k, gamma,max_iters)
        weights.append(w)
       
    mse_tr=np.mean(loss_tr)
    mse_te=np.mean(loss_te)
    mse_var_tr=np.var(loss_tr)
    mse_var_te=np.var(loss_te)
    accuracy_means=np.mean(accuracy)
   
    print("MSEtr={r},MSEte={rt}, MSEvartr={mvar}, MSEvarte={mvarte}, Accuracy={acc}".format(r=mse_tr,rt= mse_te,mvar= mse_var_tr , mvarte= mse_var_te, acc=accuracy_means))
    return weights
         
def cross_validation_demo_reg_logistic_regression(y,x,k_fold,lamb,gamma,max_iters,seed=250):
    
    k_indices = build_k_indices(y, k_fold, seed)

    loss_tr=np.zeros(k_fold)
    loss_te=np.zeros(k_fold) 
    accuracy=np.zeros(k_fold) 
    mse_tr = np.zeros(len(lamb))
    mse_te = np.zeros(len(lamb))
    mse_var_tr=np.zeros(len(lamb))
    mse_var_te=np.zeros(len(lamb))
    accuracy_means=np.zeros(len(lamb))
    weights=[]

    w_initial = np.zeros((x.shape[1]))


    for ind, lam in enumerate(lamb):
        for k in range (k_fold):
            w,loss_tr[k],loss_te[k],accuracy[k]=cross_validation_regularized_logistic_descent(y,x, k_indices, k,lam, gamma,max_iters)
            weights.append(w)
       
        mse_tr[ind]=np.mean(loss_tr)
        mse_te[ind]=np.mean(loss_te)
        mse_var_tr[ind]=np.var(loss_tr)
        mse_var_te[ind]=np.var(loss_te)
        accuracy_means[ind]=np.mean(accuracy)
   
    print("MSEtr={r},MSEte={rt}, MSEvartr={mvar}, MSEvarte={mvarte}, Accuracy={acc}".format(r=mse_tr,rt= mse_te,mvar= mse_var_tr , mvarte= mse_var_te, acc=accuracy_means))
    return weights
