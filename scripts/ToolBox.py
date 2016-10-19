# -*- coding: utf-8 -*-
import numpy as np
from costs import *
import helpers



def compute_logistic_gradient(y,tx,w):
    f=-1*(tx.dot(w))
    logistic=1/(1+np.exp(f))
    error=logistic-y
    gradient=-(1/len(y))*(np.transpose(tx)).dot(error)
    return gradient
                
def logistic_gradient_descent(y, tx,initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    #define w_initial here
    ws = [initial_w]
    #losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient=compute_logistic_gradient(y,tx,w)
        w=w-(gamma*gradient)
        ws.append(np.copy(w))
        #losses.append(loss)
    return ws
    

def compute_gradient(y, tx, w):
    """Compute the gradient."""
    gradient=-(1/len(y))*((np.transpose(tx)).dot(y-(tx.dot(w))))
    return gradient
    
def gradient_descent(y, tx,initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    #define w_initial here
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient=compute_gradient(y,tx,w)
        loss=compute_loss_MSE(y, tx, w)
        w=w-(gamma*gradient)
        ws.append(np.copy(w))
        losses.append(loss)
        # print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1},w2={w2}, w3={w3},w4={w4}, w5={w5},w6={w6}, w7={w7},w8={w8}, w9={w9},w10={w10}, w11={w11},w12={w12}, w13={w13},w14={w14}, w15={w15},w16={w16}, w17={w17},w18={w18}, w19={w19},w20={w20}, w21={w21},w22={w22}, w23={w23},w24={w24}, w25={w25},w26={w26}, w27={w27},w28={w28}, w29={w29},w30={w30}, w31={w31},w32={w32} \n".format(bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1], w2=w[2], w3=w[3],w4=w[4], w5=w[5],w6=w[6], w7=w[7],w8=w[8], w9=w[9],w10=w[10], w11=w[11],w12=w[12], w13=w[13],w14=w[14], w15=w[15],w16=w[16], w17=w[17],w18=w[18], w19=w[19],w20=w[20], w21=w[21],w22=w[22], w23=w[23],w24=w[24], w25=w[25],w26=w[26], w27=w[27],w28=w[28], w29=w[29],w30=w[30], w31=w[31],w32=w[32]))
#output minimum loss w values
    print(losses[max_iters-1])
    return losses, ws

def stochastic_gradient_descent(y, tx,batch_size, max_epochs, gamma):
    """Stochastic gradient descent algorithm."""
    # ***************************************************
    # Internally define initial_w
    ws = [initial_w]
    losses = []
    w = initial_w
    i = 0
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        i = i + 1
        if (i>max_epochs):
            return losses, ws
        gradient=compute_gradient(minibatch_y,minibatch_tx, w)
        loss=compute_loss_MSE(y, tx, w)
        w=w-(gamma*gradient)
        ws.append(np.copy(w))
        losses.append(loss)
        
def compute_subgradient(y, tx, w):
    ones_vector=np.ones(len(y))
    error=y-(tx.dot(w))
    for i in range(len(error)):
        if (error[i]<0):
            ones_vector[i]=-1  
        if (error[i]==0):
            print("encountered nondifferentiable point")
    
    subgradient=(1/len(y))*(-1*(np.transpose(tx)).dot(ones_vector))

    return subgradient

def subgradient_descent(y, tx, initial_w, max_iters, gamma): 
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient=compute_subgradient(y,tx,w)
        loss=compute_loss_MAE(y, tx, w)
        w=w-(gamma*gradient)
        ws.append(np.copy(w))
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
    return losses, ws
        
        
def least_squares(y, tx):
    phitphi=np.transpose(tx).dot(tx)
    phity=np.transpose(tx).dot(y)
    return np.linalg.solve(phitphi,phity)

def ridge_regression(y, tx, lamb):
    xtx=np.transpose(tx).dot(tx)
    lambdaprime=2*lamb*len(y)
    lambdaprime_identity=lambdaprime*np.identity(tx.shape[1])
    A=np.transpose(xtx+lambdaprime_identity)
    B=np.transpose(y).dot(tx)
    w=np.linalg.solve(A,B)
    return w



def split_data(x, y, ratio, seed=1):
    
    # set seed
    np.random.seed(seed)
    # ***************************************************
    shuffle_indices =np.random.permutation(len(x))
    shuffled_y = y[shuffle_indices]
    shuffled_x = x[shuffle_indices]
    train_end_index=int(np.ceil(len(x)*ratio))
    test_start_index=train_end_index+1
    train_data_x=shuffled_x[0:train_end_index]
    train_data_y=shuffled_y[0:train_end_index]
    test_data_x= shuffled_x[test_start_index:len(x)]
    test_data_y= shuffled_y[test_start_index:len(x)]

    return train_data_x,train_data_y,test_data_x,test_data_y