import numpy as np

# -*- coding: utf-8 -*-
""" ML-methods to implement """
########################################################################################################################################
"""This script contains the various machine learning algorithms used to predict an output given a training set"""

#The included algorithms and their main helper and cost functions are:
# 1- Least Squares with Gradient Descent:least_squares_GD(y, tx, gamma, max iters)
    # a.calculate_gradient_mse(y, tx, w): Computes the gradient required to perform gradient_descent
    # b.calculate_loss_mse(y,tx,w): Returns the mean square error obtained for the given y,tx,w
    
# 2- Least Squares with Stochastic Gradient Descent: least_squares_SGD(y, tx, gamma, max_iters):
    #a. batch_iter(y, tx, batch_size, num_batches=None, shuffle=True): Generate a minibatch iterator for a dataset.
    #b.calculate_gradient_mse(y, tx, w): Computes the gradient required to perform gradient_descent
    #c.calculate_loss_mse(y,tx,w): Returns the mean square error obtained for the given y,tx,w
    
# 3- Least Squares:least_squares(y, tx)

# 4- Ridge Regression: ridge_regression(y, tx, lambda_): Ridge regression using normal equations""
    
# 5- Logistic Regression:logistic_regression(y, tx, gamma, max_iters)
    #a. calculate_gradient_log_likelihood(y, tx, w): Determines the gradient of the negative log likelihood function
    #b. sigmoid(t): Applies sigmoid function on t while taking into consideration boundary conditions
    #c. calculate_loss_log_likelihood(y, tx, w): Returns the cost value for the given y,tx and w.
    
# 6- Regularized Logistic Regression:reg_logistic_regression(y, tx, lambda_, gamma, max_iters)
    #a. calculate_gradient_log_likelihood(y, tx, w): Determines the gradient of the negative log likelihood function
    #b. sigmoid(t): Applies sigmoid function on t while taking into consideration boundary conditions
    #c. calculate_loss_log_likelihood(y, tx, w): Returns the cost value for the given y,tx and w.

#########################################################################################################################

import numpy as np


##Machine Learning Algorithms

##Least Squares gradient descent 
#Inputs: y    vector: (N,) 
#       tx    matrix: (N,D)  
#       gamma     value of stepsize
#       max_iters # iterations

#Output: w     vector:(D,)
#Note: option of breaking sooner available at specified threshold to avoid overfitting

def least_squares_GD(y, tx, gamma, max_iters):
    """Linear regression using gradient descent"""
    # init parameters
    threshold = 1e-8
    losses = []
    w = np.zeros(tx.shape[1])
    # start the regression
    for iter in range(max_iters):
        gradient = calculate_gradient_mse(y, tx, w)
        # get loss and updated w
        loss = calculate_loss_mse(y, tx, w)
        w = w - gamma * gradient
        # log info
        if iter % 1000 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    #print("The  of loss iteration={i} is {l} ".format(l=calculate_loss_mse(y, tx, w), i=iter))
    return w


##Least Squares Stochastic gradient descent 
#Inputs: y    vector: (N,) 
#       tx    matrix: (N,D)  
#       gamma     value of stepsize
#       max_iters # iterations
#batch_size  required batch size

#Output: w     vector:(D,)
#Note: option of breaking sooner available at specified threshold to avoid overfitting

def least_squares_SGD(y, tx, gamma, max_iters,batch_size):
    """Linear regression using stochastic gradient descent"""
    # init parameters
    threshold = 1e-8
    losses = []
    w = np.zeros(tx.shape[1])

    # Perform regression on a data batch of size=100
    for iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx,batch_size):
            gradient = calculate_gradient_mse(batch_y, batch_tx, w)
        # get loss and updated w
        loss = calculate_loss_mse(y, tx, w)
        w = w - gamma * gradient
        # log info
        if iter % 50 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    print("The  of loss iteration={i} is {l} ".format(l=calculate_loss_mse(y, tx, w), i=iter))
    return w

##Least Squares using normalized equations
#Inputs: y    vector: (N,) 
#       tx    matrix: (N,D)
#Output:w     vector: (D,)
#prints the loss for the calculated w
def least_squares(y, tx):
    """calculate the least squares solution."""
    w_star = np.linalg.solve(np.transpose(tx).dot(tx), tx.T.dot(y))
    print("The loss={l}".format(l=calculate_loss_mse(y, tx, w_star)))
    return w_star


##Penalized Least Squares
#Inputs: y    vector: (N,) 
#       tx    matrix: (N,D)
#       lambda_ value of penalty
#Increasing lambda results in smaller weights

#Output:w     vector: (D,)
#prints the loss for the calculated w
def ridge_regression(y, tx, lamb):
    """Ridge regression using normal equations"""
    xtx=np.transpose(tx).dot(tx)
    lambdaprime=2*lamb*len(y)
    lambdaprime_identity=lambdaprime*np.identity(tx.shape[1])
    A=(xtx+lambdaprime_identity)
    B=np.transpose(tx).dot(y)
    w_star=np.linalg.solve(A,B)
    print("The loss={l}".format(l=calculate_loss_mse(y, tx, w_star)))
    return w_star


##Logistic Regression Using gradient descent 
#Inputs: y    vector: (N,) 
#       tx    matrix: (N,D)  
#       gamma     value of stepsize
#       max_iters # iterations

#Output: w     vector:(D,)
#Note: option of breaking sooner available at specified threshold to avoid overfitting

def logistic_regression(y, tx, gamma, max_iters):
    """Logistic regression using gradient descent"""
    # init parameters
    threshold = 1e-8
    losses = []
    w = np.zeros(tx.shape[1])
    # start the logistic regression
    for iter in range(max_iters):
        gradient = calculate_gradient_log_likelihood(y, tx, w)
        # get loss and updated w
        loss = calculate_loss_log_likelihood(y, tx, w)
        w = w - gamma * gradient 
        # log info
        if iter % 50 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    print("The  of loss iteration={i} is {l} ".format(calculate_loss_log_likelihood(y, tx, w), i=iter))
    return w
   

def regularized_logistic_regression(y, tx, lambda_, gamma, max_iters):
    """Regularized logistic regression using gradient descent"""
    # init parameters
    threshold = 1e-8
    losses = []
    w = np.zeros(tx.shape[1])

    # start the logistic regression
    for iter in range(max_iters):
        gradient = calculate_gradient_log_likelihood(y, tx, w) + (lambda_ * 2 * w)
        # get loss and updated w
        loss=calculate_loss_log_likelihood(y, tx, w)+lambda_*np.sum(w**2)
        w = w - gamma * gradient
       
        # log info
        if iter % 500 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=calculate_loss_log_likelihood(y, tx, w)+lambda_*np.sum(w**2)))
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    #print("The loss of iteration={i} is {l} ".format(l=calculate_loss_log_likelihood(y, tx, w)+lambda_*np.sum(w**2), i=iter))
    return w


                                                        ########################################
# LOSS FUNCTIONS

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def calculate_mae(e):
    """Calculate the mae for vector e."""
    return np.mean(np.abs(e))

def calculate_loss_mse(y, tx, w):
    """Calculate the loss using MSE."""
    e = y - tx.dot(w)
    return calculate_mse(e)

def calculate_loss_mae (y, tx, w):
    e = y - tx.dot(w)
    return calculate_mae(e)

def calculate_loss_log_likelihood(y, tx, w):
    """compute the cost by negative log likelihood."""
    first_sigmoid_term = np.log(sigmoid(tx.dot(w)))
    first_term = (y.T).dot(first_sigmoid_term)
    second_sigmoid_term = np.log(1 - sigmoid(tx.dot(w)))
    second_term = ((1 - y).T).dot(second_sigmoid_term)
    pos_loss = first_term + second_term
    return -1 * pos_loss


    
                                                      ########################################
##Gradient calculation helpers

def calculate_gradient_mse(y, tx, w):
    """Compute the gradient of the MSE loss function."""
    N = y.shape[0]
    e = y - (tx.dot(w))
    return (-1/N) * (np.transpose(tx).dot(e))


def sigmoid(t):
    """apply sigmoid function on t."""
    if t>0:
        z=np.exp(-1*t)
        result=1 / (1 + z)
    else: 
        z=np.exp(t)
        result= z / (1 + z)
    return result

def calculate_loss_log_likelihood(y, tx, w):
    a=tx.dot(w)
    loss_vec=np.log(1+np.exp(a))-(y*a)
    return np.sum(loss_vec)/len(loss_vec)

def calculate_gradient_log_likelihood(y, tx, w):
    """compute the gradient of negative log likelihood."""
    a=tx.dot(w)
    for i in range (len(a)):
        a[i]=sigmoid(a[i])
    gradient=np.transpose(tx).dot(a-y)
    return gradient/len(y)
