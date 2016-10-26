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

#######################################################################################################################################

import numpy as np

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
    print("The loss={l}".format(l=calculate_loss_mse(y, tx, w)))
    return w

def least_squares_SGD(y, tx, gamma, max_iters):
    """Linear regression using stochastic gradient descent"""
    # init parameters
    threshold = 1e-8
    losses = []
    w = np.zeros(tx.shape[1])

    # Perform regression on a data batch 
    for iter in range(max_iters):
        for batch_y, batch_tx in batch_iter(y, tx, 100, num_batches=1):
            gradient = calculate_gradient_mse(batch_y, batch_tx, w)
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
    print("The loss={l}".format(l=calculate_loss_mse(y, tx, w)))
    return w

def least_squares(y, tx):
    """calculate the least squares solution."""
    w_star = np.linalg.solve(tx.T @ tx, tx.T @ y)
    print("The loss={l}".format(l=calculate_loss_mse(y, tx, w_star)))
    return w_star

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations"""
    M = tx.shape[1]
    N = y.shape[0]
    lamb_prime = lambda_ * 2 * N
    w_star = np.linalg.solve((tx.T @ tx) + lambda_ * np.identity(M), tx.T @ y)
    print("The loss={l}".format(l=calculate_loss_mse(y, tx, w_star)))
    return w_star

def logistic_regression(y, tx, gamma, max_iters):
    """Logistic regression using gradient descent"""
    # init parameters
    threshold = 1e-5
    losses = []
    w = np.zeros(tx.shape[1])
    # start the logistic regression
    for iter in range(max_iters):
        gradient = calculate_gradient_log_likelihood(y, tx, w)
        
        # get loss and updated w
        loss = calculate_loss_log_likelihood(y, tx, w)
        w = w - gamma * gradient
        
        # log info
        if iter % 100 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=loss))
        # converge criteria
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    print("The loss={l}".format(l=calculate_loss_log_likelihood(y, tx, w)))
    return w

def reg_logistic_regression(y, tx, lambda_, gamma, max_iters):
    """Regularized logistic regression using gradient descent"""
    # init parameters
    threshold = 1e-5
    losses = []

    w = np.zeros(tx.shape[1])

    # start the logistic regression
    for iter in range(max_iters):
        gradient = calculate_gradient_log_likelihood(y, tx, w) + (lambda_ * 2 * w)
        
        # get loss and updated w
        w = w - gamma * gradient
        
        # log info
        if iter % 1000 == 0:
            print("Current iteration={i}, the loss={l}".format(i=iter, l=calculate_loss_log_likelihood(y, tx, w)))
        
    print("The loss={l}".format(l=calculate_loss_log_likelihood(y, tx, w)))
    return w
    
# LOSS FUNCTIONS

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2*np.mean(e**2)

def calculate_loss_mse(y, tx, w):
    """Calculate the loss using MSE."""
    e = y - tx.dot(w)
    return calculate_mse(e)

def calculate_loss_log_likelihood(y, tx, w):
    """compute the cost by negative log likelihood."""
    first_sigmoid_term = np.log(sigmoid(tx.dot(w)))
    first_term = (y.T).dot(first_sigmoid_term)
    second_sigmoid_term = np.log(1 - sigmoid(tx.dot(w)))
    second_term = ((1 - y).T).dot(second_sigmoid_term)
    pos_loss = first_term + second_term
    return -1 * pos_loss

# linear regression helpers
def calculate_gradient_mse(y, tx, w):
    """Compute the gradient of the MSE loss function."""
    N = y.shape[0]
    e = y - (tx @ w)
    return (-1/N) * (tx.T @ e)

# logistic regression helpers
def sigmoid(t):
    """apply sigmoid function on t."""
    t[t > 709] = 709
    return np.exp(t)/(1 + np.exp(t))

def calculate_gradient_log_likelihood(y, tx, w):
    """compute the gradient of negative log likelihood."""
    return tx.T @ (sigmoid(tx @ w) - y)

# stochastic gradient descent helpers
def batch_iter(y, tx, batch_size, num_batches=None, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.
    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)
    num_batches_max = int(np.ceil(data_size/batch_size))
    if num_batches is None:
        num_batches = num_batches_max
    else:
        num_batches = min(num_batches, num_batches_max)

    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_y = y[shuffle_indices]
        shuffled_tx = tx[shuffle_indices]
    else:
        shuffled_y = y
        shuffled_tx = tx
    for batch_num in range(num_batches):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        if start_index != end_index:
            yield shuffled_y[start_index:end_index], shuffled_tx[start_index:end_index]
            
# cross validation
def build_k_indices(y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)
