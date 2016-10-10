# -*- coding: utf-8 -*-
import numpy as np
import costs
import helpers


def compute_gradient(y, tx, w):
    """Compute the gradient."""
    gradient=-(1/len(y))*((np.transpose(tx)).dot(y-(tx.dot(w))))
    return gradient
    
def gradient_descent(y, tx, initial_w, max_iters, gamma):
    """Gradient descent algorithm."""
   # Define parameters to store w and loss
    #define w_initial here
    ws = [w_initial]
    losses = []
    for n_iter in range(max_iters):
        gradient=compute_gradient(y,tx,w)
        loss=compute_loss_MSE(y, tx, w)
        w=w-(gamma*gradient)
        ws.append(np.copy(w))
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))
#output minimum loss w values
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
        
def least_squares(y, tx):
    """calculate the least squares solution."""
    #w=np.linalg.solve(tx,y)
    xtxinverse=np.linalg.inv(np.transpose(tx).dot(tx))
    xtransposey=np.transpose(tx).dot(y)
    w=xtxinverse.dot(xtransposey)
    mse=compute_loss_MSE(y,tx,w)
    return w
    # ***************************************************

def ridge_regression(y, tx, lamb):
    """implement ridge regression."""
    xtx=np.transpose(tx).dot(tx)
    lambdaprime=2*lamb*len(tx)
    lambdaprime_identity=lambdaprime*np.identity(tx.shape[1])
    first_term=xtx+lambdaprime_identity
    first_term_inv=np.linalg.inv(first_term)
    second_term=np.transpose(tx).dot(y)
    w=first_term_inv.dot(second_term)
    return w
