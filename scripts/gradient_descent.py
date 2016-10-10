# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Gradient Descent
"""
import numpy as np
import costs


def compute_gradient_mse(y, tx, w):
    """Compute the gradient."""
    N = y.shape[0]
    e_vector = error_vector(y, tx, w)
    return (-1/N) * (tx.T @ e_vector)
    
def compute_subgradient_mae(y, tx, w):
    N = y.shape[0]
    e_vector = error_vector(y, tx, w)
    e_vector[e_vector == 0] = -1
    return (1/N) * (tx.T @ (-np.sign(e_vector)))

def gradient_descent(y, tx, initial_w, max_iters, gamma, compute_gradient, compute_cost): 
    """Gradient descent algorithm."""
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        gradient = compute_gradient(y, tx, w)
        loss = compute_cost(y, tx, w)
        w = w - gamma * gradient
        
        # store w and loss
        ws.append(np.copy(w))
        losses.append(loss)
        print("Gradient Descent({bi}/{ti}): loss={l}, w0={w0}, w1={w1}".format(
              bi=n_iter, ti=max_iters - 1, l=loss, w0=w[0], w1=w[1]))

    return losses, ws

def gradient_descent_mae(y, tx, initial_w, max_iters, gamma):
    return gradient_descent(y, tx, initial_w, max_iters, gamma, compute_gradient_mae, compute_cost_mae)

def gradient_descent_mse(y, tx, initial_w, max_iters, gamma):
    return gradient_descent(y, tx, initial_w, max_iters, gamma, compute_subgradient_mse, compute_cost_mse)