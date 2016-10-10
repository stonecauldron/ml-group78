# -*- coding: utf-8 -*-
"""Problem Sheet 2.

Stochastic Gradient Descent
"""
from helpers import batch_iter
import costs

def compute_stoch_gradient_mse(y, tx, w):
    """Compute a stochastic gradient for batch data."""
    return compute_gradient_mse(y, tx, w)
    
def compute_stoch_gradient_mae(y, tx, w):
    return compute_gradient_mae(y, tx,w)

def stochastic_gradient_descent(
        y, tx, initial_w, batch_size, max_epochs, gamma, compute_stoch_gradient):
    """Stochastic gradient descent algorithm."""
    w = initial_w
    losses = np.zeros(max_epochs)
    ws = np.zeros((max_epochs, w.shape[0]))
    for i in range(max_epochs):
        generator = batch_iter(y, tx, batch_size)
        y_n, tx_n = next(generator)
        g = compute_stoch_gradient(y_n, tx_n, w)
        w = w - gamma * g
        ws[i] = w
        losses[i] = compute_cost(y, tx, w)
    return losses, ws