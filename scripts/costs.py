# -*- coding: utf-8 -*-
"""a function used to compute the loss."""

import numpy as np

def error_vector(y, tx, w):
    """calculate the error vector
    """
    N = y.shape[0]
    return y - np.squeeze(tx @ w)

def compute_cost_mse(y, tx, w):
    """calculate the cost using mean square error
    """
    N = y.shape[0]
    e_vector = error_vector(y, tx, w)
    return (1/(2*N) * (e_vector.T @ e_vector))

def compute_cost_mae(y, tx, w):
    """calculate the cost using mean absolute error
    """
    N = y.shape[0]
    return (1/N) * (np.ones(N).T @ np.abs(e_vector))
