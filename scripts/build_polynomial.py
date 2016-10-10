# -*- coding: utf-8 -*-
"""implement a polynomial basis function."""

import numpy as np


def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree."""
    phi=np.matrix
    phi=np.zeros((len(x), degree+1))
    phi[:,0]=np.ones(len(x)) 
    for i in range (1 , degree+1):
        phi[:,i]=x**i
    return phi
