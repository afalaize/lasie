#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 19:16:35 2017

@author: Falaize
"""

import numpy as np


def concatenate_over_2d_axis(l):
    """
    
Parameter
----------
l : list of numpy arrays
    Each element of the list must be a 2 dimensionnal array with shape \
:code:`(n0, n1)`.

Return
------
a : numpy array
    Concatenation of the arrays in the list :code:`l`, with shape \
:code:`(n0, len(l), n1)`.
    """
    def add_axe(a):
        return a[:, np.newaxis, :]
    return np.concatenate(map(add_axe, l), axis=1)

    
def norm(a):
    """
Norm of the numpy array over the last dimension

Parameter
----------
a : numpy array with shape (Nx, Nc)

Return
------
n : numpy array with shape (Nx, Nc)
    Norm of the numpy array over the last dimension `n[i]=sqrt(sum_j a[i,j]^2)`
    """
    return np.sqrt(np.einsum('ij,ij->i', a, a))
