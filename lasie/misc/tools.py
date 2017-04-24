#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 19:16:35 2017

@author: Falaize
"""

import numpy as np


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
