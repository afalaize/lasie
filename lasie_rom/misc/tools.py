#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 19:16:35 2017

@author: Falaize
"""

import numpy as np
from ..config import ORDER

def vstack(M):
    """
    Stack elements of array M with shape (nx, nc, m) into array MM with shape
    (nx*nc, m) with MM = [M[:, :, 0].T, ..., M[:, :, ].T].T
    """
    nx, nc, m = M.shape
    return M.reshape((nx*nc, m), order=ORDER)


def unvstack(M, nc):
    """
    Stack elements of array M with shape (nx, m, nc) into array MM with shape
    (nx*m, nc) with MM = [M[:, 0, :].T, ..., M[:, m, :].T].T
    """
    n, m = M.shape
    assert n // nc == 0

    nx = n // nc
    new_shape = (nx, m, nc)
    return M.reshape(new_shape, order=ORDER)


def concatenate_in_first_axis(l):
    """

    Parameter
    ----------
    l : list of N numpy arrays
        Each element of the list must be a D dimensionnal array, and dimensions
        of every arrays must coincide.

    Return
    ------
    a : numpy array
        Concatenation of the arrays in the list :code:`l`. Number of dimensions
        is D+1, with shape along last axis equal to the lenght of list l.

    """
    def expand(a):
        return np.expand_dims(a, 0)
    try:
        return np.concatenate(map(expand, l), axis=0)
    except TypeError:
        return np.concatenate(list(map(expand, l)), axis=0)


def concatenate_in_given_axis(l, axis=0):
    """
    Parameter
    ----------
    l : list of N numpy arrays
        Each element of the list must be a D dimensionnal array, and dimensions
        of every arrays must coincide.
    Return
    ------
    a : numpy array
        Concatenation of the arrays in the list :code:`l`. Number of dimensions
        is D+1, with shape along last axis equal to the lenght of list l.
    """
    def expand(a):
        return np.expand_dims(a, axis)
    try:
        return np.concatenate(map(expand, l), axis=axis)
    except TypeError:
        return np.concatenate(list(map(expand, l)), axis=axis)


def norm(a):
    """
Norm of the numpy array over the last dimension

Parameter
----------
a : numpy array with shape (Nx, Nc)

Return
------
n : numpy array with shape (Nx,)
    Norm of the numpy array over the last dimension `n[i]=sqrt(sum_j a[i,j]^2)`
    """
    if len(a.shape) == 1:
        a = a[np.newaxis, :]
    return np.sqrt(np.einsum('ij,ij->i', a, a))


def iterarray(a, axis=0):
    naxis = len(a.shape)
    s = [slice(None), ]*naxis
    def slice_i(i):
        slicer = s
        slicer[axis] = i
        return slicer
    for i in range(a.shape[axis]):
        yield a[slice_i(i)]
