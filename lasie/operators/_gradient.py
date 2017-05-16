#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 17:18:52 2017

@author: afalaize
"""
from ..config import ORDER
import numpy as np


def gridgradient(a, shape, h, edge_order=2):
    """
Return the gradient of array `a` defined over a grid using `numpy.gradient`.

The gradient is computed using second order accurate central differences
in the interior and either first differences or second order accurate
one-sides (forward or backwards) differences at the boundaries.

Parameters
-----------

a : numpy array with shape (Nx, Nc)
    Input array with the convention that `a[i, j]` is the value of the `j`-th
    component of the quantity `a` at point x[i].

shape : iterable of floats
    Original shape of the grid as return by `lasie.grids.generate`.

h : float or iterable of floats
    Space discretization step, constant in each direction. If a single value is
    provided, it is used for every direction. Else, len(h) == Nc must hold.

edge_order : {1, 2}
    Option for `numpy.gradient`. Gradient is calculated using N\ :sup:`th`
    order accurate differences at the boundaries. Default: 2.

Return
------

grad: numpy array with shape (Nx, Nc, Nc)
    The discrete gradient of `a` with the following convention:
    grad[i,j,k] = d a_j(x_i) / d x_k.

See also
--------

`lasie.grids.generate`

    """
    # original array shape
    array_shape = a.shape
    # init gradient shape
    grad = np.zeros((array_shape[0], array_shape[1], shape[0]))
    # iterate over the components of a
    for i in range(array_shape[1]):
        # reshape to grid shape
        ai = a[:, i].reshape(shape[1:], order=ORDER)
        # compute gradient
        gi = np.gradient(ai, *h, edge_order=edge_order)
        # iterate over the components of the gradient
        for j in range(shape[0]):
            # store gradient
            grad[:, i, j] = gi[j].reshape((np.prod(shape[1:]), ),
                                          order=ORDER)
    return grad
