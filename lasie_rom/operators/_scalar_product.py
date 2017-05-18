#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri May 12 13:37:03 2017

@author: root
"""

import numpy as np

def scalar_product(u, v, W=None):
    """
Compute the (weighted) scalar product of :code:`u` and :code:`v`, with 
:code:`W`the weighting matrix.

Parameters
----------

u, v: array like
    Quantities defined over a (discrete) spatial domain represented by :code:`nx`
    grid points. Could be 
    * scalars :math:`u\in\mathbb{R}^{n_x\times 1}`
    * vectors :math:`u\in\mathbb{R}^{n_x\times n_c}` with :code:`nc` the number of components
    * tensors :math:`u\in\mathbb{R}^{n_x\times n_c\times n_c}`.

W: array like or None
    Weighting matrix with shape :code:`(nx, nx)`. If None, then \
:math:`\mathbf W \equiv \mathbf{I_d}(n_x)` (the default is None).
    
Return
------

a: numpy array
    Scalar product :math:`a = \sum_i \sum_j u_i W_{ij} v_j`.
    """
    
    u_shape = u.shape
    v_shape = v.shape
    
    if len(u_shape) == 1:
        u = u[:, np.newaxis]
    if len(v_shape) == 1:
        v = v[:, np.newaxis]
    
    u_shape = u.shape
    v_shape = v.shape
    
    if len(u_shape) == 2:
        u = u[:, np.newaxis, :]
    if len(v_shape) == 2:
        v = v[:, :, np.newaxis]
    
    u_shape = u.shape
    v_shape = v.shape
    
    assert len(u_shape) == 3
    assert len(v_shape) == 3
              
    assert u_shape[0] == v_shape[0]
    nx = u_shape[0]

    if u_shape[-1] != v_shape[1]:
        assert u_shape[-1] == 1 or v_shape[1] == 1
       
    if W is not None:
        assert W.shape == (nx, nx)
        v = np.einsum('ij,j...->i...', W, v)
    
    a = np.einsum('ijl,ilk->jk', u, v)
    if a.shape == (1, 1):
        a = a[0, 0]
    elif a.shape[0] == 1:
        a = a[0, :]
    elif a.shape[1] == 1:
        a = a[:, 0]
    return a
