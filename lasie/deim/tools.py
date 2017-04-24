#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 17:49:25 2017

@author: Falaize
"""

import numpy as np
from scipy.optimize import root
from ..misc.tools import norm

# ----------------------------  DEIM INDICES  ------------------------- #

def argmax(a):
    """
Indices of the maximum values of the module of array `a` organized as
`a[i,j] = a_j(x_i)`.
    """
    return np.argmax(np.einsum('ij,ij->i', a, a))


def idcol(i, N):
    """
Returns the i-th column of the identity matrix with shape N.
    """
    v = np.zeros(N)
    v[i] = 1.
    return v


def indices(a):
    """
Indices for the DEIM approximation of the basis represented by the array `a`, \
where `a[i,j,k]` is the component k of the basis element j at point x_i.

Algorithm 1 from reference [1]_ is used.

Reference
----------

.. [1] Chaturantabut, S., & Sorensen, D. C. (2010). Nonlinear model reduction \
via discrete empirical interpolation. SIAM Journal on Scientific Computing, \
32(5), 2737-2764.

    """
    nx, nm, nc = a.shape
    p = list()
    ai = a[:, 0, :]
    pi = argmax(ai)
    A = ai[:, np.newaxis, :]
    P = idcol(pi, nx)[:, np.newaxis]
    p.append(pi)
    for i in range(1, nm):
        ai = a[:, i, :]

        def func(c):
            t1 = np.einsum('il,ijk->ljk', P, A)
            t2 = np.einsum('ljk,j->lk', t1, c)
            t3 = np.einsum('il,ik->lk', P, ai)
            return norm(t2-t3)

        c = root(func, np.zeros(i)).x
        r = ai - np.einsum('ijk,j->ik', A, c)
        pi = argmax(r)
        A = np.concatenate((A, ai[:, np.newaxis, :]), axis=1)
        P = np.concatenate((P, idcol(pi, nx)[:, np.newaxis]), axis=1)
        p.append(pi)
    return p, P

# ----------------------------  DEIM MATRICES  ------------------------- #
