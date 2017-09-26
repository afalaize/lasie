#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 19:16:35 2017

@author: Falaize
"""

import numpy as np
import sympy as sy

from ..config import ORDER, DTYPE


def sympyExpr2numpyFunc(expr, args, subs, vectorize=True):
    """
    Build a numerical evaluation of expr(args) after substitutions of subs.

    Inputs
    ------

    expr : sympy expresion

    args : sequence of sympy symbols for arguments

    subs : dictionary {symbol: value, ...} with symbol a sympy symbol and value
    a numeric for substition.

    Return
    ------

    func : numerical function
    Evaluation expr(args, subs) => func(*args)

    """

    if isinstance(expr, (list, tuple)):
        f_list = []
        for e in expr:
            f_list.append(sympyExpr2numpyFunc(e, args, subs, vectorize=False))

        def func(*fargs):
            return np.array([f(*fargs) for f in f_list])
    else:
        if subs is None or subs == {}:
            subsed_expr = expr
        else:
            subsed_expr = expr.subs(subs)
        subsed_expr = subsed_expr.simplify()

        fs = subsed_expr.free_symbols
        diff = fs.difference(args)
        if not len(diff) == 0:
            raise AttributeError('missing replacement for symbol {}'.format(diff))

        func = sy.lambdify(args, subsed_expr, dummify=False, modules='numpy')

    if vectorize:
        return np.vectorize(func)
    else:
        return func


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
        Concatenation of the arrays in the list `l`. Number of dimensions
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

    Norm of the numpy array `a`over its last axis.

    Input
    -----
    a : numpy array with shape (Nx, Nc)

    Return
    ------
    n : numpy array with shape (Nx,)
        Norm of the numpy array over the last dimension
        `n[i]=sqrt(sum_j a[i,j]^2)`

    """
    if len(a.shape) == 1:
        a = a[np.newaxis, :]
    return np.sqrt(np.einsum('ij,ij->i', a, a))


def iterarray(a, ind=0):
    """

    Build a generator over the `i`-th axis of the array `a`.

    Inputs
    ------

    `a` : N-dimensional numpy array

    `i` : index in [0 -- N-1] of the axis of `a` for iteration

    Return
    ------

    gen : a generator of arrays associated with an iterator over the `i`-th
    axis of `a` with element j given by the (N-1)-dimensional numpy array:

    b[..., x_{i-1}, x_{i+1}, ...] = a[..., x_{i-1}, j, x_{i+1}, ...]

    """
    naxis = len(a.shape)
    s = [slice(None), ]*naxis
    def slice_i(i):
        slicer = s
        slicer[ind] = i
        return slicer
    for i in range(a.shape[ind]):
        yield a[slice_i(i)]
