#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 17:49:25 2017

@author: Falaize
"""

import numpy as np
from sympy import symbols, sqrt, Matrix, lambdify
from sympy.tensor.array import tensorcontraction, tensorproduct, Array
from scipy.optimize import minimize

TOL = 1e-16


# ----------------------------  DEIM INDICES  ------------------------- #

def argmax(a):
    """
======
argmax
======

Returns the index of the maximum value of the module of array :code:`a`:

.. math:: i=\\underset{i\\in[0 \\cdots \\mathtt{nx}-1]}{\\mbox{argmax}}\\left(\
\\sqrt{\\sum_{j=1}^{\\mathtt{nc}} a_j^2(x_{i+1})}\\right)`.

Parameters
----------
a : array_like with shape :code:`(nx, nc)`
    Input array organized as :math:`\\big[\mathbf a\\big]_{i,j} = a_j(x_i)`

Returns
-------
index : int
    Index into the array.

Notes
-----
In case of multiple occurrences of the maximum value, the index
corresponding to the first occurrence is returned.

    """
    return int(np.argmax(np.einsum('ij,ij->i', a, a)))


def idcol(i, N):
    """
Returns the i-th column of the identity matrix of dimension N.
    """
    v = np.zeros(N)
    v[i] = 1.
    return v


def objective_function(P, B, u):
    nx, ne, nc = B.shape
    c = Array(symbols('c:{}'.format(ne)))

    PB = Array(np.einsum('xe,xfc->efc', P, B))
    assert PB.shape == (ne, ne, nc)
    Pu = Array(np.einsum('xe,xc->ec', P, u))
    assert Pu.shape == (ne, nc)
    # returns einsum path 'efcf'[0], 'efcf'[1] -> 'ec'
    PB_dot_c = tensorcontraction(tensorproduct(PB, c),
                                 (1, 3))
    assert PB_dot_c.shape == (ne, nc)
    error = PB_dot_c - Pu
    assert error.shape == (ne, nc)
    temp = tensorcontraction(tensorproduct(error, error), (1, 3))
    res = Array([temp[i, i] for i in range(ne)])
    assert res.shape == (ne, )
    obj = tensorcontraction(tensorproduct(res, res), (0, 1))

    grad = jacobian(Array((obj, )), c)[0, :]
    hess = jacobian(grad, c)

    def lambd(expr):
        l = lambdify(c, expr, modules='numpy')

        def func(c):
            return np.array(l(*c))
        return func

    return map(lambd, (obj, grad, hess))


def jacobian(F, X):
    nx = X.shape[0]
    nf = F.shape[0]
    jac = Matrix(np.zeros((nf, nx)))
    for i in range(nf):
        for j in range(nx):
            jac[i, j] = F[i].diff(X[j])
    assert jac.shape == (nf, nx)
    shape = map(int, jac.shape)
    return Array(jac, shape)

    
def extend_projector_dimension(P, nc):
    return np.concatenate(map(lambda a: a[:, :, np.newaxis], (P,)*nc),
                          axis = 2)
    

    
def compute_c(P, B, u):
    nx, ne, nc = B.shape
    assert P.shape == (nx, ne)
    assert u.shape == (nx, nc)
    
    nd_P = extend_projector_dimension(P, nc)
    M = np.einsum('xic,xjc->ij', nd_P, B)
    
    Pu = np.einsum('xic,xc->i', nd_P, u)
    c = np.einsum('ij,j->i', np.linalg.inv(M), Pu)
    
    return c

def indices(basis):
    """
=======
indices
=======

Indices for the DEIM approximation of the (POD) basis represented by the numpy
array `basis`.

We use Algorithm 1 in [1]_.

Parameter
---------

basis : numpy array with shape :code:`(nx, ne, nc)`
    Basis array with :code:`basis[i,j,k]` the k-th component of the j-th basis\
 element at point :math:`x_i`. The mesh is :math:`x_i\in\
\mathbb{R}^{\mathtt{nc}}, i \in[1\cdots\mathtt{nx}]` with :math:`\mathtt{nx}` \
the number of space points, :math:`\mathtt{ne}` the number of basis elements, \
and :math:`\mathtt{nc}` the number of spatial components.

Returns
-------

p : list of :math:`\mathtt{ne}` floats
    DEIM selection indices :math:`\{p_i\}_{1\leq i \leq \mathtt{ne}}`, \
:math:`p_i \in [1\cdots\mathtt{nx}],\; \\forall i \in [1\cdots\mathtt{ne}]`.

P : numpy array with shape (:math:`\mathtt{nx}`, :math:`\mathtt{ne}`)
    DEIM projector :math:`\mathbf P[:, i] = \mathbf{I_d}[:,p_i]\
\in\mathbb R^{\mathtt{nx}}`.

Reference
----------

.. [1] Chaturantabut, S., & Sorensen, D. C. (2010). Nonlinear model reduction \
via discrete empirical interpolation. SIAM Journal on Scientific Computing, \
32(5), 2737-2764.

    """
    nx, ne, nc = basis.shape
    p = list()

    # first basis element
    bi = basis[:, 0, :]

    # first DEIM index
    pi = argmax(bi)
    p.append(pi)

    # partial basis (one element)
    B = bi[:, np.newaxis, :]

    # partial projector (one element)
    P = idcol(pi, nx)[:, np.newaxis]

    all_c = list()
    for i in range(1, ne):

        # i-th basis element
        bi = basis[:, i, :]

#        # implicite function and jacobian
#        obj, grad, hess = objective_function(P, B, bi)
#
#        # solve implicite function
#        res = minimize(obj, all_c[-1], jac=grad, hess=hess,
#                       method='Newton-CG',
#                       options={'xtol': TOL, 'disp': True},
#                       tol=TOL)
#        c = res.x
        c = compute_c(P, B, bi)
        all_c.append(c)

        # projected partial basis reconstruction error
        r = bi - np.einsum('ijk,j->ik', B, c)

        # i-th DEIM index
        pi = argmax(r)
        p.append(pi)

        # partial basis (i elements)
        B = np.concatenate((B, bi[:, np.newaxis, :]), axis=1)

        # partial projector (i elements)
        P = np.concatenate((P, idcol(pi, nx)[:, np.newaxis]), axis=1)

    # --- END FOR --- #

    # return indices and projector
    return p, P, all_c

# ----------------------------  DEIM FUNCTION  ------------------------- #

def reconstruction_matrix(P, B):
    nx, ne, nc = B.shape
    assert P.shape == (nx, ne)
    nd_P = extend_projector_dimension(P, nc)    
    M = np.einsum('xic,xjc->ij', nd_P, B)    
    return np.einsum('xic,ij->xjc', B, np.linalg.inv(M))


def interpolated_func(func, P, Phi, Psi, mean_phi=None, mean_psi=None):
    """
Return the DEIM interpolation of function func with DEIM projector P and POD 
basis Phi and Psi

Parameters
----------

func: function
    Function to interpolate :math:`\mathbf f: \mathbb R^N\rightarrow \
\mathbb R^N`. 

P: array_like
    DEIM projector
    
Phi: array_like
    POD basis for the function variable    .

Psi: array_like
    POD basis for the function evaluation.
    
Return
-------

deimfunc: function
    Interpolated function :math:`\tilde{\mathbf f}: \mathbb R^M\rightarrow \
\mathbb R^N` with M << N.
    """
    
    
    nx, ne, nc = Phi.shape    

    assert P.shape == (nx, ne)
    assert func(np.array([(1., )*nc, ])).shape == (1, nc)
    
    if mean_phi is None:
        mean_phi = np.zeros((nx, nc))
        
    if mean_psi is None:
        mean_psi = np.zeros((nx, nc))
        
    projector = np.einsum('xi,xjc->ijc', P, Phi)
    projected_mean = np.einsum('xi,xc->ic', P, mean_phi)
    assert projector.shape == (ne, ne ,nc)
    
    M = reconstruction_matrix(P, Psi)
    assert M.shape == (nx, ne, nc)
    
    def deim_func(c):
        assert c.shape == (ne, )
        return np.einsum('xec,ec->xc',
                         M, func(projected_mean+np.einsum('ejc,j->ec', projector, c)))
    deim_func.func_doc = """
DEIM interpolation of nonlinear function.

Parameter
---------

c: array_like with shape ({0}, )
    Activation coefficients for the function argument basis.
    
Return 
------

res: array_like with shape ({1}, {2})
    Evaluation of func on specially selected arguments.
""".format(ne, nx, nc)
    return deim_func
    
