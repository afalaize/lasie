#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 17:49:25 2017

@author: Falaize
"""

import numpy as np
from ..misc.tools import vstack, unstack

TOL = 1e-16


# ----------------------------  DEIM INDICES  ------------------------- #

def argmax(a):
    """
    ======
    argmax
    ======
    
    Returns the index of the maximum value of the module of array :code:`a` 
    with shape :code:`̀(nx, nc)`: 
    
    .. math:: i=\\underset{i\\in[0 \\cdots \\mathtt{nx}-1]}{\\mbox{argmax}}\\left(\\sqrt{\\sum_{j=1}^{\\mathtt{nc}} a_j^2(x_{i+1})}\\right).
    
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

Parameters
----------

i : int
    Index of the identity matrix column.
N : int
    Size of identity matrix.

Return
------

c : numpy.ndarray
    Teh 'i'-th column of identity matrix of order N.
    """
    v = np.zeros(N)
    v[i] = 1.
    return v


    
def compute_c(P, Psi, psi):
    """
    Returns the coefficients in the DEIM  procedure
    
    Parameters
    ----------
    
    P : numpy.ndarray
        Stacked components DEIM projector with shape (nx*nf, npsi).

    Psi : numpy.ndarray
        Stacked components of reduced basis for the non-linear function, with 
        shape (nx*nf, npsi).
        
    psi : numpy.ndarray
         Stacked reduced basis element.
    """

    M = np.einsum('xi,xj->ij', P, Psi)
    
    Ppsi = np.einsum('xi,x->i', P, psi)
    
    return np.einsum('ij,j->i', np.linalg.inv(M), Ppsi)


def projector(basis):
    """
projector
*********

Indices for the DEIM approximation of the (POD) basis represented by the numpy
array `basis`.

We use a slightly modified version of Algorithm 1 in [1]_.

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
    
    # stack basis components
    _basis = vstack(basis)
    
    p = list()

    # first basis element
    bi = _basis[:, 0]

    # first DEIM index
    pi = argmax(bi)
    p.append(pi)

    # partial basis (one element)
    B = bi[:, np.newaxis]

    # partial projector (one element)
    P = idcol(pi, nx*nc)[:, np.newaxis]

    for i in range(1, ne):

        # i-th basis element
        bi = basis[:, i]

        # obtain coefficients
        c = compute_c(P, B, bi)

        # projected partial basis reconstruction error
        r = bi - np.einsum('ij,j->j', B, c)

        # i-th DEIM index
        pi = argmax(r)
        p.append(pi)

        # partial basis (i elements)
        B = np.hstack((B, bi))

        # partial projector (i elements)
        P = np.hstack((P, idcol(pi, nx*nc)))

    # --- END FOR --- #

    # return projector
    return unstack(P, nc)

# ----------------------------  DEIM FUNCTION  ------------------------- #


def reconstruction_matrix(P, Phi, Psi, u_average):
    nx, nP = P.shape
    n1, nphi, nu = Phi.shape
    n2, npsi, nf = Psi.shape
    
    m1 = np.einsum('xm,xn->mn', vstack(P), vstack(Psi))
    m2 = np.einsum('xm,xn->mn', vstack(Phi), vstack(Psi))
    M = np.einsum('ij,jk->ik', m2, np.linalg.inv(m1))
    
    def Si(i):
        return np.hstack([idcol(i + n*nx, nx*nu)[:, np.newaxis] 
                          for n in range(nu)])
    SS = np.array([Si(i) for i in range(nx)])
    def N(c, i):
        return np.einsum('xm,xyj,yk->jk', P[:, i, c], SS, vstack(Phi))
    
    def umoy(c ,i):
        return np.einsum('xm,xyj,y->j', P[:, i, c], SS, vstack(u_average))
    
    return M, N, umoy


def interpolated_func(func, P, Phi, Psi, mean_phi=None, mean_psi=None):
    """
Return the DEIM interpolation of function func with DEIM projector P and POD 
basis Phi and Psi

Parameters
----------

func: function
    Function to interpolate :math:`\\mathbf f: \\mathbb R^{n_u}\\rightarrow \
\mathbb R^{n_f}`. 

P: array_like
    DEIM projector
    
Phi: array_like
    POD basis for the function variable    .

Psi: array_like
    POD basis for the function evaluation.
    
Return
-------

deimfunc: function
    Interpolated function :math:`\tilde{\mathbf f}: \mathbb R^{n_\phi}\rightarrow \
\mathbb R^{n_\phi}`.
    """

    nx, nphi, nu = Phi.shape    
    nx, npsi, nf = Psi.shape
    
    M, N, umoy = reconstruction_matrix(P, Phi, Psi)
    
    def fc(a, c):
        def prod(i):
            return np.einsum('ij,j->i', N(c, i), a)
        return np.array([func(*(umoy(c, i)+prod(i))) for i in range(nphi)])

    
    assert func(np.array([(1., )*nphi, ])).shape == (1, npsi)
    
    if mean_phi is None:
        mean_phi = np.zeros((nx, nphi))
        
    if mean_psi is None:
        mean_psi = np.zeros((nx, npsi))
    
    def deim_func(a):
        assert c.shape == (nphi, )
        return np.einsum('', M, np.add(*[fc(a, c)) for c in nf])
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
""".format(nphi, nx, npsi)
    return deim_func
    
