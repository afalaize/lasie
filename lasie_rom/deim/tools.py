#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 17:49:25 2017

@author: Falaize
"""

import numpy as np
from ..misc.tools import norm

TOL = 1e-16


# -------------------------------  DEIM INDICES  ---------------------------- #

def get_major_element(array):
    """
    ======
    argmax
    ======
    
    Returns the index of the maximum value of the module of array :code:`a` 
    with shape :code:`̀(nx,)`: 
    
    Parameters
    ----------
    a : array_like with shape :code:`(nx, )`
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
    return int(np.argmax(np.abs(array)))


def get_major_component(array):
    """
    Return the index of the component which exhibits the greatest (discrete) 
    L2 norm.
    
    Parameter
    ---------
    
    array: numpy array
        Shape is (nx, nc) with nx the number of spatial grid points and nc the 
        number of spatial component.
        
    Return
    
    i : positive int
        Index i such that for all j: || array[:,j] || <= || array[:,i] ||.
    """
    
    nx, nc = array.shape
    norms = [norm(array[:, i]) for i in range(nc)]
    return int(np.argmax(norms))


# -------------------------------  DEIM INDICES  ---------------------------- #


def temp_coeffs(P, B, b):
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
    M = np.einsum('xcm,xcn->mn', P, B)
    Ppsi = np.einsum('xcm,xc->m', P, b)
    return np.einsum('ij,j->i', np.linalg.inv(M), Ppsi)


# -------------------------------  DEIM INDICES  ---------------------------- #

def projector(B):
    """
projector
*********

Indices for the DEIM approximation of the (POD) basis represented by the numpy
array `B`.

We use a slightly modified version of Algorithm 1 in [1]_.

Parameter
---------

B : numpy array with shape :code:`(nx, ne, nc)`
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
    
    nx, nc, ne = B.shape

    rho = list()
    mu = list()
    
    rho.append(get_major_component(B[:, :, 0]))
    mu.append(get_major_element(B[:, rho[0], 0]))
    

    # partial projector (one element)
    p = np.zeros((nx, nc, 1))
    p[mu[0], rho[0], 0] = 1.
    P = p
    
    B_temp = B[:, :, :1]
    for i, b in enumerate(B[: ,: ,1:].T, 1):
        
        # i-th basis element
        b = b.T
        
        # obtain coefficients
        c = temp_coeffs(P, B_temp, b)

        # projected partial basis reconstruction error
        r = b - np.einsum('xcm,m->xc', B_temp, c)

        # i-th DEIM index
        rho_j = get_major_component(r)
        mu_j = get_major_element(r[:, rho_j])
        
        rho.append(rho_j)
        mu.append(mu_j)
        
        p = np.zeros((nx, nc, 1))
        p[mu[-1], rho[-1], 0] = 1.
         
        P = np.concatenate((P, p), axis=2)

        B_temp = np.concatenate((B_temp, b[:, :, np.newaxis]), axis=2)
        
    # --- END FOR --- #

    # return projector
    return P, rho, mu

# ----------------------------  DEIM FUNCTION  ------------------------- #

def interpolated_func(func, Phi, Psi, mean_phi=None):
    """
Return the DEIM interpolation of function func with DEIM projector P and POD 
basis Phi and Psi

Parameters
----------

func: function
    Function to interpolate :math:`\\mathbf f: \\mathbb R^{n_u}\\rightarrow \
\mathbb R^{n_f}`. 

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

    nx, nu, nphi = Phi.shape    
    nx, nf, npsi = Psi.shape    
       
    P, rho, mu = projector(Psi)
    
    M = np.einsum('mki,mkj->ij', P, Psi)
    iM = np.linalg.inv(M)

        
    if mean_phi is None:
        mean_phi = np.zeros((nx, nu))
        
    def deim_func(a):
        args = [np.einsum('km,m->k', Phi[mu[j], :, :], a) + mean_phi[mu[j], :] for j in range(npsi)]
        F = np.array([func[rho[j]](*args[j]) for j in range(npsi)])
        return np.einsum('ij,j->i', iM, F.flatten())
    
    deim_func.func_doc = """
DEIM interpolation of nonlinear function.

Parameter
---------

a: array_like with shape ({0}, )
    Activation coefficients for the solution basis functions.
    
Return 
------

res: array_like with shape ({1}, )
    Evaluation of func on specially selected arguments.
""".format(nphi, npsi)
    return deim_func
    
