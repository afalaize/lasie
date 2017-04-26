# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 11:48:23 2017

@author: afalaize
"""

from __future__ import division, absolute_import, print_function

import numpy as np
import copy


def compute_basis(ts, threshold=1e-6, nmax=None):
    """
    build the POD basis from the snapshots array :code:`ts`.
    """

    if nmax is None:
        nmax = float('Inf')
        
    eigenvals, eigenvecs = eigen_decomposition(ts)
    eigenenergy = eigen_energy(eigenvals)
    index = truncation_index(eigenenergy, threshold=threshold)
    npod = min([nmax, index])
    
    # Define POD basis
    basis = np.einsum('xtc,ti->xic', ts, eigenvecs[:, :npod])
    normalize_basis(basis)
    
    return basis

    
def eigen_energy(eigen_vals):
    modes_energy = list()
    for i, val in enumerate(eigen_vals):
        mode_energy = sum(eigen_vals[:i+1])/sum(eigen_vals)
        modes_energy.append(mode_energy)
    return modes_energy


def truncation_index(eigen_energy, threshold=1e-6):
    return [me >= 1-threshold for me in eigen_energy].index(True)+1


def check_basis_is_orthonormal(basis):
    M = np.einsum('mic,mjc->ij', basis, basis)
    mask_diag = np.ones(M.shape)-np.eye(M.shape[0])
    maxv = (M*mask_diag).max()
    print("val max out of diag from np.dot(basis.T, basis) is {}".format(maxv))
    mean = np.einsum('ii', M)/M.shape[0]
    print("mean val on diag from np.dot(basis.T, basis) is {}".format(mean))


def normalize_basis(basis):
    M = np.einsum('mic,mjc->ij', basis, basis)
    for i, row in enumerate(M):
        basis[:, i] = basis[:, i]/np.sqrt(row[i])

        
def eigen_decomposition(ts):
    """
    Compute the eigen decompositon of matrix C[i,j] = ts[x,i,c]*ts[x,j,c]
    """

    # Form correlation matrix
    C = np.einsum('xic,xjc->ij', ts, ts)
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    
    # Remove the imaginary part (which should be numerically close to zero)
    eigen_vals = np.real(eigen_vals)
    eigen_vecs = np.real(eigen_vecs)

    # sort by decreasing eigen values
    indices = sortIndices(eigen_vals)

    eigen_vals = np.array([[eigen_vals[n] for n in indices], ]).T
    eigen_vecs = eigen_vecs[:, np.array(indices)]
                            
    return eigen_vals, eigen_vecs
                              

def meanfluc(ts):
    """
========
meanfluc
========

Compute the mean of the time serie and remove that mean from each snapshot.

Parameters
----------

ts : aray_like with shape (nx, nt, nc)
    time serie, where nx is the number of spatial discretization points, nt is
    the number of snapshots and nc is the number of spatial components.
    
Returns
-------

mean : aray_like with shape (nx, nc)
    Mean of the time serie over the 2nd axis: :math:`mean[i,j]=\frac{1}{nt}\
\sum_{t=1}^{nt}ts[i, t, j]`.

fluc : aray_like with shape (nx, nt, nc)
    Time serie with the mean removed from each snapshot: :code:`fluc[:, i, :]\
=ts[:, i, :]-mean[:, :]`.
    """
    mean = np.mean(ts, axis=1)
    for i, a in enumerate(np.swapaxes(ts, 0, 1)):
        ts[:, i, :] = a - mean
    return mean, ts


    
def compute_kinetic_energy(data):
    """
    Return the kinetic energy associated with the velocity data with shape (nx, nt, nc).
    """
    return np.einsum('itc,itc->t', data, data)
    

def sortIndices(liste):
    """
    Return a list of indices so that
    [liste(i) for i in sortIndices(liste)]
    is a ordered list with decreasing values.
    """
    if not isinstance(liste, list):
        liste = [element for element in liste]
    copy_liste = copy.copy(liste)
    indices = list()
    while len(copy_liste) > 0:
        val_max = max(copy_liste)
        indices.append(liste.index(val_max))
        copy_liste.pop(copy_liste.index(val_max))
    return indices
    
    
def tensor2vector(data):
    assert len(data.shape) == 3, 'Expected a 3-D array, got data with shape {}'.format(data.shape)
    d1, d2, d3 = data.shape
    output = np.zeros((d1*d3, d2))
    for i in range(d3):
        output[i*d1:(i+1)*d1, :] = data[:, :, i]
    return output
        
    
def vector2tensor(data, dim):
    assert len(data.shape) == 2, 'Expected a 2-D array, got data with shape {}'.format(data.shape)
    d1, d2 = data.shape
    d1_reduced = d1//dim
    output = np.zeros((d1_reduced, d2, dim))
    for i in range(dim):
        output[:, :, i] = data[i*d1_reduced:(i+1)*d1_reduced:, :]
    return output
    