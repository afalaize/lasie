# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 11:48:23 2017

@author: afalaize
"""

from __future__ import division, absolute_import, print_function
from ..misc.tools import concatenate_in_given_axis

import numpy as np
import copy


def compute_basis(U, threshold=1e-6, nmax=None):
    """
    build the POD basis from the snapshots array :code:`U`.
    
    Parameters
    -----------
    
    U : numpy ndarray
        Shape is (nx, nc, nt) with nx the number of spatial nodes, nc the 
        number of data components and nt the number of data elements (times).
        
    Return
    ------
    
    B : numpy nd array
        POD basis with shape (nx, nc, nB) where nB <= nmax and chosen so that 
        the relative error on the truncated basis is <= threshold.
    """

    if nmax is None:
        nmax = float('Inf')
    eigenvals, eigenvecs = eigen_decomposition(U)
    eigenenergy = eigen_energy(eigenvals)
    index = truncation_index(eigenenergy, threshold=threshold)
    npod = min([nmax, index])
    
    # Define POD basis
    basis = np.einsum('xct,ti->xci', U, eigenvecs[:, :npod])
    normalize_basis(basis)
    
    return basis

    
def eigen_energy(eigen_vals):
    """
    Return the list of eneries associated with pod eigen values, with 
    energy[i] = sum(eigen_vals[:i+1])/sum(eigen_vals)
    """
    modes_energy = list()
    for i, val in enumerate(eigen_vals):
        mode_energy = sum(eigen_vals[:i+1])/sum(eigen_vals)
        modes_energy.append(mode_energy)
    return np.array(modes_energy)


def truncation_index(eigen_energy, threshold=1e-6):
    """
    Return the index i in eigen_energy so that 1-eigen_energy[i] < threshold.
    """
    return [me >= 1-threshold for me in eigen_energy].index(True)+1


def check_basis_is_orthonormal(basis):
    """
    Print informations about the orthonormality of array 'basis' (maximum
    valmue out of diagonal and mean of diagonal values).
    
    Parameters
    ----------
    
    B : numpy array
        Basis array with shape (nx, nc , nm) where nx is the number of sptaial 
        points, nc is the number components for each basis element an nm is the
        number of basis elements.
        
    Retrun
    ------
    
    Nothing, just print some informations.
    
    
    """
    M = np.einsum('mci,mcj->ij', basis, basis)
    mask_diag = np.ones(M.shape)-np.eye(M.shape[0])
    maxv = (M*mask_diag).max()
    print("val max out of diag from np.dot(basis.T, basis) is {}".format(maxv))
    mean = np.einsum('ii', M)/M.shape[0]
    print("mean val on diag from np.dot(basis.T, basis) is {}".format(mean))


def normalize_basis(basis):
    M = np.einsum('mci,mcj->ij', basis, basis)
    for i, row in enumerate(M):
        basis[:, :, i] = basis[:, :, i]/np.sqrt(row[i])

        
def eigen_decomposition(ts):
    """
    Compute the eigen decompositon of matrix C[i,j] = ts[x,c,i]*ts[x,c,j]
    """

    # Form correlation matrix
    C = np.einsum('xci,xcj->ij', ts, ts)
    eigen_vals, eigen_vecs = np.linalg.eig(C)
    
    # Remove the imaginary part (which should be numerically close to zero)
    eigen_vals = np.real(eigen_vals)
    eigen_vecs = np.real(eigen_vecs)

    # sort by decreasing eigen values
    indices = sortIndices(eigen_vals)

    eigen_vals = np.array([[eigen_vals[n] for n in indices], ]).T
    eigen_vecs = eigen_vecs[:, np.array(indices)]
                            
    return eigen_vals, eigen_vecs
                              

def meanfluc(A):
    """
    meanfluc
    ********
    
    Compute the mean of the time serie and remove that mean from each snapshot
    in array A
    
    Parameters
    ----------
    
    A : numpy array with shape (nx, nc, nt)
        Time serie array, where nx is the number of spatial discretization 
        points, nc is the number of spatial componentsnt is the number of 
        snapshots.
        
    Returns
    -------
    
    mean : numpy array with shape (nx, nc)
        Mean of the time serie over the last axis: 
        :math:`mean[i,j]=\\frac{1}{nt}\\sum_{t=1}^{nt}A[i, j, t]`.
    
    fluc : numpy array with shape (nx, nc, nt)
        Time serie array where the mean has been removed from each snapshot: 
        :math:`fluc[:, :, i] = A[:, :, i] - mean[:, :]`.
    """
    
    nx, nc, nt = A.shape
    mean = np.mean(A, axis=2)    
    fluc = concatenate_in_given_axis([A[:, :, t] - mean for t in range(nt)], 2)
    return mean, fluc

    
def compute_kinetic_energy(data):
    """
    Return the kinetic energy associated with the velocity data with shape (nx, nt, nc).
    """
    return np.einsum('ict,ict->t', data, data)
    

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
    