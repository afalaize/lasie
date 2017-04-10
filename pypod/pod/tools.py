# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 11:48:23 2017

@author: afalaize
"""

from __future__ import division, absolute_import, print_function

import numpy as np
import copy

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
    