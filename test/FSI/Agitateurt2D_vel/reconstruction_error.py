#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:58:34 2017

@author: afalaize
"""

import lasie_rom as lr

import numpy as np

from _0_locations import paths
from _0_parameters import parameters

import matplotlib.pyplot as plt

basis_hdf = lr.io.hdf.HDFReader(path=paths['basis'][0])
time_serie = lr.classes.TimeSerie(paths['ihdf'][0])


def norm_u(u):
    return np.sqrt(np.einsum('xc,xc', u, u))


# --------------------------------------------------------------------------- #
# Compute POD basis
meanfluc_hdf = lr.io.hdf.HDFReader(paths['meanfluc'][0])
meanfluc_hdf.openHdfFile()
mean = meanfluc_hdf.mean[:]
fluc = meanfluc_hdf.fluc[:]
meanfluc_hdf.closeHdfFile()

basis = lr.pod.compute_basis(fluc,
                             threshold=0,
                             nmax=None)

# %%
error = []
NM = basis.shape[-1]
NT = meanfluc_hdf.fluc[:].shape[-1]

NTOT = NM*NT

def compute_error(nm):
    print('nb modes = {}'.format(nm))
    error_m = []
    all_t = [(t, nm) for t in range(meanfluc_hdf.fluc[:].shape[-1])]
    def iterTS(pack):
        u_t = mean + fluc[:, :, pack[0]]
        coeffs = np.einsum('xc,xci->i', fluc[:, :, pack[0]], basis[:,:,:pack[1]])
        upod_t = mean + np.einsum('xci,i->xc', basis[:,:,:pack[1]], coeffs)
        return(norm_u(u_t-upod_t)/norm_u(u_t))
    error_m = lr.parallelization.map(iterTS, all_t)
    return np.mean(error_m)

# %%

nb_modes = [i+1 for i in range(0, NM, 5)]
error = list(map(compute_error, nb_modes))


# %%
plt.close('all')
plt.loglog(nb_modes, error)
plt.xlabel('Nb of modes (#)')
plt.ylabel('Mean realtive error ($L^2$ norm)')
