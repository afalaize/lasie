#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:30:38 2017

@author: afalaize
"""

from lasie_rom import io, rom, misc, parallelization
from lasie_rom.classes import TimeSerie

from lasie_rom.rom import fsi_relaxed_rigidity

from main import parameters
from locations import paths
from options import options

import numpy

# --------------------------------------------------------------------------- #
# Build ROM Matrices:
velocity_basis_hdf = io.hdf.HDFReader(paths['basis'][0], openFile=True)
velocity_meanfluc_hdf = io.hdf.HDFReader(paths['meanfluc'][0], openFile=True)

print('Build temp_a')
temp_a = fsi_relaxed_rigidity.temp_a(velocity_basis_hdf.basis[:])
print('Build temp_b_rho')
temp_b_rho = fsi_relaxed_rigidity.temp_b_rho(velocity_basis_hdf.basis[:],
                                              velocity_meanfluc_hdf.mean[:],
                                              velocity_basis_hdf.basis_grad[:],
                                              velocity_meanfluc_hdf.mean_grad[:])
print('Build temp_b_nu')
temp_b_nu = fsi_relaxed_rigidity.temp_b_nu(velocity_basis_hdf.basis_grad[:])
print('Build temp_c')
temp_c = fsi_relaxed_rigidity.temp_c(velocity_basis_hdf.basis[:], velocity_basis_hdf.basis_grad[:])
print('Build temp_d')
temp_d = fsi_relaxed_rigidity.temp_d(velocity_basis_hdf.basis_grad[:])
print('Build temp_f_rho')
temp_f_rho = fsi_relaxed_rigidity.temp_f_rho(velocity_basis_hdf.basis[:],
                                              velocity_meanfluc_hdf.mean[:],
                                              velocity_meanfluc_hdf.mean_grad[:])
print('Build temp_b_nu')
temp_f_nu = fsi_relaxed_rigidity.temp_f_nu(velocity_basis_hdf.basis_grad[:],
                                             velocity_meanfluc_hdf.mean_grad[:])

data = {'a': temp_a,
        'b_rho': temp_b_rho,
        'b_nu': temp_b_nu,
        'c': temp_c,
        'd': temp_d,
        'f_rho': temp_f_rho,
        'f_nu': temp_f_nu}

print('Save {}'.format(paths['matrices']))
# write hdf for rom matrices
io.hdf.data2hdf(data, paths['matrices'])

# --------------------------------------------------------------------------- #
# %% Project the snapshot on the pod basis

print('Build coeffs')
# instanciate TimeSerie
ts = TimeSerie(paths['ihdf'][0])

# Open hdf files
ts.openAllFiles()

if options['dataname'] is not None:
    dataname = options['dataname']
else:
    d = ts.data[0]
    d.openHdfFile()
    args = dir(d)
    temp = [a.startswith('f_') for a in args]
    dataname = args[temp.index(True)]
    d.closeHdfFile()

# instanciate TimeSerie
def coefficients():
    for u in ts.generator(dataname)():
        yield numpy.einsum('xc,xci->i',
                           u-velocity_meanfluc_hdf.mean[:],
                           velocity_basis_hdf.basis[:])

coeffs = misc.concatenate_in_given_axis(coefficients(), 0)

data = {'coeffs': coeffs}

print('Save {}'.format(paths['coeffs']))
# write hdf for rom matrices
io.hdf.data2hdf(data, paths['coeffs'])

# Close hdf files
velocity_basis_hdf.closeHdfFile()
velocity_meanfluc_hdf.closeHdfFile()
ts.closeAllFiles()
