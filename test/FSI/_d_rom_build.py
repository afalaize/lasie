#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:30:38 2017

@author: afalaize
"""

from lasie_rom import io, rom, misc, parallelization
from lasie_rom.rom import navierstokes_rotation
from lasie_rom.classes import TimeSerie

from main import parameters
from locations import paths
from options import options

import numpy

# --------------------------------------------------------------------------- #
# Build ROM Matrices:
velocity_basis_hdf = io.hdf.HDFReader(paths['basis'][0], openFile=True)
velocity_meanfluc_hdf = io.hdf.HDFReader(paths['meanfluc'][0], openFile=True)

temp_a = navierstokes_rotation.temp_a(velocity_basis_hdf.basis[:])
temp_b_rho = navierstokes_rotation.temp_b_rho(velocity_basis_hdf.basis[:], velocity_basis_hdf.basis_grad[:],
                                              velocity_meanfluc_hdf.mean[:], velocity_meanfluc_hdf.mean_grad[:])
temp_b_nu = navierstokes_rotation.temp_a(velocity_basis_hdf.basis_grad[:])
temp_c = navierstokes_rotation.temp_a(velocity_basis_hdf.basis[:], velocity_basis_hdf.basis_grad[:])
temp_d = navierstokes_rotation.temp_a(velocity_basis_hdf.basis_grad[:])
temp_f_rho = navierstokes_rotation.temp_b_rho(velocity_basis_hdf.basis[:],
                                              velocity_meanfluc_hdf.mean[:],
                                              velocity_meanfluc_hdf.mean_grad[:])
temp_f_nu = navierstokes_rotation.temp_b_rho(velocity_basis_hdf.basis_grad[:],
                                             velocity_meanfluc_hdf.mean_grad[:])

data = {'a': temp_a,
        'b_rho': temp_b_rho,
        'b_nu': temp_b_nu,
        'c': temp_c,
        'd': temp_d,
        'f_rho': temp_f_rho,
        'f_nu': temp_f_nu}

# write hdf for rom matrices
io.hdf.data2hdf(data, paths['matrices'])

# --------------------------------------------------------------------------- #
# Project the snapshot on the pod basis

# instanciate TimeSerie
ts = TimeSerie(paths['ihdf'])

# Open hdf files
ts.openAllFiles()

# instanciate TimeSerie
def coefficients():
    for u in ts.generator(options['dataname'])():
        yield numpy.einsum('xc,xci->i',
                           u-velocity_meanfluc_hdf.mean[:],
                           velocity_basis_hdf.basis[:])

coeffs = misc.concatenate_in_given_axis(coefficients(), 0)

data = {'coeffs': coeffs}

# write hdf for rom matrices
io.hdf.data2hdf(data, paths['coeffs'])

# Close hdf files
velocity_basis_hdf.closeHdfFile()
velocity_meanfluc_hdf.closeHdfFile()
ts.closeAllFiles()
