#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 23 12:57:49 2017

@author: afalaize
"""

from locations import paths
from options import options

from main import parameters
from lasie_rom.io.hdf import format_data_name

import numpy as np

from lasie_rom.classes import TimeSerie
from lasie_rom.io.hdf import HDFReader
from lasie_rom import plots, misc

import matplotlib.pyplot as plt

t_index = 100

# --------------------------------------------------------------------------- #
# Recover reference from interpolated HDF files
ts = TimeSerie(paths['ihdf'])
UrefHDF = ts.data[t_index]
UrefHDF.openHdfFile()
Uref = getattr(UrefHDF, options['dataname'])[:]
UrefHDF.closeHdfFile()

# --------------------------------------------------------------------------- #
# Recover mean and fluc
mfHDF = HDFReader(paths['meanfluc'])
mfHDF.openHdfFile()
mean = mfHDF.mean[:]
fluc = mfHDF.fluc[:]
mfHDF.closeHdfFile()

Ufluc = fluc[:, :, t_index]

# --------------------------------------------------------------------------- #
# Recover coefficients
coeffsHDF = HDFReader(paths['coeffs'])
coeffsHDF.openHdfFile()
coeffs = coeffsHDF.coeffs[:]
coeffsHDF.closeHdfFile()

# --------------------------------------------------------------------------- #
# Recover basis
basisHDF = HDFReader(paths['basis'])
basisHDF.openHdfFile()
basis = basisHDF.basis[:]
basisHDF.closeHdfFile()

# --------------------------------------------------------------------------- #
# Reconstruct

c = np.einsum('xcm,xc->m', basis, Ufluc)

Uold = np.einsum('xcm,m->xc', basis, coeffs[t_index, :])

Unew = np.einsum('xcm,m->xc', basis, c)

data_to_plot = [Ufluc+mean, Unew+mean, Uold+mean]

# --------------------------------------------------------------------------- #
# Recover mesh
gridHDF = HDFReader(paths['grid'])
gridHDF.openHdfFile()
grid_shape = gridHDF.shape[:].flatten()
gridHDF.closeHdfFile()

# --------------------------------------------------------------------------- #
# PLOT

plt.close('all')

plots.plot2d(misc.concatenate_in_given_axis(data_to_plot, 2), grid_shape)
