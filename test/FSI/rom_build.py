#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:30:38 2017

@author: afalaize
"""

from lasie_rom import io, rom, misc, parallelization
from lasie_rom.classes import TimeSerie

from main import parameters
from locations import paths
from options import options

import numpy


# Build ROM Matrices:
basis_hdf = io.hdf.HDFReader(paths['basis'], openFile=True)
mean_hdf = io.hdf.HDFReader(paths['meanfluc'], openFile=True)

array_A = rom.navierstokes.A(basis_hdf.basis[:])
array_B = rom.navierstokes.B(basis_hdf.basis[:], basis_hdf.basis_grad[:],
                             mean_hdf.mean[:], mean_hdf.mean_grad[:],
                             parameters['nu'], parameters['rho'],
                             options['rom']['stab'])
array_C = rom.navierstokes.C(basis_hdf.basis[:], basis_hdf.basis_grad[:])
array_F = rom.navierstokes.F(basis_hdf.basis[:], basis_hdf.basis_grad[:],
                             mean_hdf.mean[:], mean_hdf.mean_grad[:],
                             parameters['nu'], parameters['rho'],
                             options['rom']['stab'])
data = {'a': array_A,
        'b': array_B,
        'c': array_C,
        'f': array_F}

# write hdf for rom matrices
io.hdf.data2hdf(data, paths['matrices'])

# instanciate TimeSerie
ts = TimeSerie(paths['ihdf'])

# Open hdf files
ts.openAllFiles()

# instanciate TimeSerie
mfHDF = io.hdf.HDFReader(paths['meanfluc'])
mfHDF.openHdfFile()
mean = mfHDF.mean[:]
mfHDF.closeHdfFile()


basis_matrix = basis_hdf.basis[:]

def coefficients():
    for u in ts.generator(options['dataname'])():
        yield numpy.einsum('xc,xci->i', u-mean, basis_matrix)

coeffs = misc.concatenate_in_given_axis(coefficients(), 0)

data = {'coeffs': coeffs}

# write hdf for rom matrices
io.hdf.data2hdf(data, paths['coeffs'])

# Close hdf files
basis_hdf.closeHdfFile()
mean_hdf.closeHdfFile()
ts.closeAllFiles()

# Instanciate Reduced order model for Navier Stokes
rom_paths = {'basis': paths['basis'],
             'matrices': paths['matrices'],
             'original_coeffs': paths['coeffs'],
             'meanfluc':  paths['meanfluc']
             }

rom_ns = rom.navierstokes.ReducedOrderModel(rom_paths)
