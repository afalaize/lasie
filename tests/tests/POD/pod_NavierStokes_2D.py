#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:58:08 2017

@author: root
"""

from __future__ import absolute_import, division, print_function


import os
import numpy

from lasie_rom.io import hdf, pvd2hdf

from lasie_rom.classes import TimeSerie
from lasie_rom.interpolation import interp_timeserie_in_hdf

from lasie_rom import grids, pod, plots, rom, operators, misc

# Density
RHO = 1

# Dynamical viscosity
MU = 0.01

# Stabilisation coefficient
STAB = 0.2


# folder that contains the .vtu files to read
VTU_FOLDER = '/media/afalaize/DATA1/TESTS_THOST/170407_cylindre2D_SCC_windows/Results'

# options to control which snapshots are read
LOAD = {'imin': 20,     # starting index
        'imax': 270,    # stoping index 
        'decim': 1}     # read one snapshot over decim snapshots

# name of the .pvd file in VTU_FOLDER that summarize the .vtu files
PVD_NAME = 'ThostA.pvd'

# name of the folder where to write the generated hdf files (created in VTU_FOLDER)
HDF_FOLDER_NAME = 'hdf5'

# folder where to write the interpolated hdf files (this folder is created in VTU_FOLDER)
INTERP_HDF_FOLDER_NAME = 'hdf5_interpolated'

# name of the hdf file that corresponds to the interpolation grid
GRID_HDF_NAME = 'gridmesh.hdf5'

# name of the hdf file that corresponds to the mean field of the snapshots
MEANFLUC_HDF_NAME = 'meanfluc.hdf5'

# name of the hdf file that corresponds to the POD basis of the snapshots
BASIS_HDF_NAME = 'basis.hdf5'

# name of the hdf file that corresponds to the POD basis of the snapshots
ROM_MATRICES_HDF_NAME = 'rom.hdf5'

# name of the hdf file for the basis coefficients computed from the snapshots
ORIGINAL_COEFFS_HDF_NAME = 'original_coeffs.hdf5'

# =========================================================================== #

# path to the .pvd file to read
PVD_PATH = os.path.join(VTU_FOLDER, PVD_NAME)

# folder where to write the generated hdf files (created in VTU_FOLDER)
HDF_FOLDER = os.path.join(VTU_FOLDER, HDF_FOLDER_NAME)

# folder where to write the interpolated hdf files (created in VTU_FOLDER)
INTERP_HDF_FOLDER = os.path.join(VTU_FOLDER, INTERP_HDF_FOLDER_NAME)

# Path to the hdf file that corresponds to the interpolation grid
BASIS_HDF_PATH = os.path.join(INTERP_HDF_FOLDER, BASIS_HDF_NAME)

# Path to the hdf file that corresponds to the POD basis of the snapshots
GRID_HDF_PATH = os.path.join(INTERP_HDF_FOLDER, GRID_HDF_NAME)

# Path to the hdf file that corresponds to the mean field of the snapshots
MEANFLUC_HDF_PATH = os.path.join(INTERP_HDF_FOLDER, MEANFLUC_HDF_NAME)

# Path to the hdf file that contains the ROM matrices
ROM_MATRICES_HDF_PATH = os.path.join(INTERP_HDF_FOLDER, ROM_MATRICES_HDF_NAME)

# Path to the hdf file that contains the ROM matrices
ROM_MATRICES_HDF_PATH = os.path.join(INTERP_HDF_FOLDER, ROM_MATRICES_HDF_NAME)

# Path to the hdf file that contains the ROM matrices
ORIGINAL_COEFFS_HDF_PATH = os.path.join(INTERP_HDF_FOLDER, ORIGINAL_COEFFS_HDF_NAME)
# =========================================================================== #

# convert all .vtu listed in the .pvd file to hdf files
if False:
    pvd2hdf(PVD_PATH, HDF_FOLDER, **LOAD)

# =========================================================================== #

# build grid and corresponding mesh for interpolation
if False:
    # Read the .hdf files with a TimeSerie object
    ts = TimeSerie(HDF_FOLDER)
    
    # Retrieve spatial domain limits (minmax)
    d = ts.data[0]
    d.openHdfFile()
    minmax = d.getMeshMinMax()
    
    # build grid
    grid, h = grids.generate(minmax, h=0.005)
    
    # reorganize grid as mesh
    mesh = grids.to_mesh(grid)
    
    data = {'mesh': mesh,
            'grid': grid,
            'shape': numpy.array(grid.shape)[:, numpy.newaxis],
            'h': numpy.array(h)[:, numpy.newaxis]}
    
    hdf.data2hdf(data, GRID_HDF_PATH)

# =========================================================================== #

# interpolate all data from .hdf files to mesh and stores the results in new hdf files
if False:
    interp_timeserie_in_hdf(ts, mesh, INTERP_HDF_FOLDER)

# =========================================================================== #

# split mean from fluctuating snapshots fields
if False:
    # instanciate TimeSerie
    ts = TimeSerie(INTERP_HDF_FOLDER)
    # Open hdf files
    ts.openAllFiles()
    # Form snapshot matrix
    U = ts.concatenate('vitesse')
    # Compute mean and fluctuations
    mean, fluc = pod.meanfluc(U)
    # Close hdf files
    ts.closeAllFiles()
    
    # recover grid infos
    grid_hdf = hdf.HDFReader(GRID_HDF_PATH)
    grid_hdf.openHdfFile()
    grid_shape = grid_hdf.shape[:][:, 0]
    grid_h = grid_hdf.h[:][:, 0]

    # Compute mean gradient
    mean_grad = operators.gridgradient(mean, grid_shape, grid_h)

    # define data
    data = {'mesh': grid_hdf.mesh[:],
            'mean': mean,
            'mean_grad': mean_grad,
            'fluc': fluc,}
    
    # Close hdf file for grid
    grid_hdf.closeHdfFile()

    # write hdf for mean and fluct
    hdf.data2hdf(data, MEANFLUC_HDF_PATH)

    
# =========================================================================== #

# Compute POD basis
if False:
    
    mean_hdf = hdf.HDFReader(MEANFLUC_HDF_PATH)
    mean_hdf.openHdfFile()  

    basis = pod.compute_basis(mean_hdf.fluc[:])
    
    mean_hdf.closeHdfFile()

    # recover grid infos
    grid_hdf = hdf.HDFReader(GRID_HDF_PATH)
    grid_hdf.openHdfFile()
    grid_shape = grid_hdf.shape[:][:, 0]
    grid_h = grid_hdf.h[:][:, 0]

    # Compute mean gradient
    def compute_grad(a):
        return operators.gridgradient(a, grid_shape, grid_h)
    
    def grad_generator():
        for a in misc.iterarray(basis, 2):
            yield compute_grad(a)
            
    basis_grad = misc.concatenate_in_given_axis(grad_generator(), 3)
    
    data = {'mesh': grid_hdf.mesh[:],
            'basis': basis,
            'basis_grad': basis_grad}

    # Close hdf file for grid
    grid_hdf.closeHdfFile()

    # write hdf for mean and fluct
    hdf.data2hdf(data, BASIS_HDF_PATH)

# =========================================================================== #

# Plot 2D POD basis
if False:
    grid = hdf.HDFReader(GRID_HDF_PATH, openFile=True)
    basis = hdf.HDFReader(BASIS_HDF_PATH, openFile=True)
    plots.plot2d(basis.get_single_data()[:, :, :8], 
                 grid.shape[:, 0], 
                 render=0, options={'ncols':2})
    grid.closeHdfFile()
    basis.closeHdfFile()

# =========================================================================== #

# Build ROM Matrices:
if True:
    basis_hdf = hdf.HDFReader(BASIS_HDF_PATH, openFile=True)
    mean_hdf = hdf.HDFReader(MEANFLUC_HDF_PATH, openFile=True)
    
    array_A = rom.navierstokes.A(basis_hdf.basis[:])
    array_B = rom.navierstokes.B(basis_hdf.basis[:], basis_hdf.basis_grad[:],
                    mean_hdf.mean[:], mean_hdf.mean_grad[:],
                    MU, RHO, STAB)
    array_C = rom.navierstokes.C(basis_hdf.basis[:], basis_hdf.basis_grad[:])
    array_F = rom.navierstokes.F(basis_hdf.basis[:], basis_hdf.basis_grad[:],
                    mean_hdf.mean[:], mean_hdf.mean_grad[:],
                    MU, RHO, STAB)
    data = {'a': array_A,
            'b': array_B,
            'c': array_C,
            'f': array_F}
    
    # write hdf for rom matrices
    hdf.data2hdf(data, ROM_MATRICES_HDF_PATH)

    
# =========================================================================== #

# Compute snapshots coefficients
if True:

    basis_hdf = hdf.HDFReader(BASIS_HDF_PATH, openFile=True)

    # instanciate TimeSerie
    ts = TimeSerie(INTERP_HDF_FOLDER)
    # Open hdf files
    ts.openAllFiles()
    
    def coeffs():
        for u in ts.generator('vitesse')():
            yield numpy.einsum('xc,xci->i', u, basis_hdf.basis[:])
            
    coeffs = misc.concatenate_in_given_axis(coeffs(), 0)

    data = {'coeffs': coeffs,}
    
    # write hdf for rom matrices
    hdf.data2hdf(data, ORIGINAL_COEFFS_HDF_PATH)
    
    # Close hdf files
    basis_hdf.closeHdfFile()
    ts.closeAllFiles()
    
# =========================================================================== #

# Build ROM Matrices:
if True:
    paths = {'basis': BASIS_HDF_PATH,
             'rom_matrices': ROM_MATRICES_HDF_PATH,
             'original_coeffs': ORIGINAL_COEFFS_HDF_PATH}
    
    rom_ns = rom.navierstokes.ReducedOrderModel(paths)
    rom_ns.run()
    
    U_rom = rom_ns.reconstruction()

    # instanciate TimeSerie
    ts = TimeSerie(INTERP_HDF_FOLDER)
    # Open hdf files
    ts.openAllFiles()
    # Form snapshot matrix
    U_fom = ts.concatenate('vitesse').copy()
    # Close hdf files
    ts.closeAllFiles()

    # recover grid infos
    grid_hdf = hdf.HDFReader(GRID_HDF_PATH)
    grid_hdf.openHdfFile()
    grid_shape = grid_hdf.shape[:][:, 0]
    grid_h = grid_hdf.h[:][:, 0]
    grid_hdf.closeHdfFile()
    
    render = 0
    plots.plot2d(U_fom[:, :, 0:250:int(250/5.)+1],
                grid_shape, options={'ncols':1}, render=render, 
                title='Full order model')

    plots.plot2d(U_rom[:, :, 0:250:int(250/5.)+1],
                grid_shape, options={'ncols':1}, render=render, 
                title='Reduced order model')

    plots.plot2d((U_rom-U_fom)[:, :, 0:250:int(250/5.)+1],
                grid_shape, options={'ncols':1}, render=render, 
                title='Error')
