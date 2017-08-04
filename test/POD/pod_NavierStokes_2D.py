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
STAB = 0

# folder that contains the .vtu files to read
VTU_FOLDER = '/home/afalaize/dev/python/lasie_rom/test/FENICS/cylindre_fixe/RESULTS'

# options to control which snapshots are read
LOAD = {'imin': 150,     # starting index
        'imax': 250,    # stoping index
        'decim': 1}     # read one snapshot over decim snapshots


# Regular grid space-step
H = 0.1

# Name of the data to retrieve from the vtu files
DATA_NAME = 'velocity'

# name of the .pvd file in VTU_FOLDER that summarize the .vtu files
PVD_NAME = 'velocity.pvd'

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
#HDF_FOLDER = '/Users/Falaize/Desktop/TEMP_CEMEF_30mai2017/cylindre_thost'

# folder where to write the interpolated hdf files (created in VTU_FOLDER)
INTERP_HDF_FOLDER = os.path.join(HDF_FOLDER, INTERP_HDF_FOLDER_NAME)

# Path to the hdf file that corresponds to the interpolation grid
BASIS_HDF_PATH = os.path.join(INTERP_HDF_FOLDER, BASIS_HDF_NAME)

# Path to the hdf file that corresponds to the POD basis of the snapshots
GRID_HDF_PATH = os.path.join(INTERP_HDF_FOLDER, GRID_HDF_NAME)

# Path to the hdf file that corresponds to the mean field of the snapshots
MEANFLUC_HDF_PATH = os.path.join(INTERP_HDF_FOLDER, MEANFLUC_HDF_NAME)

# Path to the hdf file that contains the ROM matrices
ROM_MATRICES_HDF_PATH = os.path.join(INTERP_HDF_FOLDER, ROM_MATRICES_HDF_NAME)

# Path to the hdf file that contains the ROM matrices
ORIGINAL_COEFFS_HDF_PATH = os.path.join(INTERP_HDF_FOLDER, ORIGINAL_COEFFS_HDF_NAME)

# =========================================================================== #

# convert all .vtu listed in the .pvd file to hdf files
if True:
    pvd2hdf(PVD_PATH, HDF_FOLDER, **LOAD)

# =========================================================================== #

# build grid and corresponding mesh for interpolation
if True:
    # Read the .hdf files with a TimeSerie object
    ts = TimeSerie(HDF_FOLDER)

    # Retrieve spatial domain limits (minmax)
    d = ts.data[0]
    d.openHdfFile()
    minmax = d.getMeshMinMax()

    # build grid
    grid, h = grids.generate(minmax, h=H)

    # reorganize grid as mesh
    mesh = grids.to_mesh(grid)

    data = {'mesh': mesh,
            'grid': grid,
            'shape': numpy.array(grid.shape)[:, numpy.newaxis],
            'h': numpy.array(h)[:, numpy.newaxis]}

    hdf.data2hdf(data, GRID_HDF_PATH)

# =========================================================================== #

# interpolate all data from .hdf files to mesh and stores the results in new hdf files
if True:
    # Read the .hdf files with a TimeSerie object
    ts = TimeSerie(HDF_FOLDER)
    grid_hdf = hdf.HDFReader(GRID_HDF_PATH)
    grid_hdf.openHdfFile()
    interp_timeserie_in_hdf(ts, grid_hdf.mesh[:], INTERP_HDF_FOLDER)
    grid_hdf.closeHdfFile()

# =========================================================================== #

# split mean from fluctuating snapshots fields
if True:
    # instanciate TimeSerie
    ts = TimeSerie(INTERP_HDF_FOLDER)
    # Open hdf files
    ts.openAllFiles()
    # Form snapshot matrix
    U = ts.concatenate('f_17')
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
if True:

    mean_hdf = hdf.HDFReader(MEANFLUC_HDF_PATH)
    mean_hdf.openHdfFile()

    basis = pod.compute_basis(mean_hdf.fluc[:], threshold=1e-9)

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
    plots.plot2d(basis.basis[:, :, :8],
                 grid.shape[:, 0],
                 render='magnitude', options={'ncols': 2})
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
        for u in ts.generator('f_17')():
            yield numpy.einsum('xc,xci->i', u, basis_hdf.basis[:])

    coeffs = misc.concatenate_in_given_axis(coeffs(), 0)

    data = {'coeffs': coeffs,}

    # write hdf for rom matrices
    hdf.data2hdf(data, ORIGINAL_COEFFS_HDF_PATH)

    # Close hdf files
    basis_hdf.closeHdfFile()
    ts.closeAllFiles()

# =========================================================================== #

# Run ROM:
if True:
    # instanciate TimeSerie
    ts = TimeSerie(INTERP_HDF_FOLDER)
    # Open hdf files
    ts.openAllFiles()
    # Form snapshot matrix
    U_fom = ts.concatenate('f_17').copy()
    # Close hdf files
    ts.closeAllFiles()

    # Time-step (s)
    DT = 0.01

    # Simulation time steps run from 0 to TEND (s)
    TEND = (U_fom.shape[-1]-1)*DT

    paths = {'basis': BASIS_HDF_PATH,
             'matrices': ROM_MATRICES_HDF_PATH,
             'original_coeffs': ORIGINAL_COEFFS_HDF_PATH,
             'meanfluc': MEANFLUC_HDF_PATH}

    rom_ns = rom.navierstokes.ReducedOrderModel(paths)
    rom_ns.run(dt=DT, tend=TEND)

    U_rom = rom_ns.reconstruction()

    # recover grid infos
    grid_hdf = hdf.HDFReader(GRID_HDF_PATH)
    grid_hdf.openHdfFile()
    grid_shape = grid_hdf.shape[:][:, 0]
    grid_h = grid_hdf.h[:][:, 0]
    mesh = grid_hdf.mesh[:]
    grid_hdf.closeHdfFile()

    from lasie_rom import io, parallelization

    # name of the folder where to write the generated .vtu files
    OUTPUT_VTU_FOLDER_NAME = 'vtu_ROM'

    # path to the folder where to write the generated .vtu files
    VTU_FOLDER_PATH = os.path.join(VTU_FOLDER, OUTPUT_VTU_FOLDER_NAME)

    if not os.path.exists(VTU_FOLDER_PATH):
        os.mkdir(VTU_FOLDER_PATH)

    def write_vtu(i):
        print('write vtu #{}'.format(i))
        urom, ufom = U_rom[:, :, i], U_fom[:, :, i]
        data = {'vitesse_rom': urom,
                'vitesse_fom': ufom,
                'error': urom-ufom}
        path = os.path.join(VTU_FOLDER_PATH, 'rom_{}.vtu'.format(i))
        io.vtk.write(mesh, grid_shape, data, path)

    parallelization.map(write_vtu, range(U_fom.shape[-1]))

    rom_ns.close_hdfs()
