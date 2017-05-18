#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 11:58:08 2017

@author: root
"""

from __future__ import absolute_import, division, print_function

from lasie_rom.io.vtu2hdf import pvd2Hdf
from lasie_rom.io.hdf.write import dumpArrays2Hdf
from lasie_rom.io.hdf.read import HDFReader
from lasie_rom.classes import TimeSerie
from lasie_rom.interpolation import interpTimeSerieToHdf
from lasie_rom import grids 
from lasie_rom import pod
from lasie_rom import plots
import os
import numpy


# folder that contains the .vtu files to read
VTU_FOLDER = '/media/afalaize/DATA1/TESTS_THOST/170407_cylindre2D_SCC_windows/Results'

# name of the data to read in each .vtu file
DATA_NAMES = (r'Vitesse(m/s)', 
              r'MasseVolumique(kg/m3)',
              r'Eta')

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
MESH_HDF_NAME = 'gridmesh.hdf5'

# name of the hdf file that corresponds to the POD basis of the snapshots
BASIS_HDF_NAME = 'basis.hdf5'

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
MESH_HDF_PATH = os.path.join(INTERP_HDF_FOLDER, MESH_HDF_NAME)


# =========================================================================== #

# convert all .vtu listed in the .pvd file to hdf files
if False:
    pvd2Hdf(PVD_PATH, HDF_FOLDER, DATA_NAMES, **LOAD)

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
    
    dumpArrays2Hdf((mesh, grid, numpy.array(grid.shape)[:, numpy.newaxis]), 
                   ('mesh', 'grid', 'shape'), MESH_HDF_PATH)

# =========================================================================== #

# interpolate all data from .hdf files to mesh and stores the results in new hdf files
if False:
    interpTimeSerieToHdf(ts, mesh, INTERP_HDF_FOLDER)

# =========================================================================== #

# Compute POD basis
if True:
    ts = TimeSerie(INTERP_HDF_FOLDER)
    ts.openAllFiles()
    mean, fluc = pod.meanfluc(ts.generator('vitesse'))
    basis = pod.compute_basis(fluc)
    dumpArrays2Hdf((basis, ), 
                   ('basis', ), BASIS_HDF_PATH)

# =========================================================================== #

# Plot 2D POD basis
if True:
    grid = HDFReader(MESH_HDF_PATH, openFile=True)
    basis = HDFReader(BASIS_HDF_PATH, openFile=True)
    plots.plot2d(basis.get_single_data()[:, :12, :], 
                 grid.shape[:, 0], 
                 render=0)
    grid.closeHdfFile()
    basis.closeHdfFile()

