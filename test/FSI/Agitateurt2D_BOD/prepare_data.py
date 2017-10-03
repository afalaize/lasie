#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:39:38 2017

@author: afalaize
"""

from lasie_rom.io import hdf, pvd2hdf
from lasie_rom.interpolation import interp_timeserie_in_hdf
from lasie_rom import grids
from lasie_rom.classes import TimeSerie

import numpy

from set_parameters import parameters
from set_locations import paths
import time


# --------------------------------------------------------------------------- #
# convert all .vtu files listed in the .pvd file to new .hdf files
for i, path_to_pvd in enumerate(paths['pvd']):
    pvd2hdf(path_to_pvd, paths['hdf'][i], **parameters['load'])

# --------------------------------------------------------------------------- #
# build grid and corresponding mesh for interpolation

# Read the .hdf files with a TimeSerie object
ts = TimeSerie(paths['hdf'][0])

# Retrieve spatial domain limits (minmax)
d = ts.data[0]
d.openHdfFile()
minmax = d.getMeshMinMax()
d.openHdfFile()

# build grid
grid, h = grids.generate(minmax, h=parameters['h_mesh'])

# reorganize grid as mesh
mesh = grids.to_mesh(grid)

data = {'mesh': mesh,
        'grid': grid,
        'shape': numpy.array(grid.shape)[:, numpy.newaxis],
        'h': numpy.array(h)[:, numpy.newaxis]}

# store hdf file associated to the reference mesh
hdf.data2hdf(data, paths['grid'])

# --------------------------------------------------------------------------- #
# interpolate all data from .hdf files and stores the results in new hdf files

# Read the .hdf files with a TimeSerie object
for i, path_to_hdf in enumerate(paths['hdf']):
    ts = TimeSerie(path_to_hdf)
    ts.openAllFiles()
    grid_hdf = hdf.HDFReader(paths['grid'])
    grid_hdf.openHdfFile()
    interp_timeserie_in_hdf(ts, grid_hdf.mesh[:],
                            paths['ihdf'][i])
    grid_hdf.closeHdfFile()
    ts.closeAllFiles()
    time.sleep(2)
