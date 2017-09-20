#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 15:56:49 2017

@author: root
"""

import os
from lasie_rom import io, grids
from lasie_rom.misc.tools import concatenate_in_given_axis
import numpy as np


# =========================================================================== #
# Parameters

H = 0.1

box_limits = [(0, 1), (-2, 0), (-1, 2)]

# define vtu path
here = os.path.realpath(__file__)[:os.path.realpath(__file__).rfind(os.sep)]
path = os.path.join(here, 'test.vtu')


# =========================================================================== #
# Grid/Mesh

grid, h = grids.generate(box_limits, h=H)
gshape = grid.shape
mesh = grids.to_mesh(grid)


# =========================================================================== #
#%% DATA

data = dict()

# scalar
scalar = 1
for i, x in enumerate(mesh.T):
    scalar *= np.sin(2*np.pi*(i+1)*x)
scalar = scalar[:, np.newaxis]
scalar_name = 'my_scalar'
data.update({scalar_name: scalar})

# vector
values_x = 1
values_y = 1
values_z = 1
for i, x in enumerate(mesh.T):
    values_x *= np.sin(2*np.pi*(i+1)*x)
    values_y *= np.sin(2*np.pi*(i+1)*x)
    values_z *= np.sin(2*np.pi*(i+1)*x)

vector = concatenate_in_given_axis((values_x, values_y, values_z), 1)
vector_name = 'my_vector'
data.update({vector_name: vector})

# =========================================================================== #
#%% Wrtie vtu

io.vtk.write(mesh, gshape, data, path)

# =========================================================================== #
# READ vtu

data_read = io.vtk.read(path)

# =========================================================================== #
# TEST

for k in data.keys():
    if not all((data_read[k] == data[k]).flatten()):
        raise NotImplementedError('Error in READ/WRITE of .vtu files')
