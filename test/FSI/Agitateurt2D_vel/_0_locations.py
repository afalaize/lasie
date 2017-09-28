#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:42:17 2017

@author: afalaize
"""

import os
from _0_parameters import parameters

# ----------------------------------------------------------------------- #
def build_resultsFolderName(parameters):
    resultsFolderName = "results"
    resultsFolderName += "_radius={0}".format(parameters['lambda'][0])
    resultsFolderName += "_shape={0}".format(parameters['lambda'][1])
    resultsFolderName += "_excentr={0}".format(parameters['lambda'][2])
    resultsFolderName += "_Re={0}".format(parameters['lambda'][3])
    resultsFolderName += "_eps_tanh={0}".format(parameters['eps_tanh'])
    resultsFolderName += "_mesh={0}X{0}".format(int(parameters['h_mesh']**-1))
    return resultsFolderName

resultsFolderName = build_resultsFolderName(parameters)

# Principal folder (where are stored the .vtu files associated with Fenics simulation result)
VTU_FOLDER = resultsFolderName

# name of the folder where to store new data (POD basis, ROM results, figures, etc.)
#MAIN_FOLDER = os.path.join(VTU_FOLDER, 'POD_Results')
MAIN_FOLDER = '/Users/afalaize/Developement/hdfs_re500'

# name of the .pvd file in VTU_FOLDER that summarize the .vtu files (the .pvd extension is appended when needed)
PVD_NAMES = ['velocity', 'fnchar', 'levelset']

# name of folder where to write the hdf version of vtu files (created in MAIN_FOLDER)
HDF_FOLDER_NAME = 'hdf'

# folder where to write the interpolated hdf files (created in MAIN_FOLDER)
INTERP_HDF_FOLDER_NAME = 'hdf_interpolated'

# name of the hdf file that corresponds to the interpolation grid (created in MAIN_FOLDER_NAME)
GRID_HDF_NAME = 'gridmesh'

# name of the hdf file that corresponds to the mean field of the snapshots
MEANFLUC_HDF_NAME = 'meanfluc'

# prefix name of the hdf file that corresponds to the POD basis of the snapshots
BASIS_HDF_NAME = 'podbasis'

# name of the hdf file that corresponds to the POD basis of the snapshots
ROM_MATRICES_HDF_NAME = 'rom_matrices'

# name of the hdf file for the basis coefficients computed from the snapshots
ORIGINAL_COEFFS_HDF_NAME = 'original_coeffs'

# --------------------------------------------------------------------------- #

# path to the .pvd file to read
PVD_PATH = list(map(lambda name: os.path.join(VTU_FOLDER, name+'.pvd'),
                    PVD_NAMES))

# folder where to write the generated hdf files (created in VTU_FOLDER)
HDF_FOLDERS = list(map(lambda name: os.path.join(MAIN_FOLDER, HDF_FOLDER_NAME, name), PVD_NAMES))

# folder where to write the interpolated hdf files (created in VTU_FOLDER)
INTERP_HDF_FOLDERS = list(map(lambda name: os.path.join(MAIN_FOLDER, INTERP_HDF_FOLDER_NAME, name), PVD_NAMES))

# Path to the hdf file that corresponds to the POD basis of the snapshots
PODBASIS_HDF_PATHS = list(map(lambda name: os.path.join(MAIN_FOLDER, BASIS_HDF_NAME+'_'+name+'.hdf'), PVD_NAMES))

# Path to the hdf file that corresponds to the interpolation grid
GRID_HDF_PATH = os.path.join(MAIN_FOLDER, GRID_HDF_NAME+'.hdf')

# Path to the hdf file that corresponds to the mean field of the snapshots
MEANFLUC_HDF_PATHS = list(map(lambda name: os.path.join(MAIN_FOLDER, MEANFLUC_HDF_NAME+'_'+name+'.hdf'), PVD_NAMES))

# Path to the hdf file that contains the ROM matrices
ROM_MATRICES_HDF_PATH = os.path.join(MAIN_FOLDER, ROM_MATRICES_HDF_NAME+'.hdf')

# Path to the hdf file that contains the ROM matrices
ORIGINAL_COEFFS_HDF_PATH = os.path.join(MAIN_FOLDER, ORIGINAL_COEFFS_HDF_NAME+'.hdf')

# folder where to save the ROM Snapshots
OUT_FOLDER = os.path.join(MAIN_FOLDER, 'ROM_Results')


paths = {'pvd': PVD_PATH,
         'hdf': HDF_FOLDERS,
         'ihdf': INTERP_HDF_FOLDERS,
         'basis': PODBASIS_HDF_PATHS,
         'grid': GRID_HDF_PATH,
         'meanfluc': MEANFLUC_HDF_PATHS,
         'matrices': ROM_MATRICES_HDF_PATH,
         'coeffs': ORIGINAL_COEFFS_HDF_PATH,
         'results': MAIN_FOLDER,
         'output': OUT_FOLDER
         }

# ----------------------------------------------------------------------- #
# Save parameters
def save_parameters():
    textFileName = "parameters.txt"
    with open(os.path.join(paths['results'], textFileName),
              mode='w') as f:
        f.write("{\n")
        for k in parameters.keys():
            f.write("{0}': {1},\n".format(k, parameters[k]))
        f.write("{\n")

save_parameters()
