#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:22:11 2017

@author: afalaize
"""

import os
import matplotlib.pyplot as plt

from lasie_rom.io import hdf, vtk
from rom_build import rom_ns
from lasie_rom.classes import TimeSerie

from options import options
from locations import paths

from lasie_rom import parallelization


# --------------------------------------------------------------------------- #
# instanciate original TimeSerie
ts = TimeSerie(paths['ihdf'])
# Open hdf files
ts.openAllFiles()
# Form snapshot matrix
U_fom = ts.concatenate(options['dataname'])
# Close hdf files
ts.closeAllFiles()


# --------------------------------------------------------------------------- #
# Run ROM

# Simulation time steps run from 0 to TEND (s)
T_rom = (U_fom.shape[-1]-1)*options['rom']['dt']

rom_ns.open_hdfs()
rom_ns.run(dt=options['rom']['dt'], tend=T_rom)

# --------------------------------------------------------------------------- #

U_rom = rom_ns.reconstruction()

# --------------------------------------------------------------------------- #
# recover grid infos
grid_hdf = hdf.HDFReader(paths['grid'])
grid_hdf.openHdfFile()
grid_shape = grid_hdf.shape[:][:, 0]
mesh = grid_hdf.mesh[:]
grid_hdf.closeHdfFile()


# --------------------------------------------------------------------------- #
# write vtu results
if not os.path.exists(paths['output']):
    os.mkdir(paths['output'])


def write_vtu(i):
    "write vtu from a given index"
    print('write vtu #{}'.format(i))
    ufom = U_fom[:, :, i]
    urom = U_rom[:, :, i]
    data = {'vitesse_rom': urom,
            'vitesse_fom': ufom,
            'error': urom-ufom}
    vtk.write(mesh, grid_shape, data,
              os.path.join(paths['output'], 'output{}.vtu'.format(i)))

parallelization.map(write_vtu, range(U_fom.shape[-1]))

plt.figure()
plt.plot(rom_ns.c_rom(0), label='rom')
plt.plot(rom_ns.c_fom(0), label='fom')
plt.legend()

rom_ns.close_hdfs()
