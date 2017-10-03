#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 14:22:11 2017

@author: afalaize
"""

import os
import numpy as np
import sympy as sy

from lasie_rom.io import hdf, vtk
from lasie_rom.classes import TimeSerie

from _0_parameters import parameters
from _0_locations import paths

from lasie_rom import parallelization, rom

from lasie_rom.misc.tools import sympyExpr2numpyFunc
from ellipse.ellipse_levelset import build_Levelset_Sympy_Expression

def build_levelset_func(ell_center, ell_radius, rot_center):
    levelset = build_Levelset_Sympy_Expression(ell_center,
                                               ell_radius,
                                               rot_center)
    x = sy.symbols('x:2', real=True)
    theta_symb = sy.symbols('theta', real=True, positive=True)
    t_symb = sy.symbols('t', real=True, positive=True)
    args = [t_symb] + list(x)

    subs = {theta_symb: parameters['theta_init'] + (parameters['load']['tmin']+t_symb)*parameters['angular_vel']}

    return sympyExpr2numpyFunc(levelset, args, subs)

levelset_func = build_levelset_func(parameters['ell_center'],
                                    parameters['ell_radius'],
                                    parameters['rot_center'])

# --------------------------------------------------------------------------- #
# Instanciate Reduced order model for Navier Stokes
hdf_paths = {'basis': paths['basis'][0],
             'matrices': paths['matrices'],
             'original_coeffs': paths['coeffs'],
             'meanfluc':  paths['meanfluc'][0],
             'grid': paths['grid']
             }

rom_ns = rom.fsi_relaxed_rigidity.ReducedOrderModel(hdf_paths,
                                                    parameters,
                                                    levelset_func)

# instanciate original TimeSerie
ts = TimeSerie(paths['ihdf'][0])

# Define dataname
if parameters['dataname']['hdf'] is not None:
    dataname = parameters['dataname']['hdf']
else:
    d = ts.data[0]
    d.openHdfFile()
    args = dir(d)
    temp = [a.startswith('f_') for a in args]
    dataname = args[temp.index(True)]
    d.closeHdfFile()

# Form snapshot matrix
U_fom = ts.concatenate(dataname)

# --------------------------------------------------------------------------- #
# Run ROM

# Simulation time steps run from 0 to TEND (s)
T_rom = U_fom.shape[-1]*parameters['dt']*parameters['nb_export']

rom_ns.open_hdfs()
angle = parameters['angular_vel']*parameters['load']['tmin'] % 2*np.pi

rom_ns.run(dt=parameters['rom']['dt'], tend=T_rom)
#rom_ns.run(dt=parameters['rom']['dt'], tend=T_rom, angle=angle)

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

def myplot(i):
    import matplotlib.pyplot as plt
    plt.close('all')
    plt.plot(rom_ns.c_rom(i), '-o', label='rom')
    plt.plot(rom_ns.c_fom(i)[:len(rom_ns.c_rom(i))], '--o', label='fom')
    plt.legend(loc=0)
    plt.title(str(i))
    plt.show()

myplot(0)
myplot(1)
myplot(2)
myplot(3)
