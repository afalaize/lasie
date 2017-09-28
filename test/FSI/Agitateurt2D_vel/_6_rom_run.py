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

from lasie_rom.misc import concatenate_in_given_axis

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

# %%
def build_velocity_func(angular_vel, center):
    x = sy.symbols('x:2', real=True)
    b = sy.symbols('b:2', real=True)
    omega = sy.symbols('omega', real=True, positive=True)

    theta = sy.atan2((x[1]-b[1]), (x[0]-b[0]))

    x_vec = sy.Matrix(x)
    b_vec = sy.Matrix(b)

    dummy_vec = sy.Matrix([-sy.sin(theta), sy.cos(theta)])
    velocity_vec = omega*sy.sqrt((x_vec-b_vec).dot(x_vec-b_vec))*dummy_vec

    subs = dict([(bi, vali) for (bi, vali) in zip(b, center)])
    subs.update({omega: angular_vel})

    velocity_vec = list(velocity_vec.subs(subs))
    args = list(x)

    func_vec = [sympyExpr2numpyFunc(v, args, None) for v in velocity_vec]

    def func(*fargs):
        return np.vstack([f(*fargs) for f in func_vec]).T
    return func
velocity_func = build_velocity_func(parameters['angular_vel'],
                                    parameters['rot_center'])

#%%
# --------------------------------------------------------------------------- #

# Instanciate Reduced order model for Navier Stokes
hdf_paths = {'basis': paths['basis'][0],
             'matrices': paths['matrices'],
             'original_coeffs': paths['coeffs'],
             'meanfluc':  paths['meanfluc'][0],
             'grid': paths['grid']
             }

rom_ns = rom.fsi_relaxed_velocity.ReducedOrderModel(hdf_paths,
                                                    parameters,
                                                    levelset_func,
                                                    velocity_func)

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

#%%
# --------------------------------------------------------------------------- #
# Run ROM

# Simulation time steps run from 0 to TEND (s)
T_rom = U_fom.shape[-1]*parameters['rom']['dt']

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


#%%
# --------------------------------------------------------------------------- #
# write vtu results
if not os.path.exists(paths['output']):
    os.mkdir(paths['output'])


ls = hdf.HDFReader(paths['meanfluc'][2])
ls.openHdfFile()
ls_mean = ls.mean[:]
ls_fluc = ls.fluc[:]
ls.closeHdfFile()


def write_vtu(i):
    "write vtu from a given index"
    print('write vtu #{}'.format(i))

    ufom = U_fom[:, :, i]
    urom = U_rom[:, :, i]
    lsi = ls_mean[:] + ls_fluc[:, :, i]

    data = {'vitesse_rom': urom,
            'vitesse_fom': ufom,
            'levelset': lsi,
            'error': urom-ufom}

    vtk.write(mesh, grid_shape, data,
              os.path.join(paths['output'], 'output{}.vtu'.format(i)))

    ufom = U_fom[:, :, i] - rom_ns.meanfluc.mean[:, :]
    urom = U_rom[:, :, i] - rom_ns.meanfluc.mean[:, :]

    data = {'vitesse_rom': urom,
            'vitesse_fom': ufom,
            'levelset': lsi,
            'error': urom-ufom}

    vtk.write(mesh, grid_shape, data,
              os.path.join(paths['output'], 'output_fluc{}.vtu'.format(i)))

parallelization.map(write_vtu, range(U_rom.shape[-1]-1))

#%%
def myplot(i):
    import matplotlib.pyplot as plt
    plt.close('all')
    plt.plot(rom_ns.times[:len(rom_ns.c_rom(i))-1], rom_ns.c_rom(i)[:len(rom_ns.c_rom(i))-1], '-r', label='rom')
    plt.plot(rom_ns.times[:len(rom_ns.c_rom(i))-1], rom_ns.c_fom(i)[:len(rom_ns.c_rom(i))-1], '.b', label='fom')
    plt.xlabel('time $t$ (s)')
    plt.ylabel('$\\alpha_%i(t)$' % i)
    plt.legend(loc=0)
    plt.title('Temporal coefficient $\\alpha_%i(t)$' % i)
    plt.savefig(os.path.join(paths['results'], 'rom_coeffs_%i' % i))
    plt.show()

for n in range(10):
    myplot(n)

#%%
rom_ns.close_hdfs()
