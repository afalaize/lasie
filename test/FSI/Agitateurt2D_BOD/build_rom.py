#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:30:38 2017

@author: afalaize
"""

from lasie_rom import io, rom, misc, parallelization
from lasie_rom.classes import TimeSerie

from lasie_rom.rom import fsi_relaxed_velocity_BOD
from lasie_rom.misc.tools import sympyExpr2numpyFunc

from set_parameters import parameters
from set_locations import paths

import numpy
import sympy as sy

#%%
if True:
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
            return numpy.vstack([f(*fargs) for f in func_vec]).T
        return func

    velocity_func = build_velocity_func(parameters['angular_vel'],
                                        parameters['rot_center'])

    # --------------------------------------------------------------------------- #
    # Build ROM Matrices:
    # Open hdf files
    gridmesh_hdf = io.hdf.HDFReader(paths['grid'], openFile=True)

    vrot = velocity_func(*[xi for xi in gridmesh_hdf.mesh[:].T])

    velocity_basis_hdf = io.hdf.HDFReader(paths['basis'][0], openFile=True)
    velocity_meanfluc_hdf = io.hdf.HDFReader(paths['meanfluc'][0], openFile=True)

    fnchar_basis_hdf = io.hdf.HDFReader(paths['basis'][1], openFile=True)
    fnchar_meanfluc_hdf = io.hdf.HDFReader(paths['meanfluc'][1], openFile=True)

    print('Build temp_a')
    temp_a = fsi_relaxed_velocity_BOD.temp_a(velocity_basis_hdf.basis[:])
    print('Build temp_b_rho')
    temp_b_rho = fsi_relaxed_velocity_BOD.temp_b_rho(velocity_basis_hdf.basis[:],
                                                  velocity_meanfluc_hdf.mean[:],
                                                  velocity_basis_hdf.basis_grad[:],
                                                  velocity_meanfluc_hdf.mean_grad[:])
    print('Build temp_b_nu')
    temp_b_nu = fsi_relaxed_velocity_BOD.temp_b_nu(velocity_basis_hdf.basis_grad[:])
    print('Build temp_c')
    temp_c = fsi_relaxed_velocity_BOD.temp_c(velocity_basis_hdf.basis[:], velocity_basis_hdf.basis_grad[:])
    print('Build temp_d')
    temp_d = fsi_relaxed_velocity_BOD.temp_d(velocity_basis_hdf.basis_grad[:])
    print('Build temp_f_rho')
    temp_f_rho = fsi_relaxed_velocity_BOD.temp_f_rho(velocity_basis_hdf.basis[:],
                                                  velocity_meanfluc_hdf.mean[:],
                                                  velocity_meanfluc_hdf.mean_grad[:])
    print('Build temp_f_nu')
    temp_f_nu = fsi_relaxed_velocity_BOD.temp_f_nu(velocity_basis_hdf.basis_grad[:],
                                                 velocity_meanfluc_hdf.mean_grad[:])

    a_bar = fsi_relaxed_velocity_BOD.a_bar(temp_a, fnchar_meanfluc_hdf.mean[:], parameters['rho'], parameters['rho_delta'])
    a_tilde = fsi_relaxed_velocity_BOD.a_tilde(temp_a, fnchar_basis_hdf.basis[:], parameters['rho_delta'])

    b_bar = fsi_relaxed_velocity_BOD.b_bar(temp_b_rho, temp_b_nu, fnchar_meanfluc_hdf.mean[:], parameters['rho'], parameters['rho_delta'], parameters['nu'], parameters['nu_delta'])
    b_tilde = fsi_relaxed_velocity_BOD.b_tilde(temp_b_rho, temp_b_nu, fnchar_basis_hdf.basis[:], parameters['rho_delta'], parameters['nu_delta'])

    c_bar= fsi_relaxed_velocity_BOD.c_bar(temp_c, fnchar_meanfluc_hdf.mean[:], parameters['rho'], parameters['rho_delta'])
    c_tilde = fsi_relaxed_velocity_BOD.c_tilde(temp_c, fnchar_basis_hdf.basis[:], parameters['rho_delta'])

    f_bar = fsi_relaxed_velocity_BOD.f_bar(temp_f_rho, temp_f_nu, fnchar_meanfluc_hdf.mean[:], parameters['rho'], parameters['rho_delta'], parameters['nu'], parameters['nu_delta'])
    f_tilde = fsi_relaxed_velocity_BOD.f_tilde(temp_f_rho, temp_f_nu, fnchar_basis_hdf.basis[:], parameters['rho_delta'], parameters['nu_delta'])
    L = fsi_relaxed_velocity_BOD.L(fnchar_meanfluc_hdf.mean[:], velocity_meanfluc_hdf.mean[:], vrot, velocity_basis_hdf.basis[:])
    M = fsi_relaxed_velocity_BOD.M(fnchar_basis_hdf.basis[:], velocity_meanfluc_hdf.mean[:], vrot, velocity_basis_hdf.basis[:])
    N = fsi_relaxed_velocity_BOD.N(fnchar_meanfluc_hdf.mean[:], velocity_basis_hdf.basis[:])
    P = fsi_relaxed_velocity_BOD.P(fnchar_basis_hdf.basis[:], velocity_basis_hdf.basis[:])

    data = {'a_bar': a_bar,
            'a_tilde': a_tilde,
            'b_bar': b_bar,
            'b_tilde': b_tilde,
            'c_bar': c_bar,
            'c_tilde': c_tilde,
            'f_bar': f_bar,
            'f_tilde': f_tilde,
            'L': L,
            'M': M,
            'N': N,
            'P': P,
            'fnchar_mean': fnchar_meanfluc_hdf.mean[:],
            'fnchar_basis': fnchar_basis_hdf.basis[:]
            }

    print('Save {}'.format(paths['matrices']))
    # write hdf for rom matrices
    io.hdf.data2hdf(data, paths['matrices'])

    # Close hdf files
    velocity_basis_hdf.closeHdfFile()
    velocity_meanfluc_hdf.closeHdfFile()
    fnchar_basis_hdf.closeHdfFile()
    fnchar_meanfluc_hdf.closeHdfFile()
    gridmesh_hdf.closeHdfFile()


# %% Project the snapshot on the pod basis
if True:
    # --------------------------------------------------------------------------- #

    # Open hdf files
    velocity_basis_hdf = io.hdf.HDFReader(paths['basis'][0], openFile=True)
    velocity_meanfluc_hdf = io.hdf.HDFReader(paths['meanfluc'][0], openFile=True)

    print('Build coeffs')
    # instanciate TimeSerie
    ts = TimeSerie(paths['ihdf'][0])

    if parameters['dataname']['hdf'] is not None:
        dataname = parameters['dataname']['hdf']
    else:
        d = ts.data[0]
        d.openHdfFile()
        args = dir(d)
        temp = [a.startswith('f_') for a in args]
        dataname = args[temp.index(True)]
        d.closeHdfFile()

    # instanciate TimeSerie
    def coefficients():
        for u in ts.generator(dataname)():
            yield numpy.einsum('xc,xci->i',
                               u-velocity_meanfluc_hdf.mean[:],
                               velocity_basis_hdf.basis[:])

    coeffs = misc.concatenate_in_given_axis(coefficients(), 0)

    data = {'coeffs': coeffs}

    print('Save {}'.format(paths['coeffs']))
    # write hdf for rom matrices
    io.hdf.data2hdf(data, paths['coeffs'])

    # Close hdf files
    velocity_basis_hdf.closeHdfFile()
    velocity_meanfluc_hdf.closeHdfFile()
