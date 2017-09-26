#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 11:56:18 2017

@author: afalaize
"""

from lasie_rom import pod, operators, misc, io, parallelization
from lasie_rom.io import hdf
from lasie_rom.classes import TimeSerie

from locations import PVD_NAMES
from options import options

from plots import plot_relative_energy_of_eigen_values, plot_pod_basis

import os

import numpy as np
# --------------------------------------------------------------------------- #
# split mean field from fluctuating fields in all snapshots

from locations import paths

for i, path_to_ihdf in enumerate(paths['ihdf']):

    # instanciate TimeSerie
    ts = TimeSerie(path_to_ihdf)

    # Form snapshot matrix
    if options['dataname'] is not None:
        dataname = options['dataname']
    else:
        d = ts.data[0]
        d.openHdfFile()
        args = dir(d)
        temp = [a.startswith('f_') for a in args]
        dataname = args[temp.index(True)]
        d.closeHdfFile()

    U = ts.concatenate(dataname)

    # Compute mean and fluctuations
    mean, fluc = pod.meanfluc(U)

    # Close hdf files
    ts.closeAllFiles()

    # recover grid infos
    grid_hdf = hdf.HDFReader(paths['grid'])
    grid_hdf.openHdfFile()
    grid_shape = grid_hdf.shape[:][:, 0]
    grid_h = grid_hdf.h[:][:, 0]

    # Compute mean gradient
    mean_grad = operators.gridgradient(mean, grid_shape, grid_h)

    # define data
    data = {'mesh': grid_hdf.mesh[:],
            'mean': mean,
            'mean_grad': mean_grad,
            'fluc': fluc}

    # Close hdf file for grid
    grid_hdf.closeHdfFile()

    # write hdf for mean and fluct
    hdf.data2hdf(data, paths['meanfluc'][i])

    # --------------------------------------------------------------------------- #
    # Compute POD basis
    meanfluc_hdf = hdf.HDFReader(paths['meanfluc'][i])
    meanfluc_hdf.openHdfFile()
    basis = pod.compute_basis(meanfluc_hdf.fluc[:],
                              threshold=options['pod']['thld'],
                              nmax=options['pod']['nmax'])
    meanfluc_hdf.closeHdfFile()

    # recover grid infos
    grid_hdf = hdf.HDFReader(paths['grid'])
    grid_hdf.openHdfFile()
    grid_shape = grid_hdf.shape[:][:, 0]
    grid_h = grid_hdf.h[:][:, 0]


    # Compute pod gradient
    def compute_grad(a):
        return operators.gridgradient(a, grid_shape, grid_h)


    def grad_generator():
        for a in misc.iterarray(basis, 2):
            yield compute_grad(a)

    basis_grad = misc.concatenate_in_given_axis(grad_generator(), 3)

    data = {'mesh': grid_hdf.mesh[:],
            'basis': basis,
            'basis_grad': basis_grad}

    # write hdf for pod basis
    hdf.data2hdf(data, paths['basis'][i])

    # Close hdf file for grid
    grid_hdf.closeHdfFile()

    # --------------------------------------------------------------------------- #
    # save basis to vtu

    basis_hdf = hdf.HDFReader(paths['basis'][i], openFile=True)
    nbmodes = basis_hdf.basis[:].shape[-1]

    grid_hdf = hdf.HDFReader(paths['grid'])
    grid_hdf.openHdfFile()
    grid_shape = grid_hdf.shape[:][:, 0]
    mesh = grid_hdf.mesh[:]
    grid_hdf.closeHdfFile()


    def write_vtu(k):
        print('write mode #{}'.format(k))
        b = basis[:, :, k].copy()
        data = {'POD': b}
        path = os.path.join(paths['results'], PVD_NAMES[i]+'_mode{}.vtu'.format(k))
        io.vtk.write(mesh.copy(), grid_shape.copy(), data.copy(), path)

    parallelization.map(write_vtu, range(nbmodes))

    basis_hdf.closeHdfFile()

    # --------------------------------------------------------------------------- #
    # Some plots

    plot_relative_energy_of_eigen_values(i)
    plot_pod_basis(i)
