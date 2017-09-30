#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 22 12:14:21 2017

@author: afalaize
"""

import matplotlib.pyplot as plt
from lasie_rom import io, pod, plots
from _0_locations import paths, PVD_NAMES
from _0_parameters import parameters
import numpy
import os

# Plot
def plot_relative_energy_of_eigen_values(i):
    meanfluc_hdf = io.hdf.HDFReader(paths['meanfluc'][i])
    meanfluc_hdf.openHdfFile()
    eigen_vals, eigen_vec = pod.eigen_decomposition(meanfluc_hdf.fluc[:])
    meanfluc_hdf.closeHdfFile()

    N = len(eigen_vals)
    modes = numpy.arange(N) + 1
    energy = pod.eigen_energy(eigen_vals)
    plt.close('all')
    plt.semilogy(modes, 1-energy, label='rel. energy')
    plt.semilogy(modes, 1-numpy.ones(N)*(1-parameters['pod']['thld']),
             label='Threshold')
    plt.semilogy(numpy.ones(N)*parameters['pod']['nmax'],
                 numpy.linspace(1e-16, 1, N),
                 label='N$_{\mathrm{max}}$')
    plt.title('Selection of number of modes')
    plt.ylabel('Relative energy (d.u.)')
    plt.xlabel('Selected number of modes (#)')
    plt.savefig(os.path.join(paths['results'], PVD_NAMES[i]+'_modes_selection.pdf'))


# Plot 2D POD basis
def plot_pod_basis(i):
    grid = io.hdf.HDFReader(paths['grid'], openFile=True)
    basis = io.hdf.HDFReader(paths['basis'][i], openFile=True)
    plots.plot2d(basis.basis[:, :, :8],
                 grid.shape[:, 0],
                 render='magnitude', options={'ncols': 2})
    grid.closeHdfFile()
    basis.closeHdfFile()
    plt.savefig(os.path.join(paths['results'], PVD_NAMES[i]+'_basis.pdf'))
