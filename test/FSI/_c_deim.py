#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:54:35 2017

@author: afalaize
"""

from locations import paths, MAIN_FOLDER

import numpy as np
import os

from lasie_rom.misc import smooth
from lasie_rom import deim
from lasie_rom.io import hdf
from lasie_rom import parallelization
from lasie_rom import plots

import lasie_rom as lr

import matplotlib.pyplot as plt

from main import parameters

# Pod basis associated to the levelset
path_Psi = paths['basis'][2]
hdf_Psi = hdf.HDFReader(path_Psi)
hdf_Psi.openHdfFile()
Psi = hdf_Psi.basis[:]
hdf_Psi.closeHdfFile()

# Pod basis of the heaviside function applied to the levelset
path_Lambda = paths['basis'][1]
hdf_Lambda = hdf.HDFReader(path_Lambda)
hdf_Lambda.openHdfFile()
Lambda = hdf_Lambda.basis[:]
hdf_Lambda.closeHdfFile()

deim_P, _, indices_ones = deim.projector(Lambda)

temp_Mat = np.linalg.inv(np.einsum('xci,xcj->ij', deim_P, Lambda))

L = np.einsum('xci,ij->xcj', Lambda, temp_Mat)

hdf_fnchar = hdf.HDFReader(paths['meanfluc'][1])
hdf_fnchar.openHdfFile()
fnchar_fluc = hdf_fnchar.fluc[:]
fnchar_mean = hdf_fnchar.mean[:]
hdf_fnchar.closeHdfFile()

hdf_levelset = hdf.HDFReader(paths['meanfluc'][2])
hdf_levelset.openHdfFile()
levelset_fluc = hdf_levelset.fluc[:]
levelset_mean = hdf_levelset.mean[:]
hdf_levelset.closeHdfFile()

deim_levelset_mean = levelset_mean[indices_ones, 0]
deim_Psi = Psi[indices_ones, 0, :]


heaviside = smooth.build_vectorized_heaviside(eps=parameters['eps_tanh'])


def deim_heaviside_eps(beta):
    """
    Deim evaluation of the heaviside function (tilde{h}_{\epsilon})
    """
    return heaviside(deim_levelset_mean+deim_Psi.dot(beta))


if __name__ == '__main__':

    plt.close('all')

    list_levelset = []
    list_fnchar = []
    list_fnchar_deim = []

    list_gamma = []
    list_gamma_deim = []

    for i, fluc_i in enumerate(levelset_fluc.T):
        beta = np.einsum('xci,xc->i', Psi, fluc_i.T)
        gamma = np.einsum('xci,xc->i', Lambda, fnchar_fluc[:, :, i])

        list_gamma.append(gamma)
        list_gamma_deim.append(np.einsum('ij,j->i', temp_Mat, deim_heaviside_eps(beta)))

        list_levelset.append(levelset_mean + np.einsum('xci,i->xc', Psi, beta))
        list_fnchar.append(fnchar_mean + np.einsum('xci,i->xc', Lambda, list_gamma[-1]))
        list_fnchar_deim.append(fnchar_mean + np.einsum('xci,i->xc', Lambda, list_gamma_deim[-1]))

    data_levelset = lr.misc.concatenate_in_given_axis(list_levelset, 2)
    data_fnchar = lr.misc.concatenate_in_given_axis(list_fnchar, 2)
    data_fnchar_deim = lr.misc.concatenate_in_given_axis(list_fnchar_deim, 2)

    hdf_grid = hdf.HDFReader(paths['grid'])
    hdf_grid.openHdfFile()
    shape = hdf_grid.shape[:]
    hdf_grid.closeHdfFile()

    plots.plot2d(data_levelset[:, :, ::55], shape,
                 title='Reconstructed Level-Set',
                 savename=os.path.join(MAIN_FOLDER, 'levelset_reconstructed'), render=0)

    plots.plot2d(smooth.heaviside(data_levelset[:, :, ::55], parameters['eps_tanh']), shape,
                 title='Reconstructed Level-Set + Heaviside',
                 savename=os.path.join(MAIN_FOLDER, 'levelset_reconstructed_heaviside'), render=0)

    plots.plot2d(data_fnchar[:, :, ::55], shape,
                 title='Reconstructed Characteristic Function',
                 savename=os.path.join(MAIN_FOLDER, 'fnchar_reconstructed'), render=0)

    plots.plot2d(data_fnchar_deim[:, :, ::55], shape,
                 title='Reconstructed DEIM Characteristic Function',
                 savename=os.path.join(MAIN_FOLDER, 'fnchar_DEIM'), render=0)

    plots.plot1d([np.array(list_gamma)[:,:10], np.array(list_gamma_deim)[:,:10]],
                 title='Gamma',
                 savename=os.path.join(MAIN_FOLDER, 'gamma'))

    plt.show()
