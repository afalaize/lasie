# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 17:08:50 2017

@author: afalaize
"""

import os

from pypod.readwrite.vtu2hdf import pvd2Hdf, format_data_name, dumpArrays2Hdf

from pypod.readwrite.read_hdf import (HDFData, HDFTimeSerie,
                                      interpTimeSerieToHdf)
from pypod.config import PVDNAME

from pypod.grids.tools import buildGrid, grid2mesh

from pypod.pod.tools import compute_kinetic_energy
from pypod.pod.pod import (ts2HdfDataMatrix, dataMatrix2MeanAndFluc,
                           fluc2CorrMatrix, computePODBasis,
                           mean2MeanGradient, basis2BasisGradient,
                           checkPODBasisIsOrthonormal)

from pypod.readwrite.write_vtu import write_vtu

from pypod.rom.rom import (build_rom_coefficients_A,
                           build_rom_coefficients_B,
                           build_rom_coefficients_C,
                           build_rom_coefficients_F,
                           ReducedOrderModel)

import numpy as np

import progressbar

import matplotlib.pyplot as plt


###############################################################################

# OSX USB: /Volumes/AFALAIZE/cylindre2D_SCC_windows/Results
# WIn: F:\TESTS_THOST\cylindre2D_SCC_windows\Results

CONFIG = {'vtu_folder': r'F:\TESTS_THOST\cylindre2D_SCC_windows\Results',
          'data_names_vtu': [r'Vitesse(m/s)',
                             r'MasseVolumique(kg/m3)',
                             r'Eta'],
          'threshold': 1e-2,    # 1e-2 equiv 5 modes
          'dt': 0.01,           # original: 0.01s
          'tend': 52.5,         # original: 52.5s
          'theta': 1.,          # theta scheme
          }

###############################################################################
CONFIG['interp_hdf_folder'] = 'test_5modes'
CONFIG['hdf_path_grid'] = CONFIG['interp_hdf_folder'] + os.sep + 'grid.hdf5'
CONFIG['hdf_path_mean'] = CONFIG['interp_hdf_folder'] + os.sep + 'mean.hdf5'
CONFIG['hdf_path_meanGradient'] = CONFIG['interp_hdf_folder'] + os.sep + \
    'meanGradient.hdf5'
CONFIG['hdf_path_podBasis'] = CONFIG['interp_hdf_folder'] + os.sep + \
    'basis.hdf5'
CONFIG['hdf_path_podBasisGradient'] = CONFIG['interp_hdf_folder'] + os.sep + \
    'basisGradient.hdf5'
CONFIG['hdf_path_A'] = CONFIG['interp_hdf_folder'] + os.sep + 'coeffs_A.hdf5'
CONFIG['hdf_path_B'] = CONFIG['interp_hdf_folder'] + os.sep + 'coeffs_B.hdf5'
CONFIG['hdf_path_C'] = CONFIG['interp_hdf_folder'] + os.sep + 'coeffs_C.hdf5'
CONFIG['hdf_path_F'] = CONFIG['interp_hdf_folder'] + os.sep + 'coeffs_F.hdf5'
CONFIG['hdf_path_Thost_temporal_coeffs'] = 'test_5modes' + \
    os.sep + 'Thost_temporal_coeffs.hdf5'


###############################################################################

def form_gradients():
    print('Form mean gradient')
    mean2MeanGradient(CONFIG['hdf_path_mean'], CONFIG['hdf_path_meanGradient'],
                      CONFIG['hdf_path_grid'])
    print('Form basis gradient')
    basis2BasisGradient(CONFIG['hdf_path_podBasis'],
                        CONFIG['hdf_path_podBasisGradient'],
                        CONFIG['hdf_path_grid'])


###############################################################################


if __name__ is '__main__':
    rho = 1
    mu = 2e-4
    form_gradients()
    build_rom_coefficients_A(CONFIG['hdf_path_podBasis'],
                             CONFIG['hdf_path_A'])

    build_rom_coefficients_B(CONFIG['hdf_path_podBasis'],
                             CONFIG['hdf_path_podBasisGradient'],
                             CONFIG['hdf_path_mean'],
                             CONFIG['hdf_path_meanGradient'], mu, rho,
                             CONFIG['hdf_path_B'])

    build_rom_coefficients_C(CONFIG['hdf_path_podBasis'],
                             CONFIG['hdf_path_podBasisGradient'],
                             CONFIG['hdf_path_C'])

    build_rom_coefficients_F(CONFIG['hdf_path_podBasis'],
                             CONFIG['hdf_path_podBasisGradient'],
                             CONFIG['hdf_path_mean'],
                             CONFIG['hdf_path_meanGradient'], mu, rho,
                             CONFIG['hdf_path_F'])
    rom = ReducedOrderModel(CONFIG)
    rom.run(dt=CONFIG['dt'], tend=CONFIG['tend'], theta=CONFIG['theta'])
    for i in range(rom.npod()):
        plt.figure()
        t_fom = np.linspace(50, 52.5, len(rom.c_fom(0)))
        plt.plot(t_fom, rom.c_fom(i), ':o', label='fom')
        t_rom = np.linspace(50, rom.tend, len(rom.c_rom(0))-1)
        plt.plot(t_rom, rom.c_rom(i)[:-1], '-x', label='rom')
        plt.title('mode {}'.format(i+1))
        plt.legend()
        plt.grid('on')
        plt.savefig('mode{}_dt={:.2f}_rho={:.2f}_mu={:.5f}.png'.format(i+1, rom.dt, rho, mu), 
                    format='png')
        plt.show()
    rom.close_hdfs()
