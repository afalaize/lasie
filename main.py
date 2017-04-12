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

actions = {'ALL': False,
           'vtu2hdf': False,
           'interpolate': False,
           'ts2data': False,
           'data2meanAndFluc': False,
           'fluc2Corr': False,
           'corr2basis': True,
           'gradients': True,
           'writeVtu': False,
           'Thost_temporal_coeffs': True,
           'rom': True
           }


###############################################################################

# OSX USB: /Volumes/AFALAIZE/cylindre2D_SCC_windows/Results
# WIn: F:\TESTS_THOST\cylindre2D_SCC_windows\Results

CONFIG = {'vtu_folder': r'F:\TESTS_THOST\cylindre2D_SCC_windows\Results',
          'data_names_vtu': [r'Vitesse(m/s)',
                             r'MasseVolumique(kg/m3)',
                             r'Eta'],
          'h': (0.005, )*3,
          'threshold': 1e-1,
          'delta_t': None,
          'theta': 0.,
          'load': {'imin': 20, 'imax': 270, 'decim': 1},
          }

###############################################################################

CONFIG['hdf_folder'] = CONFIG['vtu_folder'] + os.sep + 'hdf5'
CONFIG['interp_hdf_folder'] = CONFIG['vtu_folder'] + os.sep + \
    'hdf5_interpolated'
CONFIG['hdf_path_dataMatrix'] = CONFIG['interp_hdf_folder'] + os.sep + \
    'data.hdf5'
CONFIG['hdf_path_grid'] = CONFIG['interp_hdf_folder'] + os.sep + 'grid.hdf5'
CONFIG['hdf_path_mean'] = CONFIG['interp_hdf_folder'] + os.sep + 'mean.hdf5'
CONFIG['hdf_path_meanGradient'] = CONFIG['interp_hdf_folder'] + os.sep + \
    'meanGradient.hdf5'
CONFIG['hdf_path_meanDeformation'] = CONFIG['interp_hdf_folder'] + os.sep + \
    'meanDeformation.hdf5'
CONFIG['hdf_path_weightingMatrix'] = None
CONFIG['hdf_path_fluc'] = CONFIG['interp_hdf_folder'] + os.sep + 'fluc.hdf5'
CONFIG['hdf_path_corr'] = CONFIG['interp_hdf_folder'] + os.sep + 'corr.hdf5'
CONFIG['hdf_path_podBasis'] = CONFIG['interp_hdf_folder'] + os.sep + \
    'basis.hdf5'
CONFIG['hdf_path_podBasisGradient'] = CONFIG['interp_hdf_folder'] + os.sep + \
    'basisGradient.hdf5'
CONFIG['hdf_path_A'] = CONFIG['interp_hdf_folder'] + os.sep + 'coeffs_A.hdf5'
CONFIG['hdf_path_B'] = CONFIG['interp_hdf_folder'] + os.sep + 'coeffs_B.hdf5'
CONFIG['hdf_path_C'] = CONFIG['interp_hdf_folder'] + os.sep + 'coeffs_C.hdf5'
CONFIG['hdf_path_F'] = CONFIG['interp_hdf_folder'] + os.sep + 'coeffs_F.hdf5'
CONFIG['vtu_path_podBasis'] = CONFIG['interp_hdf_folder'] + os.sep + \
    'basis.vtu'
CONFIG['hdf_path_Thost_temporal_coeffs'] = CONFIG['interp_hdf_folder'] + \
    os.sep + 'Thost_temporal_coeffs.hdf5'


###############################################################################

def plot_kinetic_energy():
    folder = CONFIG['hdf_folder']
    TS = HDFTimeSerie(folder)
    data_name = format_data_name(CONFIG['data_names_vtu'][0])
    path = folder + os.sep + 'data.hdf5'
    load = CONFIG['load']
    ts2HdfDataMatrix(TS, data_name, path, **load)
    data = HDFData(path, openFile=True)
    e = compute_kinetic_energy(data.get_single_data())
    plt.plot(TS.times, e, 'o:')
    plt.title('Energie cinetique')
    data.closeHdfFile()


def convert_vtu2hdf():
    pvd_path = CONFIG['vtu_folder'] + os.sep + PVDNAME
    pvd2Hdf(pvd_path, CONFIG['hdf_folder'], CONFIG['data_names_vtu'],
            **CONFIG['load'])
    plot_kinetic_energy()


###############################################################################

def interpolate_data_over_regular_grid():
    TS = HDFTimeSerie(CONFIG['hdf_folder'])
    TS.openAllFiles()

    # A regular (1+N)-dimensional grid. E.g with N=3, grid[c, i, j, k] is the
    # component ‘c’ of the coordinates of the point at position (i, j, k).
    grid, grid_h = buildGrid(TS.data[0].getMeshMinMax(), CONFIG['h'])
    grid_shape = grid.shape
    grid_mesh = grid2mesh(grid)
    dumpArrays2Hdf([grid_mesh, np.array(grid_shape)[:, np.newaxis],
                    np.array(grid_h)[:, np.newaxis]],
                   ['mesh', 'original_shape', 'h'], CONFIG['hdf_path_grid'])
    interpTimeSerieToHdf(TS, grid_mesh, CONFIG['interp_hdf_folder'])
    TS.closeAllFiles()


###############################################################################

def form_data_matrix():
    print('Form data matrix')
    TS = HDFTimeSerie(CONFIG['interp_hdf_folder'])
    ts2HdfDataMatrix(TS, format_data_name(CONFIG['data_names_vtu'][0]),
                     CONFIG['hdf_path_dataMatrix'], **CONFIG['load'])
    data = HDFData(CONFIG['hdf_path_dataMatrix'], openFile=True)
    e = compute_kinetic_energy(data.get_single_data())
    plt.plot(e, 'o:')
    plt.title('Energie cinetique')


###############################################################################

def split_mean_and_fluc():
    print('Split mean from fluctuating velocity')
    dataMatrix2MeanAndFluc(CONFIG['hdf_path_dataMatrix'],
                           CONFIG['hdf_path_mean'], CONFIG['hdf_path_fluc'])


###############################################################################

def form_correlation_matrix():
    print('Form correlation matrix')
    fluc2CorrMatrix(CONFIG['hdf_path_fluc'], CONFIG['hdf_path_corr'],
                    hdf_path_weightingMatrix=None)


###############################################################################

def form_pod_basis():
    print('Form pod basis')
    computePODBasis(CONFIG['hdf_path_corr'], CONFIG['hdf_path_fluc'],
                    CONFIG['hdf_path_podBasis'], threshold=CONFIG['threshold'])
    basis = HDFData(CONFIG['hdf_path_podBasis'])
    basis.openHdfFile()
    checkPODBasisIsOrthonormal(basis.get_single_data())
    basis.closeHdfFile()


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

def export_pod_basis_to_vtu():
    print('write vtu for pod basis')
    basis = HDFData(CONFIG['hdf_path_podBasis'], openFile=True)
    grid = HDFData(CONFIG['hdf_path_grid'], openFile=True)
    write_vtu(grid.mesh[:], grid.original_shape,
              [data[:, :] for data in basis.get_single_data().swapaxes(0, 1)],
              'testData', CONFIG['vtu_path_podBasis'])
    basis.closeHdfFile()
    grid.closeHdfFile()


###############################################################################


def compute_Thost_temporal_coeffs():
    HDFbasis = HDFData(CONFIG['hdf_path_podBasis'], openFile=True)
    basis = HDFbasis.get_single_data()
    HDFbasis.closeHdfFile()

    fluct = HDFData(CONFIG['hdf_path_fluc'], openFile=True)

    def compute_coeff(u):
        return np.einsum('mc,mic->i', u, basis)

    nt = fluct.vitesse.shape[1]
    temporal_coeffs = list()
    bar = progressbar.ProgressBar(widgets=[progressbar.Timer(), ' ',
                                           progressbar.Bar(), ' (',
                                           progressbar.ETA(), ')\n', ])

    for i in bar(range(nt)):
        temporal_coeffs.append(compute_coeff(fluct.vitesse[:, i, :]))

    fluct.closeHdfFile()
    dumpArrays2Hdf([np.array(temporal_coeffs)],
                   ['coeffs'],
                   CONFIG['hdf_path_Thost_temporal_coeffs'])


###############################################################################

if __name__ is '__main__':
    if actions['vtu2hdf'] or actions['ALL']:
        convert_vtu2hdf()
    if actions['interpolate'] or actions['ALL']:
        interpolate_data_over_regular_grid()
    if actions['ts2data'] or actions['ALL']:
        form_data_matrix()
    if actions['data2meanAndFluc'] or actions['ALL']:
        split_mean_and_fluc()
    if actions['fluc2Corr'] or actions['ALL']:
        form_correlation_matrix()
    if actions['corr2basis'] or actions['ALL']:
        form_pod_basis()
    if actions['gradients'] or actions['ALL']:
        form_gradients()
    if actions['writeVtu'] or actions['ALL']:
        export_pod_basis_to_vtu()
    if actions['Thost_temporal_coeffs'] or actions['ALL']:
        compute_Thost_temporal_coeffs()
    if actions['rom'] or actions['ALL']:
        ts = HDFTimeSerie(CONFIG['interp_hdf_folder'])
        ts.data[0].openHdfFile()
        rho = ts.data[0].massevolumique[:].flatten()[0]
        mu = ts.data[0].eta[:].flatten()[0]
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
        rom.run(dt=None, tend=None, istart=None, theta=CONFIG['theta'])
        for i in range(rom.npod()):
            plt.figure()
            plt.plot(rom.times, rom.c_fom(i)[:-1], ':o', label='fom')
            plt.plot(rom.times, rom.c_rom(i)[:-1], '-x', label='rom')
            plt.title('mode {}'.format(i+1))
            plt.legend()
            plt.grid('on')
            plt.savefig('coeffs_temporels_mode{}.png'.format(i+1), 
                        format='png')
            plt.show()
        rom.close_hdfs()
