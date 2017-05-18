# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 11:15:13 2017

@author: afalaize
"""

from __future__ import division, absolute_import, print_function
import os
import numpy as np
from .tools import tensor2vector, sortIndices
from ..readwrite.vtu2hdf import dumpArrays2Hdf
from ..readwrite.read_hdf import HDFData
from ..config import ORDER


def mean2MeanGradient(hdf_path_mean, hdf_path_meangradient, hdf_path_grid):
    mean = HDFData(hdf_path_mean, openFile=True)
    grid = HDFData(hdf_path_grid, openFile=True)
    grid_shape = map(int, list(grid.original_shape[:, 0]))
    h = list(grid.h[:, 0])
    gradient = formatedGradient(mean.get_single_data(), grid_shape, h)
    dumpArrays2Hdf([gradient, ], ['gradient'+mean.names[0], ],
                   hdf_path_meangradient)
    grid.closeHdfFile()
    mean.closeHdfFile()


def basis2BasisGradient(hdf_path_basis, hdf_path_basisgradient, hdf_path_grid):
    basis = HDFData(hdf_path_basis, openFile=True)
    grid = HDFData(hdf_path_grid, openFile=True)
    grid_shape = map(int, list(grid.original_shape[:, 0]))
    h = list(grid.h[:, 0])
    npod = basis.vitesse.shape[1]
    def gradient_generator():
        for i in range(npod):
            yield formatedGradient(getattr(basis, basis.names[0])[:, i, :],
                                   grid_shape, h)[:, np.newaxis, :, :]
    def form_array():
        return np.concatenate(tuple(gradient_generator()), axis=1)

    dumpArrays2Hdf([form_array(), ], ['gradient'+basis.names[0], ],
                   hdf_path_basisgradient)
    grid.closeHdfFile()
    basis.closeHdfFile()


def ts2HdfDataMatrix(ts, data_name, hdf_path, imin=0, imax=None, decim=1.):
    dataL = list()
    if imax is None:
        imax = float('Inf')
    for i, d in enumerate(ts.data):
        #if imin <= i < imax and i%decim == 0:
        d.openHdfFile()
        dataL.append(getattr(d, data_name)[:][:, np.newaxis, :])
        d.closeHdfFile()
    data = np.concatenate(dataL, axis=1)
    dumpArrays2Hdf([data, ], [data_name, ], hdf_path)


def dataMatrix2MeanAndFluc(hdf_path_dataMatrix, hdf_path_mean, hdf_path_fluc,
                           setToZero=False):
    data = HDFData(hdf_path_dataMatrix, openFile=True)
    if setToZero:
        mean = 0.*data.get_single_data()[:, 0, :]
    else:
        mean = np.mean(data.get_single_data(), axis=1)
    dumpArrays2Hdf([mean, ], [data.names[0], ], hdf_path_mean)
    fluc = data.get_single_data()
    for i, d in enumerate(data.get_single_data().swapaxes(0, 1)):
        fluc[:, i, :] = d - mean
    dumpArrays2Hdf([fluc, ],
                   [data.names[0], ], hdf_path_fluc)
    data.closeHdfFile()


def getWeightingMatrix(hdf_path_weightingMatrix=None):
    W = HDFData(hdf_path_weightingMatrix)
    if hdf_path_weightingMatrix is None:
        W.openHdfFile = lambda : None
        W.closeHdfFile = lambda : None
        W.get_single_data = lambda : 1.
    return W


def scalarProductHDFData(hdfdata_A, hdfdata_B, hdf_weightingMatrix):
    temp = np.dot(hdf_weightingMatrix.get_single_data(),
                  hdfdata_B.get_single_data())
    return np.dot(hdfdata_A.get_single_data().T, temp)


def scalarProductArray(array_A, array_B, array_weightingMatrix):
    temp = np.dot(array_weightingMatrix, array_B)
    return np.dot(array_A.T, temp)


def scalarProduct(obj_A, obj_B, obj_weightingMatrix):
    if isinstance(obj_A, HDFData):
        assert isinstance(obj_B, HDFData)
        assert isinstance(obj_weightingMatrix, HDFData)
        return scalarProductHDFData(obj_A, obj_B, obj_weightingMatrix)
    else:
        assert isinstance(obj_A, np.ndarray)
        assert isinstance(obj_B, np.ndarray)
        return scalarProductArray(obj_A, obj_B, obj_weightingMatrix)


def fluc2CorrMatrix(hdf_path_flucDataMatrix, hdf_path_CorrMatrix,
                    hdf_path_weightingMatrix=None):


    W = getWeightingMatrix(hdf_path_weightingMatrix)

    fluc = HDFData(hdf_path_flucDataMatrix, openFile=True)
    reader = fluc.get_single_data()
    shape = reader[:].shape
    newshape = (shape[1], np.prod([shape[0], ]+list(shape[2:])))

    # Temporal correlation matrix
    C = scalarProductArray(reader[:].swapaxes(0, 1).reshape(newshape,
                                                            order=ORDER).swapaxes(0, 1),
                           reader[:].swapaxes(0, 1).reshape(newshape,
                                                            order=ORDER).swapaxes(0, 1),
                           W.get_single_data())

    # Temporal correlation matrix eigen decomposition
    eigen_vals, eigen_vecs = np.linalg.eig(C)

    # Remove the imaginary part (which should be numerically close to zero)
    eigen_vals = np.real(eigen_vals)
    eigen_vecs = np.real(eigen_vecs)

    # sort by decreasing eigen values
    indices = sortIndices(eigen_vals)

    C_eigen_vals = np.array([[eigen_vals[n] for n in indices], ]).T
    C_eigen_vecs = eigen_vecs[:, np.array(indices)]

    dumpArrays2Hdf([C, C_eigen_vals, C_eigen_vecs],
                   [fluc.names[0], 'eigen_vals', 'eigen_vecs'],
                   hdf_path_CorrMatrix)

    fluc.closeHdfFile()
    W.closeHdfFile()


def computePODBasis(hdf_path_CorrMatrix, hdf_path_flucDataMatrix,
                    hdf_path_podBasis, threshold=1e-6, nmax=None,
                    hdf_path_weightingMatrix=None):
    """
    build the POD basis from the snapshots (self.basis)
    """

    C_data = HDFData(hdf_path_CorrMatrix, openFile=True)
    fluc = HDFData(hdf_path_flucDataMatrix, openFile=True)
    npod = truncatedNpod(computeModesEnergy(C_data.eigen_vals[:, 0]),
                         threshold=threshold)
    if nmax is not None:
        npod=nmax

    # Define POD basis
    basis = np.einsum('mtc,ti->mic', fluc.get_single_data(),
                      C_data.eigen_vecs[:, :npod])
    W = getWeightingMatrix(hdf_path_weightingMatrix)
    normalizePODBasis(basis, W.get_single_data())
    dumpArrays2Hdf([basis, ], [C_data.names[0], ], hdf_path_podBasis)

    fluc.closeHdfFile()
    C_data.closeHdfFile()
    W.closeHdfFile()
