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


def mean2MeanGradient(hdf_path_mean, hdf_path_meangradient, hdf_path_grid, 
                      nc=3):
    mean = HDFData(hdf_path_mean, openFile=True)
    grid = HDFData(hdf_path_grid, openFile=True)
    grid_shape = map(int, list(grid.original_shape[:, 0]))
    h = list(grid.h[:, 0])    
    gradient = formatedGradient(mean.get_single_data(), grid_shape, h)
    dumpArrays2Hdf([gradient, ], ['gradient'+mean.names[0], ], 
                   hdf_path_meangradient)
    grid.closeHdfFile()
    mean.closeHdfFile()
    

def basis2BasisGradient(hdf_path_basis, hdf_path_basisgradient, hdf_path_grid, 
                        nc=3):
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
    
    
def formatedGradient(array, grid_shape, h):
    """
    return the gradient of the grid array
    """
    array_shape = array.shape 
    out_grad = np.zeros((array_shape[0], array_shape[1], array_shape[1]))
    for i in range(array_shape[1]):
        ai = array[:, i].reshape(grid_shape[1:], order=ORDER)
        gi = np.gradient(ai, *h)
        for j in range(array_shape[1]):
            out_grad[:, i, j] = gi[j].reshape((np.prod(grid_shape[1:]), ), order=ORDER)
    return out_grad

    
def ts2HdfDataMatrix(ts, data_name, hdf_path, imin=0, imax=None, decim=1.):
    ts.openAllFiles()
    dataL = list()
    if imax is None:
        imax = float('Inf')
    for i, d in enumerate(ts.data):
        if imin <= i < imax and i%decim == 0:
            dataL.append(getattr(d, data_name)[:][:, np.newaxis, :])
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
                    hdf_path_podBasis, threshold=1e-6, 
                    hdf_path_weightingMatrix=None):
    """
    build the POD basis from the snapshots (self.basis)
    """
    
    C_data = HDFData(hdf_path_CorrMatrix, openFile=True)
    fluc = HDFData(hdf_path_flucDataMatrix, openFile=True)
    npod = truncatedNpod(computeModesEnergy(C_data.eigen_vals[:, 0]),
                         threshold=threshold)
    
    # Define POD basis
    basis = np.einsum('mkj,ki->mij', fluc.get_single_data(), 
                      C_data.eigen_vecs[:, :npod])
    W = getWeightingMatrix(hdf_path_weightingMatrix)
    normalizePODBasis(basis, W.get_single_data())
    dumpArrays2Hdf([basis, ], [C_data.names[0], ], hdf_path_podBasis)
    
    fluc.closeHdfFile()
    C_data.closeHdfFile()
    W.closeHdfFile()

    
def computeModesEnergy(C_eigen_vals):
    modes_energy = list()
    for i, val in enumerate(C_eigen_vals):
        mode_energy = sum(C_eigen_vals[:i+1])/sum(C_eigen_vals)
        modes_energy.append(mode_energy)
    return modes_energy

    
def truncatedNpod(modes_energy, threshold=1e-6):
    return [me >= 1-threshold for me in modes_energy].index(True)+1


def checkPODBasisIsOrthonormal(basis):
    M = np.abs(np.dot(basis.T, basis))
    for i in range(M.shape[0]):
        M[i, i] = 0
    print("val max out of diag from np.dot(basis.T, basis) is {}".format(M.max()))

    
def normalizePODBasis(basis, array_weightingMatrix):
    for i in range(basis.shape[1]):
        scalarprod = scalarProduct(basis[:, i, 0], basis[:, i, 0], 
                                   array_weightingMatrix)
        basis[:,i] = basis[:,i]/np.sqrt(scalarprod)
        
class POD(object):

    def __init__(self, hdf_path_mean, hdf_path_fluc):
        
        if folder is None:
            i_sep = ts.paths[0].rfind(os.sep)
            folder = ts.paths[0][:i_sep]
        self.folder = folder
            
        self.ts = ts
        
        hdf_path_data = self.folder + os.sep + 'data.hdf5'
        writeMeanData(ts, data_name, hdf_path_data)
        
        self.data = HDFData(hdf_path_data, openFile=True)

        # number of space nodes (points), time steps and spatial components
        self.nx, self.nt, self.nc = self.data_3d.shape
        # store the 2-dimensionnal array for the data
        self.data_2d = tensor2vector(data)

    def splitMeanFluct(self, set_mean_to_zero=False):
        """
        Compute the mean and fluctuating fields (self.data_mean and
        self.data_fluc, respectively).
        
        The mean can be forced to 0 by passing the option set_mean_to_zero=True
        """
        mean = np.mean(self.data_2d, axis=1)
        if set_mean_to_zero:
            mean = 0*mean
        self.data_mean = mean
        self.data_fluc = np.zeros((self.nx*self.nc, self.nt))
        for i, col in enumerate(self.data_2d.T):
            self.data_fluc[:, i] = col - mean

    def computePODBasis(self, weighting_matrix="Id"):
        """
        nuild the POD basis from the snapshots (self.basis)
        """
        if weighting_matrix=="Id":
            W = 1.  # np.eye(self.Nx)
        else:
            assert False, 'weighting matrix {} is not defined'.format(weighting_matrix)

        # Temporal correlation matrix
        self.C = np.dot(self.data_fluc.T, np.dot(W, self.data_fluc))

        # Temporal correlation matrix eigen decomposition
        eigen_vals, eigen_vecs = np.linalg.eig(self.C)

        # Remove the imaginary part (which should be numerically close to zero)
        eigen_vals = np.real(eigen_vals)
        eigen_vecs = np.real(eigen_vecs)
        
        # sort by decreasing eigen values
        indices = sortIndices(eigen_vals)
        
        self.C_eigen_vals = [eigen_vals[n] for n in indices]
        self.C_eigen_vecs = eigen_vecs[:, np.array(indices)]

        # Define POD basis
        self.basis = np.dot(self.data_fluc, self.C_eigen_vecs)
        self.npod = self.basis.shape[1]
    
    def computeModesEnergy(self):
        self.modes_energy = list()
        for i, val in enumerate(self.C_eigen_vals):
            mode_energy = sum(self.C_eigen_vals[:i+1])/sum(self.C_eigen_vals)
            self.modes_energy.append(mode_energy)

    def truncatePODBasis(self, threshold=1e-6):
        self.computeModesEnergy()
        self.threshold = threshold
        self.npod = [me >= 1-self.threshold for me in self.modes_energy].index(True)+1
        self.basis = self.basis[:, :self.npod]

    def checkPODBasisIsOrthonormal(self):
        M = np.abs(np.dot(self.basis.T, self.basis))
        for i in range(M.shape[0]):
            M[i, i] = 0
        print("val max out of diag from np.dot(basis.T, basis) is {}".format(M.max()))

    def normalizePODBasis(self):
        for i in range(self.npod):
            scalarprod = np.dot(self.basis[:,i], self.basis[:,i])
            self.basis[:,i] = self.basis[:,i]/np.sqrt(scalarprod)
            
    def temporalCoefficients(self, Nmodes=None):
        if Nmodes is None:
            Nmodes = self.npod
        return np.dot(self.data_fluc.T, self.basis[:, :Nmodes])
    
    def reconstructData(self, Nmodes=None):
        if Nmodes is None:
            Nmodes = self.npod
        coeffs = self.temporalCoefficients(Nmodes=Nmodes)
        self.reconstruction = np.zeros(self.data_2d.shape)
        for t in range(coeffs.shape[0]):
            self.reconstruction[:, t] = self.data_mean + \
                np.dot(self.basis[:, :Nmodes], coeffs[t, :].T)
        
    def reconstructionError(self, Nmodes=None):
        if Nmodes is None:
            Nmodes = self.Npod
        self.reconstructData(Nmodes=Nmodes)
        
        def norm(data_2d):
            return np.mean((np.array(data_2d)*np.array(data_2d)).sum(axis=0))

        return norm(self.data_2d-self.reconstruction)

