# -*- coding: utf-8 -*-
"""
Created on Thu Feb 02 11:53:24 2017

@author: afalaize
"""

from __future__ import division
import numpy as np
from ..readwrite.read_hdf import HDFTimeSerie, HDFData
from ..pod.pod import scalarProductArray, getWeightingMatrix
from ..readwrite.vtu2hdf import dumpArrays2Hdf
from scipy.optimize import root
import progressbar



class ReducedOrderModel(object):
    """
    """
    def __init__(self, config):
        """
        """
        self.config = config

        print('Open POD basis HDF...')
        self.basis = HDFData(config['hdf_path_podBasis'])

        for name in 'ABCF':
            print('Open element {}...'.format(name))
            hdf = HDFData(config['hdf_path_'+name])
            setattr(self, name+'_coeffs', hdf)

        print('Open ThosT Temporal Coeff...')
        self.Thost_temporal_coeffs = HDFData(config['hdf_path_Thost_temporal_coeffs'])
        
        self.open_hdfs()

    def open_hdfs(self):
        for name in 'ABCF':
            getattr(self, name+'_coeffs').openHdfFile()
        self.Thost_temporal_coeffs.openHdfFile()
        self.basis.openHdfFile()

    def close_hdfs(self):
        for name in 'ABCF':
            getattr(self, name+'_coeffs').closeHdfFile()
        self.Thost_temporal_coeffs.closeHdfFile()
        self.basis.closeHdfFile()

    def nc(self):
        """
        return the number of spatial components
        """
        return self.basis.get_single_data().shape[2]

    def npod(self):
        """
        return the number of pod basis elements
        """
        return self.basis.get_single_data().shape[1]

    def nx(self):
        """
        return the number of mesh nodes
        """
        return self.basis.get_single_data.shape[0]

    def nt(self):
        """
        return the number of time steps in the original time serie
        """
        return len(self.ts.times)

    def rho(self):
        return self.ts.data[0].massevolumique[:].flatten()[0]

    def mu(self):
        return self.ts.data[0].eta[:].flatten()[0]

    def A(self):
        return self.A_coeffs.get_single_data()

    def B(self):
        return self.B_coeffs.get_single_data()

    def C(self):
        return self.C_coeffs.get_single_data()

    def F(self):
        """
        """
        return self.F_coeffs.get_single_data()

    def imp_func(self, delta_coeffs, coeffs, t, delta_t, theta):
        tA = np.einsum('ij,j->i', self.A(), delta_coeffs)/delta_t
        tB = np.einsum('ij,j->i', self.B(), coeffs+theta*delta_coeffs)
        tC = np.einsum('ijk,j,k->i',
                       self.C(),
                       coeffs+theta*delta_coeffs,
                       coeffs+theta*delta_coeffs)
        tF = self.F()
        return tA + tB + tC + tF

    def run(self, dt=0.01, tend=52.5, theta=.5):

        self.dt = dt

        self.tstart = 50.

        self.tend = tend
        self.times = [self.tstart + n*dt for n in range(int((tend-self.tstart)/dt)+1)]

        self.coeffs = list()
        self.coeffs.append(self.Thost_temporal_coeffs.get_single_data()[0, :])
        delta_coeff = np.zeros(self.npod())

        bar = progressbar.ProgressBar(widgets=[progressbar.Timer(), ' ',
                                               progressbar.Bar(), ' (',
                                               progressbar.ETA(), ')\n', ])

        for i in bar(range(len(self.times))):
            t = self.times[i]
            c = self.coeffs[-1]
# Theta + solveur implicite
            args = (c, t, dt, theta)
            res = root(self.imp_func,
                       delta_coeff,
                       args)
            if not res.success:
                s = 'Convergence issue at time t={} (index {}):\n    {}'
                print(s.format(t, i, res.message))
            delta_coeff = res.x
# Euelr explicite
#            iA = np.linalg.inv(self.A()/dt)
#            delta_coeff = -np.dot(iA, (np.dot(self.B(), c) +
#                                       np.dot(np.dot(self.C(), c), c) +
#                                       self.F()))
#
            self.coeffs.append(c + delta_coeff)
            
    def c_rom(self, i=None):
        if i is None:
            return self.coeffs
        else:
            return [el[i] for el in self.coeffs]

    def c_fom(self, i=None):
        if i is None:
            return self.Thost_temporal_coeffs.get_single_data()
        else:
            return [el[i] for el in self.Thost_temporal_coeffs.get_single_data()]

###############################################################################


def A(phi):
    """
    Return the coefficients array A[i, j] = phi[m, i, c]*phi[m, j, c]
    with dims (npod, npod)

    Input
    ------

    phi : array with dims (nx, npod, nc)
        POD basis.

    Output
    ------

    A : array with dims (npod, npod)
        A[i, j] = phi[m, i, c]*phi[m, j, c]

    """
#    return np.einsum('mic,mjc->ij', phi, phi)
    return np.eye(phi.shape[1])


def B(phi, grad_phi, u_moy, grad_u_moy, mu, rho):

    def B_bar():
        """
        Return the coefficients array b_bar with dims (npod, npod).

        Inputs
        -------
        phi : array with dims (nx, npod, nc)
            POD basis.

        grad_phi : array with dims (nx, npod, nc, nc)
            Gradient of POD basis.

        u_moy : array with dims (nx, nc)
            Mean velocity.

        grad_u_moy : array with dims (nx, nc, nc)
            gradient of mean velocity

        Output
        -------

        b_bar : array with dims (npod, npod)
            b_bar[i, j] =  (u_moy[m, d]*grad_phi[m, j, d, c] +
                            phi[m, j, d]*grad_u_moy[m, d, c])*phi[m, i , c]
        """
        t1 = np.einsum('md,mjdc->mjc', u_moy, grad_phi)
        t2 = np.einsum('mjd,mdc->mjc', phi, grad_u_moy)
        return np.einsum('mjc,mic->ij', t1 + t2, phi)

    def B_tilde():
        """
        Return the coefficients array b_tilde with dims (npod, npod).

        Inputs
        -------

        grad_phi : array with dims (nx, npod, nc, nc)
            Gradient of POD basis.

        Output
        -------

        b_tilde : array with dims (npod, npod)
            b_tilde[i, j] = (u_moy[m, d]*grad_phi[m, j, d, c] +
                              phi[m, j, d]*grad_u_moy[m, d, c])*phi[m, i , c]
        """
        D = (grad_phi + grad_phi.swapaxes(2, 3))/2.
        trace_arg = np.einsum('mjce,mied->ijcd', D, grad_phi)
        return (2*mu/rho)*np.einsum('ijcc->ij', trace_arg)

    return B_bar() + B_tilde()


def C(phi, grad_phi):
    """
    Return the coefficients array
    C[i, j, k] = phi[m, k, c]*grad_phi[m, j, c, d]*phi[m, i, d]
    with dims (npod, npod, npod)

    Input
    ------

    phi : array with dims (nx, npod, nc)
        POD basis.

    grad_phi : array with dims (nx, npod, nc, nc)

    Output
    ------

    C : array with dims (npod, npod, npod)
        C[i, j, k] = phi[m, k, c]*gradphi[m, j, c, d]*phi[m, i, d]

    """
    temp = np.einsum('mjc,mkcd->mjkd', phi, grad_phi)
    return np.einsum('mjkd,mid->ijk', temp, phi)


def F(phi, grad_phi, u_moy, grad_u_moy, mu, rho):

    def F_bar():
        """
        """
        return np.einsum('mc,mcd,mid->i', u_moy, grad_u_moy, phi)

    def F_tilde():
        """
        """
        D = (grad_u_moy + grad_u_moy.swapaxes(1, 2))/2.
        trace_arg = np.einsum('mce,mied->icd', D, grad_phi)
        return (2*mu/rho)*np.einsum('icc', trace_arg)
    return F_bar() + F_tilde()


###############################################################################

###############################################################################


def build_rom_coefficients_A(hdf_path_podBasis, hdf_path_A):
    """
    """
    basis = HDFData(hdf_path_podBasis, openFile=True)
    array_a = A(basis.get_single_data())
    dumpArrays2Hdf([array_a, ], ['a', ], hdf_path_A)
    basis.closeHdfFile()


def build_rom_coefficients_B(hdf_path_podBasis, hdf_path_podBasisGradient,
                             hdf_path_mean, hdf_path_meanGradient, mu, rho,
                             hdf_path_B):
    """
    """
    basis = HDFData(hdf_path_podBasis, openFile=True)
    basis_gradient = HDFData(hdf_path_podBasisGradient, openFile=True)
    mean = HDFData(hdf_path_mean, openFile=True)
    mean_gradient = HDFData(hdf_path_meanGradient, openFile=True)

    array_B = B(basis.get_single_data(),
                basis_gradient.get_single_data(),
                mean.get_single_data(),
                mean_gradient.get_single_data(),
                mu, rho)

    dumpArrays2Hdf([array_B, ], ['b', ], hdf_path_B)

    for hdf in [basis, basis_gradient, mean, mean_gradient]:
        hdf.closeHdfFile()


def build_rom_coefficients_C(hdf_path_podBasis, hdf_path_podBasisGradient,
                             hdf_path_C):
    """
    """
    basis = HDFData(hdf_path_podBasis, openFile=True)
    basis_gradient = HDFData(hdf_path_podBasisGradient, openFile=True)

    array_C = C(basis.get_single_data(), basis_gradient.get_single_data())
    dumpArrays2Hdf([array_C, ], ['c', ], hdf_path_C)
    basis.closeHdfFile()
    basis_gradient.closeHdfFile()


def build_rom_coefficients_F(hdf_path_podBasis, hdf_path_podBasisGradient,
                             hdf_path_mean, hdf_path_meanGradient, mu, rho,
                             hdf_path_F):
    """
    """
    basis = HDFData(hdf_path_podBasis, openFile=True)
    basis_gradient = HDFData(hdf_path_podBasisGradient, openFile=True)
    mean = HDFData(hdf_path_mean, openFile=True)
    mean_gradient = HDFData(hdf_path_meanGradient, openFile=True)

    array_F = F(basis.get_single_data(),
                basis_gradient.get_single_data(),
                mean.get_single_data(),
                mean_gradient.get_single_data(),
                mu, rho)
    dumpArrays2Hdf([array_F, ], ['f', ], hdf_path_F)

    for hdf in [basis, basis_gradient, mean, mean_gradient]:
        hdf.closeHdfFile()


###############################################################################
