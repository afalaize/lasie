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

def build_rom_coefficients_A(hdf_path_podBasis, hdf_path_A):
    """
    """
    basis = HDFData(hdf_path_podBasis, openFile=True)
    array_a = a(basis.get_single_data())
    dumpArrays2Hdf([array_a, ], ['a', ], hdf_path_A)
    basis.closeHdfFile()


def build_rom_coefficients_B(hdf_path_podBasis, hdf_path_podBasisGradient,
                             hdf_path_mean, hdf_path_meanGradient, hdf_path_B):
    """
    """
    basis = HDFData(hdf_path_podBasis, openFile=True)
    basis_gradient = HDFData(hdf_path_podBasisGradient, openFile=True)
    mean = HDFData(hdf_path_mean, openFile=True)
    mean_gradient = HDFData(hdf_path_meanGradient, openFile=True)

    array_b_bar = b_bar(basis.get_single_data(),
                        basis_gradient.get_single_data(),
                        mean.get_single_data(),
                        mean_gradient.get_single_data())

    array_b_tilde = b_tilde(basis_gradient.get_single_data())

    dumpArrays2Hdf([array_b_bar, array_b_tilde], ['b_bar', 'b_tilde'],
                   hdf_path_B)

    for hdf in [basis, basis_gradient, mean, mean_gradient]:
        hdf.closeHdfFile()


def build_rom_coefficients_C(hdf_path_podBasis, hdf_path_podBasisGradient,
                             hdf_path_C):
    """
    """
    basis = HDFData(hdf_path_podBasis, openFile=True)
    basis_gradient = HDFData(hdf_path_podBasisGradient, openFile=True)

    array_C = c(basis.get_single_data(), basis_gradient.get_single_data())
    dumpArrays2Hdf([array_C, ], ['c', ], hdf_path_C)
    basis.closeHdfFile()
    basis_gradient.closeHdfFile()


def build_rom_coefficients_F(hdf_path_podBasis, hdf_path_podBasisGradient,
                             hdf_path_mean, hdf_path_meanGradient,
                             hdf_path_F, hdf_path_source=None):
    """
    """
    basis = HDFData(hdf_path_podBasis, openFile=True)
    basis_gradient = HDFData(hdf_path_podBasisGradient, openFile=True)
    mean = HDFData(hdf_path_mean, openFile=True)
    mean_gradient = HDFData(hdf_path_meanGradient, openFile=True)
    if hdf_path_source is not None:
        source = HDFData(hdf_path_source, openFile=True)
        array_f_bar = f_bar(basis.get_single_data(), source.get_single_data())
    else:
        array_f_bar = f_bar(basis.get_single_data())

    array_f_tilde = f_tilde(mean_gradient.get_single_data(),
                            basis_gradient.get_single_data())

    array_f_hat = f_hat(mean.get_single_data(),
                        mean_gradient.get_single_data(),
                        basis.get_single_data())

    dumpArrays2Hdf([array_f_bar, array_f_tilde, array_f_hat],
                   ['f_bar', 'f_tilde', 'f_hat'], hdf_path_F)

    for hdf in [basis, basis_gradient, mean, mean_gradient]:
        hdf.closeHdfFile()


class ReducedOrderModel(object):
    """
    """
    def __init__(self, config):
        """
        """
        self.config = config
        self.ts = HDFTimeSerie(config['interp_hdf_folder'])
        print('Open TimeSerie HDF...')
        self.ts.openAllFiles()

        print('Open POD basis HDF...')
        self.basis = HDFData(config['hdf_path_podBasis'], openFile=True)

        for name in 'ABCF':
            print('Open element {}...'.format(name))
            hdf = HDFData(config['hdf_path_'+name], openFile=True)
            setattr(self, name+'_coeffs', hdf)

        print('Open ThosT Temporal Coeff...')
        self.Thost_temporal_coeffs = HDFData(config['hdf_path_Thost_temporal_coeffs'], openFile=True)

    def nc(self):
        """
        return the number of spatial components
        """
        return self.basis.vitesse.shape[2]

    def npod(self):
        """
        return the number of pod basis elements
        """
        return self.basis.vitesse.shape[1]

    def nx(self):
        """
        return the number of mesh nodes
        """
        return self.basis.vitesse.shape[0]

    def nt(self):
        """
        return the number of time steps in the original time serie
        """
        return len(self.ts.times)

    def rho(self, t):
        return self.ts.time_interp('massevolumique', t)[:, 0]

    def mu(self, t):
        return self.ts.time_interp('eta', t)[:, 0]

    def A(self, t):
        return np.einsum('m,mij->ij',
                         self.temp_rho,
                         self.A_coeffs.a[:])

    def B(self, t):
        return (np.einsum('m,mij->ij', self.temp_rho, self.B_coeffs.b_bar[:]) +
                np.einsum('m,mij->ij', self.temp_mu, self.B_coeffs.b_tilde[:]))

    def C(self, t):
        return np.einsum('m,mijk->ijk', self.temp_rho, self.C_coeffs.c[:])

    def F(self, t):
        """
        """
        return (self.F_coeffs.f_bar[:] +
                np.einsum('m,mi->i', self.temp_rho, self.F_coeffs.f_hat[:]) +
                np.einsum('m,mi->i', self.temp_mu, self.F_coeffs.f_tilde[:]))

    def imp_func(self, delta_coeffs, coeffs, t, delta_t, beta, theta1, theta2):
        tA = np.einsum('ij,j->i', self.temp_A, delta_coeffs)/delta_t
        tB = np.einsum('ij,j->i', self.temp_B, coeffs+beta*delta_coeffs)
        tC = np.einsum('ijk,j,k->i',
                       self.temp_C,
                       coeffs+theta1*delta_coeffs,
                       coeffs+theta2*delta_coeffs)
        tF = self.temp_F
        return tA + tB + tC - tF

    def jac_imp_func(self, delta_coeffs, coeffs, t, delta_t, beta, theta1, theta2):
        tA = self.temp_A/delta_t
        tB = beta*self.temp_B
        tC1 = np.einsum('jl,k->ljk',
                        np.eye(self.npod()),
                        coeffs + theta2*delta_coeffs)
        tC2 = np.einsum('kl,j->ljk',
                        np.eye(self.npod()),
                        coeffs + theta1*delta_coeffs)
        tC = np.einsum('ijk,ljk->il', self.temp_C, theta1*tC1 + theta2*tC2)
        return tA + tB + tC

    def run(self, dt=None,
            tend=None, istart=None,
            beta=1., theta1=1., theta2=1.):

        if istart is None:
            istart = 0
        else:
            assert istart < self.nt()-2
        self.istart = istart

        if dt is None:
            dt = self.ts.times[1]-self.ts.times[0]
        self.dt = dt

        self.tstart = self.ts.times[self.istart]

        if tend is None:
            tend = self.ts.times[-1]
        self.tend = tend
        self.times = [self.tstart + n*dt for n in range(int((tend-self.tstart)/dt)+1)]

        self.coeffs = list()
        self.coeffs.append(self.Thost_temporal_coeffs.get_single_data()[self.istart, :])
        delta_coeff = np.zeros(self.npod())

        bar = progressbar.ProgressBar(widgets=[progressbar.Timer(), ' ',
                                               progressbar.Bar(), ' (',
                                               progressbar.ETA(), ')\n', ])

        for name in ['rho', 'mu', 'A', 'B', 'C', 'F']:
            print('init ' + name + '...')
            setattr(self, 'temp_'+name, getattr(self, name)(self.tstart))

        for i in bar(range(len(self.times))):
            t = self.times[i]
            args = (self.coeffs[-1], t, dt, beta, theta1, theta2)
            res = root(self.imp_func,
                       delta_coeff,
                       args,
                       jac = self.jac_imp_func)
            if not res.success:
                s = 'Convergence issue at time t={} (index {}):\n    {}'
                print(s.format(t, i, res.message))

            delta_coeff = res.x
            self.coeffs.append(self.coeffs[i] + delta_coeff)

    def c_rom(self, i=None):
        if i is None:
            return self.coeffs
        else:
            return [el[i] for el in self.coeffs]

    def c_fom(self, i=None):
        if i is None:
            return self.Thost_temporal_coeffs
        else:
            return [el[i] for el in self.Thost_temporal_coeffs.get_single_data()]

###############################################################################


def a(phi):
    """
    Return the coefficients array a[m, i, j] = phi[m, i, c]*phi[m, j, c]
    with dims (nx, npod, npod)

    Input
    ------

    phi : array with dims (nx, npod, nc)
        POD basis.

    Output
    ------

    a : array with dims (nx, npod, npod)
        a[m, i, j] = phi[m, i, c]*phi[m, j, c]

    """
    return np.einsum('mic,mjc->mij', phi, phi)


def A(rho, a):
    """
    Return the matrix A = (rho phi_j, phi_i) with dims (npod, npod)

    Inputs
    -------
    rho : array with dims (nx, )
        Pointwise fluid density (kg/m3).

    a : array with dim (nx, npod, npod)
        Coefficients a[m, i, j] returned by the function a(phi) above.

    Output
    -------

    A : array with dims (npod, npod)
        A[i, j] = rho[m]*a[m, i, j]
    """
    return np.einsum('m,mij->ij', rho, a)


def b_bar(phi, grad_phi, u_moy, grad_u_moy):
    """
    Return the coefficients array b_bar with dims (nx, npod, npod).

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

    b_bar : array with dims (nx, npod, npod)
        b_bar[m, i, j] = (u_moy[m, d]*grad_phi[m, j, d, c] +
                          phi[m, j, d]*grad_u_moy[m, d, c])*phi[m, i , c]
    """
    t1 = np.einsum('md,mjdc->mjc', u_moy, grad_phi)
    t2 = np.einsum('mjd,mdc->mjc', phi, grad_u_moy)
    return np.einsum('mjc,mic->mij', np.add(t1, t2), phi)


def b_tilde(grad_phi):
    """
    Return the coefficients array b_tilde with dims (nx, npod, npod).

    Inputs
    -------

    grad_phi : array with dims (nx, npod, nc, nc)
        Gradient of POD basis.

    Output
    -------

    b_tilde : array with dims (nx, npod, npod)
        b_tilde[m, i, j] = (u_moy[m, d]*grad_phi[m, j, d, c] +
                          phi[m, j, d]*grad_u_moy[m, d, c])*phi[m, i , c]
    """
    return np.einsum('mjce,mice->mij',
                     (grad_phi + grad_phi.swapaxes(2, 3))/2,
                     grad_phi)


def B(rho, mu, b_bar, b_tilde):
    """
    Return the matrix B = rho[m]*b_bar[m, i, j] + mu[m]*b_tilde[m, i, j]
    with dims (npod, npod)

    Inputs
    -------
    rho : array with dims (nx, )
        Pointwise fluid density (kg/m3).

    mu : array with dims (nx, )
        Pointwise fluid dynamic viscosity (Pa.s).

    b_bar : array with dim (nx, npod, npod)
        Coefficients b_bar[m, i, j] returned by the function b_bar above.

    b_tilde : array with dim (nx, npod, npod)
        Coefficients b_tilde[m, i, j] returned by the function b_tilde above.

    Output
    -------

    B : array with dims (npod, npod)
        B[i, j] = rho[m]*b_bar[m, i, j] + mu[m]*b_tilde[m, i, j]
    """
    return np.sum((np.einsum('m,mij->ij', rho, b_bar),
                   np.einsum('m,mij->ij', mu, b_tilde)))


def c(phi, grad_phi):
    """
    Return the coefficients array c[m, i, j, k] = phi[m, k, c]*grad_phi[m, j, c, d]*phi[m, i, d]
    with dims (nx, npod, npod, npod)

    Input
    ------

    phi : array with dims (nx, npod, nc)
        POD basis.

    Output
    ------

    c : array with dims (nx, npod, npod)
        a[m, i, j] = phi[m, i, c]*phi[m, j, c]

    """
    return np.einsum('mkc,mjcd,mid->mijk', phi, grad_phi, phi)


def C(rho, c):
    """
    Return the tensor C = (rho phi_k grad phi_j, phi_i) with dims (npod, npod, npod)

    Inputs
    -------
    rho : array with dims (nx, )
        Pointwise fluid density (kg/m3).

    c : array with dim (nx, npod, npod)
        Coefficients c[m, i, j, k] returned by the function c(phi, grad_phi) above.

    Output
    -------

    C : array with dims (npod, npod, npod)
        C[i, j, k] = rho[m]*phi[m, k, c]*grad_phi[m, j, c, d]*phi[m, i, d]
    """
    return np.einsum('m,mijk->ijk', rho, c)


def f_bar(phi, f=None):
    """
    """
    if f is None:
        return np.zeros((phi.shape[1], ))
    else:
        return np.einsum('mc,mic->i', f, phi)


def f_hat(u_moy, grad_u_moy, phi):
    """
    """
    return -np.einsum('mc,mcd,mid->mi', u_moy, grad_u_moy, phi)


def f_tilde(grad_u_moy, grad_phi):
    """
    """
    return -np.einsum('mcd,micd->mi',
                      grad_u_moy + grad_u_moy.swapaxes(1,2),
                      grad_phi)/2.


def f(rho, mu, f_bar, f_hat, f_tilde):
    """
    """
    return np.sum((f_bar,
                   np.einsum('m,mi->i', rho, f_hat),
                   np.einsum('m,mi->i', mu, f_tilde)))
