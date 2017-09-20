#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 11:43:09 2017

@author: afalaize

Reduced order model for fluid structure interaction with volumic penalization
for imposed velocity

"""

from __future__ import division
import numpy as np
from ..io.hdf import HDFReader
from ..misc import concatenate_in_given_axis
from scipy.optimize import root
import progressbar


class ReducedOrderModel(object):
    """
    """
    def __init__(self, paths):
        """
        """
        self.paths = paths

        for k in self.paths:
            setattr(self, k, HDFReader(self.paths[k]))

    def open_hdfs(self):
        for k in self.paths:
            hdffile = getattr(self, k)
            hdffile.openHdfFile()

    def close_hdfs(self):
        for k in self.paths:
            hdffile = getattr(self, k)
            hdffile.closeHdfFile()

    def nc(self):
        """
        return the number of spatial components
        """
        return self.basis.basis[:].shape[1]

    def npod(self):
        """
        return the number of pod basis elements
        """
        return self.basis.basis[:].shape[2]

    def nx(self):
        """
        return the number of mesh nodes
        """
        return self.basis.basis[:].shape[0]

    def nt(self):
        """
        return the number of time steps in the original time serie
        """
        return self.original_coeffs.coeffs[:].shape[1]

    def A(self):
        return self.matrices.a[:]

    def B(self):
        return self.matrices.b[:]

    def C(self):
        return self.matrices.c[:]

    def F(self):
        return self.matrices.f[:]

    def imp_func(self, delta_coeffs, coeffs, t, delta_t, theta):
        tA = np.einsum('ij,j->i', self.A(), delta_coeffs)/delta_t
        tB = np.einsum('ij,j->i', self.B(), coeffs+theta*delta_coeffs)
        tC = np.einsum('ijk,j,k->i',
                       self.C(),
                       coeffs+theta*delta_coeffs,
                       coeffs+theta*delta_coeffs)
        tF = self.F()
        return tA + tB + tC + tF

    def run(self, dt=0.01, tend=1, theta=.5):

        self.dt = dt

        self.tstart = 0.

        self.tend = tend
        self.times = [self.tstart + n*dt for n in range(int((tend-self.tstart)/dt)+1)]

        self.coeffs = list()
        self.coeffs.append(self.original_coeffs.coeffs[0, :])
        delta_coeff = np.zeros(self.npod())

        bar = progressbar.ProgressBar(widgets=['ROM',
                                               progressbar.Timer(), ' ',
                                               progressbar.Bar(), ' (',
                                               progressbar.ETA(), ')\n', ])

        for i in bar(range(len(self.times))):
            t = self.times[i]
            c = self.coeffs[-1]

            args = (c, t, dt, theta)
            res = root(self.imp_func,
                       delta_coeff,
                       args)
            if not res.success:
                s = 'Convergence issue at time t={} (index {}):\n    {}'
                print(s.format(t, i, res.message))
            delta_coeff = res.x

            self.coeffs.append(c + delta_coeff)

    def c_rom(self, i=None):
        if i is None:
            return self.coeffs
        else:
            return [el[i] for el in self.coeffs]

    def c_fom(self, i=None):
        if i is None:
            return self.original_coeffs.coeffs[:]
        else:
            return [el[i] for el in self.original_coeffs.coeffs[:]]

    def reconstruction(self, time=None):
        def reconstruct(c):
            return np.einsum('xci,i->xc', self.basis.basis[:], c) + self.meanfluc.mean[:]

        if time is not None:
            return reconstruct(self.coeffs[time])
        else:
            def generator():
                for c in self.coeffs:
                    yield reconstruct(c)
            return concatenate_in_given_axis(generator(), 2)

###############################################################################


# rhof: fluid density
# phi: pod basis associated with velocity


def A(phi, rhof):
    """
    Return the coefficients array A[i, j] = rhof*phi[x, c, i]*phi[x, c, j]
    with dims (npod, npod)

    Input
    ------

    phi : array with dims (nx, nc, npod)
        POD basis.

    rhof: float
        fluid density

    Output
    ------

    A : array with dims (npod, npod)
        A[i, j] = rhof*phi[x, c, i]*phi[x, c, j]

    """
    return rhof*np.einsum('mci,mcj->ij', phi, phi)


def mu_stab(mu, stab, nmodes):
    """
    Return a vector of viscosity coefficients with i-th element given by
    mu_i = mu*(1+stab*i)

    Parameters
    ----------
    mu : float
        reference velocity
    stab : float
        linear progression
    nmodes : int
        number of modes (i.e. length of output)
    """
    return mu*(1+stab*(1+np.array(range(nmodes))))


def B(phi, grad_phi, u_moy, grad_u_moy, muf, rhof, stab=0):
    nmodes = phi.shape[2]
    mu_modes = mu_stab(muf, stab, nmodes)

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
        t1 = np.einsum('mcdj,md->mcj', grad_phi, u_moy)
        t2 = np.einsum('mcd,mdj->mcj', grad_u_moy, phi)
        return rhof*np.einsum('mcj,mci->ij', t1 + t2, phi)

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
            b_tilde[i, j] = (u_moy[m, d]*grad_phi[m, d, c, j] +
                              phi[m, d, j]*grad_u_moy[m, d, c])*phi[m, c, i]
        """
        D = grad_phi + grad_phi.swapaxes(1, 2)
        trace_arg = np.einsum('mcdj,mcdi->cdij', D, grad_phi)
        return np.einsum('ccij,i->ij', trace_arg, mu_modes)

    return B_bar() + B_tilde()

###############################################################################
# TO DO

#
#
#def C(phi, L, umoy, rho_delta):
#    """
#    Return the coefficients array
#    C[i, j] = rho_delta*L[m, j]*umoy[m, c]*phi[m, c, i]
#    with dims (nL, nPhi)
#
#    Input
#    ------
#
#    phi : array with dims (nx, npod, nc)
#        POD basis.
#
#    grad_phi : array with dims (nx, npod, nc, nc)
#
#    Output
#    ------
#
#    C : array with dims (nL, npod)
#        C[i, j] = rho_delta*L[m, j]*umoy[m, c]*phi[m, i, c]
#
#    """
#    temp = np.einsum('mj,mc->mcj', L, umoy)
#    return rho_delta*np.einsum('mcj,mci->ij', temp, phi)
#
#
#def C(phi, L, umoy, rho_delta):
#    """
#    Return the coefficients array
#    D[i, j] = rho_delta*L[m, j]*umoy[m, c]*phi[m, c, i]
#    with dims (nL, nPhi)
#
#    Input
#    ------
#
#    phi : array with dims (nx, npod, nc)
#        POD basis.
#
#    grad_phi : array with dims (nx, npod, nc, nc)
#
#    Output
#    ------
#
#    C : array with dims (nL, npod)
#        C[i, j] = rho_delta*L[m, j]*umoy[m, c]*phi[m, i, c]
#
#    """
#    temp = np.einsum('mj,mc->mcj', L, umoy)
#    return rho_delta*np.einsum('mcj,mci->ij', temp, phi)
#
#
#def F(phi, grad_phi, u_moy, grad_u_moy, mu, rho, stab=0):
#    nmodes = phi.shape[2]
#    mu_modes = mu_stab(mu, stab, nmodes)
#
#    def F_bar():
#        """
#        """
#        return np.einsum('mc,mdc,mdi->i', u_moy, grad_u_moy, phi)
#
#    def F_tilde():
#        """
#        """
#        D = (grad_u_moy + grad_u_moy.swapaxes(1, 2))/2.
#        trace_arg = np.einsum('mcd,mdei->cei', D, grad_phi)
#        return (2./rho)*np.einsum('cci,i->i', trace_arg, mu_modes)
#    return F_bar() + F_tilde()
