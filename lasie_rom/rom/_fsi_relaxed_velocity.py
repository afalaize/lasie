# -*- coding: utf-8 -*-
"""
Created on Thu Feb 02 11:53:24 2017

@author: afalaize
"""

from __future__ import division
import numpy as np
from .. import operators
from ..io.hdf import HDFReader
from ..misc import concatenate_in_given_axis, smooth
from scipy.optimize import root
import progressbar


class ReducedOrderModel(object):
    """
    """

    def __init__(self, hdfpaths, parameters, levelset_func, velocity_func):
        """
        hdfpaths: dictionary
            {'name': 'path/to/hdf/file.hdf', ...}
            with name in :
            'grid',
            'basis',
            'matrices',
            'meanfluc'.
            'original_coeffs',

        parameters: dictionary
            Simulation, POD-ROM and runtime parameters.

        levelset_func: function(t, x1, ..., xD) s.t.
        - levelset_func(t, x)>0 for x in the t-dependent solid domain,
        - levelset_func(t, x)<0 for x in the t-dependent fluid domain,
        - levelset_func(t, x)=0 for x in the t-dependent fluid/solid interface.
        levelset_func must accept numpy.array arguments (see numpy.vectorize).
        """
        self.parameters = parameters

        # recover hdf files
        self.paths = hdfpaths
        for k in self.paths:
            setattr(self, k, HDFReader(self.paths[k]))

        # define smoothed heaviside function
        eps = self.parameters['eps_tanh']
        self.heaviside = smooth.build_vectorized_heaviside(eps)

        # define a levelset constructor that takes the time as a aprameter
        self.levelset = levelset_func
        self.velocity = velocity_func

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

    def update_Is(self, t):
        self._Is = self.heaviside(self.levelset(t, *[xi for xi in self.grid.mesh[:].T]))

    def update_rho_and_nu(self):
        self._rho = self.parameters['rho'] + self.parameters['rho_delta']*self._Is
        self._nu = self.parameters['nu'] + self.parameters['nu_delta']*self._Is

    def update_A(self):
        self._A = np.einsum('x,xij->ij', self._rho, self.matrices.a[:])

    def update_B(self):
        self._B = (np.einsum('x,xij->ij', self._rho, self.matrices.b_rho[:]) +
                   np.einsum('x,xij->ij', self._nu, self.matrices.b_nu[:]))

    def update_C(self):
        self._C = np.einsum('x,xijk->ijk', self._rho, self.matrices.c[:])

    def update_f_rho_and_nu(self):
        self._f_rho = np.einsum('x,xi->i', self._rho, self.matrices.f_rho[:])
        self._f_nu = np.einsum('x,xi->i', self._nu, self.matrices.f_nu[:])

    def update_u(self, c):
        self._u_old = self._u
        self._u = self.reconstruct(c)

    def update_F(self):
        self._F = self._f_rho + self._f_nu + np.einsum('x,xi->i',
                                                       self._Is,
                                                       self._f_lambda)

    def update_lambda(self):
        self._lambda = self._lambda - self.dt*self.parameters['c_lambda']*(self._u-self._Vs[:,:])

    def update_f_lambda(self):
        # update f_lambda
        self._f_lambda = np.einsum('xc,xci->xi', self._lambda, self.basis.basis[:])

    def imp_func(self, coeffs, coeffs_l, coeffs_n):
        return (np.einsum('ij,j->i',
                          self._A/self.dt + self._B + np.einsum('ijk,k->ij',
                                                                self._C,
                                                                coeffs_l),
                          coeffs) -
                np.einsum('ij,j->i',
                          self._A/self.dt,
                          coeffs_n) +
                self._F)

    def run(self, dt=0.01, tend=1):

        self.dt = dt

        theta = self.parameters['theta_init']

        self.tstart = 0.

        self.tend = tend
        self.times = [self.tstart + n*dt
                      for n in range(int((tend-self.tstart)/dt)+1)]

        self.coeffs = list()
        self.coeffs.append(self.original_coeffs.coeffs[0, :])

        # define the velocity associated with the the solid rotation
        self._Vs = self.velocity(*[xi for xi in self.grid.mesh[:].T])

        self._u = np.random.randn(self.nx(), self.nc())
        self.update_u(self.coeffs[-1])

        self._lambda = np.zeros((self.nx(), self.nc()))

        bar = progressbar.ProgressBar(widgets=['ROM', ' ',
                                               progressbar.Timer(), ' ',
                                               progressbar.Bar(), ' (',
                                               progressbar.ETA(), ')\n', ])

        for i in bar(range(len(self.times)-1)):

            theta += self.dt*self.parameters['angular_vel']
            self.update_Is(self.times[i+1])
            self.update_rho_and_nu()
            self.update_A()
            self.update_B()
            self.update_C()
            self.update_f_rho_and_nu()

            l = 0
            coeffs_l = self.coeffs[-1]

            test_u = float('Inf')
            test_lambda = float('Inf')

            self._lambda = np.zeros((self.nx(), self.nc()))

            while (test_u > self.parameters['eps_u'] and
                   test_lambda > self.parameters['eps_lambda']):

                # increment counter
                l += 1

                # update f_lambda
                self.update_f_lambda()

                # update F = f_rho + f_nu + f_lambda
                self.update_F()

                # solve implicit function for new coeffs_l
                args = (coeffs_l, self.coeffs[-1])
                res = root(self.imp_func,
                           coeffs_l,
                           args)
                if not res.success:
                    s = 'Convergence issue at time t={} (index {}):\n    {}'
                    print(s.format('timestep {}'.format(i), res.message))
                coeffs_l = res.x

                # update u
                self.update_u(coeffs_l)

                # update lambda
                self.update_lambda()

                # make tests
                test_u = np.linalg.norm(self._u-self._u_old)/np.linalg.norm(self._u_old)
                test_lambda = np.linalg.norm(np.einsum('xc,x->xc', self._u-self._Vs, self._Is))

                # Printing
                message = 'n={0}, l={1}, test_u={2}, test_lambda={3}'
                print(message.format(i, l, test_u, test_lambda))

            self.coeffs.append(coeffs_l)

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

    def reconstruct(self, c):
        return (np.einsum('xci,i->xc', self.basis.basis[:], c) +
                self.meanfluc.mean[:])

    def reconstruction(self, time=None):

        if time is not None:
            return self.reconstruct(self.coeffs[time])
        else:
            def generator():
                for c in self.coeffs:
                    yield self.reconstruct(c)
            return concatenate_in_given_axis(generator(), 2)

###############################################################################


def temp_a(phi):
    return np.einsum('xcj,xci->xij', phi, phi)


def temp_b_rho(phi, u_moy, grad_phi, grad_u_moy):
    temp1 = np.einsum('xcdj,xd->xcj', grad_phi, u_moy)
    temp2 = np.einsum('xcd,xdj->xcj', grad_u_moy, phi)
    return np.einsum('xcj,xci->xij', temp1+temp2, phi)


def temp_b_nu(grad_phi):
    temp = 0.5*(grad_phi+grad_phi.swapaxes(1, 2))
    arg_of_trace = np.einsum('xcej,xedi->xcdij', temp, temp)
    return 2*np.einsum('xccij->xij', arg_of_trace)


def temp_c(phi, grad_phi):
    return np.einsum('xcdj,xdk,xci->xijk', grad_phi, phi, phi)


def temp_d(grad_phi):
    return 0.5 * (grad_phi + grad_phi.swapaxes(1, 2))

def temp_f_rho(phi, u_moy, grad_u_moy):
    return np.einsum('xcd,xd,xci->xi', grad_u_moy, u_moy, phi)

def temp_f_nu(grad_phi, grad_u_moy):
    temp_phi = 0.5*(grad_phi+grad_phi.swapaxes(1, 2))
    temp_u_moy = 0.5*(grad_u_moy+grad_u_moy.swapaxes(1, 2))
    arg_of_trace = np.einsum('xcei,xed->xcdi', temp_phi, temp_u_moy)
    return 2*np.einsum('xcci->xi', arg_of_trace)


def mu_stab(mu, stab, nmodes):
    return mu*(1+stab*(1+np.array(range(nmodes))))
