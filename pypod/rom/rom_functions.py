#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 10 12:48:19 2017

@author: Falaize
"""

import numpy as np


def a(phi):
    """
    Return the coefficients array a[m, i, j] = phi[m, i, c]*phi[m, j, c]
    with dims (nx, nc, nc)

    Input
    ------

    phi : array with dims (nx, npod, nc)
        POD basis.

    Output
    ------

    a : array with dims (nx, npod, npod)
        a[m, i, j] = phi[m, i, c]*phi[m, j, c]

    """
    return np.einsum('mic, mjc->mij', phi, phi)


def A(rho, phi):
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


def b_rho(phi, grad_phi, u_moy, grad_u_moy):
    t1 = np.einsum('md,mjdc->mjc', u_moy, grad_phi)
    t2 = np.einsum('mjd,mdc->mjc', phi, grad_u_moy)
    return np.add(t1, t2)


def B_rho(phi, b_rho):
    return np.einsum('mic,mjc->mij', phi, b_rho)


def D_phi(grad_phi):
    """
    strain tensor D(phi) = (grad_phi + grad_phi.T)/2

    Input
    ------

    grad_phi : array with shape (nx, npod, nc, nc)
        Gradient of pod basis.


    Output
    -------

    D : array with shape (nx, npod, nc, nc)
        D[m, i, c, d] = (grad_phi[m, i, c, d] + grad_phi[m, i, d, c])/2

    """

    return np.einsum('micd,midc->micd', grad_phi, grad_phi)/2.


def B_mu(D_phi, grad_phi):
    return np.einsum('mjcd,micd->mij', D_phi, grad_phi)


def B(rho, B_rho, mu, B_mu):
    t_rho = np.einsum('m,mij->ij', rho, B_rho)
    t_mu = np.einsum('m,mij->ij', mu, B_mu)
    return t_rho + t_mu


def C(phi, grad_phi):
    return np.einsum('mkc,mjcd,mid->ijk', phi, grad_phi, phi)


def D_u_moy(grad_u_moy):
    """
    strain tensor D(u_moy) = (grad_u_moy + grad_u_moy.T)/2

    Input
    ------

    grad_u_moy : array with shape (nx, nc, nc)
        Gradient of pod basis.


    Output
    -------

    D : array with shape (nx, nc, nc)
        D[m, c, d] = (grad_u_moy[m, c, d] + grad_phi[m, d, c])/2

    """

    return np.einsum('mcd,mdc->mcd', grad_u_moy, grad_u_moy)/2.


def f_bar(f, phi):
    return np.einsum('mc,mic->i', f, phi)


def f_hat(u_moy, grad_u_moy, phi):
    return -np.einsum('mc,mcd,mid->mi', u_moy, grad_u_moy, phi)


def f_tilde(D_u_moy, grad_phi):
    return np.einsum('mcd,micd->mi')


def f(rho, mu, f_bar, f_hat, f_tilde):
    f_rho = np.einsum('m,mi->i', rho, f_hat)
    f_mu = np.einsum('m,mi->i', mu, f_tilde)
    return f_bar + f_rho + f_mu
