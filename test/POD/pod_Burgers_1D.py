#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:28:56 2017

@author: afalaize

Equation de Burger avec viscosité:

u_t + u.u_x - nu.u_xx = 0

"""

import numpy as np
import scipy as sc
from lasie_rom.grids import generate as gen_grid
from lasie_rom import operators
from lasie_rom import misc
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# --------------------------------------------------------------------------- #
# %% Parameters

NU = 0.1      # Viscosity

DELTAT = 0.2   # Time-step
THETA = 0.5     # Theta scheme in time (0:explicit, 0.5:midpoint, 1:implicit)

EPS = 1e-9      # Root finding: tolerance on the norm of the residu
MAXIT = 20      # Root finding: number of iterations

NM = 100      # Max number of modes
threshold = 1e-6  # threshold on energy associated to the eigne values of correlation matrix

# --------------------------------------------------------------------------- #
# Define temporal domain

T = 10*DELTAT

NT = int(T/DELTAT + 1)

Time = np.linspace(0, T, NT)


# --------------------------------------------------------------------------- #
# Define spatial domain

L = 1.

XINF = 0
XSUP = L

minmax = (XINF, XSUP)

NX = 100

grid, grid_h = gen_grid([minmax, ], npoints=NX)


# --------------------------------------------------------------------------- #
# %% Numerical scheme

# Build sparse matrix D that defines the finite difference approximation of the
# gradient with grad(f)(x_i) ~ (f(x_i+1)-f(x_i-1))/(2*DELTAX)

def row_D(i):
    """
    Return the i-th row of the matrix D associated with the finite difference
    approximation of the gradient.
    """
    Di = sc.sparse.dok_matrix((1, NX-2))
    if i > 0:
        Di[0, i-1] = -1
    if i < NX-3:
        Di[0, i+1] = +1
    return Di

ALL_Di = list()
for i in range(NX-2):
    ALL_Di.append(row_D(i))

D = sc.sparse.vstack(ALL_Di)


# Build sparse matrix E that defines the finite difference approximation of the
# Laplacian with Laplacian(f)(x_i) ~ (f(x_i+1)-2f(x_i)+f(xx_i-1))/(DELTAX**2)

def row_E(i):
    """
    Return the i-th row of the matrix E associated with the finite difference
    approximation of the laplacian.
    """
    Ei = sc.sparse.dok_matrix((1, NX-2))
    if i > 0:
        Ei[0, i-1] = +1
    Ei[0, i] = -2
    if i < NX-3:
        Ei[0, i+1] = +1
    return Ei

ALL_Ei = list()
for i in range(NX-2):
    ALL_Ei.append(row_E(i))

E = sc.sparse.vstack(ALL_Ei)


def array_to_sparse_diagonal_matrix(array):
    """
    Build a diagonal matrix from an array of values.
    """
    N = len(array)
    return sc.sparse.dia_matrix((array, [0]),
                                shape=(N, N))


def u_boundaries(u):
    """
    Return the boundary conditions from the solution array u. The length of
    Output is len(u)-2, with zero elements except first and last elements which
    are the same has in the input vector u.

    Example
    -------

    >>> import numpy as np
    >>> u = np.array([1, 2, 3, 4, 5, 6])
    >>> u_boundaries(u)
    array([ 1.,  0.,  0.,  6.])
    """
    res = np.zeros(len(u)-2)
    res[0], res[-1] = u[0], u[-1]
    return res


def F(delta_u, u):
    """
    Implicit function to solve for delta_u to recover the solution update:
    u_n+1 = u_n + delta_u
    """
    u_theta = u[1:-1]+THETA*delta_u
    M_diag = array_to_sparse_diagonal_matrix(u_theta)
    ulimits = u_boundaries(u)
    return (delta_u/DELTAT +
            M_diag*(D.dot(u_theta)-ulimits)/grid_h[0] -
            NU*(E.dot(u_theta)+ulimits)/grid_h[0]**2)


def JACF(delta_u, u):
    """
    Jacobian matrix of function F with respect to delta_u.
    """
    u_theta = u[1:-1]+THETA*delta_u
    M_diag = array_to_sparse_diagonal_matrix(u_theta)
    ulimits = u_boundaries(u)
    M_limits = array_to_sparse_diagonal_matrix(ulimits)
    ID = sc.sparse.identity(NX-2)
    return (ID/DELTAT +
            THETA*(D*M_diag-M_limits+M_diag*D)/grid_h[0] -
            NU*E/grid_h[0]**2)


# --------------------------------------------------------------------------- #
# %% Initialisation

# Initial solution
u0 = np.sin(np.pi*grid[0])
u0[0] = u0[-1] = 0

# Time serie for the solution
U = list()
U.append(u0)

# Initial increment
delta_u = np.zeros(NX-2)


# --------------------------------------------------------------------------- #
# %% Process

for t in Time[1:]:

    # Recover current value for the solution
    u = U[-1].copy()

    # Init the count for Newton-Raphson solver iterations
    i = 0

    # Newton-Raphson solver iterations
    while np.linalg.norm(F(delta_u, u)) > EPS and i < MAXIT:
        delta_u -= sc.sparse.linalg.inv(JACF(delta_u, u)).dot(F(delta_u, u))
        i += 1

    # Update of the solution
    u[1: -1] += delta_u

    # Store solution in the time serie
    U.append(u)

    # Print progression
    print('FOM process {}% done'.format(int(100*t/T)))

data = np.array(U)

# --------------------------------------------------------------------------- #
# %% Plot



if False:

    plt.close('all')
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    X, Y = np.meshgrid(grid[0], Time)

    # Plot the surface.
    surf = ax.plot_surface(X, Y, data, cmap='coolwarm',
                           linewidth=100, antialiased=True)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, aspect=30)

    plt.xlabel('Omega (m)')
    plt.ylabel('Time (s)')

    plt.title('FOM')

    plt.show()


# %% POD

from lasie_rom import pod
from lasie_rom.plots import plot1d

# Reshape data
U  = data.T[:, np.newaxis, :]

u_mean, U_fluc = pod.meanfluc(U)

basis = pod.compute_basis(U, threshold=threshold, nmax=NM)

coeffs = np.einsum('xcm,xct->mt', basis, U_fluc)

U_fluc_reconstruit = np.einsum('xcm,mt->xct', basis, coeffs)

Error = U_fluc_reconstruit-U_fluc

if False:
    plot1d(U_fluc[:, :, 0:NT:int(NT/10)], title='Original')
    plot1d(U_fluc_reconstruit[:, :, 0:NT:int(NT/10)], title='Reconstructed')
    plot1d(Error[:, :, 0:NT:int(NT/10)], title='Error')


# %% ROM

# Compute mean gradient
u_mean_grad = operators.gridgradient(u_mean, grid.shape, grid_h)


def grad_generator():
    for a in misc.iterarray(basis, 2):
        yield operators.gridgradient(a, grid.shape, grid_h)

# shape is nx * ncu * ncx * nm
basis_grad = misc.concatenate_in_given_axis(grad_generator(), 3)

B_temp_11 = np.einsum('xd,xcdj->xcj', u_mean, basis_grad)
B_temp_12 = np.einsum('xcd,xdj->xcj', u_mean_grad, basis)
B_temp_1 = np.einsum('xcj,xci->ij', B_temp_11 + B_temp_12, basis)
B_temp_2 = NU*np.einsum('xcdi, xdcj->ij', basis_grad, basis_grad)
B = B_temp_1 + B_temp_2

C_temp = np.einsum('mcdj,mdk->mcjk', basis_grad, basis)
C = np.einsum('mcjk,mci->ijk', C_temp, basis)

f_temp_1 = NU*np.einsum('xcd, xdci->i', u_mean_grad, basis_grad)

f_temp_21 = np.einsum('xd,xcd->xc', u_mean, u_mean_grad)
f_temp_2 = np.einsum('xc,xci->i', f_temp_21, basis)

f = f_temp_1 + f_temp_2


def F_ROM(delta_a, a):
    """
    Implicit function to solve for delta_u to recover the solution update:
    u_n+1 = u_n + delta_u
    """
    a_theta = a + THETA*delta_a
    tA = delta_a/DELTAT
    tB = np.einsum('ij,j->i', B, a_theta)
    tC = np.einsum('ijk,j,k->i',
                   C,
                   a_theta,
                   a_theta)
    tF = f
    return tA + tB + tC + tF

# %% Process ROM simulation

a0 = coeffs[:, 0]

A = list()
A.append(a0)

delta_a = np.zeros(a0.shape)

for t in Time[1:]:

    a = A[-1].copy()

    res = sc.optimize.root(F_ROM, delta_a, args=(a, ))

    delta_a = res.x

    # Update of the solution
    a += delta_a

    # Store solution in the time serie
    A.append(a)

    # Print progression
    print('ROM process {}% done'.format(int(100*t/T)))

A = np.array(A).T

# %%

if False:
    plt.figure()
    for i in range(min((4, NM))):
        plt.subplot(2, 2, i+1)
        plt.plot(Time, A[i, :], label='rom')
        plt.plot(Time, coeffs[i, :], label='fom')
        plt.legend()

# %%

data_rom = np.einsum('xm,mt->tx', basis[:, 0, :], A) + np.repeat(u_mean, A.shape[1], axis=1).T

if False:
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Make data.
    XX, TT = np.meshgrid(grid[0], Time)

    # Plot the surface.
    surf = ax.plot_surface(XX, TT, data_rom, cmap='coolwarm',
                           linewidth=100, antialiased=True)

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, aspect=30)

    plt.xlabel('Omega (m)')
    plt.ylabel('Time (s)')

    plt.title('ROM')

    plt.show()
