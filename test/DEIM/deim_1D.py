# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:33:49 2017

@author: afalaize
"""

from __future__ import absolute_import

import numpy as np
from numpy.random import uniform
import matplotlib.pyplot as plt
from lasie_rom import pod
from lasie_rom import deim
from lasie_rom.misc import (concatenate_in_first_axis, 
                            concatenate_in_given_axis,
                            iterarray)

plt.close('all')


# %% Spatial Domain definition
nx = 1000  # number of spatial nodes
nt = 200   # number of time steps

Omega = np.linspace(-1, 1, nx)  # spatial domain
T = np.linspace(0, 1, nt)       # Time domain

# %% ------------------ Build snapshots ---------------------------------------
def snapshot(t):
    """
    Return the 1D snapshot associated with time t.
    
    Parmaeter
    ---------
    
    t : float
        Time at wich is computed the snapshot.
        
    Return
    ------
    
    u : numpy array
        Snapshot computed for the time t, with shape (nx, 1).
    """
    
    # number of "modes" 
    N = 20  
    
    # array of wavenumbers k's
    wavenumbers = np.array(range(1, N+1))
    
    # X[i, j] = k_j(x_i)
    X = np.einsum('i,j->ij', Omega, wavenumbers)
    coeffs = (2/np.pi)*np.random.rand(N)*np.tanh(uniform(low=0.1, 
                                                         high=2.0, 
                                                         size=N)*np.pi*t)
    return np.einsum('ij,j->i', np.sin(2*np.pi*X), coeffs)[:, np.newaxis]
    
tmax = 1.
times = np.linspace(0, tmax, nt)

def U_gen():
    for t in times:
        yield snapshot(t)

U = concatenate_in_given_axis(U_gen(), axis=2)


# %%---------- Nonlinear function definition ----------------------------------

def f1(*args):
    u1 = args[0]
    return u1**3

func = (f1, )

def eval_func(U):
    all_f = list()
    for i, u in enumerate(iterarray(U, 2)):
        fi = concatenate_in_given_axis([f(u[:, 0]) for f in func], axis=1)
        all_f.append(fi)
    return concatenate_in_given_axis(all_f, axis=2)

F = eval_func(U)

plt.figure()
for i in range(0, nt, int(nt/4)):
    plt.plot(Omega, U[:, 0, i])
plt.title('U')


plt.figure()
for i in range(0, nt, int(nt/4)):
    plt.plot(Omega, F[:, 0, i])
plt.title('F')


# %%---------- Nonlinear function definition ----------------------------------

threshold = 1e-3
nmax = None
U_basis = pod.compute_basis(U, threshold=threshold, nmax=nmax)
F_basis = pod.compute_basis(F, threshold=threshold, nmax=nmax)

F_basis = F_basis[:, :, :10]

plt.figure()
if True:
    nmodes = nt
    for ts, label in [(U, 'u'), (F, 'f')]:
        evals, evec = pod.eigen_decomposition(ts)
        energie = pod.eigen_energy(evals)
        plt.semilogy(range(1, nmodes+1), 1-np.array(energie[:nmodes]), 
                     label=label)
    plt.plot(range(1, nmodes+1), threshold*np.ones(nmodes), label='threshold')
    plt.legend()


# %%----------------  DEIM ----------------------------------------------------

deim_func = deim.interpolated_func(func, U_basis, F_basis)


# %% reconstruction

coeffs = list()

reconstructed_U = list()
reconstructed_F = list()

for i, a in enumerate(U.T):
    a = a.T
    coeffs.append(np.einsum('xc,xci->i', a, U_basis))
    
    reconstructed_U.append(np.einsum('xci,i->xc', U_basis, coeffs[-1]))
    
    reconstructed_F.append(np.einsum('xci,i->xc', 
                                     F_basis, 
                                     deim_func(coeffs[-1])))
    
reconstructed_U = np.concatenate(map(lambda a: a[:, :, np.newaxis], 
                                     reconstructed_U), axis=2)

reconstructed_F = np.concatenate(map(lambda a: a[:, :, np.newaxis], 
                                     reconstructed_F), axis=2)

for i in range(0, nt, 20):
    plt.figure()
    plt.suptitle('i={}'.format(i))
    plt.subplot(2, 1, 1)
    plt.plot(F[:, 0, i], label='original')
    plt.plot(reconstructed_F[:, 0, i], '--', label='reconstructed')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(F[:, 0, i] - reconstructed_F[:, 0, i])
    