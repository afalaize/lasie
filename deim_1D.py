# -*- coding: utf-8 -*-
"""
Created on Fri Apr 28 11:33:49 2017

@author: afalaize
"""

import numpy as np
import matplotlib.pyplot as plt
from lasie import misc
from lasie import pod
from lasie import deim


plt.close('all')


# %% Domain definition
nx = 1000
Omega = np.linspace(-1, 1, nx)


# %% ------------------ Build snapshots ---------------------------------------
def snapshot(t):
    N = 20
    wavenumbers = np.array(range(1, N+1))
    X = np.einsum('i,j->ij', Omega, wavenumbers)
    coeffs = (2/np.pi)*np.random.rand(N)*np.tanh(np.random.uniform(low=0.1, high=2.0, size=N)*np.pi*t)
    return np.einsum('ij,j->i', np.sin(2*np.pi*X), coeffs)[:, np.newaxis]
    
tmax = 1.
nt = 200
times = np.linspace(0, tmax, nt)
list_of_snapshots = [snapshot(t) for t in times]
U = misc.concatenate_over_2d_axis(list_of_snapshots)

# %%---------- Nonlinear function definition ----------------------------------
def func(u):
    return np.cosh(np.tanh(u)**3)-1
   
list_of_NL_snapshots = [func(u) for u in np.swapaxes(U, 0, 1)]
F = misc.concatenate_over_2d_axis(list_of_NL_snapshots)

plt.figure()
for i in range(0, nt, int(nt/4)):
    plt.plot(U[:, i , :])

plt.figure()
for i in range(0, nt, int(nt/4)):
    plt.plot(F[:, i , :])
    
# %%---------- Nonlinear function definition ----------------------------------

threshold = 1e-3
nmax = None
U_basis = pod.compute_basis(U, threshold=threshold, nmax=nmax)
F_basis = pod.compute_basis(F, threshold=threshold, nmax=nmax)

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

p, P = deim.indices(F_basis)
deim_func = deim.interpolated_func(func, P, U_basis, F_basis)


# %% reconstruction

coeffs = list()
reconstructed_U = list()
reconstructed_F = list()
for i, a in enumerate(np.swapaxes(U, 0, 1)):
    coeffs.append(np.einsum('xc,xic->i', a, U_basis))
    reconstructed_U.append(np.einsum('xic,i->xc', U_basis, coeffs[-1]))
    reconstructed_F.append(deim_func(coeffs[-1]))
    
reconstructed_U = misc.concatenate_over_2d_axis(reconstructed_U)
reconstructed_F = misc.concatenate_over_2d_axis(reconstructed_F)

for i in range(0, nt, 20):
    plt.figure()
    plt.suptitle('i={}'.format(i))
    plt.subplot(2, 1, 1)
    plt.plot(F[:, i, 0], label='original')
    plt.plot(reconstructed_F[:, i, 0], '--', label='reconstructed')
    plt.legend()
    plt.subplot(2, 1, 2)
    plt.plot(F[:, i, 0]-reconstructed_F[:, i, 0])
    
#for i, mode in enumerate(np.swapaxes(F_basis, 0, 1)):
#    plt.figure()
#    plt.title('mode i={}'.format(i))
#    plt.plot(mode)
#   

    