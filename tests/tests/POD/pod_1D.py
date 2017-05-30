#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 23 10:28:39 2017

@author: root
"""

from lasie_rom import pod
from lasie_rom.misc.tools import concatenate_in_given_axis
from lasie_rom.plots import plot1d

import numpy as np

import matplotlib.pyplot as plt
plt.close('all')


nx = 1000
nt = 200
nm = 10

Omega = np.linspace(-1, 1, nx)
T = np.linspace(0, 1, nt)

# %% Build Snapshots

fmin = 1.
fmax = 10.

freqs = list()
for i in range(nm):
    freqs.append(np.random.uniform(fmin, fmax))

def u(t):
    out = 0
    for i in range(nm):
        fi = freqs[i]
        out += np.sin(2*np.pi*fi*t)*np.sin(2*np.pi*(i+1)*Omega)
                     
    return (out + np.random.rand(nx))[:, np.newaxis]
    
U = concatenate_in_given_axis((u(t) for t in T), 2)

plt.figure()
for i in range(0, nt, 20):
    plt.plot(U[:, 0, i])

# %% Build POD basis

u_mean, U_fluc = pod.meanfluc(U)

Phi = pod.compute_basis(U_fluc, threshold=1e-3)

coeffs = np.einsum('xcm,xct->mt', Phi, U_fluc)

U_fluc_reconstruit = np.einsum('xcm,mt->xct', Phi, coeffs)

Error = U_fluc_reconstruit-U_fluc

plot1d(U_fluc[:, :, 0:nt:int(nt/10)])
plot1d(U_fluc_reconstruit[:, :, 0:nt:int(nt/10)])
plot1d(Error[:, :, 0:nt:int(nt/10)])