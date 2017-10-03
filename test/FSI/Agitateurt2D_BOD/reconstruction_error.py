#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:58:34 2017

@author: afalaize
"""

import lasie_rom as lr
import os

import numpy as np

from _0_locations import paths
from _0_parameters import parameters

from lasie_rom.classes import TimeSerie

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

all_errors = []
labels = list(map(lambda s: os.path.basename(s).split('.')[0], paths['pvd']))
print(labels)

plt.close('all')

n_plot = 20
nb_modes = []
for m in np.logspace(0., 2., n_plot):
        while m in nb_modes: m += 1
	nb_modes.append(int(m))

for i in range(3):
	print(labels[i])
        basis_hdf = lr.io.hdf.HDFReader(path=paths['basis'][i])
	time_serie = TimeSerie(paths['ihdf'][i])


	def mynorm(u):
	    return np.sqrt(np.einsum('xc,xc', u, u))


	# --------------------------------------------------------------------------- #
	# Compute POD basis
	meanfluc_hdf = lr.io.hdf.HDFReader(paths['meanfluc'][i])
	meanfluc_hdf.openHdfFile()
	mean = meanfluc_hdf.mean[:]
	fluc = meanfluc_hdf.fluc[:]
	meanfluc_hdf.closeHdfFile()

	basis = lr.pod.compute_basis(fluc,
				     threshold=0,
				     nmax=None)

	# %%
	error = []
	NM = basis.shape[-1]
	NT = fluc.shape[-1]

	NTOT = NM*NT

	def compute_error(nm):
	    print('nb modes = {}'.format(nm))
	    error_m = []
	    all_t = [(t, nm) for t in range(NT)]
	    def iterTS(pack):
		u_t = mean + fluc[:, :, pack[0]]
		coeffs = np.einsum('xc,xci->i', fluc[:, :, pack[0]], basis[:,:,:pack[1]])
		upod_t = mean + np.einsum('xci,i->xc', basis[:,:,:pack[1]], coeffs)
		return(mynorm(u_t-upod_t)/mynorm(u_t))
	    error_m = lr.parallelization.map(iterTS, all_t)
	    return np.mean(error_m)

        error = list(map(compute_error, nb_modes))
        all_errors.append(error)

	# %%


# %%

nmin, nmax = min(nb_modes), max(nb_modes)

emin, emax = min(map(min, all_errors)), max(map(max, all_errors))

xnmax = np.ones(n_plot)*parameters['pod']['nmax']
ynmax = np.linspace(emin, emax, n_plot)

xthld = np.linspace(nmin, nmax, n_plot)
ythld = np.ones(n_plot)*parameters['pod']['thld']

pltfunc = plt.loglog
marks = '.+x'
for i, e in enumerate(all_errors):
	pltfunc(nb_modes, e, '-'+marks[i], label='$E_{N}$ ' + '({})'.format(labels[i]))
pltfunc(xthld, ythld, '--', label=r'$E_{\mathrm{max}}=10^{-2}$')
pltfunc(xnmax, ynmax, '-.', label='$N_{\mathrm{max}}$=25')
plt.xlabel('$N$')
plt.ylabel(r'$E_N$')
plt.axis([nmin, nmax, 1e-3, 1])
plt.title(r'Reconstruction Error $E_N=\left\langle\frac{\parallel \widetilde{\bf u} - \sum_{i=1}^N (\widetilde{\bf u},\,\phi_i)\,\phi_i \parallel_{L^2(\Omega)}}{\parallel {\bf u} \parallel_{L^2(\Omega)}}\right\rangle$')
plt.legend(loc=0)
plt.grid('on', which='minor')
plt.savefig(os.path.join(paths['results'], 'error_reconstruction'))
