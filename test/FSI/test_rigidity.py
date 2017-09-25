#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 13:08:51 2017

@author: afalaize
"""

import numpy as np

from main import resultsFolderName
from locations import paths
from options import TMIN
from main import parameters

from fenics_simulation.ellipse_fnchar import build_lambdified_levelset

import lasie_rom as lr

import matplotlib.pyplot as plt

grid_hdf = basis_hdf = lr.io.hdf.HDFReader(paths['grid'])
grid_hdf.openHdfFile()

basis_hdf = lr.io.hdf.HDFReader(paths['basis'][0])
basis_hdf.openHdfFile()

meanfluc_hdf = lr.io.hdf.HDFReader(paths['meanfluc'][0])
meanfluc_hdf.openHdfFile()


def Is(angle):
    ls = levelset(angle, *[x for x in grid_hdf.mesh[:].T])
    return heaviside(ls)

def deformation(u):
    grad = lr.operators.gridgradient(u, grid_hdf.shape[:, 0], grid_hdf.h[:, 0])
    return 0.5*(grad+grad.swapaxes(1, 2))

levelset = build_lambdified_levelset(parameters['ell_center'],
                                     parameters['ell_radius'],
                                     parameters['rot_center'])

heaviside = lr.misc.smooth.build_vectorized_heaviside(0.)


def mymatrixnorm(m):
    return

test = list()
for i, u in enumerate(lr.misc.iterarray(meanfluc_hdf.fluc[:], 2)):
    t = TMIN + i*parameters['dt']*parameters['nb_export']
    angle = parameters['theta_init'] + t*parameters['angular_vel']
    c = np.einsum('xci,xc->i', basis_hdf.basis[:], u)
    u_approx = meanfluc_hdf.mean[:] + np.einsum('xci,i->xc', basis_hdf.basis[:], c)
    deform = np.einsum('x,xcd->xcd', Is(angle), deformation(u_approx))
    test.append(np.linalg.norm(np.einsum('xcd,xcd->x',deform, deform)))

print(np.mean(test))


d = lr.misc.concatenate_in_given_axis([u_approx, lr.misc.concatenate_in_given_axis(2*[Is(angle)], 1)], 2)

lr.plots.plot2d(d, grid_hdf.shape[:, 0])
plt.show()