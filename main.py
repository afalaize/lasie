# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 17:08:50 2017

@author: afalaize
"""

import numpy as np
from lasie import grids
from lasie import plots
from lasie import deim
from lasie.deim.tools import objective_function
from lasie.config import ORDER
from lasie.misc.tools import norm

import matplotlib.pyplot as plt
plt.close('all')

lims = [(0, 1),
        (0, 1)]
h = (1e-2, 1e-2)

grid, h = grids.generate(lims, h)
mesh = grids.to_mesh(grid)

print('mesh shape = {}'.format(mesh.shape))


def basiselement(i):
    sin = np.sin((1)*np.pi*mesh[:, 0])*np.sin((2)*np.pi*mesh[:, 1])
    basis1 = sin*(1+np.cos((i+2)*np.pi*mesh[:, 0]))
    basis2 = sin*np.cos((i+3)*np.pi*mesh[:, 1])
    return np.concatenate(map(lambda a: a[:, np.newaxis], (basis1, basis2)),
                          axis=1)


def basiselements(M):
    elements = list()
    for i in range(M):
        elements.append(basiselement(i+1))
    return np.concatenate(map(lambda a: a[:, np.newaxis, :], elements),
                          axis=1)

basis = basiselements(5)
print('basis shape = {}'.format(basis.shape))
p, P, all_c = deim.indices(basis)

options = {'ncols': 3}
plots.plot2d(basis, grid.shape, options=options)


for im, ix in enumerate(p):
#    plt.axis((lims[0]+(-lims[1][1], lims[1][0])))
    xi = mesh[ix, :].tolist()
    bi = basis[:, im, :]
    if im > 0:
        tilde_bi = np.einsum('xec,e->xc', basis[:, :im, :], all_c[im-1])
    else:
        tilde_bi = 0*bi
    plt.figure()
    plt.imshow(norm(bi).reshape(grid.shape[1:], order=ORDER).T, cmap='RdBu_r')
    plt.colorbar()
    plt.figure()
    plt.imshow(norm(tilde_bi).reshape(grid.shape[1:], order=ORDER).T, cmap='RdBu_r')
    plt.colorbar()
    plt.figure()
    array = norm(bi-tilde_bi).reshape(grid.shape[1:], order=ORDER)
    plt.imshow(array.T, cmap='BuPu')
    plt.colorbar()
    n1, n2 = array.T.shape
    plt.plot(xi[0]*n1, xi[1]*n2, 'Xg')
    plt.text(xi[0]*n1, xi[1]*n2, str(im+1), fontsize=18)
