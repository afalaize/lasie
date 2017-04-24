# -*- coding: utf-8 -*-
"""
Created on Tue Feb 28 17:08:50 2017

@author: afalaize
"""

import numpy as np
from lasie import grids
from lasie import plots
from lasie import deim
import matplotlib.pyplot as plt
plt.close('all')

lims = [(0, 1),
        (0, 1)]
h = (1e-1, 1e-1)

grid, h = grids.generate(lims, h)
mesh = grids.to_mesh(grid)

print('mesh shape = {}'.format(mesh.shape))


def basiselement(i):
    basis1 = (1+np.tanh((i+1)**0.8*np.pi*mesh[:, 0]))*np.cos((i+1)**0.8*np.pi*mesh[:, 1])
    basis2 = np.sin((i+1)**0.8*np.pi*mesh[:, 0])*np.cos((i+1)**0.8*np.pi*mesh[:, 1])
    return np.concatenate(map(lambda a: a[:, np.newaxis], (basis1, basis2)),
                          axis=1)


def basiselements(M):
    elements = list()
    for i in range(M):
        elements.append(basiselement(i+1))
    return np.concatenate(map(lambda a: a[:, np.newaxis, :], elements),
                          axis=1)

basis = basiselements(3)
print('basis shape = {}'.format(basis.shape))
p, P = deim.indices(basis)

plots.plot2d(basis, grid.shape)
plt.figure()
plt.axis(np.array(lims).flatten().tolist())
for im, ix in enumerate(p):
    xi = mesh[ix,:].tolist()
    plt.plot(xi[0], xi[1], 'Pr')
    plt.text(xi[0], xi[1], str(im+1), fontsize=20)
