#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 12 20:04:37 2017

@author: Falaize
"""

from pypod.grids.tools import buildGrid, grid2mesh
from pypod.pod.pod import formatedGradient
import numpy as np
from pypod.config import ORDER
import matplotlib.pyplot as plt

h = 0.1
grid, h = buildGrid(((0, 1), (0, 1)), h)

func = np.sin(np.pi*grid[0])*np.cos(np.pi*grid[1])
g1 = np.pi*np.cos(np.pi*grid[0])*np.cos(np.pi*grid[1])
g2 = -np.pi*np.sin(np.pi*grid[0])*np.sin(np.pi*grid[1])

grad = np.gradient(func, *h, edge_order=2)

plt.figure()
plt.imshow(func, cmap='BuPu')
plt.colorbar()

###############################################################################

mesh = grid2mesh(grid)
func_mesh = (np.sin(np.pi*mesh[:, 0])*np.cos(np.pi*mesh[:, 1]))[:, np.newaxis]
grad_mesh = formatedGradient(func_mesh, grid.shape, h)

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(grad_mesh[:, 0, 0].reshape(grid.shape[1:], order=ORDER), cmap='BuPu')
plt.colorbar()
plt.title('formated')
plt.subplot(2, 2, 2)
plt.imshow(grad[0], cmap='BuPu')
plt.colorbar()
plt.title('numpy')
plt.subplot(2, 2, 3)
plt.imshow(g1, cmap='BuPu')
plt.colorbar()
plt.title('original')
plt.subplot(2, 2, 4)
plt.imshow(np.abs(grad_mesh[:, 0, 0].reshape(grid.shape[1:], order=ORDER)-g1), cmap='BuPu')
plt.colorbar()
plt.title('diff')

plt.figure()
plt.subplot(2, 2, 1)
plt.imshow(grad_mesh[:, 0, 1].reshape(grid.shape[1:], order=ORDER), cmap='BuPu')
plt.colorbar()
plt.title('formated')
plt.subplot(2, 2, 2)
plt.imshow(grad[1], cmap='BuPu')
plt.colorbar()
plt.subplot(2, 2, 3)
plt.imshow(g2, cmap='BuPu')
plt.colorbar()
plt.subplot(2, 2, 4)
plt.imshow(np.abs(grad_mesh[:, 0, 1].reshape(grid.shape[1:], order=ORDER)-g2),
           cmap='BuPu')
plt.colorbar()


