# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:40:13 2017

@author: afalaize
"""

from lasie import deim
from lasie import grids
from lasie import plots
from lasie import pod
from lasie import misc
import numpy as np
import matplotlib.pyplot as plt
from lasie.config import ORDER



# --------------------  CONSTRUCT SOLUTION SNAPSHOTS -------------------- #

# Time domain
tmax = 1.
nt = 50
times = np.linspace(0, tmax, nt)

# Spatial domain is a box
L1, L2 = 1., 1.  #  box shape
lims = [(0, L1), #  box limits
        (0, L2)]

# Spatial discretization step in both directions
h = (1e-2, 1e-2)  

# grid
grid, h = grids.generate(lims, h)
grid_shape = grid.shape

def reshape(a):
    return a.reshape(grid_shape[1:], order=ORDER)

# mesh
mesh = grids.to_mesh(grid)


def snapshot(t):
    """
    Return 2D snapshots defined over a 2D domain.
    """
    ft = 1;
    a = 1
    b = 50
    coeff_t = np.sin(2*np.pi*ft*t)
    fx = 3.5*coeff_t;

    sin = np.sin(fx*np.pi*mesh[:, 0])*np.sin(fx*np.pi*mesh[:, 1])
    rosen = (a*coeff_t-mesh[:, 0])**2 + b*(mesh[:, 1]-mesh[:, 0]**2)**2
    component1 = 100*sin*rosen*coeff_t
    component2 = rosen*(2+np.tanh(np.pi*(mesh[:, 0]-L1/2.)/(L1/3.)*coeff_t))
    
    return np.concatenate(map(lambda a: a[:, np.newaxis], 
                              (component1, component2)),
                          axis=1)

snapshots = [snapshot(t) for t in times]
snapshots = np.concatenate(map(lambda a: a[:, np.newaxis, :], snapshots),
                          axis=1)
    
plots.plot2d(snapshots[:, 0:nt:int(nt/9.)+1, :], 
             grid_shape, options={'ncols':3})

# --------------------  CONSTRUCT NONLINEAR SNAPSHOTS -------------------- #
def func(snapshot):
    eps = 1e4
    component1 = np.exp(-np.abs(np.tanh(np.pi*(1+snapshot[:, 0])/eps)*snapshot[:, 1]))
    component2 = np.exp(-np.abs(np.tanh(np.pi*snapshot[:, 1]/eps)*snapshot[:, 0]))
    return np.concatenate(map(lambda a: a[:, np.newaxis], 
                              (component1, component2)),
                          axis=1)
    
NLsnapshots = [func(snapshots[:, i, :]) for i in range(nt)]
NLsnapshots = np.concatenate(map(lambda a: a[:, np.newaxis, :], NLsnapshots),
                             axis=1)

plots.plot2d(NLsnapshots[:, 0:nt:int(nt/9.)+1, :], 
             grid_shape, options={'ncols':3})


# --------------------  CONSTRUCT POD BASIS  -------------------- #

mean, fluc = pod.meanfluc(snapshots)
basis = pod.compute_basis(fluc, nmax=5)
plots.plot2d(basis[:, :9, :], grid_shape, options={'ncols':3})

NLmean, NLfluc = pod.meanfluc(NLsnapshots)
NLbasis = pod.compute_basis(NLfluc, nmax=5)
plots.plot2d(NLbasis[:, :9, :], grid_shape, options={'ncols':3})

p, P, c = deim.indices(NLbasis)





