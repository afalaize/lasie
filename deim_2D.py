# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:40:13 2017

@author: afalaize

Test sur le champ total (pas le champ fluctuant)

"""

from lasie import deim
from lasie import grids
from lasie import plots
from lasie import pod
from lasie import misc
import numpy as np
import matplotlib.pyplot as plt
from lasie.config import ORDER


plt.close('all')

# %% --------------------  CONSTRUCT SOLUTION SNAPSHOTS -------------------- #

# Time domain
tmax = 1.
nt = 200
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
    # parameters
    ft = 2  #  temporal frequency
    coeff_t = np.sin(2*np.pi*ft*t)
    a = 1
    b = 100
    fx = 1  # spatial frequency

    sin = np.sin(fx*np.pi*mesh[:, 0])*np.sin(fx*np.pi*mesh[:, 1])
    rosen = (a-mesh[:, 0])**2 + b*(mesh[:, 1]-mesh[:, 0]**(2+coeff_t))**2
    component1 = rosen*np.cos(np.pi*mesh[:, 0]**2)/100
    component2 = coeff_t*(rosen*sin/10)-1.5
    
    return np.concatenate(map(lambda a: a[:, np.newaxis], 
                              (component1, component2)),
                          axis=1)

snapshots = [snapshot(t) for t in times]
snapshots = np.concatenate(map(lambda a: a[:, np.newaxis, :], snapshots),
                          axis=1)
    
plots.plot2d(snapshots[:, 0:nt:int(nt/9.)+1, :], 
             grid_shape, options={'ncols':3}, title='Snapshots', render = 1)

# %% --------------------  CONSTRUCT NONLINEAR SNAPSHOTS -------------------- #
def func(u):
    """
    Nonlinear function defined over a 2D space:
.. math:: 
    f:\mathbb R^2\\ni u \mapsto f(u) \in \mathbb R.
    """
    b1, b2 = 3, 4
    return 10*(np.exp(((u[:, 0]-b1)**2 + (u[:, 1]-b2)**2)/1e2-1)-0.5)[:, np.newaxis]

    
NLsnapshots = [func(snapshots[:, i, :]) for i in range(nt)]
NLsnapshots = np.concatenate(map(lambda a: a[:, np.newaxis, :], NLsnapshots),
                             axis=1)

plots.plot2d(NLsnapshots[:, 0:nt:int(nt/9.)+1, :], 
             grid_shape, options={'ncols':3}, title='NL Snapshots', render=0)


# %% --------------------  CONSTRUCT POD BASIS  -------------------- #

threshold = 1e-6

basis = pod.compute_basis(snapshots, threshold=threshold)
plots.plot2d(basis[:, :9, :], grid_shape, options={'ncols':3}, title='Basis for U', render=0)

NLbasis = pod.compute_basis(NLsnapshots, threshold=threshold)
plots.plot2d(NLbasis[:, :9, :], grid_shape, 
             options={'ncols':3}, title='Basis for F', render=0)


# %% --------------------  CONSTRUCT DEIM ---------------- #

p, P = deim.indices(NLbasis)
deim_func = deim.interpolated_func(func, P, basis, NLbasis)

if False:
    for im, ix in enumerate(p):
    #    plt.axis((lims[0]+(-lims[1][1], lims[1][0])))
        xi = mesh[ix, :].tolist()
        bi = NLbasis[:, im, :]
        if im > 0:
            tilde_bi = np.einsum('xec,e->xc', NLbasis[:, :im, :], c[im-1])
        else:
            tilde_bi = 0*bi
        plt.figure()
        plt.imshow(misc.norm(bi).reshape(grid.shape[1:], order=ORDER).T, cmap='RdBu_r')
        plt.colorbar()
        plt.title('original mode {}'.format(im+1))
        plt.figure()
        plt.imshow(misc.norm(tilde_bi).reshape(grid.shape[1:], order=ORDER).T, cmap='RdBu_r')
        plt.colorbar()
        plt.title('reconstruction mode {}'.format(im+1))
        plt.figure()
        array = misc.norm(bi-tilde_bi).reshape(grid.shape[1:], order=ORDER)
        plt.imshow(array.T, cmap='Reds')
        plt.colorbar()
        n1, n2 = array.T.shape
        plt.plot(xi[0]*n1, xi[1]*n2, 'Xg')
        plt.text(xi[0]*n1, xi[1]*n2, str(im+1), fontsize=18)

# %% -------------------- RECONSTRUCT SNAPSHOTS ---------------- #

coeffs = list()
reconstructed_snapshots = list()
reconstructed_NLsnapshots = list()
for a in np.swapaxes(snapshots, 0, 1):
    coeffs.append(np.einsum('xc,xic->i', a, basis))
    reconstructed_snapshots.append(np.einsum('xic,i->xc', basis, coeffs[-1]))
    reconstructed_NLsnapshots.append(deim_func(coeffs[-1]))
    
reconstructed_snapshots = np.concatenate(map(lambda a: a[:, np.newaxis, :], 
                                               reconstructed_snapshots),
                                           axis=1)
reconstructed_NLsnapshots = np.concatenate(map(lambda a: a[:, np.newaxis, :], 
                                               reconstructed_NLsnapshots),
                                           axis=1)

render = 0
plots.plot2d(snapshots[:, 0:nt:int(nt/9.)+1, :], grid_shape, options={'ncols':3}, render=render, 
             title='Snapshots', savename='Snapshots')
plots.plot2d(reconstructed_snapshots[:, 0:nt:int(nt/9.)+1, :], grid_shape, 
             options={'ncols':3}, render=render, 
             title='Reconstructed Snapshots', 
             savename='Reconstructed_Snapshots')

plots.plot2d(NLsnapshots[:, 0:nt:int(nt/9.)+1, :], 
             grid_shape, options={'ncols':3}, render=render,
             title='NL Snapshots', savename='Original_NL_Snapshots')
plots.plot2d(reconstructed_NLsnapshots[:, 0:nt:int(nt/9.)+1, :], grid_shape, 
             options={'ncols':3}, render=render, title='Reconstructed NL Snapshots',
             savename='Reconstructed_NL_Snapshots')

plots.plot2d(NLsnapshots[:, 0:nt:int(nt/9.)+1, :]-reconstructed_NLsnapshots[:, 0:nt:int(nt/9.)+1, :],
             grid_shape, options={'ncols':3}, render=render, 
            title='Reconstructopn error NL Snapshots',
             savename='error_NL_Snapshots')


# %%

render = 0 # 'magnitude'
for i, tup in enumerate(zip(np.swapaxes(NLsnapshots, 0, 1),
                            np.swapaxes(reconstructed_NLsnapshots, 0, 1))[0:nt:20]):
    a = np.concatenate(map(lambda e: e[:, np.newaxis, :], tup), axis=1)
    plots.plot2d(a, grid_shape, options={'ncols': 2}, 
                 render=render, savename='snapshot {}'.format(i))
    