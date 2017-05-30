# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 10:40:13 2017

@author: afalaize

Test sur le champ total (pas le champ fluctuant)

"""

from lasie_rom import deim
from lasie_rom import grids
from lasie_rom import plots
from lasie_rom import pod
from lasie_rom import misc
import numpy as np
import matplotlib.pyplot as plt
from lasie_rom.config import ORDER


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
                              (component1, component2)), axis=1)

U = misc.concatenate_in_given_axis([snapshot(t) for t in times], 2)
    
#plots.plot2d(U[:, :, 0:nt:int(nt/9.)+1], 
#             grid_shape, options={'ncols':3}, title='U', render = 1)

# %% --------------------  CONSTRUCT NONLINEAR SNAPSHOTS -------------------- #
def f1(*u):
    u1 = u[0]
    u2 = u[1]
    return (np.cosh(np.tanh(u1)**3)-1)*(u2)**3

def f2(*u):
    u1 = u[0]
    u2 = u[1]
    return np.tanh(u1)**3*np.sin(u2)

func = (f1, f2)

def eval_func(U):
    all_f = list()
    for i, u in enumerate(misc.iterarray(U, 2)):
        fi = misc.concatenate_in_given_axis([f(*list(misc.iterarray(u, 1))) for f in func], axis=1)
        all_f.append(fi)
    return misc.concatenate_in_given_axis(all_f, axis=2)

F = eval_func(U)

#plots.plot2d(F[:, :, 0:nt:int(nt/9.)+1], 
#             grid_shape, options={'ncols':3}, title='F', render='magnitude')


# %% --------------------  CONSTRUCT POD BASIS  -------------------- #

threshold = 1e-12

Phi = pod.compute_basis(U, threshold=threshold)
Psi = pod.compute_basis(F, threshold=threshold)


#plots.plot2d(Phi[:, :, :9], grid_shape, options={'ncols':3}, title='Basis for U', render='magnitude')
#
#plots.plot2d(Psi[:, :, :9], grid_shape, 
#             options={'ncols':3}, title='Basis for F', render='magnitude')


# %% --------------------  CONSTRUCT DEIM ---------------- #

deim_func = deim.interpolated_func(func, Phi, Psi)

# %% -------------------- RECONSTRUCT SNAPSHOTS ---------------- #

U_coeffs = list()
F_coeffs = list()
for u in misc.iterarray(U, 2):
    U_coeffs.append(np.einsum('xc,xci->i', u, Phi))
    F_coeffs.append(deim_func(U_coeffs[-1]))
    
U_coeffs = np.array(U_coeffs).T
F_coeffs = np.array(F_coeffs).T

U_reconstructed = np.einsum('xcm,mt->xct', Phi, U_coeffs)
F_reconstructed = np.einsum('xcm,mt->xct', Psi, F_coeffs)

    
render = 'magnitude'
plots.plot2d(U[:, :, 0:nt:int(nt/9.)+1], grid_shape, options={'ncols':3}, 
             render=render, title='U')
plots.plot2d(U_reconstructed[:, :, 0:nt:int(nt/9.)+1], grid_shape, 
             options={'ncols':3}, render=render, 
             title='U Reconstructed')

plots.plot2d(F[:, :, 0:nt:int(nt/9.)+1], 
             grid_shape, options={'ncols':3}, 
             render=render, title='F')
plots.plot2d(F_reconstructed[:, :, 0:nt:int(nt/9.)+1], grid_shape, 
             options={'ncols':3}, render=render, title='F Reconstructed')

Error = F_reconstructed - F
plots.plot2d(Error[:, :, 0:nt:int(nt/9.)+1],
            grid_shape, options={'ncols':3}, render=render, 
            title='F Reconstruction error')


# %%

#render = 0 # 'magnitude'
#for i, tup in enumerate(zip(np.swapaxes(NLsnapshots, 0, 1),
#                            np.swapaxes(reconstructed_NLsnapshots, 0, 1))[0:nt:20]):
#    a = np.concatenate(map(lambda e: e[:, np.newaxis, :], tup), axis=1)
#    plots.plot2d(a, grid_shape, options={'ncols': 2}, 
#                 render=render, ) #savename='snapshot {}'.format(i)
#    