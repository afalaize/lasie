# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:25:57 2017

@author: afalaize
"""

import matplotlib.pyplot as plt
import numpy as np
from main import CONFIG
from pypod.readwrite.read_hdf import HDFTimeSerie, HDFData
from pypod.grids.tools import buildGrid, grid2mesh
import os

nsnaps = 5
time_steps = range(0, 200, 200//(nsnaps))
ncols = 1
nrows = int(np.ceil(len(time_steps)/float(ncols)))

ts = HDFTimeSerie(CONFIG['interp_hdf_folder'])

mean = HDFData(CONFIG['hdf_path_mean'])
mean.openHdfFile()

grid = HDFData(CONFIG['hdf_path_grid'])
grid.openHdfFile()

cmap = 'RdBu_r'
plt.close('all')

figsize = (6, 8)

plt_format = 'png'

cbar_axes = [0.88, 0.125, 0.01, 0.75]
SubplotParams = {'left':0.05, 'bottom':0.05, 'right':0.85, 'top':0.9, 'wspace':None, 'hspace':None}
SubplotParams_fluc = {'left':0.05, 'bottom':0.05, 'right':0.8, 'top':0.9, 'wspace':None, 'hspace':None}
cbar_axes_fluc = [0.82, 0.125, 0.01, 0.75]
SubplotParams_pod = {'left':0.05, 'bottom':0.05, 'right':0.8, 'top':0.9, 'wspace':None, 'hspace':None}
cbar_axes_pod = [0.82, 0.125, 0.01, 0.75]
#%% PLOT MEAN
fig = plt.figure()
v = mean.get_single_data()
vx = v[:, 0]
vx_g = vx.reshape(map(int, grid.original_shape[1:, 0]))
plt.imshow(vx_g.T, cmap=cmap)
# plt.colorbar()
plt.axis('off')
plt.title('moyenne')
plt.tight_layout()
plt.savefig('moyenne.'+plt_format, format=plt_format)


#%% PLOT Vx FOM
fig = plt.figure(figsize=figsize)

plt.suptitle('MODELE COMPLET')
all_vx = list()
minmax = (float('Inf'), -float('Inf'))
for ind, time in enumerate(time_steps):
    d = ts.data[time]
    d.openHdfFile()
    v = d.vitesse[:]
    vx = v[:, 0]
    minmax = min([minmax[0], min(vx)]), max([minmax[1], max(vx)])
    all_vx.append(vx)
for ind, vx in enumerate(all_vx):
    vx_g = vx.reshape(map(int, grid.original_shape[1:, 0]))
    plt.subplot(nrows, ncols, ind+1)
    plt.title('$t={}$s'.format(50+time_steps[ind]*0.01))
    im = plt.imshow(vx_g.T, cmap=cmap, vmin=minmax[0], vmax=minmax[1])
    plt.axis('off')
    # plt.colorbar()
    d.closeHdfFile()
cbar_ax = fig.add_axes(cbar_axes)
fig.colorbar(im, cax=cbar_ax, label=r'$v_x$ (m/s)')
plt.tight_layout()
plt.subplots_adjust(**SubplotParams)
plt.savefig('fom.'+plt_format, format=plt_format)

#%% PLOT Vx FLUCT FOM
fig = plt.figure(figsize=figsize)
plt.suptitle('MODELE COMPLET (Champ fluctuant)')
all_vx = list()
minmax = (float('Inf'), -float('Inf'))
for ind, time in enumerate(time_steps):
    d = ts.data[time]
    d.openHdfFile()
    v = d.vitesse[:] - mean.get_single_data()
    vx = v[:, 0]
    minmax = min([minmax[0], min(vx)]), max([minmax[1], max(vx)])
    all_vx.append(vx)
    
for ind, vx in enumerate(all_vx):
    vx_g = vx.reshape(map(int, grid.original_shape[1:, 0]))
    plt.subplot(nrows, ncols, ind+1)
    plt.title('$t={}$s'.format(50+time_steps[ind]*0.01))
    im = plt.imshow(vx_g.T, cmap=cmap, vmin=minmax[0], vmax=minmax[1])
    plt.axis('off')
    # plt.colorbar()
    d.closeHdfFile()
cbar_ax = fig.add_axes(cbar_axes_fluc)
fig.colorbar(im, cax=cbar_ax, label=r'$v_x$ (m/s)')
plt.tight_layout()
plt.subplots_adjust(**SubplotParams_fluc)
plt.savefig('fom_fluc.'+plt_format, format=plt_format)


#%% PLOT Vx ROM
path_pod = r"F:/TESTS_THOST/cylindre2D_SCC_windows/Results/reconstructed_ROM"
ts_pod = HDFTimeSerie(path_pod)
fig = plt.figure(figsize=figsize)
plt.suptitle('MODELE REDUIT')
all_vx = list()
minmax = (float('Inf'), -float('Inf'))
for ind, time in enumerate(time_steps):
    d_pod = ts_pod.data[time]
    d_pod.openHdfFile()
    v_pod = d_pod.vitesse[:]
    d_pod.closeHdfFile()
    vx = v_pod[:, 0]
    minmax = min([minmax[0], min(vx)]), max([minmax[1], max(vx)])
    all_vx.append(vx)

for ind, vx in enumerate(all_vx):
    vx_g = vx.reshape(map(int, grid.original_shape[1:, 0]))
    plt.subplot(nrows, ncols, ind+1)
    plt.title('$t={}$s'.format(50+time_steps[ind]*0.01))
    im = plt.imshow(vx_g.T, cmap=cmap, vmin=minmax[0], vmax=minmax[1])
    plt.axis('off')
    # plt.colorbar()
cbar_ax = fig.add_axes(cbar_axes)
fig.colorbar(im, cax=cbar_ax, label=r'$v_x$ (m/s)')
plt.tight_layout()
plt.subplots_adjust(**SubplotParams)
plt.savefig('rom.'+plt_format, format=plt_format)


#%% PLOT Vx FLUCT ROM
path_pod = r"F:/TESTS_THOST/cylindre2D_SCC_windows/Results/reconstructed_ROM"
ts_pod = HDFTimeSerie(path_pod)
fig = plt.figure(figsize=figsize)
plt.suptitle('MODELE REDUIT (Champ fluctuant)')
all_vx = list()
minmax = (float('Inf'), -float('Inf'))
for ind, time in enumerate(time_steps):
    d_pod = ts_pod.data[time]
    d_pod.openHdfFile()
    v_pod = d_pod.vitesse[:] - mean.get_single_data()
    d_pod.closeHdfFile()
    vx = v_pod[:, 0]
    minmax = min([minmax[0], min(vx)]), max([minmax[1], max(vx)])
    all_vx.append(vx)

for ind, vx in enumerate(all_vx):
    vx_g = vx.reshape(map(int, grid.original_shape[1:, 0]))
    plt.subplot(nrows, ncols, ind+1)
    plt.title('$t={}$s'.format(50+time_steps[ind]*0.01))
    im = plt.imshow(vx_g.T, cmap=cmap, vmin=minmax[0], vmax=minmax[1])
    plt.axis('off')
    # plt.colorbar()
cbar_ax = fig.add_axes(cbar_axes_fluc)
fig.colorbar(im, cax=cbar_ax, label=r'$v_x$ (m/s)')
plt.tight_layout()
plt.subplots_adjust(**SubplotParams_fluc)
plt.savefig('rom_fluc.'+plt_format, format=plt_format)


#%% PLOT Vx ERROR  
path_pod = r"F:/TESTS_THOST/cylindre2D_SCC_windows/Results/reconstructed_ROM"
ts_pod = HDFTimeSerie(path_pod)
ts_pod.openAllFiles()
fig = plt.figure(figsize=figsize)
plt.suptitle(r'ERREUR RELATIVE $\frac{\vert \tilde v_x -v_x \vert}{\vert v_x\vert}$')
all_vx = list()
minmax = (float('Inf'), -float('Inf'))
for ind, time in enumerate(time_steps):
    d_original = ts.data[time]
    d_original.openHdfFile()
    v_original = d_original.vitesse[:]
    d_original.closeHdfFile()
    
    d_pod = ts_pod.data[time]
    d_pod.openHdfFile()
    v_pod = d_pod.vitesse[:]
    d_pod.closeHdfFile()
    
    v = np.abs(v_original-v_pod)/(1e-16+np.abs(v_original))*100
    vx = v[:, 0]
    minmax = min([minmax[0], min(vx)]), max([minmax[1], max(vx)])
    all_vx.append(vx)

for ind, vx in enumerate(all_vx):
    vx_g = vx.reshape(map(int, grid.original_shape[1:, 0]))
    plt.subplot(nrows, ncols, ind+1)
    plt.title('$t={}$s'.format(50+time_steps[ind]*0.01))
    im = plt.imshow(vx_g.T, cmap='Reds', vmin=minmax[0], vmax=100)
    plt.axis('off')
    # plt.colorbar()
cbar_ax = fig.add_axes(cbar_axes)
fig.colorbar(im, cax=cbar_ax, label=r'$\epsilon$ (%)')
plt.tight_layout()
plt.subplots_adjust(**SubplotParams)
plt.savefig('error.'+plt_format, format=plt_format)

#%% PLOT POD BASIS

basis = HDFData(CONFIG['hdf_path_podBasis'])
basis.openHdfFile()
basis_a = basis.get_single_data()
fig = plt.figure(figsize=figsize)
plt.suptitle('BASE POD')
for i in range(ncols*nrows):
    mode = basis_a[:, i, 0]
    mode_grid = mode.reshape(map(int, grid.original_shape[1:, 0]))
    plt.subplot(nrows, ncols, i+1)
    plt.title('mode {}'.format(i+1))
    im = plt.imshow(mode_grid.T, cmap=cmap)
    plt.axis('off')
    # plt.colorbar()
    d.closeHdfFile()
cbar_ax = fig.add_axes(cbar_axes_pod)
fig.colorbar(im, cax=cbar_ax, label=r'$\phi_x$ (s.u)')
plt.tight_layout()
plt.subplots_adjust(**SubplotParams_pod)
plt.savefig('basis.'+plt_format, format=plt_format)


#%% CLOSE HDF FILES
mean.closeHdfFile()
grid.closeHdfFile()


#%% SHOW
plt.show()

    