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
time_steps = range(0, 200, 200//(nsnaps+1))
ncols = 1
nrows = int(np.ceil(len(time_steps)/float(ncols)))

ts = HDFTimeSerie(CONFIG['interp_hdf_folder'])

mean = HDFData(CONFIG['hdf_path_mean'])
mean.openHdfFile()

grid = HDFData(CONFIG['hdf_path_grid'])
grid.openHdfFile()

cmap = 'BuPu'
plt.close('all')

fig = plt.figure()
v = mean.get_single_data()
modul = np.einsum('mc,mc->m', v, v)
modul_g = modul.reshape(map(int, grid.original_shape[1:, 0]))
plt.imshow(modul_g.T, cmap=cmap)
plt.axis('off')
plt.title('moyenne')
plt.savefig('vitesse_moyenne.png', format='png')    

fig = plt.figure(figsize=(6, 8))
#plt.suptitle('Snapshots')
for ind, time in enumerate(time_steps):
    d = ts.data[time]
    d.openHdfFile()
    v = d.vitesse[:]
    modul = np.einsum('mc,mc->m', v, v)
    modul_g = modul.reshape(map(int, grid.original_shape[1:, 0]))
    plt.subplot(nrows, ncols, ind+1)
    plt.title('$t={}$s'.format(50+time*0.01))
    plt.imshow(modul_g.T, cmap=cmap)
    plt.axis('off')
    d.closeHdfFile()

#plt.tight_layout()
plt.savefig('vitesse_originale.png', format='png')    


fig = plt.figure(figsize=(6, 8))
#plt.suptitle('Snapshots')
for ind, time in enumerate(time_steps):
    d = ts.data[time]
    d.openHdfFile()
    v = d.vitesse[:] - mean.get_single_data()
    modul = np.einsum('mc,mc->m', v, v)
    modul_g = modul.reshape(map(int, grid.original_shape[1:, 0]))
    plt.subplot(nrows, ncols, ind+1)
    plt.title('$t={}$s'.format(50+time*0.01))
    plt.imshow(modul_g.T, cmap=cmap)
    plt.axis('off')
    d.closeHdfFile()

#plt.tight_layout()
mean.closeHdfFile()
grid.closeHdfFile()
plt.savefig('vitesse_originale.png', format='png')    
plt.show()


