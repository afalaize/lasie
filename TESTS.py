# -*- coding: utf-8 -*-
"""
Created on Tue Apr 11 12:34:13 2017

@author: afalaize
"""

from main import CONFIG
from pypod.readwrite.read_hdf import HDFTimeSerie, HDFData
from pypod.readwrite.write_vtu import write_vtu
from pypod.grids.tools import mesh2grid, grid2mesh
from pypod.config import ORDER

for k in CONFIG.keys():
    print('{}: {}'.format(k, CONFIG[k]))
    

    
ts = HDFTimeSerie(CONFIG['interp_hdf_folder'])
ts.openAllFiles()

v = ts.data[0].vitesse[:]

grid = HDFData(CONFIG['hdf_path_grid'], openFile=True)

path = r'F:\TESTS_THOST\cylindre2D_SCC_windows\Results\hdf5_interpolated\test.vtu'

write_vtu(grid.mesh[:], grid.original_shape,
          [v, ], 'vitesse', path)

grid.closeHdfFile()

ts.closeAllFiles()
