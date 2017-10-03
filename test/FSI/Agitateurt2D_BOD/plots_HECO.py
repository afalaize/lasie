

import numpy as np

import lasie_rom as lr

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from _0_locations import paths
from lasie_rom.io.hdf import HDFReader

from lasie_rom.plots import plot2d
#ls = lr.classes.TimeSerie(paths['ihdf'][0])
grid = HDFReader(paths['grid'])

#d = ls.data[0]
d = lr.io.hdf.HDFReader(paths['basis'][1])
d.openHdfFile()
grid.openHdfFile()
shape = list(map(int, grid.shape[:, 0]))

#dataname = d.names[[n.startswith('f_') for n in d.names].index(True)]
dataname = 'basis'
for i in range(9):
	data = getattr(d, dataname)[:,:,i]
	func = lr.misc.smooth.build_vectorized_heaviside(5e-1)
	plot2d(data[:, :, np.newaxis], grid.shape[:, 0])
	plt.savefig(paths['results'] + '/' + 'velocity_mode_' + str(i))
grid.closeHdfFile()
d.closeHdfFile()
#plt.show()



