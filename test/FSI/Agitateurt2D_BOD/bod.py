
from _0_parameters import parameters
from _0_locations import paths

import lasie_rom as lr

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

import scipy.io.wavfile as wave

coeffs_hdf = lr.io.hdf.HDFReader(paths['coeffs'])
coeffs_hdf.openHdfFile()

from scipy.interpolate import CubicSpline

frate = (parameters['nb_export']*parameters['dt'])**-1

angles = np.linspace(0, 2*np.pi, 1e4)
Nt = coeffs_hdf.coeffs[:].shape[0]

t = parameters['load']['tmin'] + parameters['dt']*np.array(range(Nt))
angle = (parameters['theta_init']+(parameters['angular_vel']*))%(2*np.pi)
inds = np.argsort(angle)
all_angles = []
all_data = []

for i in range(coeffs_hdf.coeffs[:].shape[-1]):

	data = coeffs_hdf.coeffs[:, i]
	interpolator = CubicSpline(angle[inds], data[inds])

	interp = interpolator(angles)

	nangles = [0, 2*np.pi]

	e = np.abs(interp)
	error = 1

	while error > 5e-2:
	    nangles.append(angles[np.argmax(np.abs(e))])
	    nangles.sort()
	    redinterp = CubicSpline(nangles, interpolator(nangles))
	    e = interp - redinterp(angles)
	    error = np.linalg.norm(e)/np.linalg.norm(interp)

	all_angles.append(nangles)
	all_data.append(interpolator(nangles))

