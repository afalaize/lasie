
from _0_parameters import parameters
from _0_locations import paths

import lasie_rom as lr

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

import scipy.io.wavfile as wave

if True:
    # --------------------------------------------------------------------------- #

    # Open hdf files
    velocity_basis_hdf = lr.io.hdf.HDFReader(paths['basis'][1], openFile=True)
    velocity_meanfluc_hdf = lr.io.hdf.HDFReader(paths['meanfluc'][1], openFile=True)

    print('Build coeffs')
    # instanciate TimeSerie
    ts = lr.classes.TimeSerie(paths['ihdf'][1])

    if parameters['dataname']['hdf'] is not None:
        dataname = parameters['dataname']['hdf']
    else:
        d = ts.data[0]
        d.openHdfFile()
        args = dir(d)
        temp = [a.startswith('f_') for a in args]
        dataname = args[temp.index(True)]
        d.closeHdfFile()

    # instanciate TimeSerie
    def coefficients():
        for u in ts.generator(dataname)():
            yield np.einsum('xc,xci->i',
                               u-velocity_meanfluc_hdf.mean[:],
                               velocity_basis_hdf.basis[:])

    coeffs = lr.misc.concatenate_in_given_axis(coefficients(), 0)

    data = {'coeffs': coeffs}

    print('Save {}'.format(paths['coeffs']))
    # write hdf for rom matrices
    lr.io.hdf.data2hdf(data, paths['coeffs'])

    # Close hdf files
    velocity_basis_hdf.closeHdfFile()
    velocity_meanfluc_hdf.closeHdfFile()

coeffs_hdf = lr.io.hdf.HDFReader(paths['coeffs'])
coeffs_hdf.openHdfFile()

from scipy.interpolate import CubicSpline

frate = (parameters['nb_export']*parameters['dt'])**-1

angles = np.linspace(0, 2*np.pi, 1e4)
Nt = coeffs_hdf.coeffs[:].shape[0]

t = parameters['load']['tmin'] + parameters['dt']*parameters['nb_export']*np.array(range(Nt))
angle = (parameters['theta_init']+(parameters['angular_vel']*t)) % (2*np.pi)
inds = np.argsort(angle)
all_angles = []
all_data = []

for i in range(coeffs_hdf.coeffs[:].shape[-1]):

    data = coeffs_hdf.coeffs[:, i]
    interpolator = CubicSpline(angle[inds], data[inds])

    interp = interpolator(angles)

    nangles = [0., 2*np.pi]

    redinterp = CubicSpline(nangles, interpolator(nangles))
    e = interp - redinterp(angles)
    error = 1

    while error > 1e-2:
        nangles.append(angles[np.argmax(np.abs(e))])
        nangles.sort()
        redinterp = CubicSpline(nangles, interpolator(nangles))
        e = interp - redinterp(angles)
        error = np.linalg.norm(e)/np.linalg.norm(interp)

    all_angles.append(nangles)
    all_data.append(interpolator(nangles))

grid_hdf = lr.io.hdf.HDFReader(paths['grid'])
grid_hdf.openHdfFile()
grid_shape = grid_hdf.shape[:][:, 0]
grid_h = grid_hdf.h[:][:, 0]
grid_mesh = grid_hdf.mesh[:]
grid_hdf.closeHdfFile()

data = {'mesh': grid_mesh}
for i, (a, d) in enumerate(zip(all_angles, all_data)):
    data.update({'a'+str(i): np.array(a)[:, np.newaxis], 'd'+str(i): np.array(d)[:, np.newaxis]})

# write hdf for pod basis
lr.io.hdf.data2hdf(data, paths['interpolator'])

def myplot(i):
    interp_hdf = lr.io.hdf.HDFReader(paths['interpolator'])
    interp_hdf.openHdfFile()
    a = getattr(interp_hdf, 'a'+str(i))[:, 0]
    d = getattr(interp_hdf, 'd'+str(i))[:, 0]
    redinterp = CubicSpline(a, d)

    plt.close('all')
    plt.plot(t, coeffs_hdf.coeffs[:, i], '+r')
    plt.plot(t, redinterp(angle), 'xb')
    interp_hdf.closeHdfFile()

myplot(0)

coeffs_hdf.closeHdfFile()
