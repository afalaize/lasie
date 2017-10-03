
from set_parameters import parameters
from set_locations import paths

import lasie_rom as lr

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

import scipy.io.wavfile as wave

coeffs_hdf = lr.io.hdf.HDFReader(paths['coeffs'])
coeffs_hdf.openHdfFile()

frate, data = (parameters['nb_export']*parameters['dt'])**-1, coeffs_hdf.coeffs[:, 0]

N = len(data)
nc = int(N**0.5)
nl = N+1-nc

S = np.zeros((nl, nc))
for i in range(nl):
	for j in range(nc):
		S[i, j] = data[i+j]

C = np.dot(S, S.T.conjugate())
a, E = np.linalg.eig(C)
inds = list(np.argsort(np.abs(a)))
inds.reverse()
C = np.dot(S, S.T.conjugate())
a, E = a[inds], E[:, inds]

tol = np.finfo(float).eps
imax = np.nonzero(np.abs(a)/np.abs(a[0]) >= 1e-15)[0][-1]

Enoise = E[:, imax:]

def hrfunc(f):
	omega = 2*np.pi*f
	z = np.exp(1j*omega)
	vnl = z**(parameters['dt']*parameters['nb_export']*np.arange(nl))
	proj = np.dot(Enoise.T, vnl)
	return np.dot(proj, proj.conjugate().T)**-1

func = np.vectorize(hrfunc)

freqs = np.logspace(-1, 2, 1e4)
ress = func(freqs)

plt.figure()

#plt.semilogy(np.abs(a))
#plt.semilogy(imax, np.abs(a[imax]), 'or')

plt.plot(freqs, np.abs(ress))
plt.show()



