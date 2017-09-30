
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

frate, data = (parameters['nb_export']*parameters['dt'])**-1, coeffs_hdf.coeffs[:, 0]
nfft = 2**12
w = np.fft.fft(data, nfft)
freqs = np.fft.fftfreq(len(w))

# Find the peak in the coefficients
def get_imax(data):
	idx = np.argmax(data)
	freq = freqs[idx]
	freq_in_hertz = abs(freq * frate)
	print(freq_in_hertz)
	return idx, freq_in_hertz

mask = freqs > 0
F = freqs[mask]*frate
A = np.abs(w[mask])

A_diff = np.diff(A)
A_diff = np.array(list(A_diff) + [A_diff[-1]])

def get_lr_zeros(data, iref):
	ileft = iright = iref
	ileft -= 1
	while data[ileft] > 0 and ileft > 0:
		ileft -= 1
	iright += 1
	while data[iright] < 0 and iright < len(data)-1:
		iright += 1
	return ileft, iright

Nfreqs = 3


F_sig = []
I_sig = []

for i in range(Nfreqs):
	imax, fmax = get_imax(A)
	F_sig.append(fmax)
	I_sig.append(imax)
	(l, r) = get_lr_zeros(A_diff, imax)
	A[l:r] = np.zeros(r-l)

t = parameters['load']['tmin'] + np.array(range(len(data)))/frate
#plt.plot(F, np.abs(w[mask]))
for i, f in enumerate(F_sig):
	plt.plot(t, data)
	plt.plot(t, np.abs(w[I_sig[i]]*np.exp(2*1j*np.pi*f*t))) 

	#plt.plot(f*np.ones(50), np.linspace(min(np.abs(w)), max(np.abs(w)), 50))
plt.show()


