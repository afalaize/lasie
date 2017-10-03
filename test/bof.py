#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 21:07:03 2017

@author: afalaize
"""

import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import minimize


t0 = 1.12
dt = 1e-2
N = 500
t = np.array(range(N))*dt
tmax = t[-1]

asig = np.array([1.2, 2.4]) #np.random.rand(K)
psig = np.array([0.1, 0.3])
fsig = np.array([1.1, 2.2])

K = len(fsig)


def component(f, a, p, n=N, t=t):
    return a*np.exp(1j*p)*np.exp(2*1j*np.pi*f*t[:n])


def signal(F, A, P, t=t):
    return sum([component(f, a, p) for (f, a, p) in zip(F, A, P)])

d = signal(fsig, asig, psig)

plt.close('all')

plt.figure()
plt.plot(t, np.real(d))

for i in range(K):
    plt.plot(t, np.real(component(fsig[i], asig[i], psig[i])), ':')


# %%

def findNPoles(s, t):

    dt = np.diff(t)[0]
    dtemp = np.real(d.copy())

    N = len(dtemp)
    nc = int(N/2)
    nl = int(N+1-nc)

    S = np.zeros((nl, nc)) + 1j*np.zeros((nl, nc))
    for i in range(nl):
            for j in range(nc):
                    S[i, j] = d[i+j]

    C = np.dot(S, S.T.conjugate())

    def sorted_eig(M):
        vap, vep = np.linalg.eig(M)
        inds = list(np.argsort(np.abs(vap)))
        inds.reverse()
        return vap[inds], vep[:, inds]

    vapC, vepC = sorted_eig(C)

    tol = np.finfo(float).eps
    imax = np.nonzero(np.abs(vapC)/np.abs(vapC[0]) < tol)[0][0]

    fest = []
    aest = []
    pest = []

    nfft = max(N, 2**10)

    for i in range(imax):
        print('\n', i)

        w = np.fft.fft(np.real(dtemp), nfft)
        freqs = np.fft.fftfreq(len(w))

        ifmax = np.argmax(w)
        freq = freqs[ifmax]
        fmax = abs(freq/dt)

#        mask = freqs >= 0
#
#        plt.figure()
#        plt.semilogy(freqs[mask]/dt, np.abs(w[mask]))
#        plt.semilogy(freqs[ifmax]/dt, np.abs(w[ifmax]), 'or')

        def funca(v, f):
            e = d-np.real(component(f, v[0], v[1], n=N, t=t))
            return np.real(np.dot(e, e.conjugate()))

        def funcf(v, a, p):
            e = d-np.real(component(v, a, p, n=N, t=t))
            return np.real(np.dot(e, e.conjugate()))

        f, a, p = fmax, 1, 0

        delta = 1
        while delta > np.finfo(float).eps:
            pre = np.array([f, a, p])
            v = np.array([a, p])
            a, p = minimize(funca, v, (f)).x
            f = minimize(funcf, f, (a, p)).x
            temp = pre-np.array([f, a, p])
            delta = np.sqrt(np.dot(temp, temp))

        fest.append(f)
        aest.append(a)
        pest.append(p)

        dtemp -= np.real(component(f, a, p))

    return fest, aest, pest

fest, aest, pest = findNPoles(np.real(d), t)

plt.figure()
plt.plot(t, np.real(d), label='target')
plt.plot(t, np.real(signal(fest, aest, pest, t=t)), label='synthesis')
plt.legend()

#%%

from scipy.interpolate import CubicSpline

angle = ((2*np.pi*fsig[0]*t)-np.pi)%(2*np.pi)
inds = np.argsort(angle)

md = np.mean(d[inds])

interpolator = CubicSpline(angle[inds], d[inds]-md)

angles = np.linspace(0, 2*np.pi, 1e4)
interp = interpolator(angles)

nangles = [0, 2*np.pi]

e = np.abs(interp)
error = 1

while error > 1e-2:
    nangles.append(angles[np.argmax(np.abs(e))])
    nangles.sort()
    redinterp = CubicSpline(nangles, interpolator(nangles))
    e = interp - redinterp(angles)
    error = np.linalg.norm(e)/np.linalg.norm(interp)

plt.close('all')
plt.plot(angle[inds], d[inds], '-r')
plt.plot(nangles, interpolator(nangles), 'ob')
plt.plot(angles, interpolator(angles), ':b')

plt.legend()

