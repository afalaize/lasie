#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 15:09:54 2017

@author: afalaize
"""

import numpy as np
import sympy as sp

from lasie_rom import plots
from lasie_rom import grids
import matplotlib.pyplot as plt

from lasie_rom.parallelization import map

# Domain dimensions
xlims = [0, 1]
ylims = [0, 1]

# Domain dimensions
grid, h = grids.generate((xlims, ylims), h=0.01)
mesh = grids.to_mesh(grid)


def implicit(l0, l1, a0, a1, b0, b1):
    l = sp.symbols('l:2', real=True, positive=True)
    a = sp.symbols('a:2', real=True)
    b = sp.symbols('b:2', real=True)
    x = sp.symbols('x:2', real=True)
    p = sp.symbols('p:2', real=True)
    theta_symb = sp.symbols('theta', real=True, positive=True)
    t = sp.symbols('t', real=True)

    L = sp.Matrix([[l[0], 0],
                   [0, l[1]]])

    R = sp.Matrix([[sp.cos(theta_symb), -sp.sin(theta_symb)],
                   [sp.sin(theta_symb), sp.cos(theta_symb)]])

    A = R*L.inv()*R.T

    v = (sp.eye(2)-R.T)*sp.Matrix(b)-sp.Matrix(a)
    c = 2*R*L.inv()*v

    d = (v.T*L.inv()*v)[0, 0] - 1

    Q = (sp.Matrix(x).T*A*sp.Matrix(x) + c.T*sp.Matrix(x))[0, 0] + d

    Xmin = R*(sp.eye(2) + 2*t*L.inv()).inv()*(R.T*sp.Matrix(p) - t*R.T*c)

    subs = {x[0]: Xmin[0, 0],
            x[1]: Xmin[1, 0]}

    Q = Q.subs(subs)

    subs = {l[0]: l0**2,
            l[1]: l1**2,
            a[0]: a0,
            a[1]: a1,
            b[0]: b0,
            b[1]: b1}

    Xmin = Xmin.subs(subs)

    Q = Q.subs(subs)

    alpha = list(R.T*sp.Matrix(p))
    beta = list(R.T*c)

    coeffs = list()
    p0 = d + alpha[0]*beta[0] + alpha[1]*beta[1] + alpha[0]**2/l[0] + alpha[1]**2/l[1]
    coeffs.append(p0)
    p1 = 4*(d*(1/l[0] + 1/l[1]) + alpha[0]*(beta[0] + alpha[0]/l[0])/l[1] + alpha[1]*(beta[1] + alpha[1]/l[1])/l[0])- (beta[0]**2+beta[1]**2)
    coeffs.append(p1)
    p2 = 4*(d*((1/l[0] + 1/l[1])**2 + 2/(l[0]*l[1])) + alpha[0]*(beta[0] + alpha[0]/l[0])/l[1]**2 + alpha[1]*(beta[1] + alpha[1]/l[1])/l[0]**2) - beta[0]**2*(4/l[1] + 1/l[0]) - beta[1]**2*(4/l[0] + 1/l[1])
    coeffs.append(p2)
    p3 = 4*(1/l[0] + 1/l[1])*(4*d/(l[0]*l[1])-(beta[1]**2/l[0] + beta[0]**2/l[1]))
    coeffs.append(p3)
    p4 = 4*(4*d/(l[0]*l[1])-(beta[1]**2/l[0] + beta[0]**2/l[1]))/(l[0]*l[1])
    coeffs.append(p4)

    coeffs = [e.subs(subs) for e in coeffs]

    args = [theta_symb]+list(p)
    coeffs = [sp.lambdify(args, e, dummify=False) for e in coeffs]
    xmin = sp.lambdify([t, theta_symb]+list(p), list(Xmin), dummify=False)

    def poly(theta, *p):
        return sp.Poly(sum([pi(theta, *p)*t**i for i, pi in enumerate(coeffs)]))
    return poly, xmin

#(sp.lambdify([theta_symb, ]+list(p), Q, dummify=False),
#            sp.lambdify([t, theta_symb]+list(p), list(Xmin), dummify=False))

P, xmin = implicit(0.4, 0.1, 0.5, 0.5, 0.5, 0.5)
theta = 0.5

def distance(p):
    poly = P(theta, *p)
    return min([np.linalg.norm(np.array(xmin(t, theta, *p), dtype=float) - p) for t in [r.evalf() for r in poly.real_roots()]])

d = np.array(map(distance, mesh))

plots.plot2d(d[:, np.newaxis, np.newaxis], grid.shape, render=0)
