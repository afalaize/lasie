#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 18:41:37 2017

@author: afalaize
"""

import numpy as np
import sympy as sy

def sign(x, eps=0.):
    """
    eps can be regarded as an immersion precision
    """
    if eps == 0.0:
        return np.sign(x)
    elif eps > 0:
        return np.tanh(np.pi*x/eps)
    else:
        raise AttributeError('eps is negative')


def heaviside(x, eps=0.):
    """
    eps can be regarded as an immersion precision
    """
    return 0.5*(1.+sign(x, eps=eps))


def build_vectorized_heaviside(eps=0.):
    def func(x): return heaviside(x, eps)
    return np.vectorize(func)


def sysign(x, eps=0.):
    """
    eps can be regarded as an immersion precision
    """
    if eps == 0.0:
        return sy.sign(x)
    elif eps > 0:
        return sy.tanh(np.pi*x/eps)
    else:
        raise AttributeError('eps is negative')


def syheaviside(x, eps=0.):
    """
    eps can be regarded as an immersion precision
    """
    return 0.5*(1.+sysign(x, eps=eps))


def sybuild_vectorized_heaviside(eps=0.):
    def func(x): return syheaviside(x, eps)
    return np.vectorize(func)
