#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:26:42 2017

@author: root
"""

from ..io.hdf import HDFReader, readlistOfHdfFiles
from ..misc.tools import concatenate_in_given_axis


class TimeSerie(object):
    """
Time serie reader.
    """
    def __init__(self, folder, openFiles=False):
        """
        """
        self.times, self.paths = readlistOfHdfFiles(folder)
        self.data = []

        for path in self.paths:
            self.data.append(HDFReader(path))

        if openFiles:
            self.openAllFiles()

    def openAllFiles(self):
        """
        """
        for d in self.data:
            d.openHdfFile()

    def closeAllFiles(self):
        """
        """
        for d in self.data:
            d.closeHdfFile()

    def time_interp(self, name, t):
        """
        ts.times[0] <= t <= ts.times[-1]
        """
        if not t >= self.times[0]:
            raise Exception('Time smaller than minimum in time-serie: {} < {}'.format(t, self.times[0]))
        if not t <= self.times[-1]:
            raise Exception('Time greater than max in time-serie: {} > {}'.format(t, self.times[-1]))
        if t == self.times[-1]:
            return getattr(self.data[-1], name)[:]
        else:
            index_t = [t >= e for e in self.times].index(True)
            t0 = self.times[index_t]
            t1 = self.times[index_t+1]
            coeff = (t-t0)/(t1-t0)
            return ((1-coeff)*getattr(self.data[index_t], name)[:] +
                    coeff*getattr(self.data[index_t+1], name)[:])

    def generator(self, name):
        """
Parameter
---------
name: str
    The name of the attribute to retrieve from each data in TimeSerie.data

Output
------
func: generator function
    A function without argument that returns the generator.
        """
        def generate_data():
            for i, d in enumerate(self.data):
                assert hasattr(d, 'names'), 'hdf files are not opened; use TimeSerie.openAllFiles()'
                yield getattr(d, name)[:]
        return generate_data
                
    def concatenate(self, name):
        """
    Returns a single array with shape (nx, nc, ..., nt) with nx the number of
    points in the mesh, nc the number of components and nt = len(TimeSerie.data)
    for the data atttribute that corresponds to name (the '...' means that
    each array in the time serie can be N-dimensional with N>2).
        """
        shape = getattr(self.data[0], name)[:].shape
        generator = self.generator(name)
        return concatenate_in_given_axis(generator(), axis=len(shape))
