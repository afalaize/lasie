#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 12:38:45 2017

@author: root
"""

import os
import tables
from ..io.hdf.write import writeArray2Hdf
from ..parallelization.tools import parmap


def interpTimeSerieToHdf(ts, mesh, folder):
    """
====================
interpTimeSerieToHdf
====================

Interpolate every data in the time serie :code:`ts` over the given :code:`mesh`
and store the results in HDF5 files in the given :code:`folder`

Parameters
----------

ts: lasie.classes.TimeSerie
    The original time serie

mesh: numpy array with shape (nx, nc)
    The new mesh to interpolate the data in the time serie, with mesh[i, j] the
    j-th coordinate component of the i-th mesh point.

folder: str
    The folder where the resulting HDF files are generated.
    
Output
------

None:
    This function does not generates anything; all the results are stores in 
    the prescribed :code:`folder`.
    
See also
--------

:code:`lasie.classes.TimeSerie` to read the resulting hdf5 files as a time
serie, and :code:`lasie.readwrite.HDFReader` to read a single hdf5 file.
    """
    
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    arguments = list(zip(ts.paths, range(len(ts.data)), ts.times))

    def func(arg):
        hdf_path, i, t = arg
        data = ts.data[i]
        
        flag_close_data = False
        if not hasattr(data, 'names'):
            data.openHdfFile()
            flag_close_data = True
            
        if not hasattr(data, 'interpolators'):
            data.buildInterpolators()
        
        # build the path to the interpolated hdf5 file
        i_sep = hdf_path.rfind(os.sep)
        hdf_filename = hdf_path[i_sep+1:]
        print('Interpolate {}'.format(hdf_filename))
        interp_hdf_path = folder + os.sep + hdf_filename
        
        # title for the hdf file: a list of data names recovered from the .hdf5 file
        hdf_title = data.hdf_file.title
        
        # open the hdf file in write mode
        hdf_file = tables.open_file(interp_hdf_path, mode='w', title=hdf_title)
        
        for name in data.names:
            if name == 'mesh':
                writeArray2Hdf(hdf_file, mesh, name)
            else:
                writeArray2Hdf(hdf_file, data.interpolators[name](mesh), name)
                
        hdf_file.close()
        
        del(data.interpolators)
        
        if flag_close_data:
            data.closeHdfFile()
            
    
    listOfHdfFiles = open(folder + os.sep + 'listOfHdfFiles.txt', 
                          'w')

    parmap(func, arguments)

    for hdf_path, data, t in arguments:
        i_sep = hdf_path.rfind(os.sep)
        hdf_filename = hdf_path[i_sep+1:]
        print('Interpolate {}'.format(hdf_filename))
        interp_hdf_path = folder + os.sep + hdf_filename
        listOfHdfFiles.write('{} {}\n'.format(t, interp_hdf_path))
        
    listOfHdfFiles.close()     
