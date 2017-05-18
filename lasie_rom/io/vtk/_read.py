#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:28:03 2017

@author: root
"""

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy


def read_vtk(path):
    """
    read_vtk
    ********
    
    Read the vtk file pointed by path and returns a dictionary of numpy arrays.
    
    Parameters
    ----------
    
    path : str
        Path to the vtk file to read.
        
    Return
    ------
    
    data : dictionary
        Recover the mesh in data['mesh'] and every data arrays in data[name]
        with name the data name in the original vtk file. Each array is of the 
        shape (nx, nc) with nx the number of mesh points and nc the number of 
        data components.
    """
    # init vtk reader
    reader = vtk.vtkXMLUnstructuredGridReader()
    # connect to file
    reader.SetFileName(path)
    # update metadata
    reader.Update()
    
    # retrieve values
    output = reader.GetOutput()

    # retrieve mesh
    mesh = vtk_to_numpy(output.GetPoints().GetData())
    
    # store mesh in dic
    data = {'mesh': mesh}
    
    # get vtk data
    vtk_data = output.GetPointData()
    # get number of arrays in vtk data
    narrays = vtk_data.GetNumberOfArrays()
    
    for i in range(narrays):
        # recover array name
        key = vtk_data.GetArrayName(i)
        # recover array values
        array = vtk_to_numpy(vtk_data.GetArray(i))
        # add an axe if array is 1D => 2D array
        if len(array.shape) == 1:
            array = array[:, np.newaxis]
        # store array in dic
        data.update({key: array})
    return data
    
