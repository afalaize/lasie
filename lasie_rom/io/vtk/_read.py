#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 17 13:28:03 2017

@author: root
"""

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from xml.etree import cElementTree as ElementTree
from collections import OrderedDict
import os


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
    ndims_mesh = mesh.shape[1]
    nonempty_axes = range(ndims_mesh)
    for i in range(ndims_mesh):
        if len(np.nonzero(mesh[:, i])[0]) == 0:
            nonempty_axes.pop(i)
    
    # Remove empty dimensions
    mesh = mesh[:, np.array(nonempty_axes)]
    
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
        # Remove empty dimensions
        elif array.shape[1] == ndims_mesh:
            array = array[:, np.array(nonempty_axes)]
            
        # store array in dic
        data.update({key: array})
    return data

    
def pvd2files(pvd_path):
    """
    Extract a list of .vtu file paths from the .pvd file pointed by pvd_path.
    
    Warning
    -------
    Duplicate .vtu file paths are deleted, so the list can possibly not contain
    the whole list of paths.
    """
    pvdtree = ElementTree.parse(open(pvd_path))
    folder = pvd_path[:pvd_path.rfind(os.sep)]
    listOfFiles = list()
    for elem in pvdtree.getiterator('DataSet'):
        path = elem.attrib['file']
        istartname = path.rfind(os.sep)
        if istartname == -1:
            istartname = 0
        else:
            istartname += 1
        listOfFiles.append(os.path.join(folder, path[istartname:]))
    # WARNING: we remove duplicate filenames
    return list(OrderedDict.fromkeys(listOfFiles))


def pvd2times(pvd_path):
    """
    Extract a list of times from the .pvd file pointed by pvd_path.

    Warning
    --------
    Duplicate time values are deleted, so the list can possibly not contain
    the whole list of time values.
    """
    pvdtree = ElementTree.parse(open(pvd_path))
    listOfTimes = list()
    for elem in pvdtree.getiterator('DataSet'):
        listOfTimes.append(float(elem.attrib['timestep']))
    # WARNING: we remove duplicate times
    return list(OrderedDict.fromkeys(listOfTimes))


