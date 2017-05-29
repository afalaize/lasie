# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:22:27 2017

@author: afalaize
"""

import os
import tables
from scipy.interpolate import LinearNDInterpolator


class HDFReader(object):
    """
    Class for reading a HDF5 file generated .
    """

    def __init__(self, path, openFile=False):
        self.path = path
        if openFile:
            self.openHdfFile()
        
    def openHdfFile(self):
        self.hdf_file = tables.open_file(self.path, mode='r')
        self.names = self.hdf_file.title.split()
        for name in self.names:
            setattr(self, name, getattr(self.hdf_file.root, name))
            
    def closeHdfFile(self):
        self.hdf_file.close()
        
    def getMeshMinMax(self):
        minmax = list()
        for i, xi_mesh in enumerate(self.mesh[:].T):
            ximin, ximax = min(xi_mesh), max(xi_mesh)
            minmax.append((ximin, ximax))        
        return minmax
        
    def buildInterpolators(self):
        self.interpolators = {}
        for name in self.names:
            if not name == 'mesh':
                data = getattr(self, name)
                self.interpolators[name] = LinearNDInterpolator(self.mesh[:],
                                                                data[:])
                
    def evalInterpolator(self, name, mesh):
        return self.interpolators[name](mesh)

    def get_single_data(self):
        assert len(self.names) == 1
        return getattr(self, self.names[0])[:]
    
    def data_dict(self):
        """
        Return all the data in dict format {'dataname': numpy_array}.
        """
        d = dict()
        for name in self.names:
            d[name] = getattr(self, name)[:]
        

def readlistOfHdfFiles(folder):
    """
    Read the file 'listOfHdfFiles.txt' in 'folder' and retrieve:
        - list of times
        - list of corresponding hdf files
    
    Parameterfolder
    ----------
    
    folder: str
        The folder that contains the hdf files.
    
    Outputs
    --------
    
    listOfTimes: list of floats
        List of time steps
        
    listOfFiles: list of str
        List of path to hdf5 files
    
    """
    filename = folder + os.sep + r'listOfHdfFiles.txt'
    file_ = open(filename, 'r')
    listOfTimes = list()
    listOfFiles = list()
    for line in file_.readlines():
        time, path = line.split(' ')
        listOfTimes.append(float(time))
        listOfFiles.append(path[:-1])
    return listOfTimes, listOfFiles
    