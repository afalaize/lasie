# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 16:22:27 2017

@author: afalaize
"""

import os
import tables
from scipy.interpolate import LinearNDInterpolator
from .vtu2hdf import writeArray2Hdf


def readlistOfHdfFiles(folder):
    """
    Read the file 'listOfHdfFiles.txt' in 'folder' and retrieve:
        - list of times
        - list of corresponding hdf files
    
    Parameter
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
    

def interpTimeSerieToHdf(ts, mesh, output_folder):
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    listOfHdfFiles = open(output_folder + os.sep + 'listOfHdfFiles.txt', 
                          'w')

    for i, (hdf_path, data, t) in enumerate(zip(ts.paths, ts.data, ts.times)):
        print('Build interpolators {}/{}.'.format(i+1, len(ts.data)))
        
        if not hasattr(data, 'interpolators'):
            data.buildInterpolators()
        
        # build the path to the interpolated hdf5 file
        i_sep = hdf_path.rfind(os.sep)
        hdf_filename = hdf_path[i_sep+1:]
        interp_hdf_path = output_folder + os.sep + hdf_filename
        
        # title for the hdf file: a list of data names recovered from the .hdf5 file
        hdf_title = data.hdf_file.title
        
        # open the hdf file in write mode
        hdf_file = tables.open_file(interp_hdf_path, mode='w', title=hdf_title)
        
        print('Evaluate interpolators {}/{}.'.format(i+1, len(ts.data)))
        for name in data.names:
            if name == 'mesh':
                writeArray2Hdf(hdf_file, mesh, name)
            else:
                writeArray2Hdf(hdf_file, data.interpolators[name](mesh), name)
                
        hdf_file.close()
        
        del(data.interpolators)
        listOfHdfFiles.write('{} {}\n'.format(t, interp_hdf_path))
        
    listOfHdfFiles.close()     
        

class HDFData(object):
    """
    Class for reading HDF5 files
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
                print('    Build interpolator for {}'.format(name))
                data = getattr(self, name)
                self.interpolators[name] = LinearNDInterpolator(self.mesh[:],
                                                                data[:])
            
    def evalInterpolator(self, name, mesh):
        return self.interpolators[name](mesh)

    def get_single_data(self):
        assert len(self.names) == 1
        return getattr(self, self.names[0])[:]
        

class HDFTimeSerie(object):
    """
    """
    def __init__(self, folder, openFiles=False):
        """
        """
        self.times, self.paths = readlistOfHdfFiles(folder)
        self.data = []

        for path in self.paths:
            self.data.append(HDFData(path))
            
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
        assert t >= self.times[0], 'time smaller than min in time-serie: \
{} < {}'.format(t, self.times[0])
        assert t <= self.times[-1], 'time greater than max in time-serie: \
{} > {}'.format(t, self.times[-1])
        if t == self.times[-1]:
            return getattr(self.data[-1], name)[:]
        else:
            index_t = [t >= e for e in self.times].index(True)
            t0 = self.times[index_t]
            t1 = self.times[index_t+1]
            coeff = (t-t0)/(t1-t0)
            return ((1-coeff)*getattr(self.data[index_t], name)[:] + 
                    coeff*getattr(self.data[index_t+1], name)[:])
