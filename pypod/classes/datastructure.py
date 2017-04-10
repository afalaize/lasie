#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 16:36:10 2017

@author: Falaize
"""

from ..readwrite.tools import getFolderFromPath
import numpy as np
from ..config import ORDER


class Data(object):
    """
    4-dimensional data structure with shape (nx, nt, n1, n2) where 
        * nx is the number of mesh points, 
        * nt is the number of time steps, 
        * n1 and n2 are the dimensions of the tensor over axis 1 and 2.
    
    For a scalar quantity, n1=n2=1.
    For a vector quantity, n1=number of vector component and n2=1.
    For a tensor, n1 and n2 are the number of components over each axis.
    """
    
    def __init__(self, path):
        self.path = path
        
    def folder(self):
        return getFolderFromPath(self.path)

    def get_array(self):
        return self.hdf5
        
    def set_array(self):
        return self.hdf5
        
    array = property(get_array, set_array)
    
    
        
    def nx(self):
        return self._array.shape[0]

    def nc(self):
        return self.array().shape[1:]

    def order(self):
        return len(self.nc())

    def vector(self):
        return self._array.reshape((1, self.nx()*np.prod(self.nc())),
                                   order=ORDER)[0]

    def array(self):
        return self._array

    def dot(self, array):
        return np.array(map(np.dot, self.array(), array))

    @staticmethod
    def reshape(array, nc):
        if isinstance(nc, (tuple, list)):
            assert len(nc) == 2
            newshape = (array.shape[0]//np.prod(nc), nc[0], nc[1])
        else:
            assert isinstance(nc, int)
            newshape = (array.shape[0]//nc, nc)
        return array.reshape(newshape, order=ORDER)




if __name__ == '__main__':
    nc = 2
    def t(i):
        return [range((i*nc+j)*nc, (i*nc+j+1)*nc) for j in range(nc)]
    nx = 3
    d = np.array([t(i) for i in range(nx)])
    t = Tensor(d)
    print(Tensor.reshape(t.vector(), (nc, nc)))
    print(t.dot(t.array().swapaxes(1,2)))



