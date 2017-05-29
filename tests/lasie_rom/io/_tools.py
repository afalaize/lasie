# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 11:11:09 2017

@author: afalaize
"""

import os


def getFolderFromPath(path):
    """
    Get the folder that contains a given :code:`path`.
    
    Paameters
    ---------
    
    path: str
        A path to a file.
        
    Return
    ------
    
    folder: str
        The folder that contains the file pointed by :code:`path`.
        
    Examples
    --------
    On Linux:
    >>> path = '/home/user1/Documents/document.txt'
    >>> getFolderFromPath(path)
    '/home/user1/Documents'    

    On Windows:
    >>> path = 'C:\\Users\\user1\\Documents\\document.txt'
    >>> getFolderFromPath(path)
    'C:\\Users\\user1\\Documents'
    """
    index = path.rfind(os.sep)
    if index == -1:
        return os.getcwd()
    else:
     return path[:index]

