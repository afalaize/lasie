# -*- coding: utf-8 -*-
"""
Created on Tue Mar 07 11:11:09 2017

@author: afalaize
"""

import os


def getFolderFromPath(path):
    """
    Get the folder that contains a given path.
    
    Example
    --------
    >>> p = 'C:\\Users\\user1\\Documents\\a_document.txt'
    >>> getFolderFromPath(p)
    'C:\\Users\\user1\\Documents'
    """
    index = path.rfind(os.sep)
    if index == -1:
        return os.getcwd()
    else:
     return path[:index]

