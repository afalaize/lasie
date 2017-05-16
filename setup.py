#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed May 10 18:19:01 2017

@author: root
"""

from __future__ import absolute_import

from setuptools import setup, find_packages

# To use a consistent encoding
from codecs import open

from os import path

from lasie import __author__, __version__, __licence__, __author_email__


###############################################################################
here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

###############################################################################


setup(name='lasie',
      version=__version__,
      description="Development Status :: 3 - Alpha",
      long_description=long_description,
      classifiers=[
        'Natural Language :: English',
        'Development Status :: 3 - Alpha',
        'Operating System :: Unix',
        'Operating System :: MacOS',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft',
        'Operating System :: Microsoft :: Windows',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: ' + __licence__,
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Physics'
      ],
      keywords='model order reduction',
      url='https://github.com/afalaize/lasie',
      author=__author__,
      author_email=__author_email__,
      license=__licence__,
      packages=find_packages(exclude=[]),
      zip_safe=False,
      include_package_data=True,
      install_requires=[
          'numpy',
          'scipy',
          'progressbar2',
          'matplotlib',
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      )