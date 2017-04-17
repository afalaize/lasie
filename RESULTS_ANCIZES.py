# -*- coding: utf-8 -*-
"""
Created on Fri Apr 14 16:25:06 2017

@author: afalaize
"""
from pypod.readwrite.read_hdf import HDFData, HDFTimeSerie
from pypod.readwrite.write_vtu import write_vtu
from pypod.readwrite.vtu2hdf import pvd2Hdf
from pypod.readwrite.vtu2hdf import dumpArrays2Hdf
import os
import numpy as np

def reconstruct_snapshots_ROM(rom):
    config = rom.config
    HDFbasis = HDFData(config['hdf_path_podBasis'], openFile=True)
    name = HDFbasis.names[0]
    basis = HDFbasis.get_single_data()
    HDFbasis.closeHdfFile()
    
    mean = HDFData(config['hdf_path_mean'], openFile=True).get_single_data()
    grid = HDFData(config['hdf_path_grid'], openFile=True)
    folder = 'reconstructed_ROM'
    pvd_file = open(config['vtu_folder'] + os.sep + folder + os.sep + 'vitesse.pvd', 'w')
    pvd_file.write("""<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
    <Collection>""")
    template = '\n        <DataSet timestep="{0}" group="" part="0" file="{1}"/>'
 
    listOfHdfFiles = open(config['vtu_folder'] + os.sep + folder + os.sep + 'listOfHdfFiles.txt' , 'w')
    for i, coeffs in enumerate(rom.c_rom()[:-2]):
        v = mean + np.einsum('i,mic->mc', coeffs, basis)
        vtu_path = config['vtu_folder'] + os.sep + folder+ os.sep + 'vitesse_{}.vtu'.format(i+1)
        write_vtu(grid.mesh[:], grid.original_shape,
                  [v, ],
                  'vitesse', vtu_path)
        pvd_file.write(template.format(rom.times[i], vtu_path))
        hdf_path = config['vtu_folder'] + os.sep + folder+ os.sep + 'vitesse_{}.hdf5'.format(i+1)
        dumpArrays2Hdf([v, ], [name, ], hdf_path)
        listOfHdfFiles.write('{} {}\n'.format(rom.times[i], hdf_path))
    pvd_file.write("""
    </Collection>
</VTKFile>""")
    pvd_file.close()
    grid.closeHdfFile()
    listOfHdfFiles.close()     

def vtu2hdf_snapshots_ROM(rom):
    folder = 'reconstructed_ROM'
    pvdpath = rom.config['vtu_folder'] + os.sep + folder + os.sep + 'vitesse.pvd'
    pvd2Hdf(pvdpath, folder, ['vitesse1'])
    
    
def reconstruct_snapshots_error(rom):
    config = rom.config
    HDFbasis = HDFData(config['hdf_path_podBasis'], openFile=True)
    name = HDFbasis.names[0]
    basis = HDFbasis.get_single_data()
    HDFbasis.closeHdfFile()
    
    ts = HDFTimeSerie(config['interp_hdf_folder'])
    ts.openAllFiles()
    
    mean = HDFData(config['hdf_path_mean'], openFile=True).get_single_data()
    grid = HDFData(config['hdf_path_grid'], openFile=True)
    folder = 'Error'
    pvd_file = open(config['vtu_folder'] + os.sep + folder + os.sep + 'error.pvd', 'w')
    pvd_file.write("""<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
    <Collection>""")
    template = '\n        <DataSet timestep="{0}" group="" part="0" file="{1}"/>'
 
    listOfHdfFiles = open(config['vtu_folder'] + os.sep + folder + os.sep + 'listOfHdfFiles.txt' , 'w')
    for i, d in enumerate(ts.data):
        v_rom = mean + np.einsum('i,mic->mc', rom.c_rom()[i], basis)
        error = np.abs(v_rom-d.vitesse[:])
        vtu_path = config['vtu_folder'] + os.sep + folder+ os.sep + 'error_{}.vtu'.format(i+1)
        write_vtu(grid.mesh[:], grid.original_shape,
                  [error, ],
                  'error', vtu_path)
        pvd_file.write(template.format(rom.times[i], vtu_path))
        hdf_path = config['vtu_folder'] + os.sep + folder+ os.sep + 'error_{}.hdf5'.format(i+1)
        dumpArrays2Hdf([error, ], [name, ], hdf_path)
        listOfHdfFiles.write('{} {}\n'.format(rom.times[i], hdf_path))
    pvd_file.write("""
    </Collection>
</VTKFile>""")
    pvd_file.close()
    grid.closeHdfFile()
    listOfHdfFiles.close()     


def reconstruct_snapshots_FOM(rom):
    config = rom.config
    basis = HDFData(config['hdf_path_podBasis'], openFile=True).get_single_data()
    mean = HDFData(config['hdf_path_mean'], openFile=True).get_single_data()
    grid = HDFData(config['hdf_path_grid'], openFile=True)
    folder = 'reconstructed_FOM'
    pvd_file = open(config['vtu_folder'] + os.sep + folder + os.sep + 'vitesse.pvd', 'w')
    pvd_file.write("""<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
    <Collection>""")
    template = '\n        <DataSet timestep="{0}" group="" part="0" file="{1}"/>'
 
    for i, coeffs in enumerate(rom.c_fom()):
        v = mean + np.einsum('i,mic->mc', coeffs, basis)
        vtu_path = config['vtu_folder'] + os.sep + folder+ os.sep + 'vitesse_{}.vtu'.format(i+1)
        write_vtu(grid.mesh[:], grid.original_shape,
                  [v, ],
                  'vitesse', vtu_path)
        pvd_file.write(template.format(rom.times[i], vtu_path))
    pvd_file.write("""
    </Collection>
</VTKFile>""")
    pvd_file.close()
    grid.closeHdfFile()

    
def recover_snapshots_FOM(rom):
    config = rom.config
    mean = HDFData(config['hdf_path_mean'], openFile=True).get_single_data()
    ts = HDFTimeSerie(config['interp_hdf_folder'])
    ts.openAllFiles()
    grid = HDFData(config['hdf_path_grid'], openFile=True)
    folder = 'recover_FOM'
    pvd_file_origin = open(config['vtu_folder'] + os.sep + folder + os.sep + 'vitesse_fluctuante.pvd', 'w')
    pvd_file_origin.write("""<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
    <Collection>""")
    pvd_file_fluc = open(config['vtu_folder'] + os.sep + folder + os.sep + 'vitesse_originale.pvd', 'w')
    pvd_file_fluc.write("""<VTKFile type="Collection" version="0.1" byte_order="LittleEndian">
    <Collection>""")
    template = '\n        <DataSet timestep="{0}" group="" part="0" file="{1}"/>'
 
    for i, d in enumerate(ts.data):
        v = d.vitesse[:]
        v_fluc = v-mean
        vtu_path = config['vtu_folder'] + os.sep + folder+ os.sep + 'vitesse_originale{}.vtu'.format(i+1)
        write_vtu(grid.mesh[:], grid.original_shape,
                  [v, ],
                  'vitesse_originale', vtu_path)
        pvd_file_origin.write(template.format(rom.times[i], vtu_path))
        vtu_path = config['vtu_folder'] + os.sep + folder+ os.sep + 'vitesse_fluctuante{}.vtu'.format(i+1)
        write_vtu(grid.mesh[:], grid.original_shape,
                  [v_fluc, ],
                  'vitesse_fluctuante', vtu_path)
        pvd_file_fluc.write(template.format(rom.times[i], vtu_path))
    pvd_file_origin.write("""
    </Collection>
</VTKFile>""")
    pvd_file_origin.close()
    pvd_file_fluc.write("""
    </Collection>
</VTKFile>""")
    pvd_file_fluc.close()
    grid.closeHdfFile()
    ts.closeAllFiles()
