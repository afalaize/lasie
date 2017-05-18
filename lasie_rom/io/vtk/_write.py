# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:07:22 2017

@author: afalaize
"""


from __future__ import absolute_import

import numpy as np
from xml.etree import cElementTree as ElementTree
from decimal import Decimal
import vtk
from lasie_rom.grids import generate, to_mesh
from vtk.util.numpy_support import numpy_to_vtk


def setPoints(vtkUnstructuredGrid, mesh):
    nx, nc = mesh.shape

    vtkPoints = vtk.vtkPoints()
    vtkPoints.SetNumberOfPoints(nx)

    for i, x in enumerate(mesh):
#        vtkPoints.InsertPoint(i, *(list(x) + [0,]))
        vtkPoints.InsertPoint(i, *x)

    vtkUnstructuredGrid.SetPoints(vtkPoints)


def insertCells(vtkUnstructuredGrid, shape):
    try:
        nc, nx, ny, nz = shape
        for i in range(nx-1):
            for j in range(ny-1):
                for k in range(nz-1):
                    index = ny*nz*i + nz*j + k
                    vtkIdList = vtk.vtkIdList()

                    vtkIdList.InsertNextId(index)
                    vtkIdList.InsertNextId(index + nz*ny)
                    vtkIdList.InsertNextId(index + (ny+1)*nz)
                    vtkIdList.InsertNextId(index + nz)

                    vtkIdList.InsertNextId(index + 1)
                    vtkIdList.InsertNextId(index + nz*ny + 1)
                    vtkIdList.InsertNextId(index + (ny+1)*nz + 1)
                    vtkIdList.InsertNextId(index + nz + 1)

                    vtkUnstructuredGrid.InsertNextCell(vtk.VTK_HEXAHEDRON,
                                                       vtkIdList)
    except ValueError:
        nc, nx, ny = shape
        for i in range(nx-1):
            for j in range(ny-1):
                    index = ny*i + j
                    vtkIdList = vtk.vtkIdList()
                    vtkIdList.InsertNextId(index)
                    vtkIdList.InsertNextId(index + ny)
                    vtkIdList.InsertNextId(index + ny+1)
                    vtkIdList.InsertNextId(index + 1)

                    vtkUnstructuredGrid.InsertNextCell(vtk.VTK_TETRA,
                                                       vtkIdList)



def write_vtk(mesh, shape, data, path):
    """
    Parameters
    -----------

    mesh: np.array
        The (regular) mesh on wich the data are defined. Format is (n_points, n_components).

    shape: (nc, nx, ny, nz, ...)
        shape of the grid, with nc the number of spatial components, and
        (nx, ny, nz, ...) the dimensions of the grid in each direction.

    data: dic
        The data to write. Each key is the data name (xml tag) and each value 
        is the corresponding array with shape (nx, n_data_components).

    path: str
        The path to the file to write the data in.
    """
    # importé et modifié depuis code_Erwan\Routines_vtk.py, fonction "ecrivtk2"

    vtkUnstructuredGrid = vtk.vtkUnstructuredGrid()

    setPoints(vtkUnstructuredGrid, mesh)
    insertCells(vtkUnstructuredGrid, shape)

    npoints, nc = mesh.shape
    nc2 = shape[0]
    assert nc2 == nc

    for k in data.keys():
        npoints_data, nc_data = data[k].shape
        assert npoints == npoints_data   
        vtkFloatArray = numpy_to_vtk(data[k].ravel())
        vtkFloatArray.SetNumberOfComponents(nc_data)
        vtkFloatArray.SetName(k)
        vtkUnstructuredGrid.GetPointData().AddArray(vtkFloatArray)
        #vtkUnstructuredGrid.GetPointData().SetActiveVectors(k)

    vtkfile = vtk.vtkXMLUnstructuredGridWriter()
    vtkfile.SetInputData(vtkUnstructuredGrid)
    vtkfile.SetFileName(path)
    vtkfile.Write()

# TEST USING pyvtk => BUG in the conection sci.spatial.Delaunay and pyvtk.UnstructuredGrid
#def write_with_pyvtk(mesh, shape, data, path, title=None):
#    """
#    Parameters
#    -----------
#
#    mesh: np.array
#        The (regular) mesh on wich the data are defined. Format is (n_points, n_components).
#
#    shape: (nc, nx, ny, nz, ...)
#        shape of the grid, with nc the number of spatial components, and
#        (nx, ny, nz, ...) the dimensions of the grid in each direction.
#
#    data: dic
#        The data to write. Each key is the data name (xml tag) and each value 
#        is the corresponding array with shape (nx, n_data_components).
#
#    path: str
#        The path to the file to write the data in.
#    """
#    import pyvtk
#    vtk_data = list()
#    for k in data.keys():
#        d = data[k]
#        shape = d.shape
#        assert len(shape) == 2
#        if shape[1] == 1:
#            vtk_data.append(pyvtk.Scalars(d, name=k))
#        else:
#            assert shape[1] == mesh.shape[1]
#            vtk_data.append(pyvtk.Vectors(d, name=k))
#        
#    tri = Delaunay([list(e) for e in mesh])
#    
#    vtk = pyvtk.VtkData(\
#      pyvtk.UnstructuredGrid(mesh,
#        tetra=tri.simplices
#        ),
#      pyvtk.PointData(*vtk_data),
#      title
#      )
#      
#    vtk = pyvtk.VtkData(pyvtk.UnstructuredGrid(mesh),
#                        )
#    
#    vtk.tofile(path)
#


def array2text(data, dtype=None):
    if dtype is None:
        dtype = Decimal
    string = ''
    Nx, Nc = data.shape
    for i in range(Nx):
        formated_vector = map(dtype, data[i, :].tolist())
        string += ' '.join(['{:f}',]*3).format(*formated_vector) + '\n'
    return string


def getTemplate(filename):
    # Get file content
    vtufile = open(filename, mode='r')
    tree = ElementTree.parse(vtufile)
    vtufile.close()
    VTKFile = tree.getroot()
    UnstructuredGrid = VTKFile[0]
    Piece = UnstructuredGrid[0]
    for tag in ['CellData', 'UserData']:
        tags = [el.tag for el in Piece]
        index = tags.index(tag)
        del Piece[index]
    tags = [el.tag for el in Piece]
    PointData = Piece[tags.index('PointData')]
    names = [e.attrib['Name'] for e in PointData]
    for name in names:
        current_names = [e.attrib['Name'] for e in PointData]
        index = current_names.index(name)
        if name == 'Vitesse(m/s)':
            print(PointData[index].attrib)
        del PointData[index]
    return tree


ATTRIBUTES = {'NumberOfComponents': '2',
              'type': 'Float32',
              'Name': 'Vitesse(m/s)',
              'format': 'ascii'}


def setData(tree, data, label='PodBasisElement', dtype=None):
    VTKFile = tree.getroot()
    UnstructuredGrid = VTKFile[0]
    Piece = UnstructuredGrid[0]
    tags = [el.tag for el in Piece]
    PointData = Piece[tags.index('PointData')]
    Nx, Nb, Nc = data.shape
    for b in range(Nb):
        print('Writing pod basis element {}/{}'.format(b+1, Nb))
        d = data[:, b, :]
        subel = ElementTree.SubElement(PointData, 'DataArray')
        attrib = ATTRIBUTES.copy()
        attrib.update({'Name': label+str(b+1)})
        for key in attrib.keys():
            subel.set(key, attrib[key])
        subel.text = array2text(d, dtype=dtype)
    return tree


if __name__ == '__main__':
    grid, h = generate(((0., 1.), (0, 2.), (0., 3.)), 0.1)
    mesh = to_mesh(grid)
    data = np.zeros(mesh.shape)
    for i, x in enumerate(mesh):
        data[i, :] = map(lambda e: np.sin(np.pi*e), x)
    data = data.reshape((np.prod(data.shape), ))
    write_vtu(mesh, grid.shape, [data, ], 'testData', 'test.vtu')
