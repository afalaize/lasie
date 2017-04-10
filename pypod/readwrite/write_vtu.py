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
from ..grids.tools import buildGrid, grid2mesh

def setPoints(vtkUnstructuredGrid, mesh):
    nx, nc = mesh.shape

    vtkPoints = vtk.vtkPoints()
    vtkPoints.SetNumberOfPoints(nx)

    for i, x in enumerate(mesh):
        vtkPoints.InsertPoint(i, *(list(x) + [0,]))

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



def write_vtu(grid, shape, data, data_name, path):
    """
    Parameters
    -----------

    grid: np.array
        Regular grid on wich the data are defined. Format is (n_points, \
n_components).

    shape: (nc, nx, ny, nz, ...)
        shape of the grid, with nc the number of spatial components, and
        (nx, ny, nz, ...) the dimensions of the grid in each direction.

    data: list of np.arrays
        The data to write, with shape (nx, n_data_components) for every arrays
        of the list.

    data_name: str
        The name (xml tag) for the data.

    path: str
        The path to the file to write the data in.
    """
    # importé et modifié depuis code_Erwan\Routines_vtk.py, fonction "ecrivtk2"

    vtkUnstructuredGrid = vtk.vtkUnstructuredGrid()

    setPoints(vtkUnstructuredGrid, grid)
    insertCells(vtkUnstructuredGrid, shape)

    npoints, nc = grid.shape
    nc2 = shape[0]
    assert nc2 == nc
    npoints_data, nc_data = data[0].shape
    assert npoints == npoints_data

    dims = shape[1:]

    for i, d in enumerate(data):
        vtkFloatArray = vtk.vtkFloatArray()
        vtkFloatArray.SetNumberOfComponents(nc_data)
        vtkFloatArray.SetNumberOfTuples(npoints)
        vtkFloatArray.SetName(data_name+str(i+1))
        for j, dd in enumerate(d):
            if nc_data == 2:
                InsertTuple = vtkFloatArray.InsertTuple2
            elif nc_data == 3:
                InsertTuple = vtkFloatArray.InsertTuple3
            InsertTuple(j, *dd)
            vtkUnstructuredGrid.GetPointData().AddArray(vtkFloatArray)
            vtkUnstructuredGrid.GetPointData().SetActiveVectors(data_name+str(i+1))

    vtkfile = vtk.vtkXMLUnstructuredGridWriter()
    vtkfile.SetInputData(vtkUnstructuredGrid)
    vtkfile.SetFileName(path)
    vtkfile.Write()


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
    grid = buildGrid(((0., 1.), (0, 2.), (0., 3.)), 0.1)
    mesh = grid2mesh(grid)
    data = np.zeros(mesh.shape)
    for i, x in enumerate(mesh):
        data[i, :] = map(lambda e: np.sin(np.pi*e), x)
    data = data.reshape((np.prod(data.shape), ))
    write_vtu(mesh, grid.shape, [data, ], 'testData', 'test.vtu')
