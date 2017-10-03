# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 17:07:22 2017

@author: afalaize
"""


from __future__ import absolute_import

from lasie_rom.config import ORDER

import vtk
from vtk.util.numpy_support import numpy_to_vtk

from xml.etree import cElementTree as ElementTree
from decimal import Decimal

import numpy as np


def setPoints(vtkUnstructuredGrid, mesh):
    nx, nc = mesh.shape

    vtkPoints = vtk.vtkPoints()
    vtkPoints.SetNumberOfPoints(nx)

    for i, x in enumerate(mesh):
        l = int(3-x.shape[0])
        xpoint = list(x) + [0,]*l
        vtkPoints.InsertPoint(i, *xpoint)

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
        nc, nx, ny = list(map(int, shape))
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


def prepare_mesh_for_3D_vtk_rendering(mesh, shape):
    """
    Add a dimension to the mesh if the mesh is 2D. 1D meshes are not supported.
    """

    # recover mesh shape
    m_shape = list(map(int, mesh.shape))

    # 1D meshes are not supported
    if m_shape[1] == 1:
        raise NotImplemented('lasie_rom package cannot write vtk for 1d data.')

    # Add a component if the mesh is 2D
    # The thickness in the 3rd dimension is ewual to the mesh size h
    elif m_shape[1] == 2:
        x0 = np.zeros((m_shape[0], 1))

        h = np.min(np.max(np.diff(mesh, axis=0), axis=0))
        x1 = np.ones((m_shape[0], 1))*h

        additional_coords = np.vstack((x0, x1))
        mesh = np.hstack((additional_coords, np.vstack((mesh, )*2)))
        shape = map(int,
                    [3, ] +    # mesh dim is now 3
                    [2, ] +    # We added two points in the 3rd dimension
                    list(shape[1:])   # Here we recover the previous shape
                    )
    # Do nothing for 3D meshes
    else:
        pass

    return mesh, list(shape)


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

    mesh, shape = prepare_mesh_for_3D_vtk_rendering(mesh, shape)

    vtkUnstructuredGrid = vtk.vtkUnstructuredGrid()

    setPoints(vtkUnstructuredGrid, mesh)
    insertCells(vtkUnstructuredGrid, shape)

    npoints, nc = mesh.shape
    nc2 = shape[0]
    assert nc2 == nc
    for k in data.keys():
        npoints_data, nc_data = data[k].shape
        if not npoints_data == npoints:
            if nc_data == 1:
                data[k] = np.vstack((data[k], data[k]))
            elif nc_data == 2:
                additional_data = np.zeros((npoints_data, 1))
                # add a column of zeros in fisrt position
                data[k] = np.hstack((additional_data, data[k]))
                # duplicate data to introduce thickness
                data[k] = np.vstack((data[k], )*2)
        npoints_data, nc_data = data[k].shape
        vtkFloatArray = numpy_to_vtk(data[k])
        vtkFloatArray.SetNumberOfComponents(nc_data)
        vtkFloatArray.SetName(k)
        vtkUnstructuredGrid.GetPointData().AddArray(vtkFloatArray)
#        vtkUnstructuredGrid.GetPointData().SetActiveVectors(k)

    vtkfile = vtk.vtkXMLUnstructuredGridWriter()
    vtkfile.SetInputData(vtkUnstructuredGrid)
    vtkfile.SetFileName(path)
    vtkfile.Write()

# TEST USING pyvtk => BUG in the connection sci.spatial.Delaunay and pyvtk.UnstructuredGrid
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
