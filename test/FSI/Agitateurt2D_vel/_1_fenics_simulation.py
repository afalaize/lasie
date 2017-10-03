"""

Fluide 2d mis en mouvement par la rotation d'un solide de forme ellipsoidale,
dans un domaine 2d de forme carree unitaire avec condition de bord de type
Dirichlet homogenes.

Parametres
0 < lambda1 < 1: Ellipse principal radius w.r.t box side length (d.u.)
0 < lambda2 < 1: Ellipse shape parameter (d.u.) (0=>1D line, 1=>circle)
0 <= lambda2 < 1: Ellipse excentricity (d.u.) (0=>centered, 1=>rotation around ellipse tip)
lambda4 = 100.    # Reynolds Number (d.u.)

The boundary conditions are chosen ase constant homogen Neumann condition


"""

from __future__ import print_function
import numpy as np
import os

from ellipse.ellipse_fnchar import build_fnchar_Dolfin_Expression
from ellipse.ellipse_levelset import build_Levelset_Dolfin_Expression
from ellipse.ellipse_tools import build_velocity_Expression

from _0_parameters import parameters
from _0_locatios import VTU_FOLDER

def fenicsSimulation():

    import dolfin as dlf
    import matplotlib.pyplot as plt

    # turn off interactive ploting to save figures without rendering them
    plt.ioff()

    if not os.path.exists(VTU_FOLDER):
        os.mkdir(VTU_FOLDER)

    # ----------------------------------------------------------------------- #
    # Maillage

    # Create mesh
    mesh = dlf.UnitSquareMesh(int(parameters['h_mesh']**-1),
                              int(parameters['h_mesh']**-1))

    # ----------------------------------------------------------------------- #
    # Espaces EF

    degre = 1

    P1 = dlf.FiniteElement("Lagrange", mesh.ufl_cell(), degre)  # vitesse
    B = dlf.FiniteElement("Bubble", mesh.ufl_cell(), degre+2)   # Bubble

    V = dlf.VectorElement(P1 + B)

    # Functional space for velocity and pressure (monolithic formulation)
    W = dlf.FunctionSpace(mesh, V*P1)

    # Functional space for characteristic function
    Wfnchar = dlf.FunctionSpace(mesh, P1)

    # Functional space for levelset function
    Wlevelset = dlf.FunctionSpace(mesh, P1)

    # Functional space for velocity
    Wvelocity = dlf.FunctionSpace(mesh, V)

    # ----------------------------------------------------------------------- #
    # Tenseur des contraintes

    def sigma(v, w):
        """
        Tenseur des contraintes de cauchy
        """
        return (2.0*parameters['nu']*0.5*(dlf.grad(v) +
                dlf.grad(v).T) -
                w*dlf.Identity(len(v)))

    # ----------------------------------------------------------------------- #
    # Conditions limites

    class ParoiesBoundary(dlf.SubDomain):
            def inside(self, x, on_boundary):
                return (bool(x[1] < dlf.DOLFIN_EPS and on_boundary) or
                        bool(x[1] > 1 - dlf.DOLFIN_EPS and on_boundary) or
                        bool(x[0] < dlf.DOLFIN_EPS and on_boundary) or
                        bool(x[0] > 1 - dlf.DOLFIN_EPS and on_boundary))

    # 0 dirichlet boundary condition for the velocity
#    paroies = dlf.project(dlf.Constant((0)), W.sub(0).sub(1).collapse())
#    bc = dlf.DirichletBC(W.sub(0).sub(1), paroies, ParoiesBoundary())

    # 0 dirichlet boundary condition for the pressure
    paroies = dlf.project(dlf.Constant((0)), W.sub(1).collapse())
    bc = dlf.DirichletBC(W.sub(1), paroies, ParoiesBoundary())

    # Collect boundary conditions
    bcs = [bc, ]

    # ----------------------------------------------------------------------- #
    # Def fonction Characteristique
    fnchar_expr = build_fnchar_Dolfin_Expression(parameters['ell_center'],
                                          parameters['ell_radius'],
                                          parameters['rot_center'],
                                          parameters['eps_tanh'])

    # Def velocity
    velocity_expr = build_velocity_Expression(parameters['rot_center'],
                                              parameters['angular_vel'])

    # Def levelset
    levelset_expr = build_Levelset_Dolfin_Expression(parameters['ell_center'],
                                                     parameters['ell_radius'],
                                                     parameters['rot_center'])

    # ----------------------------------------------------------------------- #
    # Trials and tests functions
    u, p = dlf.TrialFunctions(W)
    v, q = dlf.TestFunctions(W)

    fnchar = dlf.Function(Wfnchar)
    fnchar.assign(fnchar_expr(parameters['theta_init'], degre))

    levelset = dlf.Function(Wlevelset)
    levelset.assign(levelset_expr(parameters['theta_init'], degre))

    # Functions
    w0 = dlf.Function(W)
    u0, p0 = w0.split(True)
    w1 = dlf.Function(W)

    # velocity in the solid
    us = dlf.Function(Wvelocity)

    # inital velocity in the whole domain
    uinit = dlf.Constant((0, 0))

    # inital velocity in the solid
    us = dlf.project(uinit, W.sub(0).collapse())

    # Source term
    f = dlf.Constant((0, 0))

    # ----------------------------------------------------------------------- #
    # Monolithic formulation
    a1 = ((parameters['rho']/parameters['dt'])*dlf.inner(u, v)*dlf.dx +
          parameters['rho']*dlf.inner(dlf.grad(u)*u0, v)*dlf.dx +
          parameters['nu']*dlf.inner(dlf.grad(u), dlf.grad(v))*dlf.dx -
          dlf.div(v)*p*dlf.dx +
          q*dlf.div(u)*dlf.dx -
          (1./parameters['pen'])*dlf.inner(fnchar*u, v)*dlf.dx)

    L1 = ((parameters['rho']/parameters['dt'])*dlf.inner(u0, v)*dlf.dx +
          dlf.inner(f, v)*dlf.dx -
          (1./parameters['pen'])*dlf.inner(fnchar*us, v)*dlf.dx)

    # ----------------------------------------------------------------------- #
    # fichiers de sauvergarde
    u_file = dlf.File(os.path.join(VTU_FOLDER, "velocity.pvd"))
    p_file = dlf.File(os.path.join(VTU_FOLDER, "pressure.pvd"))
    fnchar_file = dlf.File(os.path.join(VTU_FOLDER, "fnchar.pvd"))
    levelset_file = dlf.File(os.path.join(VTU_FOLDER, "levelset.pvd"))

#    timeseries_u = dlf.TimeSeries(os.path.join(resultsFolderName,
#                                               "velocity_serie"))
#    timeseries_p = dlf.TimeSeries(os.path.join(resultsFolderName,
#                                               "pressure_serie"))
#    timeseries_fnchar = dlf.TimeSeries(os.path.join(resultsFolderName,
#                                                    "fnchar_serie"))

    # ----------------------------------------------------------------------- #
    # Create progress bar
    progress = dlf.Progress('Time-stepping')
    dlf.set_log_level(dlf.WARNING)

    energy = list()

    # Init simulation
    nb = 0
    t = 0

    # ----------------------------------------------------------------------- #
    # Solve inital state
    dlf.solve(a1 == L1, w1, bcs)
    u1, p1 = w1.split(True)

    # Save data name
    parameters.update({'dataname': u1.name()})

    # Save inital state
    u_file << u1
    p_file << p1
    fnchar_file << fnchar
    levelset_file << levelset
    print("Courant number =",
          u0.vector().max()*parameters['dt']/mesh.hmin(),
          " t =", t)

    # ----------------------------------------------------------------------- #
    # Iterate
    while t < parameters['T']:

        theta = parameters['angular_vel']*t

        fnchar.assign(fnchar_expr(theta, degre))
        levelset.assign(levelset_expr(theta, degre))

        uinit = velocity_expr(degre)
        us.assign(dlf.project(uinit, W.sub(0).collapse()))

        dlf.solve(a1 == L1, w1, bcs)
        u1, p1 = w1.split(True)

        energy.append((t, dlf.assemble(dlf.dot(u1, u1)*dlf.dx)))

        # Save to file
        nb += 1
        if nb == parameters['nb_export']:
            u_file << u1
            p_file << p1
            fnchar_file << fnchar
            levelset_file << levelset
            np.savetxt(os.path.join(VTU_FOLDER,
                                    "energy.txt"), energy)
            plt.close('all')
            plt.plot([e[0] for e in energy],
                     [e[1] for e in energy])
            plt.savefig(os.path.join(VTU_FOLDER,
                                     "energy.png"))
#            timeseries_u.store(u1.vector(), t+parameters['dt'])
#            timeseries_p.store(p1.vector(), t+parameters['dt'])
#            timeseries_fnchar.store(fnchar.vector(), t+parameters['dt'])
            nb = 0

        # Move to next time step
        u0.assign(u1)
        t += parameters['dt']

        # Update progress bar
        progress.update(t / parameters['T'])
        courant = u0.vector().max()*parameters['dt']/mesh.hmin()
        print("Courant number =", courant, " t =", t)

    # ----------------------------------------------------------------------- #

    return parameters, VTU_FOLDER


if __name__ == '__main__':

    fenicsSimulation()
