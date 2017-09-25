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
import dolfin as dlf
import numpy as np
import os

from .ellipse_fnchar import (build_fnchar_Expression,
                             build_velocity_Expression,
                             build_Levelset_Expression)

import matplotlib.pyplot as plt


# turn off interactive ploting to save figures without rendering them
plt.ioff()

# --------------------------------------------------------------------------- #
# Parameters

def build_parameters(lambda1, lambda2, lambda3, lambda4, eps_tanh):
    rot_center = np.array((0.5, 0.5))
    ell_center = rot_center + np.array((lambda3*lambda1, 0))
    parameters = {'rho': 1.,         # masse volumique du fluide
                  'rho_delta': 0,   # rho_solide - rho_fluide
                  'dt': 0.005,       # pas de temps
                  'T': 30.,          # temps final
                  'nb_export': 2,    # number of time-steps between each saving
                  'nu': 1./lambda4,  # visco dynamique du fluide, nu = 1/Re
                  'nu_delta': 0,    # nu_solide - nu_fluide
                  'pen': 1e-9,       # Coefficient de penalisation volumique (~0)
                  'rot_center': rot_center,  # Center of rotation of ellipse
                  'ell_center': ell_center,  # Center of ellipse
                  'ell_radius': (lambda1, lambda2*lambda1),  # Ellipse Radii
                  'angular_vel': (2*(lambda1)**2)**-1,   # angular velocity (rad/s)
                  'theta_init': 0.,  # initial ellipse angle w.r.t 1st axe
                  'lambda': (lambda1, lambda2, lambda3, lambda4),  # parameters
                  'h_mesh': 0.008,  # mesh size w.r.t 1m (d.u.)
                  'eps_tanh': eps_tanh
                  }
    return parameters


# N.B
# Refernce velocity is v = r1*omega
# Reference length is d = 2*r1
# reference density is rho = 1
# so that Re = v*d*rho/nu is equivalent to nu = 1/Re


# --------------------------------------------------------------------------- #
# Set results folder

def build_resultsFolderName(parameters):
    resultsFolderName = "results"
    resultsFolderName += "_radius={0}".format(parameters['lambda'][0])
    resultsFolderName += "_shape={0}".format(parameters['lambda'][1])
    resultsFolderName += "_excentr={0}".format(parameters['lambda'][2])
    resultsFolderName += "_Re={0}".format(parameters['lambda'][3])
    resultsFolderName += "_eps_tanh={0}".format(parameters['eps_tanh'])
    resultsFolderName += "_mesh={0}X{0}".format(int(parameters['h_mesh']**-1))
    return resultsFolderName

# --------------------------------------------------------------------------- #
# Main function


def fenicsSimulation(lambda1, lambda2, lambda3, lambda4, eps_tanh):

    parameters = build_parameters(lambda1, lambda2, lambda3, lambda4, eps_tanh)
    resultsFolderName = build_resultsFolderName(parameters)

    if not os.path.exists(resultsFolderName):
        os.mkdir(resultsFolderName)

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
    fnchar_expr = build_fnchar_Expression(parameters['ell_center'],
                                          parameters['ell_radius'],
                                          parameters['rot_center'],
                                          parameters['eps_tanh'])

    # Def velocity
    velocity_expr = build_velocity_Expression(parameters['rot_center'],
                                              parameters['angular_vel'])

    # Def levelset
    levelset_expr = build_Levelset_Expression(parameters['ell_center'],
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
    u_file = dlf.File(os.path.join(resultsFolderName, "velocity.pvd"))
    p_file = dlf.File(os.path.join(resultsFolderName, "pressure.pvd"))
    fnchar_file = dlf.File(os.path.join(resultsFolderName, "fnchar.pvd"))
    levelset_file = dlf.File(os.path.join(resultsFolderName, "levelset.pvd"))

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
            np.savetxt(os.path.join(resultsFolderName,
                                    "energy.txt"), energy)
            plt.close('all')
            plt.plot([e[0] for e in energy],
                     [e[1] for e in energy])
            plt.savefig(os.path.join(resultsFolderName,
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
    # Save parameters
    def save_parameters():
        textFileName = "parameters.txt"
        with open(os.path.join(resultsFolderName, textFileName),
                  mode='w') as f:
            f.write("{\n")
            for k in parameters.keys():
                f.write("{0}': {1},\n".format(k, parameters[k]))
            f.write("{\n")

    print(parameters)
    save_parameters()

    # ----------------------------------------------------------------------- #

    return parameters, resultsFolderName


if __name__ == '__main__':
    lambda1 = 0.375/1.  # Ellipse principal radius/box side length (d.u.)
    lambda2 = .2        # Ellipse shape parameter (d.u.)
    lambda3 = 0.        # Ellipse excentricity parameter (d.u.)
    lambda4 = 100.      # Reynolds Number (d.u.)

    parameters, resultsFolderName = fenicsSimulation(lambda1,
                                                     lambda2,
                                                     lambda3,
                                                     lambda4)
