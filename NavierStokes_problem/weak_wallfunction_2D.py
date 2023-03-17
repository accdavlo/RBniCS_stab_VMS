import time as time_module
from dolfin import *
from ufl.geometry import *
import numpy as np
from ufl import Jacobian
from sys import argv
from dolfin.cpp.mesh import *
from mshr import *
import ufl
from wurlitzer import pipes
from numpy import random
from fenicstools import interpolate_nonmatching_mesh, interpolate_nonmatching_mesh_any
from utils_PETSc_scipy import PETSc2scipy, inverse_lumping_with_zeros
import pickle

import os
import petsc4py
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import bisect
import matplotlib.pyplot as plt

from rbnics.backends import BasisFunctionsMatrix, ProperOrthogonalDecomposition, max as rbnics_max, abs as rbnics_abs
from rbnics.backends.dolfin.snapshots_matrix import SnapshotsMatrix
from rbnics.utils.io import TextIO, ExportableList


from multiprocessing import Pool


from problems import Problem


giovanni = True
boundary_tag =  "spalding" # "strong"#"spalding"# "spalding"# 

parameters["linear_algebra_backend"] = "PETSc"
args = "--petsc.snes_linesearch_monitor --petsc.snes_linesearch_type bt"
parameters.parse(argv = argv[0:1] + args.split())

degree = 2


Nx = 100
problem_name = "cylinder_turb"#"cylinder"#"lid-driven_cavity"

CFL = 0.5
Nt_max = 10000
dT=Constant(1.e-5)







if problem_name=="cylinder_turb":
    # Rey = 5 10^4 
    T = 3.0    #5.0    #0.5 # 0.01
    dtplot =0.02
    nu_val =  0.000002 #0.000002 #1. # 0.001
    u_top_val = 1. #1.  #100.  # 500 
    param_range = [ [0.5,5.0],# u_in
                [0.000002,0.00002] # nu
                ]
elif problem_name=="cylinder":
    # Rey = 100 
    T = 3.0    #5.0    #0.5 # 0.01
    dtplot =0.02
    nu_val =  0.001 #0.000002 #1. # 0.001
    u_top_val = 1. #1.  #100.  # 500

    param_range = [ [0.5,5.0],# u_in
                [0.0002,0.005] # nu
                ]



def generate_inner_products(V, V0):

    trial = TrialFunction(V)
    test = TestFunction(V)

    trial_v0 = TrialFunction(V0)
    test_v0 = TestFunction(V0)

    u, p = split(trial)
    v, q = split(test)
    # Inner products for orthonormalization
    X = dict()

    # Definition of the scalar product in the space V1 (used for pressure supremizers as well)
    
    X["u"] = assemble(inner(grad(u), grad(v)) * dx + inner(u,v)*dx)
    X["p"] = assemble(inner(p, q) * dx)
    X["tau"] = assemble(inner(trial_v0,test_v0)*dx)

    return X

def project_onto_RB(RB, u_FOM, tau_FOM=None ):
    uRB = dict()
    for comp in ("u","p"):
        NRB  = len(RB[comp])
        uRB[comp] = np.zeros(NRB)
        for j in range(NRB):
            uRB[comp][j] = (X_inner[comp]*RB[comp][j]).inner(u_FOM.vector())
    if tau_FOM is not None:
        comp = "tau"
        uRB[comp] = np.zeros(NRB)
        for j in range(NRB):
            uRB[comp][j] = (X_inner[comp]*RB[comp][j]).inner(tau_FOM.vector())
    return uRB 

def reconstruct_RB(RB, u_RB, u_hat, tau_hat=None):
    u_tmp = dict()
    for comp in ("u","p"):
        NRB = len(RB[comp])  
        u_tmp[comp] = Function(W)
        u_tmp[comp].vector()[:] = sum([u_RB[comp][j]*RB[comp][j].vector() for j in range(NRB)])
    assign(u_hat, [u_tmp["u"].sub(0), u_tmp["p"].sub(1)])
    if tau_hat is not None:
        comp = "tau"
        NRB = len(RB[comp])  
        tau_hat.vector()[:] = sum([u_RB[comp][j]*RB[comp][j].vector() for j in range(NRB)])

# POD_variables = dict() 
# ProperOrthogonalDecomposition(V, X_rho)

C_I = Constant(36.0)
C_t = Constant(4.0)
C_b_I_value = 4.0
C_b_I = Constant(C_b_I_value)

physical_problem = Problem(problem_name, Nx)
mesh = physical_problem.mesh
space_dim = physical_problem.space_dim

if giovanni:
    out_folder = "out_"+problem_name+"_test_"+boundary_tag+"_gio_P"+str(degree)+"_N_"+str(Nx)
else:
    out_folder = "out_"+problem_name+"_test_"+boundary_tag+"_hughes_P"+str(degree)+"_N_"+str(Nx)

try:
    os.mkdir(out_folder)
except:
    print("Folder %s already exists"%(out_folder))

# Set parameter values
nu = Constant(nu_val)
f = Constant((0., 0.))
u_top = Constant(u_top_val)

Re_val = physical_problem.get_reynolds(u_top_val,nu_val)

print("Reynolds Number = %e"%Re_val)

q_degree = 5
dx = dx(metadata={'quadrature_degree': q_degree})


"""### Function spaces"""
#pbc = PeriodicBoundary()
V_element = VectorElement("Lagrange", mesh.ufl_cell(), degree)
Q_element = FiniteElement("Lagrange", mesh.ufl_cell(), degree)
W_element = MixedElement(V_element, Q_element) 
W = FunctionSpace(mesh, W_element)#, constrained_domain=PeriodicBoundary())
print(W.dim())
physical_problem.define_boundaries(W)
subdomains = physical_problem.subdomains
boundaries = physical_problem.boundaries
bmesh      = physical_problem.bmesh

# Save to xml file
File(out_folder+"/%s.xml"%(physical_problem.mesh_name)) << physical_problem.mesh
File(out_folder+"/%s_physical_region.xml"%(physical_problem.mesh_name)) << physical_problem.subdomains
File(out_folder+"/%s_facet_region.xml"%(physical_problem.mesh_name)) << physical_problem.boundaries

# Save to pvd file for visualization
xdmf = XDMFFile(mesh.mpi_comm(), out_folder+"/Mesh.xdmf")
xdmf.write(mesh)
xdmf = XDMFFile(mesh.mpi_comm(), out_folder+"/Physical_region.xdmf")
subdomains.rename("subdomains", "subdomains")
xdmf.write(physical_problem.subdomains)
xdmf = XDMFFile(mesh.mpi_comm(), out_folder+"/Facet_region.xdmf")
boundaries.rename("boundaries", "boundaries")
xdmf.write(boundaries)

h = function.specialfunctions.CellDiameter(physical_problem.mesh)
hmin = physical_problem.mesh.hmin()


dg0_element = FiniteElement("DG", mesh.ufl_cell(),0)
V0 = FunctionSpace(mesh, dg0_element)
dg0_bound_element = FiniteElement("DG", bmesh.ufl_cell(),0)
V0_bound = FunctionSpace(bmesh, dg0_bound_element)

z_test_V0 = TestFunction(V0)


def get_DG_inverse_matrix(VDG):
    """
    Building the matrix that returns the inverse of the mass matrix only on boundary cells 
    Piecewise constant polynomials considered
    Input: VDG (FunctionalSpace) of discontinuous piecewise constant polynomials
    """
    z_trial = TrialFunction(VDG)
    z_test = TestFunction(VDG)
    lhs = inner(z_trial,z_test)*ds
    LHS = PETSc2scipy(assemble(lhs))
    LHS_inv=inverse_lumping_with_zeros(LHS)
    return LHS_inv


def get_DG_matrix(VDG):
    """
    Building the matrix that returns the inverse of the mass matrix only on boundary cells 
    Piecewise constant polynomials considered
    Input: VDG (FunctionalSpace) of discontinuous piecewise constant polynomials
    """
    z_trial = TrialFunction(VDG)
    z_test = TestFunction(VDG)
    lhs = inner(z_trial,z_test)*ds
    LHS = assemble(lhs)
    return LHS

def solve_boundary_problem(f, VDG, LHS):
    """
    Extend function f which is evaluatable only at boundaries onto the whole domain
    the result lives in VDG which must be piecewise constant polynomails space.
    """
    rhs = inner(f,z_test_V0)*ds
    RHS = assemble(rhs)
    x = Function(VDG)
    solve(LHS, x.vector(), RHS)
    return x



def solve_spalding_law(u,hb):
    """Finding the solution to Spalding's Law at boundary cells
    Interpolating all functions onto the piecewise discontinuous polynomials on the boundary mesh 
    (the restriction to the boundary of the cell)
    In each cell finds the solution to Spalding's law through bisection 
    (gradient is very steep in a very small region, the function is monotone increasing)
    """
    # computing the norm of u and interpolating hb on boundaries
    u_norm = interpolate_nonmatching_mesh(project(sqrt(dot(u,u)),V0),V0_bound)
    hb_bound = interpolate_nonmatching_mesh(solve_boundary_problem(hb,V0,DG_matrix) ,V0_bound)

    # solution of Spalding's law
    tau_B_bound = Function(V0_bound)

    # Loop over boundary cells
    for i in map_cell_dof_bound:
        # solving Spalding's law through bisection in a cell
        tau_B_bound.vector()[i] = bisect(spalding_func, 1e-14, 1e4, xtol=1e-12, \
            args=(hb_bound.vector()[i], u_norm.vector()[i], \
                Chi_Spalding_value, B_Spalding_value, C_b_I_value, nu_val ) )
    # Extending the solution vector to the whole mesh
    tau_B_ext = Function(V0)
    tau_B_ext.vector()[interp_V0_V0_bound.row] = tau_B_bound.vector()[interp_V0_V0_bound.col]

    return tau_B_ext


def bisection_wrapper(inputs):
    (hb_dof,u_norm_dof) = inputs
    for tau_guess in [1.,10.,1.e2,1.e3,1.e4,1.e5]:
        if spalding_func(tau_guess,hb_dof, u_norm_dof, \
                Chi_Spalding_value, B_Spalding_value, C_b_I_value, nu_val ) >=0:
            return bisect(spalding_func, 1e-14, tau_guess, xtol=1e-12, \
                args=(hb_dof, u_norm_dof, \
                Chi_Spalding_value, B_Spalding_value, C_b_I_value, nu_val ) )

def solve_spalding_law_inout(u,hb, tau_B_ext):
    """Finding the solution to Spalding's Law at boundary cells
    Interpolating all functions onto the piecewise discontinuous polynomials on the boundary mesh 
    (the restriction to the boundary of the cell)
    In each cell finds the solution to Spalding's law through bisection 
    (gradient is very steep in a very small region, the function is monotone increasing)
    """
    # computing the norm of u and interpolating hb on boundaries
    u_norm = interpolate_nonmatching_mesh(project(sqrt(dot(u,u)),V0),V0_bound)
    hb_bound = interpolate_nonmatching_mesh(solve_boundary_problem(hb,V0,DG_matrix) ,V0_bound)

    # solution of Spalding's law
    tau_B_bound = Function(V0_bound)

    inputs = [(hb_bound.vector()[i], u_norm.vector()[i]) for i in map_cell_dof_bound]
    outputs = [bisection_wrapper(inp) for inp in inputs]
    
    tau_B_bound.vector()[map_cell_dof_bound] = outputs[:]

    # Extending the solution vector to the whole mesh
    tau_B_ext.vector()[interp_V0_V0_bound.row] = tau_B_bound.vector()[interp_V0_V0_bound.col]

    return tau_B_ext




def spalding_func(tau, hb, unorm, Chi, B, C_b_I, nu):
    """ Spalding's law
    SL(tau) = y^+-u^+-e^{-\Chi B}(e^{\Chi u^+} -1 -\Chi u^+ -(\Chi u^+)^2/2-(\Chi u^+)^3/6)
    Where it holds tau_B = (u^*)^2/||u||, y^+=yu^*/\nu , u^+=||u||/u^*, y = h_b/C_b^I
    Hence, one can write SL(tau) as a function of tau, hb, ||u|| and constants
    Details: https://www.doi.org/10.1016/j.cma.2007.06.026
    Inputs:
    tau unknown parameter
    h_b = 2*sqrt(dot(n,G*n)) mesh dependent parameter
    ||u|| function of the solution velocity u 
    Chi, B, C_b^I, nu constants
    """
    if unorm <1e-16:
        return 0.
    y = hb/C_b_I 
    us = np.sqrt(tau*unorm) # u^*
    yp = y*us/nu            # y^+
    up = unorm/us           # u^+
    Chiup = Chi*up          # \Chi*u^+
    sp = yp-up-np.exp(-Chi*B)*(np.exp(Chiup)-1.-Chiup-Chiup**2/2.-Chiup**3/6.) # Spalding's law
    return sp
    
def spalding_der(tau, hb, unorm, Chi, B, C_b_I, nu):
    """
    Derivative of Spalding's law in tau
    """
    y = hb/C_b_I
    us = np.sqrt(tau*unorm)
    yp = y*us/nu
    up = unorm/us
    Chiup = Chi*up
    sp_der = y*np.sqrt(unorm/tau)/2./nu+np.sqrt(unorm/tau**3)/2.*(\
        1+ np.exp(-Chi*B)*Chi*(\
            np.exp(Chiup)-1.-Chiup-Chiup**2/2.\
        )\
    )
    return sp_der


def sigmaVisc(u,nu):
    """
    The viscous part of the Cauchy stress, in terms of velocity ``u`` and
    dynamic viscosity ``nu``.
    """
    return 2.0*nu*sym(grad(u))

def sigma(u,p,nu):
    """
    The fluid Cauchy stress, in terms of velocity ``u``, pressure ``p``, 
    and dynamic viscosity ``nu``.
    """
    return sigmaVisc(u,nu) - p*Identity(ufl.shape(u)[0])

def materialTimeDerivative(u,u_t=None,f=None):
    """
    The fluid material time derivative, in terms of the velocity ``u``, 
    the partial time derivative ``u_t`` (which may be omitted for steady
    problems), and body force per unit mass, ``f``.
    """
    DuDt = dot(u,nabla_grad(u))
    if(u_t != None):
        DuDt += u_t
    if(f != None):
        DuDt -= f
    return DuDt

def meshMetric(mesh):
    """
    Extract mesh size tensor from a given ``mesh``.
    This returns the physical element metric tensor, ``G`` as a 
    UFL object.
    """
    dx_dxiHat = 0.5*ufl.Jacobian(mesh)
    dxiHat_dx = inv(dx_dxiHat)
    G = dxiHat_dx.T*dxiHat_dx
    return G

def stableNeumannBC(traction,u,v,n,g=None,ds=ds,gamma=Constant(1.0)):
    """
    This function returns the boundary contribution of a stable Neumann BC
    corresponding to a boundary ``traction`` when the velocity ``u`` (with 
    corresponding test function ``v``) is flowing out of the domain, 
    as determined by comparison with the outward-pointing normal, ``n``.  
    The optional velocity ``g`` can be used to offset the boundary velocity,
    as when this term is used to obtain a(n inflow-
    stabilized) consistent traction for weak enforcement of Dirichlet BCs.  
    The paramter ``gamma`` can optionally be used to scale the
    inflow term.  The BC is integrated using the optionally-specified 
    boundary measure ``ds``, which defaults to the entire boundary.
    NOTE: The sign convention here assumes that the return value is 
    ADDED to the residual given by ``interiorResidual``.
    NOTE: The boundary traction enforced differs from ``traction`` if 
    ``gamma`` is nonzero.  A pure traction BC is not generally stable,
    which is why the default ``gamma`` is one.  See
    https://www.oden.utexas.edu/media/reports/2004/0431.pdf
    for theory in the advection--diffusion model problem, and 
    https://doi.org/10.1007/s00466-011-0599-0
    for discussion in the context of Navier--Stokes.  
    """
    if(g==None):
        u_minus_g = u
    else:
        u_minus_g = u-g
    return -(inner(traction,v)
            + gamma*ufl.Min(inner(u,n),Constant(0.0))
            *inner(u_minus_g,v))*ds(physical_problem.walls_ID)

def weakDirichletBC(u,p,u_prev,v,q,g,nu,mesh,ds=ds, tau_pen = None, G=None,
                    sym=True, spalding=True, C_pen=Constant(1e3),
                    overPenalize=False):
    """
    This returns the variational form corresponding to a weakly-enforced 
    velocity Dirichlet BC, with data ``g``, on the boundary measure
    given by ``ds``, defaulting to the full boundary of the domain given by
    ``mesh``.  It takes as parameters an unknown velocity, ``u``, 
    unknown pressure ``p``, corresponding test functions ``v`` and ``q``, 
    mass density ``rho``, and viscosity ``nu``.  Optionally, the 
    non-symmetric variant can be used by overriding ``sym``.  ``C_pen`` is
    a dimensionless scaling factor on the penalty term.  The penalty term
    is omitted if ``not sym``, unless ``overPenalize`` is 
    optionally set to ``True``.  The argument ``G`` can optionally be given 
    a non-``None`` value, to use an alternate mesh size tensor.  If left 
    as ``None``, it will be set to the output of ``meshMetric(mesh)``.
    NOTE: The sign convention here assumes that the return value is 
    ADDED to the residual given by ``interiorResidual``.
    For additional information on the theory, see
    https://doi.org/10.1016/j.compfluid.2005.07.012
    """
    n = FacetNormal(mesh)
    sgn = 1.0
    if(not sym):
        sgn = -1.0
    if G == None:
        G =  meshMetric(mesh) # $\sim h^{-2}$
    traction = sigma(u,p,nu)*n
    consistencyTerm = stableNeumannBC(traction,u,v,n,g=g,ds=ds)
    # Note sign of ``q``, negative for stability, regardless of ``sym``.    
    adjointConsistency = -sgn*dot(sigma(v,-sgn*q,nu)*n,u-g)*ds(physical_problem.walls_ID)
    # Only term we need to change
    hb = 2/sqrt(dot(n,G*n))
    
    # Weak penalty coefficient or Spalding's law coefficient
    if tau_pen is None:
        if spalding:
            tau_pen  = solve_spalding_law(u_prev,hb)
        else:
            tau_pen =  C_pen*nu/hb 

    penalty = tau_pen*dot((u-g),v)*ds(physical_problem.walls_ID)
    retval = consistencyTerm + adjointConsistency
    if(overPenalize or sym):
        retval += penalty
        print("Passing here")
    return retval

def weakHughesBC(u,p,u_prev,v,q,g,nu,mesh,ds=ds, tau_pen = None,G=None,
                    symmetric=True, spalding=True, gamma=Constant(1.0),
                    C_pen=Constant(1e3), overPenalize=False):
    """
    This returns the variational form corresponding to a weakly-enforced 
    velocity Dirichlet BC, with data ``g``, on the boundary measure
    given by ``ds``, defaulting to the full boundary of the domain given by
    ``mesh``.  It takes as parameters an unknown velocity, ``u``, 
    unknown pressure ``p``, corresponding test functions ``v`` and ``q``, 
    mass density ``rho``, and viscosity ``nu``.  Optionally, the 
    non-symmetric variant can be used by overriding ``symmetric``.  ``C_pen`` is
    a dimensionless scaling factor on the penalty term.  The penalty term
    is omitted if ``not sym``, unless ``overPenalize`` is 
    optionally set to ``True``.  The argument ``G`` can optionally be given 
    a non-``None`` value, to use an alternate mesh size tensor.  If left 
    as ``None``, it will be set to the output of ``meshMetric(mesh)``.
    NOTE: The sign convention here assumes that the return value is 
    ADDED to the residual given by ``interiorResidual``.
    For additional information on the theory, see
    https://doi.org/10.1016/j.compfluid.2005.07.012
    """
    n = FacetNormal(mesh)
    sgn = 1.0
    if(not symmetric):
        sgn = -1.0
    if G == None:
        G = meshMetric(mesh) # $\sim h^{-2}$
    consistencyTerm = -sgn*inner(2*nu*sym(grad(u))*n,v)*ds(physical_problem.walls_ID)
    adjointConsistency = -sgn*inner(2*gamma*nu*sym(grad(v))*n,u-g)*ds(physical_problem.walls_ID)

    # Only term we need to change
    hb = 2/sqrt(dot(n,G*n))
    
    # Weak penalty coefficient or Spalding's law coefficient
    if tau_pen is None:
        if spalding:
            tau_pen  = solve_spalding_law(u_prev,hb)
        else:
            tau_pen =  C_pen*nu/hb
    
    penalty = dot(tau_pen*(u-g),v)*ds(physical_problem.walls_ID)
    retval = consistencyTerm + adjointConsistency
    if(overPenalize or symmetric):
        retval += penalty
        print("Passing here")
    return retval


def strongResidual(u,p,nu,u_t=None,f=None):
    """
    The momentum and continuity residuals, as a tuple, of the strong PDE,
    system, in terms of velocity ``u``, pressure ``p``, dynamic viscosity
    ``nu``, mass density ``rho``, and, optionally, the partial time derivative
    of velocity, ``u_t``, and a body force per unit mass, ``f``.  
    """
    DuDt = materialTimeDerivative(u,u_t,f)
    i,j = ufl.indices(2)
    r_M = DuDt - as_tensor(grad(sigma(u,p,nu))[i,j,j],(i,))
    r_C = div(u)
    return r_M, r_C


def strongResidualHughes(u,p,nu,u_t,f=None):
    """
    The momentum and continuity residuals, as a tuple, of the strong PDE,
    system, in terms of velocity ``u``, pressure ``p``, dynamic viscosity
    ``nu``, mass density ``rho``, and, optionally, the partial time derivative
    of velocity, ``u_t``, and a body force per unit mass, ``f``.  
    """
    if f is None:
        f = Constant((0.,0.))
    r_M = u_t  + ufl.nabla_div(outer(u,u)) +grad(p) \
        - ufl.nabla_div(2*nu*sym(grad(u))) -f
    r_C = div(u)
    return r_M, r_C

def stabilizationParameters(u,nu,G,C_I,C_t,Dt=None,scale=Constant(1.0)):
    """
    Compute SUPS and LSIC/grad-divx stabilization parameters (returned as a
    tuple, in that order).  Input parameters are the velocity ``u``,  the mesh
    velocity ``vhat``, the kinematic viscosity ``nu``, the mesh metric ``G``,
    order-one constants ``C_I`` and ``C_t``, a time step ``Dt`` (which may be
    omitted for steady problems), and a scaling factor that defaults to unity.  
    """
    # The additional epsilon is needed for zero-velocity robustness
    # in the inviscid limit.
    denom2 = inner(u,G*u) + C_I*nu*nu*inner(G,G) + DOLFIN_EPS
    if(Dt != None):
        denom2 += C_t/Dt**2
    tau_M = scale/sqrt(denom2)
    tau_C = 1.0/(tau_M*tr(G))
    return tau_M, tau_C



DG_matrix     = get_DG_matrix(V0)
DG_matrix_inv = get_DG_inverse_matrix(V0)
dofmapV0 = V0.dofmap()

dofmapV0_bound = V0_bound.dofmap()


def interp_domain_to_bound_matrix():



    row= []
    column = []
    data = []

    for cell in cells(mesh):
        for myFacet in facets(cell):
            if myFacet.exterior():
                mid_facet = myFacet.midpoint().array()
                for cell_b in cells(bmesh):
                    if np.linalg.norm(mid_facet-cell_b.midpoint().array(),np.inf)<1e-10:
                        row.append(dofmapV0.dofs()[cell.index()])
                        column.append(dofmapV0_bound.dofs()[cell_b.index()])
                        data.append(1.)
    
    interp_mat = sparse.coo_matrix((data, (row, column)), shape=(mesh.num_cells(),bmesh.num_cells()))

    return interp_mat

interp_V0_V0_bound = interp_domain_to_bound_matrix()


tau_penalty = Function(V0)

Chi_Spalding_value = 0.4
Chi_Spalding = Constant(Chi_Spalding_value)
B_Spalding_value = 5.5
B_Spalding = Constant(B_Spalding_value)
C_pen_value = 1e3
C_pen=Constant(C_pen_value)

n = FacetNormal(mesh)
G = meshMetric(mesh) # proportional to h^{-2}
hb = 2./sqrt(dot(n,G*n)) # proportional to h in normal direction



map_cell_dof_bound = [] #np.zeros(len(cells(bmesh)),dtype=np.int64)
# Loop over boundary cells
for cell in cells(bmesh):
    cell_index = cell.index() 
    mid_pt = cell.midpoint()
    map_cell_dof_bound.append(dofmapV0_bound.dofs()[cell_index])

map_cell_dof_bound = np.array(map_cell_dof_bound, dtype=np.int64)


"""### Test and trial functions (for the increment)"""
vq                 = TestFunction(W) # Test function in the mixed space
delta_up           = TrialFunction(W) # Trial function in the mixed space (XXX Note: for the increment!)
(delta_u, delta_p) = split(delta_up) # Function in each subspace to write the functional  (XXX Note: for the increment!)
(v, q)             = split(      vq) # Test function in each subspace to write the functional

"""### <font color="red">Solution (which will be obtained starting from the increment at the current time)</font>"""
up = Function(W)
(u, p) = split(up)

"""### <font color="red">Solution at the previous time</font>"""
up_prev = Function(W)
(u_prev, _) = split(up_prev)
up_bc = Function(W)
(u_bc, _) = split(up_bc)
u_bc = Constant((0.,0.))

X_inner = generate_inner_products(W,V0)

nu = Constant(nu_val)

u_top = Constant(u_top_val)

# ## BCs for weak boundaries and initial condition perturbation
# ## BCs for weak boundaries and initial condition perturbation
# u_pert = Expression(('u_max*0.01*sin(25*x[1]*2*pi/delta_z)',\
#     '1e-3*u_max*sin(41*x[0]*2*pi/delta_x)'), \
#     degree=4, u_max=u_max, delta_x=delta_x, delta_z=delta_z)

# #u_0 = interpolate(u_in, W.sub(0).collapse())
# u_0 = project(u_in+u_pert, W.sub(0).collapse())
# #u_0 = interpolate(u_in, W.sub(0).collapse())+project(u_pert, W.sub(0).collapse())

# u_bc = interpolate(u_in, W.sub(0).collapse())

# p_0 = interpolate(p_in, W.sub(1).collapse())

u_t = (u - u_prev)/dT

# Preparation of Variational forms.
#K = JacobianInverse(mesh)
#G = K.T*K

#gg = (K[0,0] + K[1,0] + K[2,0])**2 + (K[0,1] + K[1,1] + K[2,1])**2 + (K[0,2] + K[1,2] + K[2,2])**2
#rm = (u-u_prev)/dt + grad(p) + grad(u)*u - nu*(u.dx(0).dx(0) + u.dx(1).dx(1) + u.dx(2).dx(2)) - f

DuDt = materialTimeDerivative(u,u_t,f)
i,j = ufl.indices(2)
rm = DuDt - as_tensor(grad(sigma(u,p,nu))[i,j,j],(i,)) # if u \in P1 last term is zero (second derivative)! 
rc = div(u)


denom2 = inner(u,G*u) + C_I*nu*nu*inner(G,G) + DOLFIN_EPS
if(dT != None):
    denom2 += C_t/dT**2

tm = 1/sqrt(denom2)
#tm=(4*((dt)**(-2)) + 36*(nu**2)*inner(G,G) + inner(u,G*u))**(-0.5)
tc=1.0/(tm*tr(G))#!/usr/bin/env python
tcross = outer((tm*rm),(tm*rm))


uPrime = -tm*rm
pPrime = -tc*rc

# Giovanni's version
if giovanni:
    F = (inner(DuDt,v) + inner(sigma(u,p,nu),grad(v))
        + inner(div(u),q)
        - inner(dot(u,nabla_grad(v)) + grad(q), uPrime)
        - inner(pPrime,div(v))
        + inner(v,dot(uPrime,nabla_grad(u)))
        - inner(grad(v),outer(uPrime,uPrime)))*dx

    if boundary_tag=="weak":
        F += weakDirichletBC(u,p,u_prev,v,q,u_bc,nu,mesh,physical_problem.ds_bc, spalding=False, C_pen = C_b_I)
    elif boundary_tag=="spalding":
        F += weakDirichletBC(u,p,u_prev,v,q,u_bc,nu,mesh,physical_problem.ds_bc, tau_penalty, spalding=True)
else:
    #Hughes' version
    r_M, r_C = strongResidual(u,p,nu,u_t,f)
#    r_M, r_C = strongResidualHughes(u,p,nu,u_t,f)
    tau_M, tau_C = stabilizationParameters(u,nu,G,C_I,C_t,Dt=dT,scale=Constant(1.0))

    stab =  (inner( dot(u,2*sym(grad(v)))+grad(q), tau_M * r_M )
        + inner(div(v), tau_C * r_C ))*dx

    F = (inner(u_t + dot(u,nabla_grad(u))-f,v) # = inner(DuDt, v) 
        + inner(div(u),q)  
        - inner(p,div(v)) + inner(2*nu*sym(grad(u)),sym(grad(v))) )*dx\
        + stab
    # F = (inner(u_t -f,v) -  inner(outer(u,u),grad(v)) # = inner(DuDt, v) 
    #     + inner(div(u),q)  
    #     - inner(p,div(v)) + inner(2*nu*sym(grad(u)),sym(grad(v))) )*dx\
    #     + stab \
    #     + dot(u,v)*dot(u,n)*ds


    if boundary_tag=="weak":
        F += weakHughesBC(u,p,u_prev,v,q,u_bc,nu,mesh,physical_problem.ds_bc, G=G, spalding=False, C_pen = C_b_I)
    elif boundary_tag=="spalding":
        F += weakHughesBC(u,p,u_prev,v,q,u_bc,nu,mesh,physical_problem.ds_bc, tau_penalty, G=G, spalding=True)




J = derivative(F, up, delta_up)







def solve_FOM(param, folder_simulation, RB = None, with_plot = False):
    try:
        os.mkdir(folder_simulation)
    except:
        print("Probably folder %s already exists "%folder_simulation)
    u_top_val = param[0]
    nu_val = param[1]

    nu.assign(Constant(nu_val))
    u_top.assign(Constant(u_top_val))

    Re_val = physical_problem.get_reynolds(u_top_val,nu_val)

    print("Reynolds Number = %e"%Re_val)
    physical_problem.define_bc(W, u_top)



    """### Boundary conditions (for the solution)"""

    # walls_bc       = DirichletBC(W.sub(0), Constant((0., 0.)), boundaries, walls_ID )
    # #sides_bc       = DirichletBC(W.sub(0).sub(1), Constant(0.), boundaries, sides_ID )
    # #inlet_bc       = DirichletBC(W.sub(1), Constant(0.),       boundaries, inlet_ID )
    # #outlet_bc       = DirichletBC(W.sub(1), Constant(0.),      boundaries, outlet_ID )
    # onePoint_bc     = DirichletBC(W.sub(1), Constant(0.),      boundaries, onePoint_ID) #OnePoint(), method='pointwise')# 


    if boundary_tag == "strong":
        bc = physical_problem.bcs # [onePoint_bc, walls_bc] #, sides_bc
    else:
        bc = physical_problem.bc_no_walls # [onePoint_bc]#, walls_bc] #, sides_bc

    snes_solver_parameters = {"nonlinear_solver": "snes",
                            "snes_solver": {"linear_solver": "mumps",
                                            "maximum_iterations": 20,
                                            "report": True,
                                            "error_on_nonconvergence": True}}

    problem = NonlinearVariationalProblem(F, up, bc, J)
    solver  = NonlinearVariationalSolver(problem)
    solver.parameters.update(snes_solver_parameters)

    u_max = max(u_top.values())

    dt = CFL*hmin/u_max
    time=0.
    



    u_0, p_0 = physical_problem.get_IC()
    u_0 = project(u_0, W.sub(0).collapse())
    p_0 = project(p_0, W.sub(1).collapse())

    assign(up_prev , [u_0,p_0])
    assign(up , [u_0,p_0])


    outfile_u = File(folder_simulation+"/u.pvd")
    outfile_p = File(folder_simulation+"/p.pvd")
    if boundary_tag in ["spalding", "weak"]:
        outfile_tau = File(folder_simulation+"/tau.pvd")


    outxdmf_u = XDMFFile(folder_simulation+"/u.xdmf")
    outxdmf_p = XDMFFile(folder_simulation+"/p.xdmf")
    if boundary_tag in ["spalding", "weak"]:
        outxdmf_tau = XDMFFile(folder_simulation+"/tau.xdmf")


    if RB is not None:
        up_hat = Function(W)
        if boundary_tag=="spalding":
            up_RB = project_onto_RB(RB, up, tau_penalty)
            tau_hat = Function(V0)
        else:
            up_RB = project_onto_RB(RB, up)
            tau_hat = None
        reconstruct_RB(RB, up_RB, up_hat, tau_hat )

        outfile_uRB = File(folder_simulation+"/uRB.pvd")
        outfile_pRB = File(folder_simulation+"/pRB.pvd")
        if boundary_tag in ["spalding"]:
            outfile_tauRB = File(folder_simulation+"/tauRB.pvd")



        outxdmf_uRB = XDMFFile(folder_simulation+"/uRB.xdmf")
        outxdmf_pRB = XDMFFile(folder_simulation+"/pRB.xdmf")
        if boundary_tag in ["spalding"]:
            outxdmf_tauRB = XDMFFile(folder_simulation+"/tauRB.xdmf")

        (u_hat, p_hat) = up_hat.split()
        outfile_uRB << u_hat
        outfile_pRB << p_hat

        outxdmf_uRB.write_checkpoint(u_hat, "u", time, XDMFFile.Encoding.HDF5, append=False)
        outxdmf_pRB.write_checkpoint(p_hat, "p", time, XDMFFile.Encoding.HDF5, append=False)

        if boundary_tag in ["spalding"]:
            outxdmf_tauRB.write_checkpoint(tau_hat, "tau", time, XDMFFile.Encoding.HDF5, append=False) 



    if boundary_tag=="weak":
        trial_v0 = TrialFunction(V0)
        tau_penalty_bound = Function(V0)
        test_v0 = TrialFunction(V0)
        F_tau = inner(tau_penalty_bound,test_v0)*dx - inner(test_v0,C_pen*nu/hb)*physical_problem.ds_bc
        solve(F_tau==0,tau_penalty_bound )
        # outfile_tau << tau_penalty_bound
        outxdmf_tau.write_checkpoint(tau_penalty_bound, "tau", time, XDMFFile.Encoding.HDF5, False)
        

    (u, p) = up.split()
    outfile_u << u
    outfile_p << p

    outxdmf_u.write_checkpoint(u, "u", time, XDMFFile.Encoding.HDF5, append=False)
    outxdmf_p.write_checkpoint(p, "p", time, XDMFFile.Encoding.HDF5, append=False)


    tic= time_module.time()


    times = [time]
    times_plot = [time]
    if RB is not None:
        errors = dict()
        err = Function(W)
        err.assign(up-up_hat)
        for comp in ("u","p"):
            errors[comp] = [ (X_inner[comp]*err.vector()).inner(err.vector()) ]
        if boundary_tag=="spalding":
            comp= "tau"
            tau_err = Function(V0)
            tau_err.assign(tau_penalty-tau_hat)
            errors["tau"] = [ (X_inner[comp]*tau_err.vector()).inner(tau_err.vector())]
    
    it=0
    tplot=0.
    u_norm = interpolate(u_top,V0)
    while time < T and it < Nt_max:
        tic_one_step= time_module.time()
        if u_norm.vector().max()<1e-8:
            dt=CFL*hmin
        else:
            dt = CFL*project(h/u_norm,V0).vector().min()
        dt = min(dt, T-time)
        dT.assign(dt)
        print("Maximum speed %g"%(u_norm.vector().max()))
        print("Time %1.5e, final time = %1.5e, dt = %1.5e"%(time,T,dt))
        # Compute the current time
        # Update the time for the boundary condition
        physical_problem.u_in.t = time
        # Solve the nonlinear problem
        # with pipes() as (out, err):
        if boundary_tag =="spalding":
            tic_spalding= time_module.time()
            solve_spalding_law_inout(u_prev,hb,tau_penalty)
            toc_spalding= time_module.time() - tic_spalding
            print("Spalding time %e"%toc_spalding)

        #solver.solve()
        solve(F == 0, up, bcs=bc)#, solver_parameters={"newton_solver":{"relative_tolerance":1e-8} })
        # Store the solution in up_prev
        assign(up_prev, up)
        # Plot
        (u, p) = up.split()

        u_norm = project(sqrt(u[0]**2+u[1]**2),V0)

        toc_one_step = time_module.time() - tic_one_step
        print("Time one step %e"%(toc_one_step))
        if boundary_tag=="spalding":
            print("Percentage spalding %g%%"%(100.*toc_spalding/toc_one_step))
        tplot+= dt
        time+= dt
        it+=1

        times.append(time)

        if it<10 or tplot > dtplot:
            times_plot.append(time)
            print("time = %g"%time)
            tplot = 0.
            outxdmf_u.write_checkpoint(u, "u", time, XDMFFile.Encoding.HDF5, append=True)
            outxdmf_p.write_checkpoint(p, "p", time, XDMFFile.Encoding.HDF5, append=True)
            if boundary_tag in ["spalding"]:
                outxdmf_tau.write_checkpoint(tau_penalty, "tau", time, XDMFFile.Encoding.HDF5, append=True)

            outfile_u << u
            outfile_p << p
            if boundary_tag in ["spalding"]:
                outfile_tau << tau_penalty
            
            if RB is not None:
                if boundary_tag=="spalding":
                    up_RB = project_onto_RB(RB, up, tau_penalty)
                    tau_hat = Function(V0)
                else:
                    up_RB = project_onto_RB(RB, up)
                    tau_hat = None
                reconstruct_RB(RB, up_RB, up_hat, tau_hat )
                outxdmf_uRB.write_checkpoint(u_hat, "u", time, XDMFFile.Encoding.HDF5, append=True)
                outxdmf_pRB.write_checkpoint(p_hat, "p", time, XDMFFile.Encoding.HDF5, append=True)
                if boundary_tag in ["spalding"]:
                    outxdmf_tauRB.write_checkpoint(tau_hat, "tau", time, XDMFFile.Encoding.HDF5, append=True)

                outfile_uRB << u_hat
                outfile_pRB << p_hat
                if boundary_tag in ["spalding"]:
                    outfile_tauRB << tau_hat

                # Computing errors

                err.assign(up-up_hat)
                for comp in ("u","p"):
                    errors[comp].append(\
                        (X_inner[comp]*err.vector()).inner(err.vector())/\
                        (X_inner[comp]*up.vector()).inner(up.vector())) 
                if boundary_tag=="spalding":
                    comp= "tau"
                    tau_err.assign(tau_penalty-tau_hat)
                    errors["tau"].append((X_inner[comp]*tau_err.vector()).inner(tau_err.vector()))
            

    computational_time =  time_module.time()-tic
    print("computational time %g"%computational_time)
    times = np.array(times)

    data_file = folder_simulation+"/data.npy"
    np.savez(data_file, times, param, computational_time)


    times_plot.append(time)
    outxdmf_u.write_checkpoint(u, "u", time, XDMFFile.Encoding.HDF5, True)
    outxdmf_p.write_checkpoint(p, "p", time, XDMFFile.Encoding.HDF5, True)
    if boundary_tag in ["spalding"]:
        outxdmf_tau.write_checkpoint(tau_penalty, "tau", time, XDMFFile.Encoding.HDF5, True)

    outfile_u << u
    outfile_p << p
    if boundary_tag in ["spalding"]:
        outfile_tau << tau_penalty

    if RB is not None:
        if boundary_tag=="spalding":
            up_RB = project_onto_RB(RB, up, tau_penalty)
            tau_hat = Function(V0)
        else:
            up_RB = project_onto_RB(RB, up)
            tau_hat = None
        reconstruct_RB(RB, up_RB, up_hat, tau_hat )
        outxdmf_uRB.write_checkpoint(u_hat, "u", time, XDMFFile.Encoding.HDF5, append=True)
        outxdmf_pRB.write_checkpoint(p_hat, "p", time, XDMFFile.Encoding.HDF5, append=True)
        if boundary_tag in ["spalding"]:
            outxdmf_tauRB.write_checkpoint(tau_hat, "tau", time, XDMFFile.Encoding.HDF5, append=True)

        outfile_uRB << u_hat
        outfile_pRB << p_hat
        if boundary_tag in ["spalding"]:
            outfile_tauRB << tau_hat

        # Computing errors

        err.assign(up-up_hat)
        for comp in ("u","p"):
            errors[comp].append(\
                (X_inner[comp]*err.vector()).inner(err.vector())/\
                (X_inner[comp]*up.vector()).inner(up.vector())) 
        if boundary_tag=="spalding":
            comp= "tau"
            tau_err.assign(tau_penalty-tau_hat)
            errors["tau"].append(\
                (X_inner[comp]*tau_err.vector()).inner(tau_err.vector())/\
                (X_inner[comp]*tau_penalty.vector()).inner(tau_penalty.vector()))

        with open(folder_simulation+"/RB_proj_errors.pickle", "wb") as ff:
            pickle.dump(errors,ff, protocol=pickle.HIGHEST_PROTOCOL)

    if with_plot:
        plt.figure()
        pp=plot(p); plt.colorbar(pp)
        plt.title("Pressure")
        plt.savefig(folder_simulation+"/p_final.png")
        plt.show(block=False)

        plt.figure()
        pp=plot(u); plt.colorbar(pp)
        plt.title("Velocity")
        plt.savefig(folder_simulation+"/u_final.png")
        plt.show(block=False)

        plt.figure()
        pp=plot(u[0]); plt.colorbar(pp)
        plt.title("u")
        plt.show(block=False)

        plt.figure()
        pp=plot(u[1]); plt.colorbar(pp)
        plt.title("v")
        plt.show(block=False)


        if boundary_tag in ["spalding"]:
            plt.figure()
            pp=plot(tau_penalty); plt.colorbar(pp)
            plt.title("Tau penalty")
            plt.savefig(folder_simulation+"/tau_final.png")
            plt.show(block=False)

        if RB is not None:
            plt.figure()
            pp=plot(u_hat); plt.colorbar(pp)
            plt.title("Velocity")
            plt.savefig(folder_simulation+"/u_RB_proj_final.png")
            plt.show(block=False)

            plt.figure()
            pp=plot(p_hat); plt.colorbar(pp)
            plt.title("Pressure")
            plt.savefig(folder_simulation+"/p_RB_proj_final.png")
            plt.show(block=False)

            plt.figure()
            pp=plot(u_hat[0]); plt.colorbar(pp)
            plt.title("uRB")
            plt.show(block=False)

            plt.figure()
            pp=plot(u_hat[1]); plt.colorbar(pp)
            plt.title("vRB")
            plt.show(block=False)


            if boundary_tag in ["spalding"]:
                plt.figure()
                pp=plot(tau_hat); plt.colorbar(pp)
                plt.title("Tau penalty")
                plt.savefig(folder_simulation+"/tau_RB_proj_final.png")
                plt.show(block=False)

            plt.figure()
            pp=plot(err.sub(0)); plt.colorbar(pp)
            plt.title("Velocity")
            plt.savefig(folder_simulation+"/err_u_RB_proj_final.png")
            plt.show(block=False)

            plt.figure()
            pp=plot(err.sub(1)); plt.colorbar(pp)
            plt.title("Pressure")
            plt.savefig(folder_simulation+"/err_p_RB_proj_final.png")
            plt.show(block=False)


            if boundary_tag in ["spalding"]:
                plt.figure()
                pp=plot(tau_err); plt.colorbar(pp)
                plt.title("Tau penalty error")
                plt.savefig(folder_simulation+"/err_tau_RB_proj_final.png")
                plt.show(block=False)

            plt.figure()
            plt.plot(times_plot, errors["u"], label="error u")
            plt.plot(times_plot, errors["p"], label="error p")
            if boundary_tag in ["spalding"]:
                plt.plot(times_plot, errors["tau"], label="error tau")
            plt.savefig(folder_simulation+"/errors_vs_time.pdf")
            plt.show(block=False)

def compute_FOM_RB_projection(RB, param, folder_simulation):
    try:
        os.mkdir(folder_simulation)
    except:
        print("Probably folder %s already exists "%folder_simulation)
    u_top_val = param[0]
    nu_val = param[1]

    nu.assign(Constant(nu_val))
    u_top.assign(Constant(u_top_val))

    Re_val = physical_problem.get_reynolds(u_top_val,nu_val)

    print("Reynolds Number = %e"%Re_val)
    physical_problem.define_bc(W, u_top)



    """### Boundary conditions (for the solution)"""

    # walls_bc       = DirichletBC(W.sub(0), Constant((0., 0.)), boundaries, walls_ID )
    # #sides_bc       = DirichletBC(W.sub(0).sub(1), Constant(0.), boundaries, sides_ID )
    # #inlet_bc       = DirichletBC(W.sub(1), Constant(0.),       boundaries, inlet_ID )
    # #outlet_bc       = DirichletBC(W.sub(1), Constant(0.),      boundaries, outlet_ID )
    # onePoint_bc     = DirichletBC(W.sub(1), Constant(0.),      boundaries, onePoint_ID) #OnePoint(), method='pointwise')# 


    if boundary_tag == "strong":
        bc = physical_problem.bcs # [onePoint_bc, walls_bc] #, sides_bc
    else:
        bc = physical_problem.bc_no_walls # [onePoint_bc]#, walls_bc] #, sides_bc

    snes_solver_parameters = {"nonlinear_solver": "snes",
                            "snes_solver": {"linear_solver": "mumps",
                                            "maximum_iterations": 20,
                                            "report": True,
                                            "error_on_nonconvergence": True}}

    problem = NonlinearVariationalProblem(F, up, bc, J)
    solver  = NonlinearVariationalSolver(problem)
    solver.parameters.update(snes_solver_parameters)

    u_max = max(u_top.values())

    dt = CFL*hmin/u_max





    u_0, p_0 = physical_problem.get_IC()
    u_0 = project(u_0, W.sub(0).collapse())
    p_0 = project(p_0, W.sub(1).collapse())

    assign(up_prev , [u_0,p_0])
    assign(up , [u_0,p_0])



    outfile_u = File(folder_simulation+"/u.pvd")
    outfile_p = File(folder_simulation+"/p.pvd")
    if boundary_tag in ["spalding", "weak"]:
        outfile_tau = File(folder_simulation+"/tau.pvd")


    outxdmf_u = XDMFFile(folder_simulation+"/u.xdmf")
    outxdmf_p = XDMFFile(folder_simulation+"/p.xdmf")
    if boundary_tag in ["spalding", "weak"]:
        outxdmf_tau = XDMFFile(folder_simulation+"/tau.xdmf")


    time=0.
    if boundary_tag=="weak":
        trial_v0 = TrialFunction(V0)
        tau_penalty_bound = Function(V0)
        test_v0 = TrialFunction(V0)
        F_tau = inner(tau_penalty_bound,test_v0)*dx - inner(test_v0,C_pen*nu/hb)*physical_problem.ds_bc
        solve(F_tau==0,tau_penalty_bound )
        # outfile_tau << tau_penalty_bound
        outxdmf_tau.write_checkpoint(tau_penalty_bound, "tau", time, XDMFFile.Encoding.HDF5, False)
        

    (u, p) = up.split()
    outfile_u << u
    outfile_p << p

    outxdmf_u.write_checkpoint(u, "u", time, XDMFFile.Encoding.HDF5, append=False)
    outxdmf_p.write_checkpoint(p, "p", time, XDMFFile.Encoding.HDF5, append=False)


    tic= time_module.time()

    times = [time]
    it=0
    tplot=0.
    u_norm = interpolate(u_top,V0)
    while time < T and it < Nt_max:
        tic_one_step= time_module.time()
        if u_norm.vector().max()<1e-8:
            dt=CFL*hmin
        else:
            dt = CFL*project(h/u_norm,V0).vector().min()
        dt = min(dt, T-time)
        dT.assign(dt)
        print("Maximum speed %g"%(u_norm.vector().max()))
        print("Time %1.5e, final time = %1.5e, dt = %1.5e"%(time,T,dt))
        # Compute the current time
        # Update the time for the boundary condition
        physical_problem.u_in.t = time
        # Solve the nonlinear problem
        # with pipes() as (out, err):
        if boundary_tag =="spalding":
            tic_spalding= time_module.time()
            solve_spalding_law_inout(u_prev,hb,tau_penalty)
            toc_spalding= time_module.time() - tic_spalding
            print("Spalding time %e"%toc_spalding)

        #solver.solve()
        solve(F == 0, up, bcs=bc)#, solver_parameters={"newton_solver":{"relative_tolerance":1e-8} })
        # Store the solution in up_prev
        assign(up_prev, up)
        # Plot
        (u, p) = up.split()

        u_norm = project(sqrt(u[0]**2+u[1]**2),V0)

        ## Project solution onto RB


        toc_one_step = time_module.time() - tic_one_step
        print("Time one step %e"%(toc_one_step))
        if boundary_tag=="spalding":
            print("Percentage spalding %g%%"%(100.*toc_spalding/toc_one_step))
        tplot+= dt
        time+= dt
        it+=1

        times.append(time)

        if it<10 or tplot > dtplot:
            print("time = %g"%time)
            tplot = 0.
            outxdmf_u.write_checkpoint(u, "u", time, XDMFFile.Encoding.HDF5, append=True)
            outxdmf_p.write_checkpoint(p, "p", time, XDMFFile.Encoding.HDF5, append=True)
            if boundary_tag in ["spalding"]:
                outxdmf_tau.write_checkpoint(tau_penalty, "tau", time, XDMFFile.Encoding.HDF5, append=True)

            outfile_u << u
            outfile_p << p
            if boundary_tag in ["spalding"]:
                outfile_tau << tau_penalty

        
    computational_time =  time_module.time()-tic
    print("computational time %g"%computational_time)
    times = np.array(times)

    data_file = folder_simulation+"/data.npy"
    np.savez(data_file, times, param, computational_time)

    outxdmf_u.write_checkpoint(u, "u", time, XDMFFile.Encoding.HDF5, True)
    outxdmf_p.write_checkpoint(p, "p", time, XDMFFile.Encoding.HDF5, True)
    if boundary_tag in ["spalding"]:
        outxdmf_tau.write_checkpoint(tau_penalty, "tau", time, XDMFFile.Encoding.HDF5, True)

    outfile_u << u
    outfile_p << p
    if boundary_tag in ["spalding"]:
        outfile_tau << tau_penalty