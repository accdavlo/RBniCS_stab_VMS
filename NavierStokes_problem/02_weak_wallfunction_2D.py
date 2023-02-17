from dolfin import *
from ufl.geometry import *
import numpy as np
from ufl import Jacobian
from sys import argv
from dolfin.cpp.mesh import *
from mshr import *
from matplotlib import pyplot
import ufl
from wurlitzer import pipes
from numpy import random
from fenicstools import interpolate_nonmatching_mesh, interpolate_nonmatching_mesh_any
from utils_PETSc_scipy import PETSc2scipy, inverse_lumping_with_zeros


import petsc4py
from scipy import sparse
from scipy.sparse.linalg import spsolve
from scipy.optimize import bisect

giovanni = False
boundary_tag = "strong" # "spalding"#"weak" # 

parameters["linear_algebra_backend"] = "PETSc"
args = "--petsc.snes_linesearch_monitor --petsc.snes_linesearch_type bt"
parameters.parse(argv = argv[0:1] + args.split())




delta_x = 2*pi
#delta_y = 2
delta_z = 2/3*pi
rx = 10
#ry = 5
rz = 10
Nx = int(rx*delta_x)
#Ny = int(ry*delta_y)
Nz = int(rz*delta_z)

dx_mesh = delta_x/Nx
dz_mesh = delta_z/Nx


delta_pressure_val = 3.37204e-3 
delta_pressure = Constant(delta_pressure_val)
nu_val = 1.472e-4
nu = Constant(nu_val)
f = Constant((delta_pressure_val,0.))
q_degree = 3
dx = dx(metadata={'quadrature_degree': q_degree})
 
# Create mesh
mesh = RectangleMesh(Point(0,0),Point(delta_x,delta_z),Nx,Nz)

# Create subdomains
subdomains = MeshFunction("size_t", mesh, 2)
subdomains.set_all(0)

# Create boundaries
class Walls(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary and \
            (abs(x[1]) < DOLFIN_EPS or abs(x[1] - delta_z) < DOLFIN_EPS)

# Sub domain for Periodic boundary condition
class PeriodicBoundary(SubDomain):

    # Left boundary is "target domain" G
    def inside(self, x, on_boundary):
        return on_boundary and bool(near(x[0],0))

    # Map right boundary (H) to left boundary (G)
    def map(self, x, y):
        #if near(x[0],delta_x):
            y[0] = x[0] - delta_x
            y[1] = x[1]

class OnePoint(SubDomain):
    # x=y=z=0      to set pressure in one point
    def inside(self, x, on_boundary):
        return on_boundary and \
           (abs(x[0]-1.5*dx_mesh) < 1.01*dx_mesh and abs(x[1]) < dz_mesh)


class AllBoundary(SubDomain):
    def inside(self, x, on_boundary):
        return on_boundary

boundaries = MeshFunction("size_t", mesh, mesh.topology().dim()-1, 0)
boundaries.set_all(0)
walls_ID = 1
walls = Walls()
walls.mark(boundaries, walls_ID)
onePoint_ID = 5
onePoint = OnePoint()
onePoint.mark(boundaries, onePoint_ID)

bmesh = BoundaryMesh(mesh, 'exterior')

ds_bc = Measure('ds', domain=mesh, subdomain_data=boundaries, subdomain_id=1, metadata = {'quadrature_degree': 2})

# Save to xml file
File("Rectangular.xml") << mesh
File("Rectangular_physical_region.xml") << subdomains
File("Rectangular_facet_region.xml") << boundaries

# Save to pvd file for visualization
xdmf = XDMFFile(mesh.mpi_comm(), "Rectangular_mesh.xdmf")
xdmf.write(mesh)
xdmf = XDMFFile(mesh.mpi_comm(), "Rectangular_physical_region.xdmf")
subdomains.rename("subdomains", "subdomains")
xdmf.write(subdomains)
xdmf = XDMFFile(mesh.mpi_comm(), "Rectangular_facet_region.xdmf")
boundaries.rename("boundaries", "boundaries")
xdmf.write(boundaries)


Re = 395. # 120. # not used!!
#u_bar = Re*nu_val/delta_z
char_L=delta_z/2.
print("Theoretical u_max ", delta_pressure_val/2/nu_val*(char_L)**2)
u_max =   delta_pressure_val/2/nu_val*(char_L)**2 #1.5#
print("Max velocity", u_max)
Re_posteriori = u_max/nu_val*char_L
print("Reynolds a posteriori ",Re_posteriori)
#u_in = Expression(("delta_pressure/2/nu*x[2]*(delta_z-x[2])", "0."), t=0, delta_z = delta_z, u_bar=u_bar, nu=nu, delta_pressure=delta_pressure, degree=2) 
u_in = Expression(("u_max*4*x[1]*(delta_z-x[1])/delta_z/delta_z", "0."), t=0, delta_z = delta_z, u_max=u_max, degree=2) 
p_in = Expression('0', degree=1)
# nu = Constant(u_bar*0.1/Re) # obtained from the definition of Re = u_bar * diam / nu. In our case diam = 0.1.

dt = 10*delta_x/Nx/u_max
T = 2000 * dt # should be 15 to generate the video

"""### Function spaces"""
pbc = PeriodicBoundary()
V_element = VectorElement("Lagrange", mesh.ufl_cell(), 1)
Q_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W_element = MixedElement(V_element, Q_element) 
W = FunctionSpace(mesh, W_element, constrained_domain=PeriodicBoundary())
print(W.dim())



dg0_element = FiniteElement("DG", mesh.ufl_cell(),0)
V0 = FunctionSpace(mesh, dg0_element)
dg0_bound_element = FiniteElement("DG", bmesh.ufl_cell(),0)
V0_bound = FunctionSpace(bmesh, dg0_bound_element)


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

def solve_boundary_problem(f, VDG, LHS_inv):
    """
    Extend function f which is evaluatable only at boundaries onto the whole domain
    the result lives in VDG which must be piecewise constant polynomails space.
    """
    z_test = TestFunction(VDG)
    rhs = inner(f,z_test)*ds
    RHS = assemble(rhs)
    x = Function(VDG)
    x.vector()[:]=LHS_inv@RHS
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
    hb_bound = interpolate_nonmatching_mesh(solve_boundary_problem(hb,V0,DG_matrix_inv) ,V0_bound)

    # solution of Spalding's law
    tau_B_bound = Function(V0_bound)

    # Loop over boundary cells
    for cell in cells(bmesh):
        cell_index = cell.index() 
        mid_pt = cell.midpoint()
        i = dofmapV0.dofs()[cell_index]
        # solving Spalding's law through bisection in a cell
        tau_B_bound.vector()[i] = bisect(spalding_func, 1e-14, 1e5, xtol=1e-12, \
            args=(hb_bound.vector()[i], u_norm.vector()[i], \
                Chi_Spalding(mid_pt), B_Spalding(mid_pt), C_b_I(mid_pt), nu(mid_pt) ) )
    # Extending the solution vector to the whole mesh
    tau_B_ext = Function(V0)
    tau_B_ext.vector()[:] = interp_V0_V0_bound@tau_B_bound.vector()

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
             *inner(u_minus_g,v))*ds

def weakDirichletBC(u,p,u_prev,v,q,g,nu,mesh,ds=ds,G=None,
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
        G = meshMetric(mesh) # $\sim h^{-2}$
    traction = sigma(u,p,nu)*n
    consistencyTerm = stableNeumannBC(traction,u,v,n,g=g,ds=ds)
    # Note sign of ``q``, negative for stability, regardless of ``sym``.    
    adjointConsistency = -sgn*dot(sigma(v,-sgn*q,nu)*n,u-g)*ds
    # Only term we need to change
    hb = 2*sqrt(dot(n,G*n))
    
    # Weak penalty coefficient or Spalding's law coefficient
    if spalding:
        tau_pen  = solve_spalding_law(u_prev,hb)
    else:
        tau_pen =  C_pen*nu/hb
    
    penalty = tau_pen*dot((u-g),v)*ds
    retval = consistencyTerm + adjointConsistency
    if(overPenalize or sym):
        retval += penalty
        print("Passing here")
    return retval

def weakHughesBC(u,p,u_prev,v,q,g,nu,mesh,ds=ds,G=None,
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

    consistencyTerm = -sgn*inner(2*nu*sym(grad(u))*n,v)*ds
    adjointConsistency = -sgn*inner(2*gamma*nu*sym(grad(v))*n,u-g)*ds

    # Only term we need to change
    hb = 2*sqrt(dot(n,G*n))
    
    # Weak penalty coefficient or Spalding's law coefficient
    if spalding:
        tau_pen  = solve_spalding_law(u_prev,hb)
    else:
        tau_pen =  C_pen*nu/hb
    
    penalty = tau_pen*dot((u-g),v)*ds
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

def strongResidual(u,p,nu,u_t,f):
    """
    The momentum and continuity residuals, as a tuple, of the strong PDE,
    system, in terms of velocity ``u``, pressure ``p``, dynamic viscosity ``nu``,
    the partial time derivative of velocity ``u_t``, and a body force per unit mass
    ``f``.  
    """
    DuDt = materialTimeDerivative(u,u_t,f)
    i,j = ufl.indices(2)
    r_M = DuDt - as_tensor(grad(sigma(u,p,nu))[i,j,j],(i,))  # if P1 last term zero
    r_C = div(u)
    return r_M, r_C



DG_matrix_inv = get_DG_inverse_matrix(V0)
dofmapV0 = V0.dofmap()

dofmapV0_bound = V0_bound.dofmap()

interp_V0_V0_bound=PETSc2scipy(PETScDMCollection.create_transfer_matrix(V0_bound,V0))

tau_B= Function(V0)
Chi_Spalding = Constant(0.4)
B_Spalding = Constant(5.5)

n = FacetNormal(mesh)
G = meshMetric(mesh)
hb = 2*sqrt(dot(n,G*n))


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

## BCs for weak boundaries and initial condition perturbation
## BCs for weak boundaries and initial condition perturbation
u_pert = Expression(('u_max*0.01*sin(25*x[1]*2*pi/delta_z)',\
    '1e-3*u_max*sin(41*x[0]*2*pi/delta_x)'), \
    degree=4, u_max=u_max, delta_x=delta_x, delta_z=delta_z)

#u_0 = interpolate(u_in, W.sub(0).collapse())
u_0 = project(u_in+u_pert, W.sub(0).collapse())
#u_0 = interpolate(u_in, W.sub(0).collapse())+project(u_pert, W.sub(0).collapse())

u_bc = interpolate(u_in, W.sub(0).collapse())

p_0 = interpolate(p_in, W.sub(1).collapse())

assign(up_prev , [u_0,p_0])
assign(up , [u_0,p_0])
u_t = (u - u_prev)/dt

# Preparation of Variational forms.
#K = JacobianInverse(mesh)
#G = K.T*K

#gg = (K[0,0] + K[1,0] + K[2,0])**2 + (K[0,1] + K[1,1] + K[2,1])**2 + (K[0,2] + K[1,2] + K[2,2])**2
#rm = (u-u_prev)/dt + grad(p) + grad(u)*u - nu*(u.dx(0).dx(0) + u.dx(1).dx(1) + u.dx(2).dx(2)) - f

DuDt = materialTimeDerivative(u,u_t,f)
i,j = ufl.indices(2)
rm = DuDt - as_tensor(grad(sigma(u,p,nu))[i,j,j],(i,)) # if u \in P1 last term is zero (second derivative)! 
rc = div(u)

C_I = Constant(36.0)
C_t = Constant(4.0)
C_b_I = Constant(4.0)

denom2 = inner(u,G*u) + C_I*nu*nu*inner(G,G) + DOLFIN_EPS
if(dt != None):
    denom2 += C_t/dt**2

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
        F += weakDirichletBC(u,p,u_prev,v,q,u_bc,nu,mesh,ds_bc, spalding=False)
    elif boundary_tag=="spalding":
        F += weakDirichletBC(u,p,u_prev,v,q,u_bc,nu,mesh,ds_bc, spalding=True)
else:
    #Hughes' version
    r_M, r_C = strongResidual(u,p,nu,u_t,f)
    tau_M, tau_C = stabilizationParameters(u,nu,G,C_I,C_t,Dt=None,scale=Constant(1.0))

    stab =  (inner( dot(u,2*sym(grad(v)))+grad(q), tau_M * r_M )
           + inner(div(v), tau_C * r_C ))*dx

    F = (inner(u_t,v) - inner(outer(u,u), grad(v))- inner(f,v) # = inner(DuDt, v) 
         + inner(div(u),q)  
         - inner(p,div(v)) + inner(2*nu*sym(grad(u)),sym(grad(v))) )*dx\
        + stab

    if boundary_tag=="weak":
        F += weakHughesBC(u,p,u_prev,v,q,u_bc,nu,mesh,ds_bc, G=G, spalding=False)
    elif boundary_tag=="spalding":
        F += weakHughesBC(u,p,u_prev,v,q,u_bc,nu,mesh,ds_bc, G=G, spalding=True)




J = derivative(F, up, delta_up)



"""### Boundary conditions (for the solution)"""

walls_bc       = DirichletBC(W.sub(0), Constant((0., 0.)), boundaries, walls_ID )
#sides_bc       = DirichletBC(W.sub(0).sub(1), Constant(0.), boundaries, sides_ID )
#inlet_bc       = DirichletBC(W.sub(1), Constant(0.),       boundaries, inlet_ID )
#outlet_bc       = DirichletBC(W.sub(1), Constant(0.),      boundaries, outlet_ID )
onePoint_bc     = DirichletBC(W.sub(1), Constant(0.),      boundaries, onePoint_ID) #OnePoint(), method='pointwise')# 


if boundary_tag == "strong":
    bc = [onePoint_bc, walls_bc] #, sides_bc
else:
    bc = [onePoint_bc]#, walls_bc] #, sides_bc

snes_solver_parameters = {"nonlinear_solver": "snes",
                          "snes_solver": {"linear_solver": "mumps",
                                          "maximum_iterations": 20,
                                          "report": True,
                                          "error_on_nonconvergence": True}}

problem = NonlinearVariationalProblem(F, up, bc, J)
solver  = NonlinearVariationalSolver(problem)
solver.parameters.update(snes_solver_parameters)

if giovanni:
    outfile_u = File("out_giovanni_2D_"+boundary_tag+"/u.pvd")
    outfile_p = File("out_giovanni_2D_"+boundary_tag+"/p.pvd")
else:
    outfile_u = File("out_Hughes_2D_"+boundary_tag+"/u.pvd")
    outfile_p = File("out_Hughes_2D_"+boundary_tag+"/p.pvd")


(u, p) = up.split()
outfile_u << u
outfile_p << p

K = int(T/dt)
for i in range(1, K):
    # Compute the current time
    t = i*dt
    print("t =", t)
    # Update the time for the boundary condition
    u_in.t = t
    # Solve the nonlinear problem
    # with pipes() as (out, err):
    #solver.solve()
    solve(F == 0, up, bcs=bc, solver_parameters={"newton_solver":{"relative_tolerance":1e-8} })
    # Store the solution in up_prev
    assign(up_prev, up)
    # Plot
    (u, p) = up.split()
    outfile_u << u
    outfile_p << p
