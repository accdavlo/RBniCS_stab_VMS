from dolfin import *
from ufl.geometry import *
parameters["form_compiler"]["representation"] = "uflacs"
import numpy as np
from ufl import Jacobian
from sys import argv
from dolfin.cpp.mesh import *
from mshr import *

class InflowBoundary(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and x[0] < xmin + bmarg

class NoslipBoundary(SubDomain):
	def inside(self, x, on_boundary):
		dx = x[0] 
		dy = x[1] 
		r = sqrt(dx*dx + dy*dy)
		return on_boundary and \
			(x[1] < ymin + bmarg or x[1] > ymax - bmarg or \
			r < radius + bmarg)
			
class Cylinder(SubDomain):
	def inside(self, x, on_boundary):
		rx = x[0] 
		ry = x[1] 
		r = sqrt(rx*rx + ry*ry)
		return on_boundary and r < radius + bmarg

# Outflow boundary
class OutflowBoundary(SubDomain):
	def inside(self, x, on_boundary):
		return on_boundary and x[0] > cyl_out - bmarg

def integrateFluidStress(a_V, a_P, a_Nu, a_Normal, a_GammaP):
	
	eps   = 0.5*(nabla_grad(a_V) + nabla_grad(a_V).T)
	sig   = -a_P*Identity(len(a_V)) + 2.0*a_Nu*eps
	traction  = dot(sig, a_Normal)
	forceX  = traction[0]*a_GammaP
	forceY  = traction[1]*a_GammaP
	fX      = assemble(forceX)
	fY      = assemble(forceY)
	return (fX, fY)
	
	
# Here go some input paramaters about the geometry
bmarg = 1.e-3 + DOLFIN_EPS
radius = 0.1 # radius of the cylinder
cyl_in = 8*radius*2 #Distance of the inlet respect to the center of the cylinder
cyl_out = 50*radius*2 #Distance of the outlet respect to the center of the cylinder
dom_wid = 20*radius*2 #Width of the domain
ref_in = 2*radius*2   #Size of the refinement box, size towards the inlet direction
ref_out = 10*radius*2  #Size of the refinement box, size towards the outlet direction
ref_w = 4*radius*2    #width of the refinement box
ymax = dom_wid/2
ymin = -dom_wid/2
xmax = cyl_out
xmin = -cyl_in

nref = 2 #number of refinement cycles
nref = range(0,nref)

res = 50 #resolution of the base mesh
n_circ= 100

# Generate mesh with a hole.
base = Rectangle(Point(-cyl_in, -dom_wid/2), Point(cyl_out, dom_wid/2))
hole = Circle(Point(0, 0), radius, n_circ)

# Mark hole in full mesh using mesh function.
mesh = generate_mesh(base-hole, res)

# Refine the mesh around the cylinder
for refinements in nref:
	cell_markers = MeshFunction("bool",mesh,mesh.topology().dim())
	cell_markers.set_all(False)
	for cell in cells(mesh):
		p = cell.midpoint()
		if  (-ref_in < p[0] < ref_out) and \
			(-ref_w < p[1] < ref_w) :
			cell_markers[cell] = True
	mesh = refine(mesh, cell_markers)

mark = {"generic": 0,
"cyl": 1,
"left": 2,
"right": 3 }
subdomains = MeshFunction("size_t", mesh, 1)
subdomains.set_all(mark["generic"])
cylinder = Cylinder()
cylinder.mark(subdomains, mark["cyl"])
ds = Measure("ds")[subdomains]
normal = FacetNormal(mesh)
GammaP = ds(mark["cyl"])

# More verbose output by SNES
parameters["linear_algebra_backend"] = "PETSc"
args = "--petsc.snes_linesearch_monitor --petsc.snes_linesearch_type bt"
parameters.parse(argv = argv[0:1] + args.split())

#     The aim of this script is to present you the implementation of an (XXX) unsteady Navier-Stokes problem

# Mesh import
#mesh = Mesh("mesh/Cylinder.xml")
#subdomains = MeshFunction("size_t", mesh, "mesh/Cylinder_physical_region.xml")
#boundaries = MeshFunction("size_t", mesh, "mesh/Cylinder_facet_region.xml")
#
## Mesh labels
#walls_ID = 1
#outlet_ID = 2
#inlet_ID = 3
#circle_ID = 4

# Constitutive parameters
Re = 10000.
u_bar = 1.
u_in = Expression(("(-u_bar/(ymax*ymax)*(x[1]*x[1])+u_bar)", "0."), t=0, u_bar=u_bar, degree=2, ymax=ymax) # XXX gradually increases
nu = Constant(u_bar*2*radius/Re) # obtained from the definition of Re = u_bar * diam / nu. In our case diam = 0.1.
f = Constant((0., 0.))

# XXX Time discretization
dt = 0.005
T = 30

# Function spaces
V_element = VectorElement("Lagrange", mesh.ufl_cell(), 2)
Q_element = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
W_element = MixedElement(V_element, Q_element)
W = FunctionSpace(mesh, W_element)

# Test and trial functions
vq                 = TestFunction(W) # Test function in the mixed space
delta_up           = TrialFunction(W) # Trial function in the mixed space (Note: for the increment!)
(delta_u, delta_p) = split(delta_up) # Function in each subspace to write the functional  (Note: for the increment!)
(v, q)             = split(      vq) # Test function in each subspace to write the functional

# Solution, obtained starting from the increment (XXX) at the current time
up = Function(W)
(u, p) = split(up)
# XXX Solution at the previous time
up_prev = Function(W)
(u_prev, _) = split(up_prev)


# Preparation of Variational forms.

K = JacobianInverse(mesh)
G = K.T*K
gg = (K[0,0] + K[1,0])**2 + (K[0,1] + K[1,1])**2

rm = (u-u_prev)/dt + grad(p) + grad(u)*u - nu*(u.dx(0).dx(0) + u.dx(1).dx(1)) - f
rc = div(u)


tm=(4*((dt)**(-2)) + 36*(nu**2)*inner(G,G) + inner(u,G*u))**(-0.5)
tc=(tm*gg)**(-1)

tcross = outer((tm*rm),(tm*rm))


# Variational forms.
F = (   inner(grad(v)*u+grad(q),tm*rm)*dx
        +inner(div(v),tc*rc)*dx
        +inner(grad(v).T*u,tm*rm)*dx
        -inner(grad(v),tcross)*dx
        + inner(v,(u-u_prev)/dt)*dx
      - inner(grad(v),outer(u,u))*dx 
      - inner(div(v),p)*dx + inner(q,div(u))*dx 
      + inner(sym(grad(v)),2*nu*sym(grad(u)))*dx  
       -inner(f,v)*dx  )
J = derivative(F, up, delta_up)

# Boundary conditions
# no-slip velocity b.c.
noslipBoundary = NoslipBoundary()
g0 = Constant( (0.,0.) )
bc0 = DirichletBC(W.sub(0), g0, noslipBoundary)
# inlet velocity b.c.
inflowBoundary = InflowBoundary()
bc1 = DirichletBC(W.sub(0), u_in, inflowBoundary)
# outflow pressure b.c.
class CenterDomain(SubDomain):
	def inside(self, x, on_boundary):
		return near(x[0], cyl_out, DOLFIN_EPS) and near(x[1], 0., DOLFIN_EPS)
center_domain = CenterDomain()
g2 = Constant(0.)
bc2 = DirichletBC(W.sub(1), g2, center_domain, method='pointwise')
# outflow velocity b.c., same as inlet
outflowBoundary = OutflowBoundary()
bc3 = DirichletBC(W.sub(0), u_in, outflowBoundary)
# collect b.c.
bcs = [bc0, bc1, bc2, bc3]
#walls_bc       = DirichletBC(W.sub(0), Constant((0., 0.)), boundaries, walls_ID )
#circle_bc      = DirichletBC(W.sub(0), Constant((0., 0.)), boundaries, circle_ID)
#inlet_bc       = DirichletBC(W.sub(0), u_in,               boundaries, inlet_ID )
#bc = [walls_bc, circle_bc, inlet_bc]

# Prepare nonlinear solver
snes_solver_parameters = {"nonlinear_solver": "snes",
	                      "snes_solver": {"linear_solver": "mumps",
	                                      "maximum_iterations": 20,
	                                      "report": True,
	                                      "error_on_nonconvergence": True}}
problem = NonlinearVariationalProblem(F, up, bcs, J)
solver  = NonlinearVariationalSolver(problem)
#solver.parameters.update(snes_solver_parameters)

# Export the initial solution (zero)
outfile_u = File("out_6/u.pvd")
outfile_p = File("out_6/p.pvd")
outfile_ld = File("out_6/ld.pvd")
out_file = open("test.txt","w")
(u, p) = up.split()
outfile_u << u
outfile_p << p

out_file.write("time\tDrag\t\tLift\n")


sav_ts = float (5) #time steps
# XXX Time loop
K = int(T/dt)
for i in range(1,K):
    # Compute the current time
    t = i*dt
    print("t =", t)
    # Update the time for the boundary condition
    u_in.t = t
    # Solve the nonlinear problem
    solver.solve()
    # Store the solution in up_prev
    assign(up_prev, up)
    # Plot
    (u, p) = up.split()
    if (i/sav_ts).is_integer():
        outfile_u << u
        outfile_p << p
	
    (fX,fY) = integrateFluidStress(u, p, nu, normal, GammaP)
    CD = fX/(0.5*u_bar**2*2*radius)
    CL = fY/(0.5*u_bar**2*2*radius)
    print("Drag=", CD, "Lift=", CL)
    out_file.write(str(t)+"\t"+str(CD)+"\t"+str(CL)+"\n")
    
out_file.close()
