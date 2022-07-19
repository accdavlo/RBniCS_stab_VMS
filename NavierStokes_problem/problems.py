from dolfin import *
from ufl.geometry import *
from dolfin.cpp.mesh import *
from mshr import *

# Print log messages only from the root process in parallel
parameters["std_out_all_processes"] = False;

class Problem:
    def __init__(self, name, Nx=50):
        self.name = name
        self.Nx = Nx
        self.space_dim = 2
        self.setup_mesh()

    def setup_mesh(self):
        if self.name == "lid-driven_cavity":
            # Create mesh
            self.domain = Rectangle(Point(0., 0.), Point(1., 1.))

            self.mesh = generate_mesh(self.domain,self.Nx)
        elif self.name == "cylinder":
            # Create mesh
            channel = Rectangle(Point(0, 0), Point(2.2, 0.41))
            self.cylinder_diam = 0.1
            cylinder = Circle(Point(0.2, 0.2), self.cylinder_diam/2.)
            self.domain = channel - cylinder

            self.mesh = generate_mesh(self.domain,self.Nx)

    def define_bc(self,W, u_in = None):
        if self.name == "lid-driven_cavity":
            if u_in is None:
                u_in = Constant(1.)
            # Define boundary conditions
            noslip  = DirichletBC(W.sub(0), (0, 0),
                                  "on_boundary && \
                                   (x[0] < DOLFIN_EPS | x[1] < DOLFIN_EPS | \
                                    x[0] > 1.0 - DOLFIN_EPS)")
            inflow  = DirichletBC(W.sub(0), (u_in, 0), "x[1] > 1.0 - DOLFIN_EPS")
            #outflow = DirichletBC(Q, 0, "x[0] > 1.0 - DOLFIN_EPS")

            class CenterDomain(SubDomain):
                def inside(self, x, on_boundary):
                    return near(x[0], 0.5, DOLFIN_EPS) and near(x[1], 0.5, DOLFIN_EPS)
            center_domain = CenterDomain()

            g2 = Constant(0.)
            bc_one_point = DirichletBC(W.sub(1), g2, center_domain, method='pointwise')


            self.bcs = [inflow, noslip, bc_one_point]
            return self.bcs
        elif self.name == "cylinder":
            if u_in is None:
                u_in = Constant(500.)
            #boundaries
            inflow   = 'near(x[0], 0)'
            outflow  = 'near(x[0], 2.2)'
            walls    = 'near(x[1], 0) || near(x[1], 0.41)'
            cylinder = 'on_boundary && x[0]>0.1 && x[0]<0.3 && x[1]>0.1 && x[1]<0.3'

            # Define inflow profile
            inflow_profile = ('u_in*4.0*1.5*x[1]*(0.41 - x[1]) / pow(0.41, 2)', '0')

            bcu_inflow = DirichletBC(W.sub(0), Expression(inflow_profile, degree=2,u_in=u_in), inflow)
            bcu_walls = DirichletBC(W.sub(0), Constant((0, 0)), walls)
            bcu_cylinder = DirichletBC(W.sub(0), Constant((0, 0)), cylinder)
            bcp_outflow = DirichletBC(W.sub(1), Constant(0), outflow)
            bcu = [bcu_inflow, bcu_walls, bcu_cylinder]
            bcp = [bcp_outflow]

            self.bcs = [bcu_inflow, bcu_walls, bcu_cylinder,bcp_outflow]
            return self.bcs

    def get_reynolds(self,u,nu):
        if self.name=="lid-driven_cavity":
            return u/nu
        elif self.name=="cylinder":
            return u/nu * self.cylinder_diam

    def get_viscosity(self,u,Re):
        if self.name=="lid-driven_cavity":
            return u/Re
        elif self.name=="cylinder":
            return u/Re * self.cylinder_diam