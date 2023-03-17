from dolfin import *
from ufl.geometry import *
from dolfin.cpp.mesh import *
from mshr import *
import numpy as np

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
        elif "cylinder" in self.name:
            # Create mesh
            self.mesh_name = "Cylinder"

            self.x_left = 0.
            self.x_right = 2.2
            self.y_bottom = -0.41
            self.y_top = 0.41
            self.x_cylinder_center = 0.2
            self.y_cylinder_center = 0.0
            
            self.cylinder_diam = 0.1

            channel = Rectangle(Point(self.x_left, self.y_bottom),\
                                Point(self.x_right, self.y_top))
            cylinder = Circle(Point(self.x_cylinder_center, self.y_cylinder_center), \
                              self.cylinder_diam/2., 40)
            self.domain = channel - cylinder

            self.mesh = generate_mesh(self.domain,self.Nx)


    def define_boundaries(self, W):
        if "cylinder" in self.name:
            PD = self

            # Create boundaries
            class Walls(SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary and \
                        (near(x[1], PD.y_bottom) or near(x[1], PD.y_top))
                    
            class Outflow(SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary and near(x[0], PD.x_right)
                    
            class Inflow(SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary and near(x[0], PD.x_left)

            class Cylinder(SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary and \
                        (x[0]>PD.x_cylinder_center-PD.cylinder_diam\
                         and x[0]<PD.x_cylinder_center+PD.cylinder_diam\
                         and x[1]>PD.y_cylinder_center-PD.cylinder_diam\
                         and x[1]<PD.y_cylinder_center+PD.cylinder_diam)

            dist = np.inf
            for i in range(self.mesh.num_vertices()):
                if near(self.mesh.coordinates()[i][0],self.x_right):
                    dist_tmp = np.abs(self.mesh.coordinates()[i][1]-self.y_cylinder_center)
                    if dist_tmp<dist:
                        center_y = self.mesh.coordinates()[i][1]
                        dist = dist_tmp

            class CenterDomain(SubDomain):
                def inside(self, x, on_boundary):
                    return near(x[0], PD.x_right, DOLFIN_EPS) and near(x[1], center_y, DOLFIN_EPS)


            class AllBoundary(SubDomain):
                def inside(self, x, on_boundary):
                    return on_boundary                        
            # Create subdomains
            self.subdomains = MeshFunction("size_t", self.mesh, 2)
            self.subdomains.set_all(0)


            self.boundaries = MeshFunction("size_t", self.mesh, self.mesh.topology().dim()-1, 0)
            self.boundaries.set_all(0)
            self.walls_ID = 1
            self.walls = Walls()
            self.walls.mark(self.boundaries, self.walls_ID)
            self.cylinder = Cylinder()
            self.cylinder.mark(self.boundaries, self.walls_ID)

            self.outflow_ID = 2
            self.outflow = Outflow()
            self.outflow.mark(self.boundaries, self.outflow_ID)
            self.inflow_ID = 3
            self.inflow = Inflow()
            self.inflow.mark(self.boundaries, self.inflow_ID)

            self.center_domain_ID = 4
            self.center_domain = CenterDomain()
            self.center_domain.mark(self.boundaries, self.center_domain_ID)

            self.bmesh = BoundaryMesh(self.mesh, 'exterior')

            self.ds_bc = Measure('ds', domain=self.mesh, \
                subdomain_data=self.boundaries, \
                subdomain_id=self.walls_ID, \
                metadata = {'quadrature_degree': 3})

    def define_bc(self,W, u_in = None):
        if self.name == "lid-driven_cavity":
            if u_in is None:
                self.u_in = Constant(1.)
            else:
                self.u_in = u_in

            # Define boundary conditions
            noslip  = DirichletBC(W.sub(0), (0, 0),
                                  "on_boundary && \
                                   (x[0] < DOLFIN_EPS | x[1] < DOLFIN_EPS | \
                                    x[0] > 1.0 - DOLFIN_EPS)")
            inflow  = DirichletBC(W.sub(0), (self.u_in, 0), "x[1] > 1.0 - DOLFIN_EPS")
            #outflow = DirichletBC(Q, 0, "x[0] > 1.0 - DOLFIN_EPS")

            class CenterDomain(SubDomain):
                def inside(self, x, on_boundary):
                    return near(x[0], 0.5, DOLFIN_EPS) and near(x[1], 0.5, DOLFIN_EPS)
            center_domain = CenterDomain()

            g2 = Constant(0.)
            bc_one_point = DirichletBC(W.sub(1), g2, center_domain, method='pointwise')


            self.bcs = [inflow, noslip, bc_one_point]
            return self.bcs
        elif "cylinder" in self.name:
            if u_in is None:
                self.u_in = Constant(1.)
            else:
                self.u_in = u_in

            # # Define inflow profile
            # inflow_profile = ('u_in*4.0*(x[1] %+g)*(%g - x[1]) / pow(%g, 2)'%(\
            #                     -self.y_bottom, self.y_top, self.y_top-self.y_bottom), '0')

            # Define inflow profile
            inflow_profile = ('u_in', '0')



            self.bcu_inflow = DirichletBC(W.sub(0), Expression(inflow_profile, degree=2,u_in=self.u_in),\
                 self.boundaries, self.inflow_ID)
            self.bcu_walls = DirichletBC(W.sub(0), Constant((0, 0)), self.boundaries, self.walls_ID)
            self.bcp_outflow  = DirichletBC(W.sub(1), Constant(0), self.boundaries, self.outflow_ID)
            self.bcp_onePoint = DirichletBC(W.sub(1), Constant(0), self.center_domain, method='pointwise')

            self.bcu = [self.bcu_inflow, self.bcu_walls]
#            self.bcp = [self.bcp_outflow]
            self.bcp = [self.bcp_onePoint]

#            self.bc_no_walls = [self.bcu_inflow, self.bcp_outflow]
            self.bc_no_walls = [self.bcu_inflow, self.bcp_onePoint]
            self.bc_walls = [self.bcu_walls]

            #self.bcs = [self.bcu_inflow, self.bcu_walls, self.bcp_outflow]
            self.bcs = [self.bcu_inflow, self.bcu_walls, self.bcp_onePoint]
            return self.bcs

    def get_IC(self):
        if self.name in ["lid-driven_cavity", "cylinder", "cylinder_turb"]:
            return Constant((0.0,0.0)) , Constant(0.)


    def get_reynolds(self,u,nu):
        if self.name=="lid-driven_cavity":
            return u/nu
        elif "cylinder" in self.name:
            return u/nu * self.cylinder_diam

    def get_viscosity(self,u,Re):
        if self.name=="lid-driven_cavity":
            return u/Re
        elif "cylinder" in self.name:
            return u/Re * self.cylinder_diam