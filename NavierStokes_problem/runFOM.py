from weak_wallfunction_2D import *
import rbnics 
import random
import matplotlib.pyplot as plt
random.seed(a=5)
markers = ["1","2","3","+","x","o","s","d","."]

class Parameter():
    def __init__(self,param_range):
        self.dim = len(param_range)
        self.param_range = param_range
    def generate_training_set(self, N=100, plot_folder = None):
        self.N_train = N
        self.training_set = []
        for i in range(self.N_train):
            unif_rand = [random.random() for k in range(self.dim)]
            new_param = [ self.param_range[k][0] +\
                 unif_rand[k]*\
                (self.param_range[k][1]-self.param_range[k][0]) \
                for k in range(self.dim) ]
            self.training_set.append(new_param)
        if plot_folder is not None:

            plt.figure()
            for i, param in enumerate(self.training_set):
                plt.plot(param[0], param[1], marker = markers[i%len(markers)], label=f"par {i}")
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
            plt.xlabel("u inlet")
            plt.ylabel("viscosity")
            plt.tight_layout()
            plt.savefig(plot_folder+"/training_set.pdf")
            plt.close()
    
    def generate_test_set(self, N=100):
        self.N_train = N
        self.test_set = []
        for i in range(self.N_train):
            unif_rand = np.random.rand(self.dim)
            new_param = [ self.param_range[k][0] +\
                 unif_rand[k]*\
                (self.param_range[k][1]-self.param_range[k][0]) \
                for k in range(self.dim) ]
            self.test_set.append(new_param)


param_set = Parameter(param_range)




class Snapshots():
    def __init__(self, param_range, N = 100, snap_folder = "FOM_snapshots", with_lifting = False ):
        self.training_set = Parameter(param_range)
        self.training_set.generate_training_set( N, plot_folder =  snap_folder )
        self.snap_folder = snap_folder
        self.with_lifting = with_lifting
        try:      
            os.mkdir(self.snap_folder)
        except:
            print("folder %s already exists"%self.snap_folder)
        
    def compute_snapshots(self):
        for i in range(self.training_set.N_train):
            param = self.training_set.training_set[i]
            simul_folder = self.snap_folder+"/param_%03d"%i
            solve_FOM(param, simul_folder)

    def read_snapshots(self):
        self.POD = dict()
        X_inner = generate_inner_products(W, V0)
        for comp in ("u","p"):
            self.POD[comp] = ProperOrthogonalDecomposition(W, X_inner[comp])

        Spaces = dict()
        Spaces["u"] = FunctionSpace(mesh, V_element)
        Spaces["p"] = FunctionSpace(mesh, Q_element)
        Spaces["tau"] = V0
        
        tmp = dict()
        for comp in ("u","p"):
            tmp[comp] = Function(Spaces[comp])
        if boundary_tag in ["spalding"]:
            comp = "tau"
            self.POD[comp] = ProperOrthogonalDecomposition(V0, X_inner[comp])
            Spaces[comp] = V0
            tmp[comp] = Function(Spaces[comp])

        up = Function(W)
        
        i=0
        param = self.training_set.training_set[0]
        simul_folder = self.snap_folder+"/param_%03d"%i


        print("Reading %s"%(simul_folder))
        print("for parameter ",param)
            
        inp_xdmf = dict()
        for comp in ("u","p"):
            inp_xdmf[comp] = XDMFFile(simul_folder+"/%s.xdmf"%comp)
        if boundary_tag in ["spalding"]:
            comp = "tau"
            inp_xdmf[comp] = XDMFFile(simul_folder+"/%s.xdmf"%comp)
        
        lift_xdmf = dict()
        for comp in ("u","p"):
            lift_xdmf[comp] = XDMFFile(self.snap_folder+"/lift_%s.xdmf"%comp)

            
            
        self.lift = Function(W)
        if self.with_lifting:
            it=100
            reading=True
            while reading:
                try:
                    inp_xdmf["u"].read_checkpoint(tmp["u"],"u",it)
                    lift_factor = get_lifting_factor(param)
                    assign(self.lift,[tmp["u"],tmp["p"]])
                    self.lift.assign(1./lift_factor*self.lift)
                    (u_lift, p_lift) = self.lift.split(deepcopy=True)
                    lift_xdmf["u"].write_checkpoint(u_lift, "u", 0, XDMFFile.Encoding.HDF5, append=False)
                    lift_xdmf["p"].write_checkpoint(p_lift, "p", 0, XDMFFile.Encoding.HDF5, append=False)
                    reading = False
                except:
                    it-=1
                if it==1:
                    print(f"no solution found for parameter {i}")
                    break

        up_lift = Function(W)
        for i in range(self.training_set.N_train):
            it=1
            param = self.training_set.training_set[i]
            simul_folder = self.snap_folder+"/param_%03d"%i
            print("Reading %s"%(simul_folder))
            print("for parameter ",param)
                
            inp_xdmf = dict()
            for comp in ("u","p"):
                inp_xdmf[comp] = XDMFFile(simul_folder+"/%s.xdmf"%comp)
            if boundary_tag in ["spalding"]:
                comp = "tau"
                inp_xdmf[comp] = XDMFFile(simul_folder+"/%s.xdmf"%comp)

            reading=True
            param = self.training_set.training_set[i]
            while reading:
                try:
                    for comp in ("u","p"):
                        inp_xdmf[comp].read_checkpoint(tmp[comp],comp,it)
                    assign(up,[tmp["u"],tmp["p"]])
                    for comp in ("u","p"):
                        up_lift.assign(self.lift * get_lifting_factor(param))
                        self.POD[comp].store_snapshot(up-up_lift)
                    if boundary_tag=="spalding":
                        comp = "tau"
                        inp_xdmf[comp].read_checkpoint(tmp[comp],comp,it)
                        self.POD["tau"].store_snapshot(tmp["tau"])
                except:
                    reading = False
                    print("Time %d does not exists of solution %s"%(it,simul_folder))
                it+=1

    def compute_POD(self, N_POD = 100, tol = 1e-15):
        self.POD_folder = self.snap_folder+"/POD"
        try: 
            os.mkdir(self.POD_folder)
        except:
            print("POD folder already exists")
        self.eigs = dict()
        self.Z_comp = dict()
        for ic, comp in enumerate(("u","p")):
            print("Computing the POD for component %s"%comp)
            self.eigs[comp],_,self.Z_comp[comp], N = self.POD[comp].apply(N_POD, tol=tol)
            #save bases
            outxdmf_RB = XDMFFile(self.POD_folder+"/POD_basis_%s.xdmf"%comp)
            for ib, basis in enumerate(self.Z_comp[comp]):
                basis_components = basis.split(deepcopy=True)
                if ib == 0:
                    outxdmf_RB.write_checkpoint(basis_components[ic], comp, ib, XDMFFile.Encoding.HDF5, append=False)
                else:
                    outxdmf_RB.write_checkpoint(basis_components[ic], comp, ib, XDMFFile.Encoding.HDF5, append=True)
            np.save(self.POD_folder+"/eigs_"+comp+".npy",self.eigs[comp])

            plt.figure()
            plt.semilogy(self.eigs[comp])
            plt.savefig(self.POD_folder+"/eigs_"+comp+".pdf")
            plt.close('all')


        if boundary_tag in ["spalding"]:
            comp="tau"
            print("Computing the POD for component %s"%comp)
            self.eigs[comp],_,self.Z_comp[comp], N = self.POD[comp].apply(N_POD, tol=tol)
            #save bases
            outxdmf_RB = XDMFFile(self.POD_folder+"/POD_basis_%s.xdmf"%comp)
            for ib, basis in enumerate(self.Z_comp[comp]):
                if ib == 0:
                    outxdmf_RB.write_checkpoint(basis, comp, ib, XDMFFile.Encoding.HDF5, append=False)
                else:
                    outxdmf_RB.write_checkpoint(basis, comp, ib, XDMFFile.Encoding.HDF5, append=True)
            np.save(self.POD_folder+"/eigs_"+comp+".npy",self.eigs[comp])
        
            plt.figure()
            plt.semilogy(self.eigs[comp])
            plt.savefig(self.POD_folder+"/eigs_"+comp+".pdf")
            plt.close('all')
        
        # convert RB into matrix
        self.RB_mat = BasisFunctionsMatrix(W)
        self.RB_mat.init(["u","p"])
        self.RB_mat.enrich(self.Z_comp["u"],component="u")
        self.RB_mat.enrich(self.Z_comp["p"],component="p")
        self.RB_mat.save(self.POD_folder, "POD_up")

        if boundary_tag =="spalding":
            comp ="tau"
            self.RB_mat_tau = BasisFunctionsMatrix(V0)
            self.RB_mat_tau.init([comp])
            self.RB_mat_tau.enrich(self.Z_comp[comp])
            self.RB_mat_tau.save(self.POD_folder, "POD_tau")

    def load_RB(self, NPOD_max = None):
        """NPOD_max is a list of maximum mode we want to
        retain for each component
        """
        self.POD_folder = self.snap_folder+"/POD"
        self.eigs = dict()
        self.Z_comp = dict()
        self.N_POD = dict()
        for ic, comp in enumerate(("u","p")):
            print("Read the POD bases for component %s"%comp)
            self.eigs[comp]=np.load(self.POD_folder+"/eigs_"+comp+".npy")
            self.N_POD[comp] = len(self.eigs[comp])
            if NPOD_max is not None:
                self.N_POD[comp] = min(self.N_POD[comp], NPOD_max[ic])
            self.Z_comp[comp] = rbnics.backends.dolfin.functions_list.FunctionsList(W)
            outxdmf_RB = XDMFFile(self.POD_folder+"/POD_basis_%s.xdmf"%comp)
            for i in range(self.N_POD[comp]):
                up_tmp = up.split(deepcopy=True)
                outxdmf_RB.read_checkpoint(up_tmp[ic],comp,i)
                assign(up, [up_tmp[0],up_tmp[1]])
                self.Z_comp[comp].enrich(up)
        
            plt.figure()
            plt.semilogy(self.eigs[comp])
            plt.savefig(self.snap_folder+"/eigs_"+comp+".pdf")
            plt.close('all')

        if boundary_tag in ["spalding"]:
            comp = "tau"
            print("Read the POD bases for component %s"%comp)
            self.eigs[comp]=np.load(self.POD_folder+"/eigs_"+comp+".npy")
            self.N_POD[comp] = len(self.eigs[comp])
            self.Z_comp[comp] = rbnics.backends.dolfin.functions_list.FunctionsList(V0)
            outxdmf_RB = XDMFFile(self.POD_folder+"/POD_basis_%s.xdmf"%comp)
            for ib in range(self.N_POD[comp]):
                outxdmf_RB.read_checkpoint(tau_penalty,comp,ib)
                self.Z_comp[comp].enrich(tau_penalty)

            plt.figure()
            plt.semilogy(self.eigs[comp])
            plt.savefig(self.snap_folder+"/eigs_"+comp+".pdf")
            plt.close('all')

        # convert RB into matrix
        self.RB_mat = BasisFunctionsMatrix(W)
        self.RB_mat.init(["u","p"])
        self.RB_mat.enrich(self.Z_comp["u"],component="u")
        self.RB_mat.enrich(self.Z_comp["p"],component="p")
        self.RB_mat.save(self.POD_folder, "POD_up")

        if boundary_tag =="spalding":
            comp ="tau"
            self.RB_mat_tau = BasisFunctionsMatrix(V0)
            self.RB_mat_tau.init([comp])
            self.RB_mat_tau.enrich(self.Z_comp[comp])
            self.RB_mat_tau.save(self.POD_folder, "POD_tau")
        if self.with_lifting:
            lift_xdmf = dict()
            for comp in ("u","p"):
                lift_xdmf[comp] = XDMFFile(self.snap_folder+"/lift_%s.xdmf"%comp)
            
            self.lift = Function(W)
            (u_tmp, p_tmp) = self.lift.split(deepcopy=True)
            lift_xdmf["u"].read_checkpoint(u_tmp ,"u",0)
            lift_xdmf["p"].read_checkpoint(p_tmp ,"p",0)
            assign(self.lift, [u_tmp, p_tmp])



    def project_snapshots(self):
        # Projecting onto RB space all training snapshots and saving 
        # coefficients and parameters for learning phase
        self.all_inputs = []  # time, params
        self.all_outputs = dict()
        for comp in self.Z_comp.keys():
            self.all_outputs[comp] = []

        for i in range(self.training_set.N_train):
            try:
                param = self.training_set.training_set[i]
                simul_folder = self.snap_folder+"/param_%03d"%i
                up_lift = Function(W)
                up_lift.assign(get_lifting_factor(param)*self.lift)
                if boundary_tag =="spalding":
                    times_plot, RB_coef, errors = read_FOM_and_project(simul_folder, self.Z_comp, RB_tau = self.RB_mat_tau, u_lift = up_lift, with_plot=True)
                else:
                    times_plot, RB_coef, errors = read_FOM_and_project(simul_folder, self.Z_comp, u_lift = up_lift, with_plot=True)
                for it, time in enumerate(times_plot):
                    tmp = np.zeros(len(param)+1)
                    tmp[0] = time
                    tmp[1:] = param
                    self.all_inputs.append(tmp)
                    for comp in self.Z_comp.keys():
                        self.all_outputs[comp].append(RB_coef[comp][it])
            except:
                print(f"Parameter {i} not computed")
        np.save(self.snap_folder+"/training_input.npy",self.all_inputs)
        for comp in self.Z_comp.keys():
            np.save(self.snap_folder+f"/training_output_{comp}.npy",self.all_outputs[comp])

    # def compute_tau_errors(self):


def get_lifting_factor(param):
    return param[0]

    
#u_top_val = 5.1
#nu_val = 1e-3

# solve_FOM([u_top_val,nu_val], out_folder+"/param_trial", with_plot=True)

snapshots = Snapshots(param_range, N=10, with_lifting=True, snap_folder =out_folder)
#snapshots.compute_snapshots()

# param = snapshots.training_set.training_set[0]
# u_top_val = param[0]
# nu_val = param[1]

# solve_FOM([u_top_val,nu_val], out_folder+"/param_trial")


# snapshots.read_snapshots()
# snapshots.compute_POD(N_POD = 30)

snapshots.load_RB()#([10,10])
snapshots.project_snapshots()


# up_lift = Function(W)
# param = [u_top_val,nu_val]
# lift_factor = get_lifting_factor(param)
# up_lift.assign(lift_factor*snapshots.lift)

# if boundary_tag=="spalding":
#     solve_FOM(param, out_folder+"/param_trial_RB_proj", \
#             RB=snapshots.Z_comp, RB_tau = snapshots.RB_mat_tau, with_plot=True, u_lift = up_lift)

#     solve_POD_Galerkin(param, out_folder+"/param_trial_RB", \
#             snapshots.Z_comp, RB_tau = snapshots.RB_mat_tau, with_plot=True, u_lift = up_lift, FOM_comparison= True)
# else:
#     solve_FOM(param, out_folder+"/param_trial_RB_proj", \
#             RB=snapshots.Z_comp, with_plot=True, u_lift = up_lift)

#     solve_POD_Galerkin(param, out_folder+"/param_trial_RB", \
#             snapshots.Z_comp, with_plot=True, u_lift = up_lift, FOM_comparison= True)

# param = snapshots.training_set.training_set[5]
# up_lift = Function(W)
# lift_factor = get_lifting_factor(param)
# up_lift.assign(lift_factor*snapshots.lift)
# times_plot, RB_coef, errors = read_FOM_and_project(out_folder+"/param_005", snapshots.Z_comp,\
#                                 u_lift = up_lift, with_plot=True)
#stop()