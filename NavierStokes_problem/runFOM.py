from weak_wallfunction_2D import *
import rbnics 
import random
random.seed(a=5)


class Parameter():
    def __init__(self,param_range):
        self.dim = len(param_range)
        self.param_range = param_range
    def generate_training_set(self, N=100):
        self.N_train = N
        self.training_set = []
        for i in range(self.N_train):
            unif_rand = [random.random() for k in range(self.dim)]
            new_param = [ self.param_range[k][0] +\
                 unif_rand[k]*\
                (self.param_range[k][1]-self.param_range[k][0]) \
                for k in range(self.dim) ]
            self.training_set.append(new_param)
    
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
    def __init__(self, param_range, N = 100, snap_folder = "FOM_snapshots" ):
        self.training_set = Parameter(param_range)
        self.training_set.generate_training_set( N )
        self.snap_folder = snap_folder
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
        for i in range(self.training_set.N_train):
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

            i=0
            reading=True
            while reading:
                try:
                    for comp in ("u","p"):
                        inp_xdmf[comp].read_checkpoint(tmp[comp],comp,i)
                    assign(up,[tmp["u"],tmp["p"]])
                    for comp in ("u","p"):
                        self.POD[comp].store_snapshot(up)
                    if boundary_tag=="spalding":
                        self.POD["tau"].store_snapshot(tmp["tau"])
                except:
                    reading = False
                    print("Time %d does not exists of solution %s"%(i,simul_folder))
                i+=1

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
            print("Computing the POD for component %s"%comp)
            comp="tau"
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

    def load_RB(self):
        self.POD_folder = self.snap_folder+"/POD"
        self.eigs = dict()
        self.Z_comp = dict()
        self.N_POD = dict()
        for ic, comp in enumerate(("u","p")):
            print("Read the POD bases for component %s"%comp)
            self.eigs[comp]=np.load(self.POD_folder+"/eigs_"+comp+".npy")
            self.N_POD[comp] = len(self.eigs[comp])
            self.Z_comp[comp] = rbnics.backends.dolfin.functions_list.FunctionsList(W)
            outxdmf_RB = XDMFFile(self.snap_folder+"/POD_basis_%s.xdmf"%comp)
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
            print("Read the POD bases for component %s"%comp)
            self.eigs[comp]=np.load(self.POD_folder+"/eigs_"+comp+".npy")
            self.N_POD[comp] = len(self.eigs[comp])
            self.Z_comp[comp] = rbnics.backends.dolfin.functions_list.FunctionsList(V0)
            outxdmf_RB = XDMFFile(self.snap_folder+"/POD_basis_%s.xdmf"%comp)
            for ib, basis in enumerate(self.Z_comp[comp]):
                outxdmf_RB.read_checkpoint(tau_penalty,comp,i)
                self.Z_comp[comp].enrich(tau_penalty)

            plt.figure()
            plt.semilogy(self.eigs[comp])
            plt.savefig(self.snap_folder+"/eigs_"+comp+".pdf")
            plt.close('all')


        # with open(self.snap_folder+"/RB_basis.pickle", "wb") as ff:
        #     pickle.dump([self.eigs, self.Z_comp], ff)

    # def load_RB(self):
    #     with open(self.snap_folder+"/RB_basis.pickle","rb") as ff:
    #         RB_str = pickle.load(ff)
    #     self.eigs = RB_str[0] 
    #     self.Z_comp = RB_str[1] 

    #     param = self.training_set.training_set[0]
    #     param_problem  = self.problem(param)
    #     self.PODs = []
    #     for i in range(param_problem.sys_dim):
    #         self.PODs.append( ProperOrthogonalDecomposition(W, X) )
                
    #     for i in range(self.training_set.N_train):
    #         param_problem  = self.problem(param)
    #         simul_folder = self.snap_folder+"/n_%03d"%i
    #         param_problem.compute_numerical_solution(V, mesh= mesh, folder =simul_folder, skip_if_computed=skip_simul_if_computed )
    #         param_problem.load_all_solutions(self.PODs, simul_folder)


#u_top_val = 5.1
#nu_val = 1e-3

#solve_FOM([u_top_val,nu_val], out_folder+"/param_trial", with_plot=True)

snapshots = Snapshots(param_range, N=20, snap_folder =out_folder)
snapshots.compute_snapshots()

#param = snapshots.training_set.training_set[0]
#u_top_val = param[0]
#nu_val = param[1]

#solve_FOM([u_top_val,nu_val], out_folder+"/param_trial")


#snapshots.read_snapshots()
#snapshots.compute_POD(N_POD = 30)

# snapshots.load_RB()

# solve_FOM([u_top_val,nu_val], out_folder+"/param_trial_RB_proj", \
#           RB=snapshots.Z_comp, with_plot=True)

#stop()