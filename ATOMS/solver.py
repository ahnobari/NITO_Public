import logging

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import reverse_cuthill_mckee
from scipy.sparse.linalg import bicgstab, cg, splu
from scipy.spatial import KDTree
from sksparse.cholmod import cholesky
from tqdm import tqdm, trange

from .geometry import generate_structured_mesh
from .MaterialModels import SingleMaterial
from .stiffness import auto_stiffness
from .utils import kernels2D, kernels3D, kernels2D_nf, kernels3D_nf
from .vis import plot_field, plot_mesh, plot_problem

logger = logging.getLogger(__name__)

try:
    import cupy as cp
    from cupyx.scipy.sparse.linalg import cg as cg_gpu
except ImportError:
    logger.warning("Cannot import cupy, GPU acceleration will not be available.")
    cp = None
    cg_gpu = None


class Solver:
    def __init__(
        self,
        mesh=None,
        domain=None,
        material_model=SingleMaterial(),
        filter_size=None,
        max_iter=500,
        move=0.2,
        ch_tol=1e-4,
        fun_tol=1e-6,
        structured=False,
        reordering=True,
        solver="cholesky",
        solver_max_iter=500,
        solve_tol=1e-5,
        solve_min_tol = 1e-3,
        min_tol_patience = 10,
        abandon_patience = 10,
        compute_engine="cpu",
        filter_kernel=None,
        **kwargs,
    ):
        """
        This class initializes the Topology Optimization solver object.

        Parameters:
            mesh (tuple): Tuple with two numpy arrays, the first array is the nodal positions and the second array is the elements. If not provided, domain must be provided.
            domain (np.array): Array with the number of elements in each dimension. If not provided, mesh must be provided.
            volume_fraction (float): Volume fraction constraint for TO.
            E (float or np.array): Base Young's Modulus for building stiffness matrices. Default is 1.0.
            Nu (float): Poisson's Ratio of the material. Default is 0.33.
            filter_size (float): Size of the filter to be used in the optimization. If not provided, it is set to 1.5 times the mean element size.
            penalty (float): Penalty factor for the SIMP method.
            penalty_start (float): Initial penalty factor for the SIMP method.
            penalty_steps (int): Number of steps to reach the final penalty factor.
            penalty_growth_stage (float): Fraction of max_iter steps to reach the final penalty factor.
            max_iter (int): Maximum number of iterations for the optimization.
            void (float): Void material Youngs Modulus value.
            move (float): Move limit for the design update.
            ch_tol (float): Change tolerance for the optimization.
            fun_tol (float): Function tolerance for the optimization.
            gradual_pen (bool): If True, the penalty factor will increase gradually.
            structured (bool): If True, the mesh is assumed to be structured with identical elements (both node ordering and shape/size are assumed to be the same, only the first element in the mesh is processed).
            reordering (bool): If True, the reordering of the stiffness matrix will be performed.
            solver (str): Solver to be used in the optimization. Options are 'cholesky','splu','cg','bicgstab','gpu', 'gpuchol'.
            solver_max_iter (int): Maximum number of iterations for the solver at each iteration of optimization (used only if iterative solver is passed).
            solve_tol (float): Tolerance for the inner solver. Default is 1e-4.
            compute_engine (str): Compute engine to be used for the optimization. Options are 'cpu' and 'gpu'.
        """

        # verify compute engine
        if compute_engine == "gpu" and not "gpu" in solver:
            raise Exception(
                "Compute engine is set to GPU but solver is not set to GPU, please set solver to 'gpu'."
            )
        self.compute_engine = compute_engine

        if compute_engine != "gpu" and "gpu" in solver:
            logger.warning(
                "Compute engine is set to CPU, while solver is set to GPU. The performance benefits of GPU will be limited by CPU/GPU communication and stiffness assembly will be accelerated on GPU as well. This option is left to allow for GPU solve on very large meshes where kernels cannot fit in GPU but solving on CPU is not practical."
            )

        # verify and set geometry
        if mesh is None and domain is None:
            raise Exception("Either mesh or domain must be provided")

        if mesh is not None and domain is not None:
            raise Exception("Only one of mesh or domain must be provided")

        self.structured = structured

        if mesh is not None:
            self.mesh = mesh
        else:
            domain = np.array(domain)
            self.mesh = generate_structured_mesh(domain / domain.max(), domain)
            self.structured = True
        
        # Clean Up Mesh To Remove Redundant Nodes
        logger.info("Checking Mesh ...")
        useful_idx = np.unique(np.concatenate(mesh[1]))
        if useful_idx.shape[0] != mesh[0].shape[0]:
            logger.info("Mesh has redundant nodes. Cleaning up ...")
            mapping = np.arange(mesh[0].shape[0])
            mapping = mapping[useful_idx]
            sorter = np.argsort(mapping)
            e_ids = np.searchsorted(mapping, mesh[1].reshape(-1), sorter=sorter)
            e_ids = e_ids.reshape(mesh[1].shape[0], -1)
            mesh = (mesh[0][useful_idx], e_ids)
        logger.info("Mesh Cleaned!")

        self.node_positions, self.elements = mesh
        self.dim = self.node_positions.shape[1]
        self.n_elements = len(self.elements)

        # verify and set material properties
        self.material_model = material_model
        props = self.material_model.base_properties()
        self.E = props["E"]
        self.Nu = props["nu"]

        # verify and set optimization parameters
        self.filter_size = filter_size
        self.move = move

        self.max_iter = max_iter
        self.ch_tol = ch_tol
        self.fun_tol = fun_tol
        self.reordering = reordering

        if solver not in ["cholesky", "splu", "cg", "bicgstab", "gpu"]:
            raise Exception(
                "Solver not supported, please choose from 'cholesky','splu','cg','gpu'"
            )

        if self.n_elements > 1000000 and (solver == "cholesky" or solver == "splu"):
            raise Exception(
                "Cholesky and splu solvers are not supporter for large problems (higher than 0.5M elements), use and interative solver either 'cg' or 'bicgstab' or 'gpu' instead. Note that 'gpu' solver is memory hungary and requires high end GPU."
            )

        if solver in ["splu", "cholesky"] and self.reordering:
            self.reordering = False
            logger.info(
                "Reordering is disabled for cholesky and splu solvers as solvers handle reordering internally."
            )

        self.solver = solver
        self.solver_max_iter = solver_max_iter
        self.solve_tol = solve_tol
        self.solve_min_tol = solve_min_tol
        self.min_tol_patience = min_tol_patience
        self.abandon_patience = abandon_patience

        # setup stiffness matrix
        self.Ks = []
        self.As = []
        self.Ds = []
        self.Bs = []

        self.tris = []
        self.quads = []
        self.tets = []
        self.hexas = []

        logger.info("Setting up Stiffness Matrix ...")
        for i, e in enumerate(
            tqdm(self.elements, disable=logger.getEffectiveLevel() > logging.INFO)
        ):
            # check if E is a numpy array
            if not structured or i == 0:
                K, D, B, A = auto_stiffness(
                    self.node_positions[self.elements[i]], self.E, self.Nu
                )

            if self.dim == 2:
                if len(e) == 3:
                    self.tris.append(i)
                else:
                    self.quads.append(i)
            else:
                if len(e) == 4:
                    self.tets.append(i)
                else:
                    self.hexas.append(i)

            self.Ks.append(K)
            self.Ds.append(D)
            self.Bs.append(B)
            self.As.append(A)
        logger.info("Stiffness Matrix Set!")

        self.dG = np.array(self.As)

        if self.filter_size is None and filter_kernel is None:
            self.filter_size = self.dG.mean() ** (1 / self.dim) * np.ceil(np.sqrt(self.dim)*10)/10
            logger.info(
                f"Filter Size not provided, setting it to {self.filter_size}, based on the mean element size (apprximatly 1.5, exactly 1.5 if quad or hex in stuctured mesh)."
            )

        logger.info("Building Kernels ...")
        self.filter_kernel = filter_kernel
        if self.dim == 2:
            if filter_kernel is None:
                (
                    self.K_kernel,
                    self.k_map,
                    self.dk,
                    self.grad_map,
                    self.filter_kernel,
                    self.element_centroids,
                ) = kernels2D(self.elements, self.Ks, self.node_positions, self.filter_size)
            else:
                if self.filter_kernel.shape[0] != self.n_elements or self.filter_kernel.shape[1] != self.n_elements:
                    raise Exception("Filter Kernel must be of shape (n_elements, n_elements)")
                (
                    self.K_kernel,
                    self.k_map,
                    self.dk,
                    self.grad_map,
                    self.element_centroids,
                ) = kernels2D_nf(
                    self.elements, self.Ks, self.node_positions
                )
        else:
            if self.filter_kernel is None:
                (
                    self.K_kernel,
                    self.k_map,
                    self.dk,
                    self.grad_map,
                    self.filter_kernel,
                    self.element_centroids,
                ) = kernels3D(self.elements, self.Ks, self.node_positions, self.filter_size)
            else:
                if self.filter_kernel.shape[0] != self.n_elements or self.filter_kernel.shape[1] != self.n_elements:
                    raise Exception("Filter Kernel must be of shape (n_elements, n_elements)")
                (
                    self.K_kernel,
                    self.k_map,
                    self.dk,
                    self.grad_map,
                    self.element_centroids,
                ) = kernels3D_nf(
                    self.elements, self.Ks, self.node_positions
                )

        logger.info("Kernels Built!")

        # Set up boundary conditions
        self.c = np.zeros([self.node_positions.shape[0], self.dim])
        self.f = np.zeros([self.node_positions.shape[0], self.dim])
        self.non_constrained_map = np.where(self.c.reshape(-1) == 0)[0]

        self.KDTree = None

    def reset_BC(self):
        """
        This function resets the boundary conditions to the initial state.
        """
        self.c = np.zeros_like(self.c)
        self.non_constrained_map = np.where(self.c.reshape(-1) == 0)[0]
        self.F = self.f.reshape(-1)[self.non_constrained_map]
        self.dK = self.dk[:, self.non_constrained_map]

    def reset_F(self):
        """
        This function resets the forces to the initial state.
        """
        self.f = np.zeros_like(self.f)
        self.F = self.f.reshape(-1)[self.non_constrained_map]

    def add_BCs(self, positions, BCs):
        """
        This function adds boundary conditions to the solver.

        Parameters:
            positions (np.array): Array with the nodal positions where the BCs are to be applied.
            BCs (np.array): Array with the BCs to be applied 0 on dimensions not to be constrained and 1 for constrained directions. Shape (n_nodes, dim)
        """
        if self.KDTree is None:
            self.KDTree = KDTree(self.node_positions)
        _, idx = self.KDTree.query(positions)
        self.c[idx] += BCs
        self.non_constrained_map = np.where(self.c.reshape(-1) == 0)[0]
        self.F = self.f.reshape(-1)[self.non_constrained_map]
        self.dK = self.dk[:, self.non_constrained_map]

    def add_Forces(self, positions, Fs):
        """
        This function adds forces to the solver.

        Parameters:
            positions (np.array): Array with the nodal positions where the forces are to be applied.
            Fs (np.array): Array with the forces to be applied. Shape (n_nodes, dim)
        """
        if self.KDTree is None:
            self.KDTree = KDTree(self.node_positions)
        _, idx = self.KDTree.query(positions)
        self.f[idx] += Fs
        self.F = self.f.reshape(-1)[self.non_constrained_map]

    def add_BC_nodal(self, node_ids, BCs):
        """
        This function adds boundary conditions to the solver.

        Parameters:
            node_ids (np.array): Array with the nodal indecies where the BCs are to be applied.
            BCs (np.array): Array with the BCs to be applied 0 on dimensions not to be constrained and 1 for constrained directions. Shape (n_nodes, dim)
        """
        self.c[node_ids] = BCs
        self.non_constrained_map = np.where(self.c.reshape(-1) == 0)[0]
        self.F = self.f.reshape(-1)[self.non_constrained_map]
        self.dK = self.dk[:, self.non_constrained_map]

    def add_F_nodal(self, node_ids, Fs):
        """
        This function adds forces to the solver.

        Parameters:
            node_ids (np.array): Array with the nodal indecies where the forces are to be applied.
            Fs (np.array): Array with the forces to be applied. Shape (n_nodes, dim)
        """
        self.f[node_ids] = Fs
        self.F = self.f.reshape(-1)[self.non_constrained_map]

    def plot_mesh(self, **kwargs):
        """
        This function plots the mesh.

        Parameters:
            plotter (pyvista.Plotter or matplotlib.pyplot): Plotter to be used for the plot. For 3D pyvista plotter and for 2D matplotlib axes. If not provided, a new plotter will be created.
            face_color (str): Color of the faces in the plot.
            edge_color (str): Color of the edges in the plot.
            x_color (str): Color of the x constraint symbol.
            y_color (str): Color of the y constraint symbol.
            z_color (str): Color of the z constraint symbol.
            force_color (str): Color of the force symbol.
        """
        return plot_mesh(self.node_positions, self.elements, **kwargs)

    def plot_field(self, field, rho=None, **kwargs):
        """
        This function plots the field.

        Parameters:
            field (np.array): Field to be plotted. Shape (n_nodes,)
            rho (np.array): Density array (boolean) for topology. Shape (n_elements,)
            plotter (pyvista.Plotter): Plotter to be used for the plot. If not provided, a new plotter will be created.
            power_scaling (float): Power scale to be used for the plot.
            cmap (str): Colormap to be used for the plot.
        """
        if rho is None:
            return plot_field(
                self.node_positions,
                self.element_centroids,
                self.elements,
                field[rho],
                **kwargs,
            )
        return plot_field(
            self.node_positions,
            self.element_centroids[rho],
            [self.elements[i] for i in np.where(rho)[0]],
            field[rho],
            **kwargs,
        )

    def plot_problem(self, **kwargs):
        """
        This function plots the problem.

        Parameters:
            plotter (pyvista.Plotter or matplotlib.pyplot): Plotter to be used for the plot. For 3D pyvista plotter and for 2D matplotlib axes. If not provided, a new plotter will be created.
            face_color (str): Color of the faces in the plot.
            edge_color (str): Color of the edges in the plot.
            x_color (str): Color of the x constraint symbol.
            y_color (str): Color of the y constraint symbol.
            z_color (str): Color of the z constraint symbol.
            force_color (str): Color of the force symbol.
        """
        return plot_problem(
            self.node_positions, self.elements, self.c, self.f, **kwargs
        )

    def plot_topology(self, rho, **kwargs):
        """
        This function plots the topology.

        Parameters:
            rho (np.array): Density array (boolean). Shape (n_elements,)
            plotter (pyvista.Plotter or matplotlib.pyplot): Plotter to be used for the plot. For 3D pyvista plotter and for 2D matplotlib axes. If not provided, a new plotter will be created.
            face_color (str): Color of the faces in the plot.
            edge_color (str): Color of the edges in the plot.
            x_color (str): Color of the x constraint symbol.
            y_color (str): Color of the y constraint symbol.
            z_color (str): Color of the z constraint symbol.
            force_color (str): Color of the force symbol.
        """
        return plot_problem(
            self.node_positions,
            [self.elements[i] for i in np.where(rho)[0]],
            self.c,
            self.f,
            **kwargs,
        )

    def FEA(self, rho=None, c=None, f=None):
        """
        This function performs a Finite Element Analysis and also returns stresses and strain energy.

        Parameters:
            rho (np.array): Density array. Shape (n_elements,) If not provided, it is set to ones.
            c (np.array): Boundary conditions array. Shape (n_nodes, dim) If not provided, it is set to solver BCs.
            f (np.array): Forces array. Shape (n_nodes, dim) If not provided, it is set to solver forces.

        Returns:
            U (np.array): Displacement array. Shape (n_nodes, dim)
            compliance (float): Compliance of the structure.
            S (np.array): Stresses array. Shape (n_elements, n_stresses)
            SE (np.array): Strain Energy array. Shape (n_elements,)
        """

        if self.compute_engine == "gpu":
            return self.FEA_GPU(rho, c, f)

        if rho is None:
            rho = np.ones([len(self.elements)])
        if c is None:
            c = self.c
        if f is None:
            f = self.f

        # make sure rho is not zero
        rho = self.material_model(rho, 0, plain=True)

        if self.reordering:
            ordr, inv_order = self.compute_ordering(
                rho, self.K_kernel, self.k_map, self.non_constrained_map
            )
        else:
            ordr = None
            inv_order = None

        non_constrained_map = np.where(c.reshape(-1) == 0)[0]
        F = f.reshape(-1)[non_constrained_map]

        if self.solver == "cholesky":
            solver_fn = self.solve_cholesky
        elif self.solver == "splu":
            solver_fn = self.solve_splu
        elif self.solver == "cg":
            solver_fn = self.solve_cg
        elif self.solver == "bicgstab":
            solver_fn = self.solve_bicgstab
        elif self.solver == "gpu":
            solver_fn = self.solve_gpu

        U, info = self.solve(
            rho,
            self.K_kernel,
            self.k_map,
            non_constrained_map,
            F,
            solver_fn,
            mb_order=ordr,
            inv_order=inv_order,
            max_iter=max(self.solver_max_iter, 5000),
            tol=self.solve_tol,
        )

        flag = True
        if info[-1] > self.solve_tol:
            flag = False
            logger.warning(f"Solver did not converge, residual: {info[-1]}")

        compliance = F.dot(U)
        U_full = np.zeros_like(c.reshape(-1))
        U_full[non_constrained_map] = U
        U_full = U_full.reshape([-1, self.dim])

        strain_energy = np.zeros([len(self.elements), 1])
        stresses = np.zeros([len(self.elements), self.Ds[0].shape[0]])

        if len(self.tris):
            Ks = np.array([self.Ks[i] for i in self.tris])
            Bs = np.array([self.Bs[i] for i in self.tris])
            Ds = np.array([self.Ds[i] for i in self.tris])
            elements = np.array([self.elements[i] for i in self.tris])

            strain_eneregy_tris = 0.5 * (
                np.transpose(
                    U_full[elements].reshape(-1, elements.shape[1] * self.dim, 1),
                    [0, 2, 1],
                )
                @ Ks
                @ U_full[elements].reshape(-1, elements.shape[1] * self.dim, 1)
            )
            stresses_tris = (Ds @ Bs) @ U_full[elements].reshape(
                -1, elements.shape[1] * self.dim, 1
            )

            strain_energy[self.tris, 0] = strain_eneregy_tris.squeeze()
            stresses[self.tris] = stresses_tris.squeeze()

        if len(self.quads):
            Ks = np.array([self.Ks[i] for i in self.quads])
            Bs = np.array([self.Bs[i] for i in self.quads])
            Ds = np.array([self.Ds[i] for i in self.quads])
            elements = np.array([self.elements[i] for i in self.quads])

            strain_eneregy_quads = 0.5 * (
                np.transpose(
                    U_full[elements].reshape(-1, elements.shape[1] * self.dim, 1),
                    [0, 2, 1],
                )
                @ Ks
                @ U_full[elements].reshape(-1, elements.shape[1] * self.dim, 1)
            )
            stresses_quads = (Ds @ Bs) @ U_full[elements].reshape(
                -1, elements.shape[1] * self.dim, 1
            )

            strain_energy[self.quads, 0] = strain_eneregy_quads.squeeze()
            stresses[self.quads] = stresses_quads.squeeze()

        if len(self.tets):
            Ks = np.array([self.Ks[i] for i in self.tets])
            Bs = np.array([self.Bs[i] for i in self.tets])
            Ds = np.array([self.Ds[i] for i in self.tets])
            elements = np.array([self.elements[i] for i in self.tets])

            strain_eneregy_tets = 0.5 * (
                np.transpose(
                    U_full[elements].reshape(-1, elements.shape[1] * self.dim, 1),
                    [0, 2, 1],
                )
                @ Ks
                @ U_full[elements].reshape(-1, elements.shape[1] * self.dim, 1)
            )
            stresses_tets = (Ds @ Bs) @ U_full[elements].reshape(
                -1, elements.shape[1] * self.dim, 1
            )

            strain_energy[self.tets, 0] = strain_eneregy_tets.squeeze()
            stresses[self.tets] = stresses_tets.squeeze()

        if len(self.hexas):
            Ks = np.array([self.Ks[i] for i in self.hexas])
            Bs = np.array([self.Bs[i] for i in self.hexas])
            Ds = np.array([self.Ds[i] for i in self.hexas])
            elements = np.array([self.elements[i] for i in self.hexas])

            strain_eneregy_hexas = 0.5 * (
                np.transpose(
                    U_full[elements].reshape(-1, elements.shape[1] * self.dim, 1),
                    [0, 2, 1],
                )
                @ Ks
                @ U_full[elements].reshape(-1, elements.shape[1] * self.dim, 1)
            )
            stresses_hexas = (Ds @ Bs) @ U_full[elements].reshape(
                -1, elements.shape[1] * self.dim, 1
            )

            strain_energy[self.hexas, 0] = strain_eneregy_hexas.squeeze()
            stresses[self.hexas] = stresses_hexas.squeeze()

        # Should scale stress and strain energy by module for true stress values but for now we will keep it as is since in single material this multiplier is one for all and zero for void. but in multi material this will be different.
        # stresses = stresses * rho.reshape(-1,1)
        # strain_energy = strain_energy * rho.reshape(-1,1)

        return U_full, compliance, stresses, strain_energy, flag

    def FEA_GPU(self, rho=None, c=None, f=None):
        """
        This function performs a Finite Element Analysis and also returns stresses and strain energy on GPU.

        Parameters:
            rho (np.array): Density array. Shape (n_elements,) If not provided, it is set to ones.
            c (np.array): Boundary conditions array. Shape (n_nodes, dim) If not provided, it is set to solver BCs.
            f (np.array): Forces array. Shape (n_nodes, dim) If not provided, it is set to solver forces.

        Returns:
            U (np.array): Displacement array. Shape (n_nodes, dim)
            compliance (float): Compliance of the structure.
            S (np.array): Stresses array. Shape (n_elements, n_stresses)
            SE (np.array): Strain Energy array. Shape (n_elements,)
        """
        if rho is None:
            rho = np.ones([len(self.elements)])
        if c is None:
            c = self.c
        if f is None:
            f = self.f

        # make sure rho is not zero
        rho = self.material_model(rho, 0, plain=True)

        if self.reordering:
            ordr, inv_order = self.compute_ordering(
                rho, self.K_kernel, self.k_map, self.non_constrained_map
            )
        else:
            ordr = None
            inv_order = None

        non_constrained_map = np.where(c.reshape(-1) == 0)[0]
        F = f.reshape(-1)[non_constrained_map]

        # Move to GPU
        solver_fn = self.solve_gpu_
        rho = cp.array(rho)
        F = cp.array(F)
        K_kernel = cp.sparse.csr_matrix(self.K_kernel)
        k_map = cp.array(self.k_map)
        non_constrained_map = cp.array(non_constrained_map)
        if ordr is not None:
            ordr = cp.array(ordr)
            inv_order = cp.array(inv_order)

        U, info = self.solve(
            rho,
            K_kernel,
            k_map,
            non_constrained_map,
            F,
            solver_fn,
            mb_order=ordr,
            inv_order=inv_order,
            coo_matrix=cp.sparse.coo_matrix,
            max_iter=max(self.solver_max_iter, 50000),
            tol=self.solve_tol,
        )

        flag = True
        if info[-1].get() > self.solve_tol:
            flag = False
            logger.warning(f"Solver did not converge, residual: {info[-1].get()}")

        compliance = F.dot(U)

        U_full = cp.zeros_like(c.reshape(-1))
        U_full[non_constrained_map] = U
        U_full = U_full.reshape([-1, self.dim])

        strain_energy = cp.zeros([len(self.elements), 1])
        stresses = cp.zeros([len(self.elements), self.Ds[0].shape[0]])

        if len(self.tris):
            Ks = cp.array([cp.array(self.Ks[i]) for i in self.tris])
            Bs = cp.array([cp.array(self.Bs[i]) for i in self.tris])
            Ds = cp.array([cp.array(self.Ds[i]) for i in self.tris])
            elements = cp.array([cp.array(self.elements[i]) for i in self.tris])

            strain_eneregy_tris = 0.5 * (
                cp.transpose(
                    U_full[elements].reshape(-1, elements.shape[1] * self.dim, 1),
                    [0, 2, 1],
                )
                @ Ks
                @ U_full[elements].reshape(-1, elements.shape[1] * self.dim, 1)
            )
            stresses_tris = (Ds @ Bs) @ U_full[elements].reshape(
                -1, elements.shape[1] * self.dim, 1
            )

            strain_energy[self.tris, 0] = strain_eneregy_tris.squeeze()
            stresses[self.tris] = stresses_tris.squeeze()

        if len(self.quads):
            Ks = cp.array([cp.array(self.Ks[i]) for i in self.quads])
            Bs = cp.array([cp.array(self.Bs[i]) for i in self.quads])
            Ds = cp.array([cp.array(self.Ds[i]) for i in self.quads])
            elements = cp.array([cp.array(self.elements[i]) for i in self.quads])

            strain_eneregy_quads = 0.5 * (
                cp.transpose(
                    U_full[elements].reshape(-1, elements.shape[1] * self.dim, 1),
                    [0, 2, 1],
                )
                @ Ks
                @ U_full[elements].reshape(-1, elements.shape[1] * self.dim, 1)
            )
            stresses_quads = (Ds @ Bs) @ U_full[elements].reshape(
                -1, elements.shape[1] * self.dim, 1
            )

            strain_energy[self.quads, 0] = strain_eneregy_quads.squeeze()
            stresses[self.quads] = stresses_quads.squeeze()

        if len(self.tets):
            Ks = cp.array([cp.array(self.Ks[i]) for i in self.tets])
            Bs = cp.array([cp.array(self.Bs[i]) for i in self.tets])
            Ds = cp.array([cp.array(self.Ds[i]) for i in self.tets])
            elements = cp.array([cp.array(self.elements[i]) for i in self.tets])

            strain_eneregy_tets = 0.5 * (
                cp.transpose(
                    U_full[elements].reshape(-1, elements.shape[1] * self.dim, 1),
                    [0, 2, 1],
                )
                @ Ks
                @ U_full[elements].reshape(-1, elements.shape[1] * self.dim, 1)
            )
            stresses_tets = (Ds @ Bs) @ U_full[elements].reshape(
                -1, elements.shape[1] * self.dim, 1
            )

            strain_energy[self.tets, 0] = strain_eneregy_tets.squeeze()
            stresses[self.tets] = stresses_tets.squeeze()

        if len(self.hexas):
            Ks = cp.array([cp.array(self.Ks[i]) for i in self.hexas])
            Bs = cp.array([cp.array(self.Bs[i]) for i in self.hexas])
            Ds = cp.array([cp.array(self.Ds[i]) for i in self.hexas])
            elements = cp.array([cp.array(self.elements[i]) for i in self.hexas])

            strain_eneregy_hexas = 0.5 * (
                cp.transpose(
                    U_full[elements].reshape(-1, elements.shape[1] * self.dim, 1),
                    [0, 2, 1],
                )
                @ Ks
                @ U_full[elements].reshape(-1, elements.shape[1] * self.dim, 1)
            )
            stresses_hexas = (Ds @ Bs) @ U_full[elements].reshape(
                -1, elements.shape[1] * self.dim, 1
            )

            strain_energy[self.hexas, 0] = strain_eneregy_hexas.squeeze()
            stresses[self.hexas] = stresses_hexas.squeeze()

        return U_full.get(), compliance.get(), stresses.get(), strain_energy.get(), flag

    @staticmethod
    def solve_cholesky(K, F, cholmod=None, **kwargs):
        """
        This function solves the linear system using cholesky decomposition.

        Parameters:
            K (scipy.sparse.csr_matrix): Stiffness matrix.
            F (np.array): Forces array.

        Returns:
            U (np.array): Displacement array.
        """

        if cholmod is None:
            cholmod = cholesky(K)
        else:
            cholmod.cholesky_inplace(K)
        U = cholmod(F)
        residual = np.linalg.norm(K.dot(U) - F)
        return U, cholmod, residual

    @staticmethod
    def solve_splu(K, F, **kwargs):
        """
        This function solves the linear system using sparse LU decomposition.

        Parameters:
            K (scipy.sparse.csr_matrix): Stiffness matrix.
            F (np.array): Forces array.

        Returns:
            U (np.array): Displacement array.
        """

        splu = splu(K)
        U = splu.solve(F)
        residual = np.linalg.norm(K.dot(U) - F)
        return U, 0, residual

    @staticmethod
    def solve_cg(K, F, max_iter=500, u0=None, tol=1e-4, **kwargs):
        """
        This function solves the linear system using conjugate gradient method.

        Parameters:
            K (scipy.sparse.csr_matrix): Stiffness matrix.
            F (np.array): Forces array.
            max_iter (int): Maximum number of iterations for the solver.
            u0 (np.array): Initial guess for the solution.
        Returns:
            U (np.array): Displacement array.
            info (int): Convergence information.
        """

        U, info = cg(K, F, maxiter=max_iter, x0=u0, rtol=tol)
        residual = np.linalg.norm(K.dot(U) - F)
        return U, info, residual

    @staticmethod
    def solve_bicgstab(K, F, max_iter=500, u0=None, tol=1e-4, **kwargs):
        """
        This function solves the linear system using biconjugate gradient stabilized method.

        Parameters:
            K (scipy.sparse.csr_matrix): Stiffness matrix.
            F (np.array): Forces array.
            max_iter (int): Maximum number of iterations for the solver.
            u0 (np.array): Initial guess for the solution.
        Returns:
            U (np.array): Displacement array.
            info (int): Convergence information.
        """

        U, info = bicgstab(K, F, maxiter=max_iter, x0=u0, rtol=tol)
        residual = np.linalg.norm(K.dot(U) - F)
        return U, info, residual

    @staticmethod
    def solve_gpu(K, F, max_iter=500, u0=None, tol=1e-4, **kwargs):
        """
        This function solves the linear system using conjugate gradient method on GPU.

        Parameters:
            K (scipy.sparse.csr_matrix): Stiffness matrix.
            F (np.array): Forces array.
            max_iter (int): Maximum number of iterations for the solver.
            u0 (np.array): Initial guess for the solution.
        Returns:
            U (np.array): Displacement array.
            info (int): Convergence information.
        """
        K_gpu = cp.sparse.csr_matrix(K)
        F_gpu = cp.array(F)

        if u0 is not None:
            u0_gpu = cp.array(u0)
        else:
            u0_gpu = None

        U, info = cg_gpu(K_gpu, F_gpu, maxiter=max_iter, x0=u0_gpu, tol=tol)
        residual = cp.linalg.norm(K_gpu.dot(U) - F_gpu)
            
        return U.get(), info, residual.get()

    @staticmethod
    def solve_gpu_(K, F, max_iter=500, u0=None, tol=1e-4, **kwargs):
        """
        This function solves the linear system using conjugate gradient method on GPU.

        Parameters:
            K (scipy.sparse.csr_matrix): Stiffness matrix.
            F (np.array): Forces array.
            max_iter (int): Maximum number of iterations for the solver.
            u0 (np.array): Initial guess for the solution.
        Returns:
            U (np.array): Displacement array.
            info (int): Convergence information.
        """

        U, info = cg_gpu(K, F, maxiter=max_iter, x0=u0, tol=tol)
        residual = cp.linalg.norm(K.dot(U) - F)

        return U, info, residual

    @staticmethod
    def solve(
        rho,
        K_Kernel,
        k_map,
        non_constrained_map,
        F,
        solver,
        mb_order=None,
        inv_order=None,
        coo_matrix=coo_matrix,
        **kwargs,
    ):
        """
        This function solves the linear system using the selected solver.

        Parameters:
            rho (np.array): Density array.
            K_Kernel (scipy.sparse.csr_matrix): Kernel stiffness matrix.
            k_map (np.array): Map of the kernel stiffness matrix.
            non_constrained_map (np.array): Map of the non-constrained nodes.
            F (np.array): Forces array.
            mb_order (np.array): Map of the bandwidth reduction.
            inv_order (np.array): Map of the bandwidth reduction.
            solver (function): Solver function to be used.

        Returns:
            U (np.array): Displacement array.
            solver_info (list): Solver Auxiliary Information.
        """

        k_vals = K_Kernel.dot(rho.reshape([-1, 1]))[:, 0]
        K = coo_matrix((k_vals, (k_map[:, 0], k_map[:, 1]))).tocsc()
        if mb_order is not None:
            K = K[non_constrained_map[mb_order], :][:, non_constrained_map[mb_order]]
            solver_output = solver(K, F[mb_order], **kwargs)
            U = solver_output[0][inv_order]
        else:
            K = K[non_constrained_map, :][:, non_constrained_map]
            solver_output = solver(K, F, **kwargs)
            U = solver_output[0]

        solver_info = solver_output[1:]

        return U, solver_info

    @staticmethod
    def system_solve(
        rho,
        material_model,
        K_Kernel,
        k_map,
        F,
        dK,
        grad_map,
        non_constrained_map,
        filter_kernel,
        solver,
        iteration,
        mb_order=None,
        inv_order=None,
        np=np,
        coo_matrix=coo_matrix,
        **kwargs,
    ):
        """
        This function solves the TO problem using the passed solver.

        Parameters:
            rho (np.array): Density array.
            K_Kernel (scipy.sparse.csr_matrix): Kernel stiffness matrix.
            k_map (np.array): Map of the kernel stiffness matrix.
            F (np.array): Forces array.
            dK (scipy.sparse.csr_matrix): Kernel gradient of the stiffness matrix.
            grad_map (np.array): Map of the kernel gradient.
            non_constrained_map (np.array): Map of the non-constrained nodes.
            filter_kernel (scipy.sparse.csr_matrix): Filter kernel.
            penalty (float): Penalty factor for the SIMP method.
            void (float): Void material Youngs Modulus value.
            solver (function): Solver function to be used.
        """
        rho_ = filter_kernel.dot(rho)
        rho__ = material_model(rho_, iteration, np=np)

        U, solver_info = Solver.solve(
            rho__,
            K_Kernel,
            k_map,
            non_constrained_map,
            F,
            solver,
            mb_order=mb_order,
            inv_order=inv_order,
            coo_matrix=coo_matrix,
            **kwargs,
        )

        compliance = F.dot(U)

        df = (
            -coo_matrix((dK.dot(U), (grad_map[:, 0], grad_map[:, 1])))
            .tocsc()[:, non_constrained_map]
            .dot(U)
            .reshape(-1, 1)
        )

        dr = material_model.grad(rho_, iteration, np=np) * df

        df = filter_kernel.T.dot(dr.reshape(dr.shape[0], -1))

        return compliance, df, U, solver_info

    @staticmethod
    def update_design(rho, df, dG, material_model, move=0.2, np=np):
        """
        This function updates the design based on the SIMP method.

        Parameters:
            rho (np.array): Density array.
            df (np.array): Sensitivity array.
            V (float): Total volume constraint. volume_fraction * total_area_or_volume.
            dG (np.array): Gradient of the density array.
            void (float): Void material Youngs Modulus value.
            move (float): Move limit for the design update.
            eta (float): SIMP method exponent.
            np (module): Numpy or Cupy module.

        Returns:
            rho_new (np.array): Updated density array.
        """

        xU = np.clip(rho + move, 0., 1.0)
        xL = np.clip(rho - move, 0., 1.0)

        ocP = material_model.ocP(df, dG, rho, np=np)
        ocP = np.maximum(1e-12, ocP)

        size = material_model.base_properties()["n_sets"]
        l1 = 1e-9 * np.ones(size)
        l2 = 1e9 * np.ones(size)

        lmid = 0.5 * (l1 + l2)

        rho_new = np.maximum(
            0, np.maximum(xL, np.minimum(1.0, np.minimum(xU, ocP / lmid)))
        )

        while np.any((l2 - l1) / (l2 + l1) > 1e-4):
            lmid = 0.5 * (l1 + l2)
            rho_new = np.maximum(
                0, np.maximum(xL, np.minimum(1.0, np.minimum(xU, ocP / lmid)))
            )

            valids = material_model.evaluate_constraint(rho_new, dG, np=np)

            l2[valids] = lmid[valids]
            l1[~valids] = lmid[~valids]

        return rho_new

    @staticmethod
    def compute_ordering(rho, K_Kernel, inds_full, non_constrained_map):
        vals_full = K_Kernel.dot(rho.reshape([-1, 1]))[:, 0]

        K = coo_matrix((vals_full, (inds_full[:, 0], inds_full[:, 1]))).tocsc()
        K = K[non_constrained_map, :][:, non_constrained_map]

        ordr = reverse_cuthill_mckee(K, symmetric_mode=True)
        inv_order = np.argsort(ordr)

        return ordr, inv_order
    
    def fs_optimize(
        self, rho, n_step = 10, chk_steps = 5, save_comp_history=False, save_change_history=False, save_rho_history=False
    ):
        """
        This function performs the topology optimization.

        Returns:
            rho (np.array): Optimized Density array.
            flag (bool): Flag indicating if the optimization converged.
        """

        if self.compute_engine == "gpu":
            return self.optimize_gpu(
                save_comp_history, save_change_history, save_rho_history
            )

        flag = False

        # rho = np.ones([len(self.elements)])

        if self.reordering:
            logger.info("Computing Reordering For Minimum Bandwidth...")
            ordr, inv_order = self.compute_ordering(
                np.ones(len(self.elements)),
                self.K_kernel,
                self.k_map,
                self.non_constrained_map,
            )
        else:
            ordr = None
            inv_order = None

        if self.solver == "cholesky":
            solver_fn = self.solve_cholesky
        elif self.solver == "splu":
            solver_fn = self.solve_splu
        elif self.solver == "cg":
            solver_fn = self.solve_cg
        elif self.solver == "bicgstab":
            solver_fn = self.solve_bicgstab
        elif "gpu" in self.solver:
            # solver_fn = self.solve_gpu
            def solver_fn(K, F, max_iter=500, u0=None, tol=1e-4, **kwargs):
                return self.solve_gpu(K, F, max_iter, u0, tol, solver=self.solver, **kwargs)

        prog = trange(n_step, disable=logger.getEffectiveLevel() > logging.INFO)

        fun_change = np.inf
        change = np.inf

        # rho = self.material_model.init_desvars(len(self.elements))

        # Initial Solve
        comp, df, U, solver_info = self.system_solve(
            rho,
            self.material_model,
            self.K_kernel,
            self.k_map,
            self.F,
            self.dK,
            self.grad_map,
            self.non_constrained_map,
            self.filter_kernel,
            solver_fn,
            0,
            mb_order=ordr,
            inv_order=inv_order,
            max_iter=self.solver_max_iter,
            tol=self.solve_tol,
        )

        if self.reordering:
            u0 = U[ordr]
        else:
            u0 = U

        if self.solver == "cholesky":
            cholmod = solver_info[0]
        else:
            cholmod = None

        if save_comp_history:
            comp_history = []

        if save_change_history:
            change_history = []

        if save_rho_history:
            rho_history = []

        hist = {}

        for i in prog:

            rho_old = np.copy(rho)
            comp_old = np.copy(comp)

            rho = self.update_design(
                rho, df, self.dG, self.material_model, self.move, np
            )

            comp, df, U, solver_info = self.system_solve(
                rho,
                self.material_model,
                self.K_kernel,
                self.k_map,
                self.F,
                self.dK,
                self.grad_map,
                self.non_constrained_map,
                self.filter_kernel,
                solver_fn,
                i,
                mb_order=ordr,
                inv_order=inv_order,
                max_iter=self.solver_max_iter,
                cholmod=cholmod,
                u0=u0,
                tol=self.solve_tol,
            )

            if self.reordering:
                u0 = U[ordr]
            else:
                u0 = U

            change = np.linalg.norm(rho - rho_old) / np.sqrt(rho.size)
            change_f = np.abs(comp - comp_old) / comp

            if save_change_history:
                change_history.append(change)

            if save_rho_history:
                rho_history.append(rho)

            if save_comp_history:
                comp_history.append(comp)

            prog.set_postfix_str(
                f"Compliance: {comp:.4e}, Change: {change:.4e}, Function Change: {change_f:.4e}, Residual: {solver_info[-1]:.4e}"
            )

            if (i+1) == chk_steps:
                rho_chk = np.copy(rho)

        if save_comp_history:
            hist["comp_history"] = comp_history
        if save_change_history:
            hist["change_history"] = change_history
        if save_rho_history:
            hist["rho_history"] = rho_history

        return rho, rho_chk, hist

    def optimize(
        self, save_comp_history=False, save_change_history=False, save_rho_history=False
    ):
        """
        This function performs the topology optimization.

        Returns:
            rho (np.array): Optimized Density array.
            flag (bool): Flag indicating if the optimization converged.
        """

        if self.compute_engine == "gpu":
            return self.optimize_gpu(
                save_comp_history, save_change_history, save_rho_history
            )

        flag = False

        rho = np.ones([len(self.elements)])

        if self.reordering:
            logger.info("Computing Reordering For Minimum Bandwidth...")
            ordr, inv_order = self.compute_ordering(
                np.ones(len(self.elements)),
                self.K_kernel,
                self.k_map,
                self.non_constrained_map,
            )
        else:
            ordr = None
            inv_order = None

        if self.solver == "cholesky":
            solver_fn = self.solve_cholesky
        elif self.solver == "splu":
            solver_fn = self.solve_splu
        elif self.solver == "cg":
            solver_fn = self.solve_cg
        elif self.solver == "bicgstab":
            solver_fn = self.solve_bicgstab
        elif "gpu" in self.solver:
            # solver_fn = self.solve_gpu
            def solver_fn(K, F, max_iter=500, u0=None, tol=1e-4, **kwargs):
                return self.solve_gpu(K, F, max_iter, u0, tol, solver=self.solver, **kwargs)

        prog = trange(self.max_iter, disable=logger.getEffectiveLevel() > logging.INFO)

        fun_change = np.inf
        change = np.inf

        rho = self.material_model.init_desvars(len(self.elements))

        # Initial Solve
        comp, df, U, solver_info = self.system_solve(
            rho,
            self.material_model,
            self.K_kernel,
            self.k_map,
            self.F,
            self.dK,
            self.grad_map,
            self.non_constrained_map,
            self.filter_kernel,
            solver_fn,
            0,
            mb_order=ordr,
            inv_order=inv_order,
            max_iter=self.solver_max_iter,
            tol=self.solve_tol,
        )

        if self.reordering:
            u0 = U[ordr]
        else:
            u0 = U

        if self.solver == "cholesky":
            cholmod = solver_info[0]
        else:
            cholmod = None

        if save_comp_history:
            comp_history = []

        if save_change_history:
            change_history = []

        if save_rho_history:
            rho_history = []

        hist = {}

        for i in prog:

            rho_old = np.copy(rho)
            comp_old = np.copy(comp)

            rho = self.update_design(
                rho, df, self.dG, self.material_model, self.move, np
            )

            comp, df, U, solver_info = self.system_solve(
                rho,
                self.material_model,
                self.K_kernel,
                self.k_map,
                self.F,
                self.dK,
                self.grad_map,
                self.non_constrained_map,
                self.filter_kernel,
                solver_fn,
                i,
                mb_order=ordr,
                inv_order=inv_order,
                max_iter=self.solver_max_iter,
                cholmod=cholmod,
                u0=u0,
                tol=self.solve_tol,
            )

            if self.reordering:
                u0 = U[ordr]
            else:
                u0 = U

            change = np.linalg.norm(rho - rho_old) / np.sqrt(rho.size)
            change_f = np.abs(comp - comp_old) / comp

            if save_change_history:
                change_history.append(change)

            if save_rho_history:
                rho_history.append(rho)

            if save_comp_history:
                comp_history.append(comp)

            prog.set_postfix_str(
                f"Compliance: {comp:.4e}, Change: {change:.4e}, Function Change: {change_f:.4e}, Residual: {solver_info[-1]:.4e}"
            )

            if (
                change < self.ch_tol
                and change_f < self.fun_tol
                and self.material_model.is_terminal(i)
            ):
                flag = True
                break

        if save_comp_history:
            hist["comp_history"] = comp_history
        if save_change_history:
            hist["change_history"] = change_history
        if save_rho_history:
            hist["rho_history"] = rho_history

        return rho, flag, hist

    def optimize_gpu(
        self, save_comp_history=False, save_change_history=False, save_rho_history=False
    ):

        flag = False

        rho = np.ones([len(self.elements)])

        if self.reordering:
            ordr, inv_order = self.compute_ordering(
                np.ones(len(self.elements)),
                self.K_kernel,
                self.k_map,
                self.non_constrained_map,
            )
            ordr = cp.array(ordr)
            inv_order = cp.array(inv_order)
        else:
            ordr = None
            inv_order = None

        # solver_fn = self.solve_gpu_

        def solver_fn(K, F, max_iter=500, u0=None, tol=1e-4, **kwargs):
            return self.solve_gpu_(K, F, max_iter, u0, tol, solver=self.solver, **kwargs)

        
        prog = trange(self.max_iter, disable=logger.getEffectiveLevel() > logging.INFO)

        fun_change = cp.inf
        change = cp.inf

        rho = self.material_model.init_desvars(len(self.elements))

        # Move to GPU
        rho = cp.array(rho)
        dG = cp.array(self.dG)
        dK = cp.sparse.csr_matrix(self.dK)
        F = cp.array(self.F)
        K_kernel = cp.sparse.csr_matrix(self.K_kernel)
        k_map = cp.array(self.k_map)
        grad_map = cp.array(self.grad_map)
        non_constrained_map = cp.array(self.non_constrained_map)
        filter_kernel = cp.sparse.csr_matrix(self.filter_kernel)

        # Initial Solve
        comp, df, U, solver_info = self.system_solve(
            rho,
            self.material_model,
            K_kernel,
            k_map,
            F,
            dK,
            grad_map,
            non_constrained_map,
            filter_kernel,
            solver_fn,
            0,
            np=cp,
            coo_matrix=cp.sparse.coo_matrix,
            mb_order=ordr,
            inv_order=inv_order,
            max_iter=self.solver_max_iter,
            tol=self.solve_tol,
        )
        if self.reordering:
            u0 = U[ordr]
        else:
            u0 = U

        if save_comp_history:
            comp_history = []

        if save_change_history:
            change_history = []

        if save_rho_history:
            rho_history = []

        hist = {}

        inner_fails = 0
        
        for i in prog:
            rho_old = cp.copy(rho)
            comp_old = cp.copy(comp)

            rho = self.update_design(rho, df, dG, self.material_model, self.move, np=cp)

            comp, df, U, solver_info = self.system_solve(
                rho,
                self.material_model,
                K_kernel,
                k_map,
                F,
                dK,
                grad_map,
                non_constrained_map,
                filter_kernel,
                solver_fn,
                i,
                np=cp,
                coo_matrix=cp.sparse.coo_matrix,
                mb_order=ordr,
                inv_order=inv_order,
                max_iter=self.solver_max_iter,
                u0=u0,
                tol=self.solve_tol,
            )
            
            if self.reordering:
                u0 = U[ordr]
            else:
                u0 = U
            
            residual = solver_info[-1]
            
            counter = 0
            while residual > self.solve_min_tol and self.material_model.is_terminal(i):
                comp, df, U, solver_info = self.system_solve(
                    rho,
                    self.material_model,
                    K_kernel,
                    k_map,
                    F,
                    dK,
                    grad_map,
                    non_constrained_map,
                    filter_kernel,
                    solver_fn,
                    i,
                    np=cp,
                    coo_matrix=cp.sparse.coo_matrix,
                    mb_order=ordr,
                    inv_order=inv_order,
                    max_iter=self.solver_max_iter,
                    u0=u0,
                    tol=self.solve_tol,
                )
                residual = solver_info[-1]
                if self.reordering:
                    u0 = U[ordr]
                else:
                    u0 = U

                counter += 1
                
                if counter >= self.min_tol_patience:
                    inner_fails += 1
                    break
            
            if residual <= self.solve_min_tol and self.material_model.is_terminal(i):
                inner_fails = 0
            
            if inner_fails >= self.abandon_patience:
                logger.warning(f"Solver failed to find solution (System is ill-conditioned), residual: {residual}")
                break

            change = cp.linalg.norm(rho - rho_old) / cp.sqrt(len(self.elements))
            change_f = cp.abs(comp - comp_old) / comp

            if save_change_history:
                change_history.append(change.get())

            if save_rho_history:
                rho_history.append(rho.get())

            if save_comp_history:
                comp_history.append(comp.get())

            prog.set_postfix_str(
                f"Compliance: {comp:.4e}, Change: {change:.4e}, Function Change: {change_f:.4e}, Residual: {solver_info[-1]:.4e}"
            )

            if (
                change < self.ch_tol
                and change_f < self.fun_tol
                and self.material_model.is_terminal(i)
            ):
                flag = True
                break

        if save_comp_history:
            hist["comp_history"] = comp_history
        if save_change_history:
            hist["change_history"] = change_history
        if save_rho_history:
            hist["rho_history"] = rho_history

        del dG, dK, F, K_kernel, k_map, grad_map, non_constrained_map, filter_kernel
        cp._default_memory_pool.free_all_blocks()

        return rho.get(), flag, hist
