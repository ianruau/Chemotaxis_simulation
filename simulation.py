#!/usr/bin/env python3
"""
Chemotaxis Simulation Tool

This script simulates a chemotaxis system using numerical methods. It solves a system
of partial differential equations (PDEs) that model the interaction between cells (u)
and chemical signals (v).

Usage:
    python simulation.py [options]

Parameters:
    Model Parameters:
    --m FLOAT         Parameter m for reaction term (default: 1)
    --beta FLOAT      Parameter beta for reaction term (default: 1)
    --alpha FLOAT     Parameter alpha for logistic term (default: 1)
    --chi FLOAT       Parameter chi for chemotaxis (default: -1)
    --a FLOAT         Parameter a for logistic growth (default: 1)
    --b FLOAT         Parameter b for logistic growth (default: 1)
    --mu FLOAT        Parameter mu for v equation (default: 1)
    --nu FLOAT        Parameter nu for v equation (default: 1)
    --gamma FLOAT     Parameter gamma for v equation (default: 1)

    Simulation Parameters:
    --meshsize INT    Number of spatial grid points (default: 50)
    --time FLOAT      Total simulation time (default: 2.5)
    --eigen_index INT  Parameter eigen_index (default: 0, letting system choose)
    --epsilon FLOAT   Parameter perturbation epsilon (default: 0.001)

    Output Control:
    --confirm        Skip confirmation prompt if set to yes (default: no)
    --generate_video Generate MP4 animation (default: no)
    --verbose        Enable verbose output (default: no)

Example:
    python simulation.py --m 2 --beta 1 --time 10 --meshsize 100 --confirm yes

Output:
    - Generates numerical solutions for u and v
    - Creates .jpeg and .png files (identical) with static plots of the solutions
    - Optionally creates animation of the evolution
    - All output files use a basename containing parameter values
"""

import argparse
from dataclasses import dataclass, field
from typing import Final, List

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import questionary
import termplotlib as tpl
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
# from scipy.linalg import solve_banded
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from tabulate import tabulate
from tqdm import tqdm  # Import tqdm for progress bar

# Matplotlib configurations
rc("text", usetex=True)  # Enable LaTeX rendering


@dataclass(frozen=True)
class SimulationConfig:
    """
    A data class to store simulation configuration parameters.

    Attributes:
    - m (float): Exponent of u(t,x) in the chemotaxis term.
    - beta (float): Exponent of the denominator (1+v) in the chemotaxis function Chi(v).
    - alpha (float): Exponent of the nonlinear term in the logistic source.
    - chi (float): Constant numerator of the chemotaxis function Chi(v).
    - a (float): Linear reaction coefficient in the logistic source.
    - b (float): Nonlinear reaction coefficient in the logistic source.
    - mu (float): Coefficient of v in the elliptic PDE.
    - nu (float): Coefficient of u^gamma in the elliptic PDE.
    - gamma (float): Exponent of the source term u^gamma in the elliptic equation.
    - L (float): The length of the spatial domain (default is 1.0).

    Simulation Parameters:
    - meshsize (int): The number of spatial grid points, determining the resolution of the simulation.
    - time (float): The total simulation time.
    - eigen_index (int): An index used for eigenvalue-related computations.
    - epsilon (float): A small parameter used for numerical stability or perturbations.

    Output Control:
    - confirm (str): A flag to confirm simulation execution.
    - generate_video (str): A flag to enable or disable video generation ()default is 'no').
    - verbose (str): A flag to enable verbose output for detailed logs.
    - diagnostic (bool): A flag to enable or disable numerical diagnostic output (default is False).
    """

    # Model parameters
    m: float = 1.0
    beta: float = 1.0
    alpha: float = 1.0
    chi: float = 25.0
    a: float = 1.0
    b: float = 1.0
    mu: float = 1.0
    nu: float = 1.0
    gamma: float = 1.0
    L: float = 1.0  # Length of the domain

    # Simulation parameters
    meshsize: int = 50
    time: float = 2.5
    eigen_index: int = 0
    epsilon: float = 0.001

    # Output control
    confirm: str = "no"
    generate_video: str = "no"
    verbose: str = "no"
    diagnostic: bool = False

    # Computed values
    uStar: float = field(init=False, default=None)
    vStar: float = field(init=False, default=None)
    ChiStar: float = field(init=False, default=None)
    ChiDStar: float = field(init=False, default=None)
    betaTilde: float = field(init=False, default=None)
    positive_sigmas: List[float] = field(init=False, default_factory=list)
    lambdas: List[float] = field(init=False, default_factory=list)
    chi_vector: List[float] = field(init=False, default_factory=list)
    uinit: np.ndarray = field(init=False, default=None)
    vinit: np.ndarray = field(init=False, default=None)

    def __post_init__(self):
        # Using object.__setattr__ because the class is frozen
        object.__setattr__(self, "uStar", (self.a / self.b)
                           ** (1 / self.alpha))
        object.__setattr__(
            self,
            "vStar",
            self.nu / self.mu * (self.a / self.b) ** (self.gamma / self.alpha),
        )

        # Compute ChiStar
        chistar = (
            (1 + self.vStar) ** self.beta
            * (np.sqrt(self.a * self.alpha) + np.sqrt(self.mu)) ** 2
            / (self.nu * self.gamma * self.uStar ** (self.m + self.gamma - 1) + 1e-10)
        )
        object.__setattr__(self, "ChiStar", chistar)

        # Compute betaTilde and ChiDStar
        betatilde = 0 if self.beta < 0.5 else min(1, 2 * self.beta - 1)
        object.__setattr__(self, "betaTilde", betatilde)

        chidstar = np.sqrt(
            self.b
            * 16
            * (1 + betatilde * self.vStar)
            * self.mu
            / (self.nu**2 * self.uStar ** (2 - self.alpha) + 1e-10)
        )
        object.__setattr__(self, "ChiDStar", chidstar)

        # Compute Lambdas and Chi vector
        lambdas = [-(((n + 1) * np.pi / self.L) ** 2) for n in range(6)]
        object.__setattr__(self, "lambdas", lambdas)

        chi_vector = []
        for lam in lambdas:
            if lam == self.mu:
                continue  # Avoid division by zero
            chi_val = (
                ((self.a * self.alpha - lam) / (self.nu * self.gamma + 1e-10))
                * (
                    ((1 + self.vStar) ** self.beta)
                    / ((self.uStar) ** (self.m + self.gamma - 1) + 1e-10)
                )
                * ((lam - self.mu) / (lam + 1e-10))
            )
            chi_vector.append(chi_val)
        object.__setattr__(self, "chi_vector", chi_vector)
        object.__setattr__(self, "ChiStar_min", min(chi_vector, default=0))

        # Compute positive sigmas
        positive_sigmas = []
        if self.chi >= self.ChiStar:
            n = 0
            sigma_n = 1.0
            max_iterations = 100  # Prevent infinite loop
            while sigma_n > 0 and n < max_iterations:
                n += 1
                lambda_n = -((n * np.pi / self.L) ** 2)
                sigma_n = (
                    lambda_n
                    + self.chi
                    * self.nu
                    * self.gamma
                    * (
                        (self.uStar ** (self.m + self.gamma - 1))
                        / ((1 + self.vStar) ** self.beta + 1e-10)
                    )
                    * (1 - self.mu / (self.mu - lambda_n + 1e-10))
                    - self.a * self.alpha
                )
                if sigma_n > 0:
                    positive_sigmas.append(sigma_n)
        object.__setattr__(self, "positive_sigmas", positive_sigmas)

        if self.eigen_index == 0:
            if len(positive_sigmas) > 0:
                object.__setattr__(self, "eigen_index", 2)
                print("Second (first nonconstant) eigenfunction is chosen.\n")
            else:
                object.__setattr__(self, "eigen_index", 1)
                print("first (constant) eigenfunction is chosen.\n")

        x_values = np.linspace(
            0, self.L, int(
                self.meshsize) + 1, dtype=np.float64)
        object.__setattr__(
            self,
            "uinit",
            self.uStar
            + self.epsilon
            * np.cos(((self.eigen_index - 1) * np.pi / self.L) * x_values),
        ),

        # Initial condition (must match exactly)
        object.__setattr__(
            self,
            "vinit",
            solve_v(
                vector_u=self.uinit,
                L=self.L,
                Nx=self.meshsize,
                mu=self.mu,
                nu=self.nu,
                gamma=self.gamma,
                diagnostic=self.diagnostic,
            ),
        )

    def display_parameters(self) -> None:
        """Display all computed parameters in a formatted way."""

        # Now access them via config['m'], config['beta'], etc.
        print("Model Parameters:")
        print("1. Logistic term: ")
        print(f"\ta = {self.a}, b = {self.b}, alpha = {self.alpha}")
        print("2. Reaction term: ")
        print(f"\tm = {self.m}, beta = {self.beta}, chi = {self.chi}")
        print("3. The v equation: ")
        print(f"\tmu = {self.mu}, nu = {self.nu}, gamma = {self.gamma}")
        print("4. Initial condition: ")
        print(f"\tepsilon = {self.epsilon}, eigen_index = {self.eigen_index}")
        print("5. Simulation Parameters:")
        print(f"\tMeshSize = {self.meshsize}, time = {self.time}")

        print("\n# Asymptotic solutions and related constants")
        print(
            f"Asymptotic solutions: u^* = {self.uStar:.2f} and v^* = {self.vStar:.2f}"
        )
        print(f"A lower bound for Chi* is {self.ChiStar:.2f}")
        print(
            f"Chi** = {self.ChiDStar:.2f} and beta tilde = {self.betaTilde:.2f}")

        print("\n# Chi* values\n")
        data = [
            [f"Lambda_{i + 2}^*", f"{lam:.3f}",
                f"Chi_{0,i + 2}^*", f"{chi:.3f}"]
            for i, (lam, chi) in enumerate(zip(self.lambdas, self.chi_vector))
        ]
        headers = ["Lambda", "Value", "Chi", "Value"]
        print(tabulate(data, headers=headers, tablefmt="grid"))
        print(
            f"\nChi* = {self.ChiStar:.3f} and the choice of chi = {self.chi:.3f}")

        if self.positive_sigmas:
            print("\n# Positive sigma values")
            for i, sigma in enumerate(self.positive_sigmas, 1):
                print(f"sigma_{i}= {sigma}")

        def plot_initial_condition(data, label):
            """
            Plots the initial condition of the given data.

            Args:
                data (array-like): The data to be plotted.
                label (str): A label describing the data (e.g., 'u' or 'v').

            This function prints a description of the initial condition and
            generates a textual plot using the `tpl` library.
            """
            print(f"\n# Initial condition for {label}")
            fig = tpl.figure()
            fig.plot(range(len(data)), data, label=label, width=100, height=36)
            fig.show()

        plot_initial_condition(self.uinit, "u_0")
        plot_initial_condition(self.vinit, "v_0")


def solve_v(
    vector_u: np.ndarray,
    L: float,
    Nx: int,
    mu: float,
    nu: float,
    gamma: float,
    diagnostic: bool = False,
) -> np.ndarray:
    """
    Solves a linear system to compute the vector 'v' from the elliptic equation
    '0 = v_xx - mu*v + nu*u^gamma' based on the given parameters. The derivatives
    involved are computed using a central difference scheme with the use of
    ghost points to treat the Neumann boundary conditions at the endpoints.

    Parameters:
    - vector_u (np.ndarray): Input vector 'u' of size 'Nx' (default is a zero vector).
    - L (float): Length of the domain.
    - Nx (int): Number of mesh points of the space domain.
    - mu (float): Coefficient of v in the elliptic equation.
    - nu (float): Coefficient of u^gamma in the elliptic equation.
    - gamma (float): Power of 'u' in the elliptic equation.
    - diagnostic (bool): Flag for enabling diagnostic output (default is False).

    Returns:
    - np.ndarray: Solution vector 'v' of size 'Nx + 1'.

    Notes:
    - The function constructs a sparse tridiagonal matrix 'A' using finite difference discretization.
    - Neumann boundary conditions are applied by modifying the first and last off-diagonal elements.
    - The right-hand side vector 'b' is computed based on the input vector 'u' and parameters.
    - The system 'A * v = b' is solved using a sparse solver.

    Steps:
    1. Compute the mesh spacing 'dx' based on the domain length 'L' and number of mesh points 'Nx'.
    2. Define the main, upper, and lower diagonals of the sparse matrix 'A'.
    3. Apply special handling for Neumann boundary conditions by modifying the first and last off-diagonal elements.
    4. Construct the sparse matrix 'A' using the diagonals and offsets.
    5. Compute the right-hand side vector 'b' based on the input vector 'u' and parameters.
    6. Solve the linear system 'A * v = b' using a sparse solver.
    7. Return the solution vector 'v'.
    """
    dx = L / Nx

    # Define the diagonals
    main_diag = np.full(Nx + 1, -(2 + mu * dx**2))
    upper_diag = np.ones(Nx)
    lower_diag = np.ones(Nx)

    # Special handling for Neumann BC
    upper_diag[0] = 2
    lower_diag[-1] = 2

    # Create sparse matrix
    diagonals = [main_diag, upper_diag, lower_diag]
    offsets = [0, 1, -1]
    A = diags(diagonals, offsets, format="csr")

    # Define right-hand side
    b = -(dx**2) * nu * (vector_u**gamma)

    # Solve system
    v = spsolve(A, b)

    # Print matrix A in a readable format
    if diagnostic:
        print("\nMatrix A:")
        print("-" * 50)
        A_dense = A.toarray()
        for i in range(Nx + 1):
            row = [f"{x:8.3f}" for x in A_dense[i]]
            print(f"Row {i:2d}: {' '.join(row)}")
        print("-" * 50 + "\n")
        # Print out v in the same format
        # print("\nVector v:")
        row = [f"{x:8.3f}" for x in vector_u]
        print(f"vector_u {i:2d}: {' '.join(row)}")
        row = [f"{x:8.3f}" for x in b]
        print(f"b {i:2d}: {' '.join(row)}")
        row = [f"{x:8.3f}" for x in v]
        print(f"v {i:2d}: {' '.join(row)}")
        print("-" * 50 + "\n")

    return v


def first_derivative_NBC(L: float, Nx: int,
                         vector_f: np.ndarray) -> np.ndarray:
    """
    Computes the first derivative of a vector 'vector_f' using finite differences
    with Neumann boundary conditions (NBC). The derivatives involved are computed
    using a central difference scheme with the use of ghost points to treat the
    Neumann boundary conditions at the endpoints.

    Parameters:
    - L (float): Length of the domain.
    - Nx (int): Number of mesh points of the space domain.
    - vector_f (np.ndarray): Input vector for which the derivative is computed.

    Returns:
    - np.ndarray: The computed first derivative of 'vector_f'.

    Notes:
    - The function constructs a sparse matrix 'A' to approximate the derivative.
    - The matrix 'A' has:
        - -1 on the lower diagonal.
        - 1 on the upper diagonal.
        - Zeros in the first and last rows to enforce Neumann boundary conditions.
    - The derivative is scaled by the grid spacing 'dx = L / Nx'.
    """

    # Define the diagonals
    upper_diag = np.ones(Nx)
    lower_diag = -np.ones(Nx)

    # Special handling for Neumann BC
    upper_diag[0] = 0
    lower_diag[-1] = 0
    # Create diagonals
    # diagonals = {
    #     1: np.ones(Nx),  # Upper diagonal
    #     -1: -np.ones(Nx),  # Lower diagonal
    # }

    # Create sparse matrix
    diagonals = [upper_diag, lower_diag]
    offsets = [1, -1]
    A = diags(diagonals, offsets, format="csr")
    # print(A.toarray())

    dx = L / Nx
    return A.dot(vector_f / (2 * dx))


def laplacian_NBC(L: float, Nx: int, vector_f: np.ndarray) -> np.ndarray:
    """
    Create a sparse Nx x Nx square matrix representing the Laplacian operator
    with Neumann Boundary Conditions (NBC) and apply it to a given vector. The 
    derivatives involved are computed using a central difference scheme with
    the use of ghost points to treat the Neumann boundary conditions at the endpoints.

    Parameters:
    - L (float): The length of the domain.
    - Nx (int): Number of mesh points of the space domain.
    - vector_f (numpy.ndarray): The input vector to which the Laplacian operator is applied.

    Returns:
    - numpy.ndarray: The result of applying the Laplacian operator to the input vector.

    Notes:
    - The Laplacian operator is discretized using a finite difference method.
    - Neumann Boundary Conditions are applied by modifying the first and last off-diagonal elements.
    - The resulting sparse matrix is in Compressed Sparse Row (CSR) format for efficient computation.
    - The input vector is scaled by the square of the mesh spacing (dx^2) before applying the operator.
    """
    # Define the diagonals
    main_diag = np.full(Nx + 1, -2)
    upper_diag = np.ones(Nx)
    lower_diag = np.ones(Nx)

    # Special handling for Neumann BC
    upper_diag[0] = 2
    lower_diag[-1] = 2

    # Create sparse matrix
    diagonals = [main_diag, upper_diag, lower_diag]
    offsets = [0, 1, -1]
    A = diags(diagonals, offsets, format="csr")

    dx = L / Nx

    return A.dot(vector_f / (dx**2))


def rhs(u: np.ndarray, v: np.ndarray, config: SimulationConfig) -> np.ndarray:
    """
    Compute the right-hand side (RHS) of the partial differential equation
    for the given variables and parameters.

    Parameters:
    u (numpy.ndarray): The primary variable (e.g., population density).
    v (numpy.ndarray): The secondary variable (e.g., chemoattractant concentration).

    Returns:
    numpy.ndarray: The computed RHS of the equation.

    Notes:
    - The function uses several parameters from the global `config` dictionary:
      - m, beta, alpha, chi, a, b, mu, nu, gamma.
    - The terms in the equation include:
      - Diffusion term (u_xx).
      - Chemotaxis-related terms (term1, term2, term3).
      - Logistic growth term (logistic).
    """
    L = config.L
    Nx = config.meshsize
    m = config.m
    beta = config.beta
    alpha = config.alpha
    chi = config.chi
    a = config.a
    b = config.b
    mu = config.mu
    nu = config.nu
    gamma = config.gamma

    u_xx = laplacian_NBC(L, Nx, u)
    u_x = first_derivative_NBC(L, Nx, u)
    v_x = first_derivative_NBC(L, Nx, v)
    term1 = ((beta * chi) / ((1 + v) ** (beta + 1))) * (v_x**2) * (u**m)
    # print("term1=", term1)
    term2 = ((m * chi) / (1 + v) ** beta) * (u ** (m - 1)) * u_x * v_x
    # print("term2=", term2)
    term3 = (chi / ((1 + v) ** beta)) * (u**m) * (mu * v - nu * u**gamma)
    # print("term3=", term3)
    # chemotaxis = -1 * first_derivative_NBC(L, Nx, u ** m * (chi /((1 + v)**beta)) * v_x)
    logistic = a * u - b * u ** (1 + alpha)
    # print("logistic=", logistic)
    return u_xx + term1 - term2 - term3 + logistic


def RK4(config: SimulationConfig, FileBaseName="Simulation") -> tuple:
    """
    Perform numerical simulation using the Runge-Kutta 4th order (RK4) method.

    Parameters:
    FileBaseName (str): Base name for output files.

    Returns:
    tuple: x_values (numpy array), u_num (numpy array), v_num (numpy array)
        - x_values: Spatial grid points.
        - u_num: Numerical solution for u over time.
        - v_num: Numerical solution for v over time.
    """
    L = config.L
    Nx = config.meshsize
    epsilon = config.epsilon
    eigen_index = config.eigen_index
    T = config.time
    m = config.m
    beta = config.beta
    alpha = config.alpha
    chi = config.chi
    a = config.a
    b = config.b
    mu = config.mu
    nu = config.nu
    gamma = config.gamma
    diagnostic = config.diagnostic
    uStar = config.uStar
    vStar = config.vStar

    # Here we make sure that Delta t/Delta x^2 is small by letting it equal to 1/4.
    # We multiply by 2 to make sure that the time step is small enough. This
    # factor should be adjusted based on the problem.
    Nt = 2 * (int(4 * T * Nx * Nx / L**2) + 1)
    # dx = L / Nx
    dt = T / Nt

    # Initialize solutions
    u_num = np.zeros((Nt + 1, Nx + 1))
    v_num = np.zeros((Nt + 1, Nx + 1))

    x_values = np.linspace(0, L, int(Nx) + 1, dtype=np.float64)

    u_num[0, :] = config.uinit
    v_num[0, :] = config.vinit

    # Time integration
    for n in tqdm(range(Nt), desc="Progress..."):
        # rk4 steps
        k1 = rhs(u_num[n, :], v_num[n, :], config)
        v1 = solve_v(
            vector_u=u_num[n, :] + 0.5 * dt * k1,
            L=L,
            Nx=Nx,
            mu=mu,
            nu=nu,
            gamma=gamma,
            diagnostic=diagnostic,
        )

        k2 = rhs(u_num[n, :] + 0.5 * dt * k1, v1, config)
        v2 = solve_v(
            vector_u=u_num[n, :] + 0.5 * dt * k2,
            L=L,
            Nx=Nx,
            mu=mu,
            nu=nu,
            gamma=gamma,
            diagnostic=diagnostic,
        )

        k3 = rhs(u_num[n, :] + 0.5 * dt * k2, v2, config)
        v3 = solve_v(
            vector_u=u_num[n, :] + dt * k3,
            L=L,
            Nx=Nx,
            mu=mu,
            nu=nu,
            gamma=gamma,
            diagnostic=diagnostic,
        )

        k4 = rhs(u_num[n, :] + dt * k3, v3, config)

        # Update
        u_num[n + 1, :] = u_num[n, :] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        v_num[n + 1, :] = solve_v(
            vector_u=u_num[n + 1, :],
            L=L,
            Nx=Nx,
            mu=mu,
            nu=nu,
            gamma=gamma,
            diagnostic=diagnostic,
        )

    # Convert lists to numpy arrays
    u_num = np.array(u_num).T  # Convert list to numpy array and transpose
    v_num = np.array(v_num).T  # Convert list to numpy array and transpose
    t_values = np.linspace(0, T, Nt + 1)

    # Setup description for the title
    SetupDes = rf"""
    $a$ = {a}, $b$ = {b}, $\alpha$ = {alpha};
    $m$ = {m}, $\beta$ = {beta}, $\chi_0$ = {chi};
    $\mu$ = {mu}, $\nu$ = {nu}, $\gamma$ = {gamma}; $N$ = {Nx}, $T$ = {T};
    $u^*$ = {uStar}, $\epsilon$ = {epsilon}, $n$ = {eigen_index}.
    """

    # Create static plots
    create_static_plots(
        t_values,
        x_values,
        u_num,
        v_num,
        uStar,
        vStar,
        SetupDes,
        FileBaseName)

    # Create animation if requested
    if config.generate_video == "yes":
        create_animation(t_values, u_num, v_num, uStar, vStar, SetupDes, FileBaseName)

    return x_values, u_num, v_num


def create_static_plots(
    t_mesh: np.ndarray,
    x_mesh: np.ndarray,
    u_data: np.ndarray,
    v_data: np.ndarray,
    uStar: float,
    vStar: float,
    SetupDes: str,
    FileBaseName: str,
) -> None:
    """
    Create and save static 2D and 3D plots of the simulation data.

    Parameters:
    - t_mesh (np.ndarray): Temporal grid points for the simulation.
    - x_mesh (np.ndarray): Spatial grid points for the simulation.
    - u_data (np.ndarray): 2D array of simulation data values for each (time, space) pair.
    - v_data (np.ndarray): 2D array of additional simulation data values for each (time, space) pair.
    - uStar (float): Reference value for creating a constant plane in the 3D plot.
    - SetupDes (str): Description of the setup, used as the title of the plot.
    - FileBaseName (str): Base name for saving the output plot files.

    The function generates:
    - A 3D surface plot of `u_data` and `v_data` over time and space.
    - A reference plane at `uStar` and another at zero for comparison.
    - Saves the plot as PNG and JPEG files with the specified base name.

    Returns:
    - None: The function saves the plots to files and does not return any value.
    """
    # Create two subplots side by side
    fig_3d = plt.figure(figsize=(15, 6), dpi=300)

    # First subplot for u(t,x)
    ax_3d_u = fig_3d.add_subplot(121, projection="3d")
    T_grid, X_grid = np.meshgrid(t_mesh, x_mesh, indexing="xy")
    ax_3d_u.plot_surface(T_grid, X_grid, u_data, cmap="viridis", alpha=0.8)

    # Adjust the spacing between subplots
    # Reduce horizontal space between subplots, default 0.2
    plt.subplots_adjust(wspace=-0.7)

    # # Add colorbar for u
    # fig_3d.colorbar(surf_u, ax=ax_3d_u, label='u(t,x)')

    # Plot reference planes for u at the levels uStar and the tx-plane
    U_grid = np.full_like(T_grid, uStar)
    Zero_grid = np.full_like(T_grid, 0)

    ax_3d_u.plot_surface(
        T_grid,
        X_grid,
        U_grid,
        alpha=0.5,
        rstride=100,
        cstride=100,
        color="r",
    )

    ax_3d_u.plot_surface(
        T_grid,
        X_grid,
        Zero_grid,
        alpha=0.2,
        rstride=100,
        cstride=100,
        color="lightgray",
    )

    # Setup the u plot
    ax_3d_u.set_xlabel(r"Time $t$")
    ax_3d_u.set_ylabel(r"Space $x$")
    # ax_3d_u.set_zlabel(r"$u(t,x)$")
    ax_3d_u.set_zlim(-0.05, u_data.max())
    ax_3d_u.set_zticks(np.linspace(0, u_data.max(), 5))
    ax_3d_u.zaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax_3d_u.set_title("Solution u(t,x)", pad=10)

    # Second subplot for v(t,x)
    ax_3d_v = fig_3d.add_subplot(122, projection="3d")
    surf_v = ax_3d_v.plot_surface(
        T_grid, X_grid, v_data, cmap="viridis", alpha=0.8)

    # # Add colorbar for v
    # fig_3d.colorbar(surf_v, ax=ax_3d_v, label='v(t,x)')

    # Plot reference planes for v at the levels vStar and the tx-plane
    V_grid = np.full_like(T_grid, vStar)

    ax_3d_v.plot_surface(
        T_grid,
        X_grid,
        V_grid,
        alpha=0.5,
        rstride=100,
        cstride=100,
        color="r",
    )

    ax_3d_v.plot_surface(
        T_grid,
        X_grid,
        Zero_grid,
        alpha=0.2,
        rstride=100,
        cstride=100,
        color="lightgray",
    )
    # Setup the v plot
    ax_3d_v.set_xlabel(r"Time $t$")
    ax_3d_v.set_ylabel(r"Space $x$")
    # ax_3d_v.set_zlabel(r"$v(t,x)$")
    ax_3d_v.set_zlim(-0.05, v_data.max())
    ax_3d_v.set_zticks(np.linspace(0, v_data.max(), 5))
    ax_3d_v.zaxis.set_major_formatter(FormatStrFormatter("%.3f"))
    ax_3d_v.set_title("Solution v(t,x)", pad=10)

    # Add overall title
    fig_3d.suptitle(SetupDes, fontsize=10)
    plt.tight_layout()
    # Adjust layout with more right margin
    # plt.tight_layout(rect=[0, 0, 1.10, 1])  # [left, bottom, right, top]

    # Save the plot as PNG and JPEG
    fig_3d.savefig(f"{FileBaseName}.png", bbox_inches="tight")
    fig_3d.savefig(f"{FileBaseName}.jpeg", bbox_inches="tight")
    print(
        f"""
    Output files saved:
    - Image: {FileBaseName}.png
    - Image: {FileBaseName}.jpeg
    """
    )


def create_animation(
    time_data: np.ndarray,
    u_data: np.ndarray,
    v_data: np.ndarray,
    uStar: float,
    vStar: float,
    SetupDes: str,
    FileBaseName: str,
) -> None:
    """
    Create and save an animation of u and v data over time, side by side.

    Parameters:
        time_data (ndarray): 1D array of time steps
        u_data (ndarray): 2D array of u values over time
        v_data (ndarray): 2D array of v values over time
        uStar (float): Reference value for u
        SetupDes (str): Description of the setup
        FileBaseName (str): Base name for the output video file
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    fig.subplots_adjust(top=0.80)

    # Initialize plots
    (line_u,) = ax1.plot(u_data[:, 0], label="u(t)")
    (line_v,) = ax2.plot(v_data[:, 0], label="v(t)")

    # Setup u plot
    ax1.set_ylim(u_data.min() - 0.1, u_data.max() + 0.1)
    ax1.axhline(y=uStar, color="r", linestyle="--", label=r"$u^*$")
    ax1.legend(loc="upper right")
    ax1.set_title("Solution u(t,x)")

    # Setup v plot
    ax2.set_ylim(v_data.min() - 0.1, v_data.max() + 0.1)
    ax2.axhline(y=vStar, color="r", linestyle="--", label=r"$v^*$")
    ax2.legend(loc="upper right")
    ax2.set_title("Solution v(t,x)")

    def update(frame):
        line_u.set_ydata(u_data[:, frame])
        line_v.set_ydata(v_data[:, frame])
        fig.suptitle(SetupDes, fontsize=10, y=0.98)
        return line_u, line_v

    # Take every nth frame to reduce total frames
    frame_stride = max(1, len(time_data) // 200)  # Aim for ~200 frames total
    frames = range(0, len(time_data), frame_stride)

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=frames,
        interval=50,
        blit=True)

    print("\n# Saving video file.\n")
    writer = animation.FFMpegWriter(
        fps=30,  # Increased FPS for smoother playback
        metadata=dict(artist="Me"),
        bitrate=1800,
    )

    with tqdm(total=len(frames), desc="Saving", unit="frame") as pbar:
        ani.save(
            f"{FileBaseName}.mp4",
            writer=writer,
            progress_callback=lambda i, n: pbar.update(1),
        )

    plt.close()  # Clean up
    print(f"Video saved as: {FileBaseName}.mp4")


def parse_args() -> SimulationConfig:
    """
    Parse command-line arguments for configuring simulation parameters.

    Returns:
    argparse.Namespace: Parsed arguments as an object with attributes corresponding to the parameters.

    Command-line Arguments:
    --confirm (str): Skip confirmation prompt if set to 'yes' (default: 'yes').
    --generate_video (str): Generate MP4 animation if set to 'yes' (default: 'no').
    --verbose (str): Enable verbose output if set to 'yes' (default: 'no').
    --m (float): Parameter m (default: 1).
    --beta (float): Parameter beta (default: 1).
    --alpha (float): Parameter alpha (default: 1).
    --chi (float): Parameter chi (default: -1).
    --a (float): Parameter a (default: 1).
    --b (float): Parameter b (default: 1).
    --mu (float): Parameter mu (default: 1).
    --nu (float): Parameter nu (default: 1).
    --gamma (float): Parameter gamma (default: 1).
    --meshsize (int): Parameter for spatial mesh size (default: 50).
    --time (float): Parameter for time to lapse (default: 2.5).
    --eigen_index (int): Parameter eigen index (default: 0, letting system choose).
    --epsilon (float): Parameter perturbation epsilon (default: 0.001).
    """
    parser = argparse.ArgumentParser(
        description="A CLI tool for configuring parameters"
    )
    parser.add_argument(
        "--confirm",
        choices=["yes", "no"],
        default="yes",
        help="Skip confirmation prompt if set to yes (default: no)",
    )
    parser.add_argument(
        "--generate_video",
        choices=["yes", "no"],
        default="no",
        help="Generate MP4 animation (default: no)",
    )
    parser.add_argument(
        "--verbose",
        choices=["yes", "no"],
        default="no",
        help="Enable verbose output (default: no)",
    )
    parser.add_argument(
        "--m",
        type=float,
        default=1.0,
        help="Parameter m (default: 1.0)")
    parser.add_argument(
        "--beta", type=float, default=1.0, help="Parameter beta (default: 1.0)"
    )
    parser.add_argument(
        "--alpha", type=float, default=1.0, help="Parameter alpha (default: 1.0)"
    )
    parser.add_argument(
        "--chi", type=float, default=-1.0, help="Parameter chi (default: -1.0)"
    )
    parser.add_argument(
        "--a",
        type=float,
        default=1.0,
        help="Parameter a (default: 1.0)")
    parser.add_argument(
        "--b",
        type=float,
        default=1.0,
        help="Parameter b (default: 1.0)")
    parser.add_argument(
        "--mu",
        type=float,
        default=1.0,
        help="Parameter mu (default: 1.0)")
    parser.add_argument(
        "--nu",
        type=float,
        default=1,
        help="Parameter nu (default: 1.0)")
    parser.add_argument(
        "--gamma", type=float, default=1.0, help="Parameter gamma (default: 1.0)"
    )
    parser.add_argument(
        "--meshsize",
        type=int,
        default=50,
        help="Parameter for spatial mesh size (default: 50)",
    )
    parser.add_argument(
        "--time",
        type=float,
        default=2.5,
        help="Parameter for time to lapse (default: 2.5)",
    )
    parser.add_argument(
        "--eigen_index",
        type=int,
        default=0,
        help="Parameter eigen index (default: 0, letting system to choose)",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.001,
        help="Parameter perturbation epsilon (default: 0.001)",
    )

    args = parser.parse_args()
    return SimulationConfig(**vars(args))


def main():
    """
    Main function to parse arguments, display simulation parameters, and run the simulation.

    Steps:
    1. Parse command-line arguments into a SimulationConfig object
    2. Display the parsed model and simulation parameters
    3. Generate a base name for output files based on the parameters
    4. Prompt the user for confirmation to proceed with the simulation
    5. Run the simulation using the specified parameters or exit if declined
    """
    # config = parse_args()  # Returns SimulationConfig directly
    # Get immutable config
    config: Final[SimulationConfig] = parse_args()

    # # Display the parsed arguments
    config.display_parameters()

    # Using the above parameters to generate a file base name string
    basename = f"a={config.a}_b={config.b}_alpha={config.alpha}_m={config.m}_beta={config.beta}_chi={config.chi}_mu={config.mu}_nu={config.nu}_gamma={config.gamma}_meshsize={config.meshsize}_time={config.time}_epsilon={config.epsilon}_eigen_index={config.eigen_index}".replace(
        ".", "-"
    )
    print(f"Output files will be saved with the basename:\n\t {basename}\n")

    # Run the solver
    if (
        config.confirm == "yes"
        or questionary.confirm("Do you want to continue the simulation?").ask()
    ):
        print("Continuing simulation...")
        x, u, v = RK4(config=config, FileBaseName=basename)
    else:
        print("Exiting simulation.")
        exit()


if __name__ == "__main__":
    main()
