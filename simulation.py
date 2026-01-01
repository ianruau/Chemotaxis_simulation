#!/usr/bin/env python3
# pylint: disable=invalid-name
# pylint: disable=too-many-locals
# pylint: disable=too-many-arguments
# pylint: disable=too-many-positional-arguments
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
    --epsilon2 FLOAT   Parameter perturbation epsilon2 (default: 0.0)

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

import os
import tempfile

# Matplotlib writes cache files (including TeX-related caches when usetex=True).
# Ensure a writable cache directory even in sandboxed / restricted environments.
if "MPLCONFIGDIR" not in os.environ:
    mpl_config_dir = os.path.join(tempfile.gettempdir(), "chemotaxis-sim-mplconfig")
    os.makedirs(mpl_config_dir, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = mpl_config_dir

import argparse
import json
import shutil
from dataclasses import asdict, dataclass, field
from collections import deque
from typing import Final, List, Optional
from matplotlib import animation

# import matplotlib.animation as animation
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

#
# NOTE: keep the simulator self-contained: prefer NumPy `.npz` files for saved
# data instead of pickled formats.

# Matplotlib configurations
# Use LaTeX rendering by default, but allow disabling it for robustness in long
# batch runs (e.g., missing TeX packages, TeX errors, or restricted environments).
_USE_TEX = os.environ.get("CHEMOTAXIS_SIM_USETEX", "yes").strip().lower() not in ("0", "no", "false")
rc("text", usetex=_USE_TEX)


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
    - c (float): Parameter in the denominator of chi(v).
    - mu (float): Coefficient of v in the elliptic PDE.
    - nu (float): Coefficient of u^gamma in the elliptic PDE.
    - gamma (float): Exponent of the source term u^gamma in the elliptic 
      equation.
    - L (float): The length of the spatial domain (default is 1.0).

    Simulation Parameters:
    - meshsize (int): The number of spatial grid points, determining the resolution of the simulation.
    - time (float): The total simulation time.
    - eigen_index (int): An index used for eigenvalue-related computations.
    - epsilon (float): A small parameter used for numerical stability or perturbations.
    - epsilon2 (float): A small parameter used for numerical stability or perturbations.

    Output Control:
    - confirm (str): A flag to confirm simulation execution.
    - generate_video (str): A flag to enable or disable video generation (default is `no`).
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
    c: float = 1.0
    mu: float = 1.0
    nu: float = 1.0
    gamma: float = 1.0
    L: float = 1.0  # Length of the domain

    # Simulation parameters
    meshsize: int = 50
    time: float = 2.5
    eigen_index: int = 0
    epsilon: float = 0.001
    epsilon2: float = 0.0

    # Output control
    confirm: str = "no"
    generate_video: str = "no"
    verbose: str = "no"
    diagnostic: bool = False
    until_converged: str = "no"
    convergence_tol: float = 1e-4
    convergence_window_time: float = 5.0
    convergence_min_time: float = 10.0
    max_saved_frames: int = 2000
    save_data: str = "yes"
    data_format: str = "npz"
    save_max_frames: int = 2000
    save_summary6: str = "yes"
    basename: str = ""
    output_dir: str = ""

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
        object.__setattr__(self, "uStar", (self.a / self.b) ** (1 / self.alpha))
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

        x_values = np.linspace(0, self.L, int(self.meshsize) + 1, dtype=np.float64)
        object.__setattr__(
            self,
            "uinit",
            self.uStar
            + self.epsilon
            * np.cos(((self.eigen_index - 1) * np.pi / self.L) * x_values)
            + self.epsilon2
            * np.cos(((self.eigen_index) * np.pi / self.L) * x_values),
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
        print(f"\ta = {self.a}, b = {self.b}, c = {self.c}, alpha = {self.alpha}")
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
        print(f"Chi** = {self.ChiDStar:.2f} and beta tilde = {self.betaTilde:.2f}")

        print("\n# Chi* values\n")
        data = [
            [f"Lambda_{i + 2}^*", f"{lam:.3f}", f"Chi_{0,i + 2}^*", f"{chi:.3f}"]
            for i, (lam, chi) in enumerate(zip(self.lambdas, self.chi_vector))
        ]
        headers = ["Lambda", "Value", "Chi", "Value"]
        print(tabulate(data, headers=headers, tablefmt="grid"))
        print(f"\nChi* = {self.ChiStar:.3f} and the choice of chi = {self.chi:.3f}")

        if self.positive_sigmas:
            print("\n# Positive sigma values")
            for i, sigma in enumerate(self.positive_sigmas, 1):
                print(f"sigma_{i}= {sigma}")

        def plot_initial_condition(data, label):
            """
            Plots the initial condition of the given data.

            Args:
                data (array-like): The data to be plotted.
                label (str): A label describing the data (e.g., `u` or `v`).

            This function prints a description of the initial condition and
            generates a textual plot using the `tpl` library.
            """
            print(f"\n# Initial condition for {label}")
            if shutil.which("gnuplot") is None:
                print("Terminal plot skipped (requires `gnuplot` for termplotlib).")
                return
            try:
                fig = tpl.figure()
                fig.plot(range(len(data)), data, label=label, width=100, height=36)
                fig.show()
            except FileNotFoundError as exc:
                if exc.filename == "gnuplot":
                    print(
                        "Terminal plot skipped (`gnuplot` not found; install it to enable termplotlib plots)."
                    )
                    return
                raise

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
    Solves a linear system to compute the vector `v` from the elliptic equation
    `0 = v_xx - mu*v + nu*u^gamma` based on the given parameters. The derivatives
    involved are computed using a central difference scheme with the use of
    ghost points to treat the Neumann boundary conditions at the endpoints.

    Parameters:
    - vector_u (np.ndarray): Input vector `u` of size `Nx` (default is a zero vector).
    - L (float): Length of the domain.
    - Nx (int): Number of mesh points of the space domain.
    - mu (float): Coefficient of v in the elliptic equation.
    - nu (float): Coefficient of u^gamma in the elliptic equation.
    - gamma (float): Power of `u` in the elliptic equation.
    - diagnostic (bool): Flag for enabling diagnostic output (default is 
      False).

    Returns:
    - np.ndarray: Solution vector `v` of size `Nx + 1`.

    Notes:
    - The function constructs a sparse tridiagonal matrix `A` using finite difference discretization.
    - Neumann boundary conditions are applied by modifying the first and last off-diagonal elements.
    - The right-hand side vector `b` is computed based on the input vector 
      `u` and parameters.
    - The system `A * v = b` is solved using a sparse solver.

    Steps:
    1. Compute the mesh spacing `dx` based on the domain length `L` and number of mesh points `Nx`.
    2. Define the main, upper, and lower diagonals of the sparse matrix `A`.
    3. Apply special handling for Neumann boundary conditions by modifying the first and last off-diagonal elements.
    4. Construct the sparse matrix `A` using the diagonals and offsets.
    5. Compute the right-hand side vector `b` based on the input vector `u` and parameters.
    6. Solve the linear system `A * v = b` using a sparse solver.
    7. Return the solution vector `v`.
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


@dataclass(frozen=True)
class SimulationResult:
    x_values: np.ndarray
    t_values: np.ndarray
    u_num: np.ndarray
    v_num: np.ndarray
    stop_time: float
    stop_reason: str
    dt: float
    setup_description: str


def _downsample_time_indices(n_frames: int, max_frames: int) -> np.ndarray:
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if n_frames <= max_frames:
        return np.arange(n_frames, dtype=np.int64)
    raw = np.linspace(0, n_frames - 1, num=max_frames)
    idx = np.unique(np.round(raw).astype(np.int64))
    if idx[0] != 0:
        idx = np.insert(idx, 0, 0)
    if idx[-1] != n_frames - 1:
        idx = np.append(idx, n_frames - 1)
    return idx


def _config_metadata(config: SimulationConfig) -> dict:
    d = asdict(config)
    for key in ("uinit", "vinit"):
        d.pop(key, None)
    d["usetex"] = bool(_USE_TEX)
    return d


def save_simulation_data_npz(
    filename: str,
    config: SimulationConfig,
    result: SimulationResult,
    *,
    max_frames: Optional[int] = None,
) -> str:
    max_frames = config.save_max_frames if max_frames is None else max_frames
    idx = _downsample_time_indices(int(result.t_values.shape[0]), int(max_frames))
    t_saved = result.t_values[idx]
    u_saved = result.u_num[:, idx]
    v_saved = result.v_num[:, idx]

    config_json = json.dumps(_config_metadata(config), sort_keys=True)
    np.savez_compressed(
        filename,
        schema_version=np.asarray(1, dtype=np.int64),
        config_json=np.asarray(config_json),
        setup_description=np.asarray(result.setup_description),
        stop_reason=np.asarray(result.stop_reason),
        stop_time=np.asarray(result.stop_time, dtype=np.float64),
        dt=np.asarray(result.dt, dtype=np.float64),
        x_values=np.asarray(result.x_values, dtype=np.float64),
        t_values=np.asarray(t_saved, dtype=np.float64),
        u_num=np.asarray(u_saved, dtype=np.float64),
        v_num=np.asarray(v_saved, dtype=np.float64),
        downsample_indices=np.asarray(idx, dtype=np.int64),
    )
    return filename


def load_simulation_data_npz(filename: str) -> dict:
    with np.load(filename, allow_pickle=False) as data:
        out = {k: data[k] for k in data.files}
    out["schema_version"] = int(out["schema_version"].item())
    out["config"] = json.loads(str(out["config_json"].item()))
    out["setup_description"] = str(out["setup_description"].item())
    out["stop_reason"] = str(out["stop_reason"].item())
    out["stop_time"] = float(out["stop_time"].item())
    out["dt"] = float(out["dt"].item())
    out["config_json"] = str(out["config_json"].item())
    return out


def create_six_frame_summary(
    x_values: np.ndarray,
    t_values: np.ndarray,
    u_num: np.ndarray,
    v_num: np.ndarray,
    uStar: float,
    vStar: float,
    setup_description: str,
    file_base_name: str,
) -> None:
    if t_values.size == 0:
        return
    t_end = float(t_values[-1])
    fractions = np.linspace(0.0, 1.0, 6)
    targets = t_end * fractions
    indices: List[int] = []
    for t_target in targets:
        idx = int(np.argmin(np.abs(t_values - t_target)))
        indices.append(idx)

    u_min = float(np.min(u_num))
    u_max = float(np.max(u_num))
    v_min = float(np.min(v_num))
    v_max = float(np.max(v_num))
    u_pad = 0.05 * max(1e-12, u_max - u_min)
    v_pad = 0.05 * max(1e-12, v_max - v_min)

    cmap = plt.get_cmap("viridis")
    color_positions = np.linspace(0.15, 0.9, len(indices))
    colors = [cmap(float(p)) for p in color_positions]

    fig, (ax_u, ax_v) = plt.subplots(1, 2, figsize=(14, 5), dpi=300, sharex=True)
    for j, (idx, frac, color) in enumerate(zip(indices, fractions, colors)):
        t_here = float(t_values[idx])
        pct = int(round(float(frac) * 100))
        label = rf"{pct}\% ($t={t_here:.2f}$)"
        ax_u.plot(x_values, u_num[:, idx], color=color, linewidth=1.6, label=label)
        ax_v.plot(x_values, v_num[:, idx], color=color, linewidth=1.6, label=label)

    ax_u.axhline(y=uStar, color="red", linestyle="--", linewidth=0.9, label=r"$u^*$")
    ax_v.axhline(y=vStar, color="red", linestyle="--", linewidth=0.9, label=r"$v^*$")

    ax_u.set_title(rf"$u(x,t)$ slices ($T_{{stop}}={t_end:.2f}$)")
    ax_v.set_title(rf"$v(x,t)$ slices ($T_{{stop}}={t_end:.2f}$)")

    ax_u.set_xlabel(r"$x$")
    ax_v.set_xlabel(r"$x$")
    ax_u.set_ylabel(r"$u(x,t)$")
    ax_v.set_ylabel(r"$v(x,t)$")
    ax_u.set_ylim(u_min - u_pad, u_max + u_pad)
    ax_v.set_ylim(v_min - v_pad, v_max + v_pad)

    ax_u.legend(loc="best", fontsize=8, frameon=False)
    ax_v.legend(loc="best", fontsize=8, frameon=False)

    fig.suptitle(setup_description, fontsize=9)
    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{file_base_name}_summary6.png", bbox_inches="tight")
    fig.savefig(f"{file_base_name}_summary6.jpeg", bbox_inches="tight")
    plt.close(fig)


def first_derivative_NBC(L: float, Nx: int, vector_f: np.ndarray) -> np.ndarray:
    """
    Computes the first derivative of a vector `vector_f` using finite differences
    with Neumann boundary conditions (NBC). The derivatives involved are computed
    using a central difference scheme with the use of ghost points to treat the
    Neumann boundary conditions at the endpoints.

    Parameters:
    - L (float): Length of the domain.
    - Nx (int): Number of mesh points of the space domain.
    - vector_f (np.ndarray): Input vector for which the derivative is computed.

    Returns:
    - np.ndarray: The computed first derivative of `vector_f`.

    Notes:
    - The function constructs a sparse matrix `A` to approximate the derivative.
    - The matrix `A` has:
        - -1 on the lower diagonal.
        - 1 on the upper diagonal.
        - Zeros in the first and last rows to enforce Neumann boundary conditions.
    - The derivative is scaled by the grid spacing `dx = L / Nx`.
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
    - vector_f (numpy.ndarray): The input vector to which the Laplacian 
      operator is applied.

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
    `u_t = u_xx - chemotaxis term + logistic source` for the given variables
    and parameters.

    Parameters:
    - u (numpy.ndarray): The primary variable (e.g., population density).
    - v (numpy.ndarray): The secondary variable (e.g., chemoattractant concentration).

    Returns:
    - numpy.ndarray: The computed RHS of the equation.

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
    c = config.c
    mu = config.mu
    nu = config.nu
    gamma = config.gamma

    u_xx = laplacian_NBC(L, Nx, u)
    u_x = first_derivative_NBC(L, Nx, u)
    v_x = first_derivative_NBC(L, Nx, v)
    term1 = ((beta * chi) / ((c + v) ** (beta + 1))) * (v_x**2) * (u**m)
    # print("term1=", term1)
    term2 = ((m * chi) / (c + v) ** beta) * (u ** (m - 1)) * u_x * v_x
    # print("term2=", term2)
    term3 = (chi / ((c + v) ** beta)) * (u**m) * (mu * v - nu * u**gamma)
    # print("term3=", term3)
    # chemotaxis = -1 * first_derivative_NBC(L, Nx, u ** m * (chi /((1 + v)**beta)) * v_x)
    logistic = a * u - b * u ** (1 + alpha)
    # print("logistic=", logistic)
    return u_xx + term1 - term2 - term3 + logistic


def RK4(config: SimulationConfig, FileBaseName="Simulation") -> SimulationResult:
    """
    Perform numerical simulation using the Runge-Kutta 4th order (RK4) method.

    Parameters:
    FileBaseName (str): Base name for output files.

    Returns:
    tuple: x_values (numpy array), u_num (numpy array), v_num (numpy array)
        - x_values: Spatial mesh points.
        - u_num: Numerical solution for u over time (matrix of size (Nt + 1) x (Nx + 1)).
        - v_num: Numerical solution for v over time (matrix of size (Nt + 1) x (Nx + 1)).
    """
    L = config.L
    Nx = config.meshsize
    epsilon = config.epsilon
    epsilon2 = config.epsilon2
    eigen_index = config.eigen_index
    T = config.time
    m = config.m
    beta = config.beta
    alpha = config.alpha
    chi = config.chi
    a = config.a
    b = config.b
    c = config.c
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
    $a$ = {a}, $b$ = {b}, $c$ = {c}, $\alpha$ = {alpha};
    $m$ = {m}, $\beta$ = {beta}, $\chi_0$ = {chi};
    $\mu$ = {mu}, $\nu$ = {nu}, $\gamma$ = {gamma}; $N$ = {Nx}, $T$ = {T};
    $u^*$ = {uStar}, $\epsilon$ = {epsilon}, $\epsilon2$ = {epsilon2}, $n$ = {eigen_index}.
    """

    # Create static plots
    create_static_plots(
        t_values, x_values, u_num, v_num, uStar, vStar, SetupDes, FileBaseName
    )

    # Create animation if requested
    if config.generate_video == "yes":
        create_animation(t_values, u_num, v_num, uStar, vStar, SetupDes, FileBaseName)

    return SimulationResult(
        x_values=np.asarray(x_values, dtype=np.float64),
        t_values=np.asarray(t_values, dtype=np.float64),
        u_num=np.asarray(u_num, dtype=np.float64),
        v_num=np.asarray(v_num, dtype=np.float64),
        stop_time=float(T),
        stop_reason="t_max",
        dt=float(dt),
        setup_description=SetupDes,
    )


def RK4_until_converged(config: SimulationConfig, FileBaseName="Simulation") -> SimulationResult:
    """
    Run RK4 time-stepping up to `config.time`, but stop early once the solution
    is approximately steady over a fixed time window.

    This mode stores at most `config.max_saved_frames` snapshots for plotting and
    saving (to avoid allocating huge arrays when `time` is large).
    """
    L = config.L
    Nx = config.meshsize
    T_max = config.time
    mu = config.mu
    nu = config.nu
    gamma = config.gamma
    diagnostic = config.diagnostic

    # Match the legacy dt choice derived from `Nt = 2*(int(4*T*Nx^2/L^2)+1)`.
    Nt_max = 2 * (int(4 * T_max * Nx * Nx / L**2) + 1)
    dt = T_max / Nt_max

    window_steps = max(1, int(config.convergence_window_time / dt))
    min_steps = max(0, int(config.convergence_min_time / dt))

    # Store a small number of snapshots for convergence checks.
    max_check_points = 200
    check_stride = max(1, window_steps // max_check_points)
    history: deque[tuple[int, np.ndarray, np.ndarray]] = deque()

    # Store snapshots for output (downsampled in time).
    save_stride = max(1, Nt_max // max(1, config.max_saved_frames))
    saved_times: List[float] = []
    saved_u: List[np.ndarray] = []
    saved_v: List[np.ndarray] = []

    x_values = np.linspace(0, L, int(Nx) + 1, dtype=np.float64)
    u_current = np.array(config.uinit, dtype=np.float64, copy=True)
    v_current = np.array(config.vinit, dtype=np.float64, copy=True)

    def save_snapshot(step: int, t: float) -> None:
        saved_times.append(t)
        saved_u.append(u_current.copy())
        saved_v.append(v_current.copy())
        if config.verbose == "yes":
            print(f"[save] step={step} t={t:.6g} (frames={len(saved_times)})")

    def check_convergence(step: int) -> bool:
        if step < max(window_steps, min_steps):
            return False
        if not history:
            return False
        target = step - window_steps
        ref_step = None
        ref_u = None
        ref_v = None
        for s, u_snap, v_snap in reversed(history):
            if s <= target:
                ref_step = s
                ref_u = u_snap
                ref_v = v_snap
                break
        if ref_u is None or ref_v is None:
            return False

        u_amp = float(np.max(u_current) - np.min(u_current))
        v_amp = float(np.max(v_current) - np.min(v_current))
        u_scale = max(1.0, u_amp)
        v_scale = max(1.0, v_amp)
        du = float(np.max(np.abs(u_current - ref_u)))
        dv = float(np.max(np.abs(v_current - ref_v)))
        ok = (du <= config.convergence_tol * u_scale) and (dv <= config.convergence_tol * v_scale)
        if config.verbose == "yes" and ref_step is not None:
            t_ref = ref_step * dt
            t_now = step * dt
            print(
                f"[conv] t={t_now:.6g} vs t_ref={t_ref:.6g} (dt*steps={window_steps}): "
                f"du={du:.3g} dv={dv:.3g} amp_u={u_amp:.3g} amp_v={v_amp:.3g} ok={ok}"
            )
        return ok

    save_snapshot(0, 0.0)
    history.append((0, u_current.copy(), v_current.copy()))

    stop_step = Nt_max
    stop_reason = "max_time"
    for step in tqdm(range(Nt_max), desc="Progress..."):
        # RK4 stages (same structure as in RK4()).
        k1 = rhs(u_current, v_current, config)
        v1 = solve_v(
            vector_u=u_current + 0.5 * dt * k1,
            L=L,
            Nx=Nx,
            mu=mu,
            nu=nu,
            gamma=gamma,
            diagnostic=diagnostic,
        )
        k2 = rhs(u_current + 0.5 * dt * k1, v1, config)
        v2 = solve_v(
            vector_u=u_current + 0.5 * dt * k2,
            L=L,
            Nx=Nx,
            mu=mu,
            nu=nu,
            gamma=gamma,
            diagnostic=diagnostic,
        )
        k3 = rhs(u_current + 0.5 * dt * k2, v2, config)
        v3 = solve_v(
            vector_u=u_current + dt * k3,
            L=L,
            Nx=Nx,
            mu=mu,
            nu=nu,
            gamma=gamma,
            diagnostic=diagnostic,
        )
        k4 = rhs(u_current + dt * k3, v3, config)

        u_current = u_current + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        v_current = solve_v(
            vector_u=u_current,
            L=L,
            Nx=Nx,
            mu=mu,
            nu=nu,
            gamma=gamma,
            diagnostic=diagnostic,
        )

        now_step = step + 1
        now_time = now_step * dt

        if now_step % save_stride == 0 or now_step == Nt_max:
            save_snapshot(now_step, now_time)

        if now_step % check_stride == 0 or now_step == Nt_max:
            history.append((now_step, u_current.copy(), v_current.copy()))
            cutoff = now_step - window_steps - check_stride
            while history and history[0][0] < cutoff:
                history.popleft()

            if check_convergence(now_step):
                stop_step = now_step
                stop_reason = "converged"
                if saved_times and saved_times[-1] != now_time:
                    save_snapshot(now_step, now_time)
                if config.verbose == "yes":
                    print(f"[stop] converged at step={stop_step} t={stop_step*dt:.6g}")
                break

    stop_time = stop_step * dt
    if stop_reason == "converged":
        print(
            f"Stopped early at T_stop={stop_time:.6g} (converged; "
            f"tol={config.convergence_tol}, window={config.convergence_window_time}, min_time={config.convergence_min_time})."
        )
    else:
        print(
            f"Reached T_max={T_max:.6g} without convergence ("
            f"tol={config.convergence_tol}, window={config.convergence_window_time}, min_time={config.convergence_min_time})."
        )

    t_values = np.asarray(saved_times, dtype=np.float64)
    u_num = np.column_stack(saved_u)
    v_num = np.column_stack(saved_v)

    SetupDes = rf"""
    $a$ = {config.a}, $b$ = {config.b}, $c$ = {config.c}, $\alpha$ = {config.alpha};
    $m$ = {config.m}, $\beta$ = {config.beta}, $\chi_0$ = {config.chi};
    $\mu$ = {config.mu}, $\nu$ = {config.nu}, $\gamma$ = {config.gamma}; $N$ = {Nx}, $T_{{\max}}$ = {T_max};
    $T_{{stop}}$ = {stop_time:.2f};
    $u^*$ = {config.uStar}, $\epsilon$ = {config.epsilon}, $\epsilon2$ = {config.epsilon2}, $n$ = {config.eigen_index}.
    """

    create_static_plots(
        t_values, x_values, u_num, v_num, config.uStar, config.vStar, SetupDes, FileBaseName
    )
    if config.generate_video == "yes":
        create_animation(t_values, u_num, v_num, config.uStar, config.vStar, SetupDes, FileBaseName)

    return SimulationResult(
        x_values=np.asarray(x_values, dtype=np.float64),
        t_values=np.asarray(t_values, dtype=np.float64),
        u_num=np.asarray(u_num, dtype=np.float64),
        v_num=np.asarray(v_num, dtype=np.float64),
        stop_time=float(stop_time),
        stop_reason=stop_reason,
        dt=float(dt),
        setup_description=SetupDes,
    )


def create_static_plots( t_mesh: np.ndarray, x_mesh: np.ndarray,
    u_data: np.ndarray, v_data: np.ndarray, uStar: float,
    vStar: float, SetupDes: str, FileBaseName: str,) -> None:
    """
    Create and save static 3D plots of the simulation data.

    Parameters:
    - t_mesh (np.ndarray): Temporal mesh points for the simulation.
    - x_mesh (np.ndarray): Spatial mesh points for the simulation.
    - u_data (np.ndarray): 2D array of simulation data values for each (time, space) pair.
    - v_data (np.ndarray): 2D array of additional simulation data values for each (time, space) pair.
    - uStar (float): Reference value for creating a constant plane in the 3D plot of `u`.
    - vStar (float): Reference value for creating a constant plane in the 3D plot of `v`.
    - SetupDes (str): Description of the setup, used as the title of the plot.
    - FileBaseName (str): Base name for saving the output plot files.

    The function generates:
    - A 3D surface plot of `u_data` and `v_data` over time and space.
    - A reference plane at `uStar` and another at zero for comparison in the plot of `u`.
    - A reference plane at `vStar` and another at zero for comparison in the plot of `v`.
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
    #surf_v = ax_3d_v.plot_surface(T_grid, X_grid, v_data, cmap="viridis", alpha=0.8)

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
    Create and save an animation of u and v data over time, side by side. The
    values of the steady states uStar and vStar are plotted with u and v,
    respectively.

    Parameters:
        time_data (ndarray): 1D array of time steps
        u_data (ndarray): 2D array of u values over time
        v_data (ndarray): 2D array of v values over time
        uStar (float): Reference value for u (steady constant state).
        vStar (float): Reference value for v (steady constant state).
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

    ani = animation.FuncAnimation(fig, update, frames=frames, interval=50, blit=True)

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
    argparse.Namespace: Parsed arguments as an object with attributes 
    corresponding to the parameters.

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
    --c (float): Parameter c (default: 1).
    --mu (float): Parameter mu (default: 1).
    --nu (float): Parameter nu (default: 1).
    --gamma (float): Parameter gamma (default: 1).
    --meshsize (int): Parameter for spatial mesh size (default: 50).
    --time (float): Parameter for time to lapse (default: 2.5).
    --eigen_index (int): Parameter eigen index (default: 0, letting system choose).
    --epsilon (float): Parameter perturbation epsilon (default: 0.001).
    --epsilon2 (float): Parameter perturbation epsilon2 (default: 0.0).
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
        "--m", type=float, default=1.0, help="Parameter m (default: 1.0)"
    )
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
        "--a", type=float, default=1.0, help="Parameter a (default: 1.0)"
    )
    parser.add_argument(
        "--b", type=float, default=1.0, help="Parameter b (default: 1.0)"
    )
    parser.add_argument(
        "--c", type=float, default=1.0, help="Parameter c (default: 1.0)"
    )
    parser.add_argument(
        "--mu", type=float, default=1.0, help="Parameter mu (default: 1.0)"
    )
    parser.add_argument(
        "--nu", type=float, default=1, help="Parameter nu (default: 1.0)"
    )
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
    parser.add_argument(
        "--epsilon2",
        type=float,
        default=0.0,
        help="Parameter perturbation epsilon2 (default: 0.0)",
    )
    parser.add_argument(
        "--until_converged",
        choices=["yes", "no"],
        default="no",
        help="Stop early once the solution is approximately steady (default: no)",
    )
    parser.add_argument(
        "--convergence_tol",
        type=float,
        default=1e-4,
        help="Convergence tolerance (default: 1e-4)",
    )
    parser.add_argument(
        "--convergence_window_time",
        type=float,
        default=5.0,
        help="Time window length used for convergence checks (default: 5.0)",
    )
    parser.add_argument(
        "--convergence_min_time",
        type=float,
        default=10.0,
        help="Minimum time before convergence checks can stop the run (default: 10.0)",
    )
    parser.add_argument(
        "--max_saved_frames",
        type=int,
        default=2000,
        help="Maximum number of time snapshots saved when --until_converged=yes (default: 2000)",
    )
    parser.add_argument(
        "--save_data",
        choices=["yes", "no"],
        default="yes",
        help="Save numerical simulation data to disk (default: yes)",
    )
    parser.add_argument(
        "--data_format",
        choices=["npz"],
        default="npz",
        help="Format for saved numerical data (default: npz)",
    )
    parser.add_argument(
        "--save_max_frames",
        type=int,
        default=2000,
        help="Maximum number of frames stored in the saved data (default: 2000)",
    )
    parser.add_argument(
        "--save_summary6",
        choices=["yes", "no"],
        default="yes",
        help="Save a 6-frame (0,20,...,100 percent) summary figure (default: yes)",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default="",
        help="Override output filename basename (default: auto from parameters)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory for saved files (default: current working directory)",
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

    output_dir = (config.output_dir or "").strip() or "."
    os.makedirs(output_dir, exist_ok=True)

    user_basename = (config.basename or "").strip()
    if user_basename and (os.sep in user_basename or (os.altsep and os.altsep in user_basename)):
        raise ValueError("`--basename` must not contain path separators; use `--output_dir` for directories.")

    # Using the above parameters to generate a file base name string
    auto_basename = (
        f"a={config.a}_b={config.b}_c={config.c}_alpha={config.alpha}_m={config.m}_beta={config.beta}_chi={config.chi}_mu={config.mu}_nu={config.nu}_gamma={config.gamma}_meshsize={config.meshsize}_time={config.time}_epsilon={config.epsilon}_epsilon2={config.epsilon2}_eigen_index={config.eigen_index}"
    )
    basename = (user_basename or auto_basename).replace(".", "-")
    file_base = os.path.join(output_dir, basename)
    print(f"Output files will be saved with the basename:\n\t {file_base}\n")

    # Run the solver
    if (
        config.confirm == "yes"
        or questionary.confirm("Do you want to continue the simulation?").ask()
    ):
        print("Continuing simulation...")
        if config.until_converged == "yes":
            result = RK4_until_converged(config=config, FileBaseName=file_base)
        else:
            result = RK4(config=config, FileBaseName=file_base)

        if config.save_summary6 == "yes":
            create_six_frame_summary(
                x_values=result.x_values,
                t_values=result.t_values,
                u_num=result.u_num,
                v_num=result.v_num,
                uStar=config.uStar,
                vStar=config.vStar,
                setup_description=result.setup_description,
                file_base_name=file_base,
            )

        if config.save_data == "yes":
            if config.data_format != "npz":
                raise ValueError(f"Unsupported data_format: {config.data_format}")
            npz_filename = f"{file_base}.npz"
            save_simulation_data_npz(npz_filename, config=config, result=result)
            print(f"Simulation data saved to {npz_filename}")
    else:
        print("Exiting simulation.")
        exit()


if __name__ == "__main__":
    main()
