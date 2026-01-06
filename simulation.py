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
import shutil
import sys
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from collections import deque
from typing import Any, Final, List, Optional
from matplotlib import animation

# import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import questionary
import termplotlib as tpl
from plots import USE_TEX, create_six_frame_summary, create_static_plots

# from scipy.linalg import solve_banded
from scipy.sparse import diags
from scipy.sparse.linalg import factorized, spsolve
from tabulate import tabulate
from tqdm import tqdm  # Import tqdm for progress bar

from npz_io import _downsample_time_indices
from npz_io import load_simulation_data_npz as _load_simulation_data_npz
from npz_io import save_simulation_data_npz as _save_simulation_data_npz
from thresholds import chi_star_threshold_continuum_1d

# Optional acceleration: stencil derivatives can be JIT-compiled with Numba.
try:  # pragma: no cover
    from numba import njit  # type: ignore
except Exception:  # pragma: no cover
    njit = None  # type: ignore

#
# NOTE: keep the simulator self-contained: prefer NumPy `.npz` files for saved
# data instead of pickled formats.

# Matplotlib rc configuration (including usetex) is handled in `plots.py`.


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
    eigen_mode_n: Optional[int] = None
    epsilon: float = 0.001
    epsilon2: float = 0.0
    restart_from: str = ""

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
    save_static_plots: str = "yes"
    save_summary6: str = "yes"
    dt_factor: float = 2.0
    stop_on_nonfinite: str = "yes"
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
    eigen_mode_n_resolved: int = field(init=False, default=None)

    def __post_init__(self):
        # Using object.__setattr__ because the class is frozen
        object.__setattr__(self, "uStar", (self.a / self.b) ** (1 / self.alpha))
        object.__setattr__(
            self,
            "vStar",
            self.nu / self.mu * (self.a / self.b) ** (self.gamma / self.alpha),
        )

        # Compute ChiStar
        if self.c + self.vStar <= 0:
            raise ValueError("Expected c + vStar > 0 for sensitivity (c+v)^(-beta).")
        chistar = (
            (self.c + self.vStar) ** self.beta
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
                    ((self.c + self.vStar) ** self.beta)
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
                        / ((self.c + self.vStar) ** self.beta + 1e-10)
                    )
                    * (1 - self.mu / (self.mu - lambda_n + 1e-10))
                    - self.a * self.alpha
                )
                if sigma_n > 0:
                    positive_sigmas.append(sigma_n)
        object.__setattr__(self, "positive_sigmas", positive_sigmas)

        # Eigenmode indexing (paper vs legacy):
        #
        # - Paper II indexes Neumann modes by n>=0 with n=0 the constant mode.
        # - The legacy CLI flag `--eigen_index k` uses k=n+1, with k=0 reserved for auto selection.
        # - The new flag `--eigen_mode_n n` directly specifies the paper mode index n (0-based).
        #
        # Precedence: if `--eigen_mode_n` is provided, it overrides `--eigen_index`.
        if self.eigen_mode_n is not None:
            if int(self.eigen_mode_n) < 0:
                raise ValueError("--eigen_mode_n must be >= 0")
            mode_n = int(self.eigen_mode_n)
        else:
            if self.eigen_index == 0:
                if len(positive_sigmas) > 0:
                    object.__setattr__(self, "eigen_index", 2)
                    print(
                        "Auto eigenmode: first nonconstant selected "
                        "(mode n=1; --eigen_index=2).\n"
                    )
                else:
                    object.__setattr__(self, "eigen_index", 1)
                    print("Auto eigenmode: constant selected (mode n=0; --eigen_index=1).\n")

            if int(self.eigen_index) < 1:
                raise ValueError("--eigen_index must be >= 0")
            mode_n = int(self.eigen_index) - 1

        object.__setattr__(self, "eigen_mode_n_resolved", int(mode_n))

        x_values = np.linspace(0, self.L, int(self.meshsize) + 1, dtype=np.float64)
        if (self.restart_from or "").strip():
            data = load_simulation_data_npz(self.restart_from)
            u_num = np.asarray(data.get("u_num"))
            if u_num.ndim != 2:
                raise ValueError(
                    f"`--restart_from` expects a 2D `u_num` array in the .npz, got shape {u_num.shape}"
                )
            u0 = np.asarray(u_num[:, -1], dtype=np.float64)
            if u0.shape[0] != x_values.shape[0]:
                raise ValueError(
                    "`--restart_from` mesh mismatch: expected u0 with "
                    f"{x_values.shape[0]} points (meshsize={self.meshsize}), got {u0.shape[0]}."
                )
            u0 = (
                u0
                + self.epsilon
                * np.cos(((self.eigen_mode_n_resolved) * np.pi / self.L) * x_values)
                + self.epsilon2
                * np.cos(((self.eigen_mode_n_resolved + 1) * np.pi / self.L) * x_values)
            )
        else:
            u0 = (
                self.uStar
                + self.epsilon
                * np.cos(((self.eigen_mode_n_resolved) * np.pi / self.L) * x_values)
                + self.epsilon2
                * np.cos(((self.eigen_mode_n_resolved + 1) * np.pi / self.L) * x_values)
            )

        object.__setattr__(self, "uinit", u0)

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
        emn = "unset" if self.eigen_mode_n is None else str(self.eigen_mode_n)
        print(
            f"\tepsilon = {self.epsilon}, eigen_index = {self.eigen_index}, "
            f"eigen_mode_n = {emn} (resolved mode n = {self.eigen_mode_n_resolved})"
        )
        print("5. Simulation Parameters:")
        print(f"\tL = {self.L}, MeshSize = {self.meshsize}, time = {self.time}")

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
        print(
            f"\nChi* (lower bound) = {self.ChiStar:.3f}; "
            f"min discrete chi_(0,n)^* = {self.ChiStar_min:.3f}; "
            f"chosen chi = {self.chi:.3f}"
        )

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
    dx, A, solver = _get_v_solver(L=L, Nx=Nx, mu=mu)

    # Define right-hand side
    b = -(dx**2) * float(nu) * np.power(vector_u, float(gamma))

    # Solve system
    if solver is None:
        v = spsolve(A, b)
    else:
        v = solver(b)

    # Print matrix A in a readable format
    if diagnostic:
        print("\nMatrix A:")
        print("-" * 50)
        A_dense = A.toarray()
        for i in range(Nx + 1):
            row = [f"{x:8.3f}" for x in A_dense[i]]
            print(f"Row {i:2d}: {' '.join(row)}")
        print("-" * 50 + "\n")
        row = [f"{x:8.3f}" for x in vector_u]
        print(f"vector_u: {' '.join(row)}")
        row = [f"{x:8.3f}" for x in b]
        print(f"b:        {' '.join(row)}")
        row = [f"{x:8.3f}" for x in v]
        print(f"v:        {' '.join(row)}")
        print("-" * 50 + "\n")

    return v


@lru_cache(maxsize=32)
def _get_v_solver(*, L: float, Nx: int, mu: float) -> tuple[float, Any, Optional[Any]]:
    """
    Return (dx, A, solver) for the elliptic solve in `solve_v`.

    The sparse matrix A depends only on (L, Nx, mu), so we cache its factorization
    to avoid rebuilding/factorizing it at every RK4 stage.
    """
    L = float(L)
    Nx = int(Nx)
    mu = float(mu)
    dx = L / Nx

    main_diag = np.full(Nx + 1, -(2 + mu * dx**2))
    upper_diag = np.ones(Nx)
    lower_diag = np.ones(Nx)
    upper_diag[0] = 2
    lower_diag[-1] = 2

    # Use CSC for factorization/backsolves.
    A = diags([main_diag, upper_diag, lower_diag], [0, 1, -1], format="csc")

    try:
        solver = factorized(A)
    except Exception:  # pragma: no cover (fallback path depends on SciPy internals)
        solver = None
        # Keep A in a usable format for spsolve fallback.
        A = A.tocsr()

    return dx, A, solver


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


def _config_metadata(config: SimulationConfig) -> dict:
    d = asdict(config)
    for key in ("uinit", "vinit"):
        d.pop(key, None)
    d["usetex"] = bool(USE_TEX)
    return d


def save_simulation_data_npz(
    filename: str,
    config: SimulationConfig,
    result: SimulationResult,
    *,
    max_frames: Optional[int] = None,
) -> str:
    max_frames = config.save_max_frames if max_frames is None else int(max_frames)
    return _save_simulation_data_npz(
        filename,
        config_metadata=_config_metadata(config),
        setup_description=result.setup_description,
        stop_reason=result.stop_reason,
        stop_time=result.stop_time,
        dt=result.dt,
        x_values=result.x_values,
        t_values=result.t_values,
        u_num=result.u_num,
        v_num=result.v_num,
        max_frames=int(max_frames),
    )


def load_simulation_data_npz(filename: str) -> dict:
    return _load_simulation_data_npz(filename)


def chi_star_threshold_discrete(
    config: SimulationConfig, *, n_max: int = 5000
) -> float:
    return chi_star_threshold_continuum_1d(
        u_star=float(config.uStar),
        v_star=float(config.vStar),
        c=float(config.c),
        a=float(config.a),
        alpha=float(config.alpha),
        mu=float(config.mu),
        nu=float(config.nu),
        gamma=float(config.gamma),
        m=float(config.m),
        beta=float(config.beta),
        L=float(config.L),
        n_max=int(n_max),
    )


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

    dx = float(L) / int(Nx)
    f = np.ascontiguousarray(vector_f, dtype=np.float64)
    if njit is not None:
        return _first_derivative_nbc_numba(f, dx)
    return _first_derivative_nbc_numpy(f, dx)


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
    dx = float(L) / int(Nx)
    f = np.ascontiguousarray(vector_f, dtype=np.float64)
    if njit is not None:
        return _laplacian_nbc_numba(f, dx)
    return _laplacian_nbc_numpy(f, dx)


def _first_derivative_nbc_numpy(vector_f: np.ndarray, dx: float) -> np.ndarray:
    """
    Central difference approximation of u_x with Neumann BC enforced by setting
    endpoint derivatives to 0.
    """
    out = np.empty_like(vector_f, dtype=np.float64)
    out[0] = 0.0
    out[-1] = 0.0
    out[1:-1] = (vector_f[2:] - vector_f[:-2]) / (2.0 * dx)
    return out


def _laplacian_nbc_numpy(vector_f: np.ndarray, dx: float) -> np.ndarray:
    """
    Central difference approximation of u_xx with Neumann BC enforced via
    ghost-point reflection: f[-1]=f[1], f[N+1]=f[N-1].
    """
    out = np.empty_like(vector_f, dtype=np.float64)
    inv_dx2 = 1.0 / (dx * dx)
    out[1:-1] = (vector_f[2:] - 2.0 * vector_f[1:-1] + vector_f[:-2]) * inv_dx2
    out[0] = 2.0 * (vector_f[1] - vector_f[0]) * inv_dx2
    out[-1] = 2.0 * (vector_f[-2] - vector_f[-1]) * inv_dx2
    return out


if njit is not None:  # pragma: no cover

    # Numba caching is fragile on some filesystems / editable installs (e.g. symlinks),
    # so keep caching disabled by default for robustness.
    @njit(cache=False)
    def _first_derivative_nbc_numba(vector_f: np.ndarray, dx: float) -> np.ndarray:
        out = np.empty_like(vector_f)
        n = vector_f.shape[0]
        out[0] = 0.0
        out[n - 1] = 0.0
        inv_2dx = 1.0 / (2.0 * dx)
        for i in range(1, n - 1):
            out[i] = (vector_f[i + 1] - vector_f[i - 1]) * inv_2dx
        return out

    @njit(cache=False)
    def _laplacian_nbc_numba(vector_f: np.ndarray, dx: float) -> np.ndarray:
        out = np.empty_like(vector_f)
        n = vector_f.shape[0]
        inv_dx2 = 1.0 / (dx * dx)
        for i in range(1, n - 1):
            out[i] = (vector_f[i + 1] - 2.0 * vector_f[i] + vector_f[i - 1]) * inv_dx2
        out[0] = 2.0 * (vector_f[1] - vector_f[0]) * inv_dx2
        out[n - 1] = 2.0 * (vector_f[n - 2] - vector_f[n - 1]) * inv_dx2
        return out


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
    L = float(config.L)
    Nx = int(config.meshsize)
    m = float(config.m)
    beta = float(config.beta)
    alpha = float(config.alpha)
    chi = float(config.chi)
    a = float(config.a)
    b = float(config.b)
    c = float(config.c)
    mu = float(config.mu)
    nu = float(config.nu)
    gamma = float(config.gamma)

    u_vec = np.ascontiguousarray(u, dtype=np.float64)
    v_vec = np.ascontiguousarray(v, dtype=np.float64)

    u_xx = laplacian_NBC(L, Nx, u_vec)
    u_x = first_derivative_NBC(L, Nx, u_vec)
    v_x = first_derivative_NBC(L, Nx, v_vec)

    # Base term: diffusion u_xx
    out = u_xx

    # Reuse shared quantities to reduce allocations.
    cv = c + v_vec
    inv_cv_beta = np.power(cv, -beta)  # (c+v)^{-beta}

    # u^m (used by term1 and term3). For m==1, reuse u directly.
    if m == 1.0:
        u_m = u_vec
    else:
        u_m = np.power(u_vec, m)

    # term1: beta*chi*(v_x^2)*u^m*(c+v)^{-(beta+1)} (skip when beta==0).
    if beta != 0.0:
        tmp = np.multiply(v_x, v_x)  # v_x^2
        tmp *= u_m
        tmp *= inv_cv_beta / cv  # (c+v)^{-(beta+1)}
        out += (beta * chi) * tmp

    # term2: m*chi*u^{m-1}*u_x*v_x*(c+v)^{-beta} (skip when chi==0 or m==0).
    if chi != 0.0 and m != 0.0:
        tmp = np.multiply(u_x, v_x)
        if m != 1.0:
            tmp *= np.power(u_vec, m - 1.0)
        tmp *= inv_cv_beta
        out -= (m * chi) * tmp

    # term3: chi*u^m*(mu*v - nu*u^gamma)*(c+v)^{-beta}
    if chi != 0.0:
        tmp = mu * v_vec
        if nu != 0.0:
            tmp = tmp - nu * np.power(u_vec, gamma)
        tmp = tmp * u_m
        tmp = tmp * inv_cv_beta
        out -= chi * tmp

    # logistic: a*u - b*u^{1+alpha}
    if a != 0.0:
        out += a * u_vec
    if b != 0.0:
        out -= b * np.power(u_vec, 1.0 + alpha)

    return out


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
    if float(config.dt_factor) <= 0:
        raise ValueError(f"dt_factor must be positive, got {config.dt_factor!r}")
    Nt = max(1, int(float(config.dt_factor) * (int(4 * T * Nx * Nx / L**2) + 1)))
    # dx = L / Nx
    dt = T / Nt

    x_values = np.linspace(0, L, int(Nx) + 1, dtype=np.float64)
    u_current = np.array(config.uinit, dtype=np.float64, copy=True)
    v_current = np.array(config.vinit, dtype=np.float64, copy=True)

    save_indices = _downsample_time_indices(int(Nt) + 1, int(config.save_max_frames))
    n_saved = int(save_indices.size)
    if n_saved <= 0:
        raise RuntimeError("Internal error: no saved frames selected.")

    saved_times = np.empty(n_saved, dtype=np.float64)
    saved_u = np.empty((n_saved, int(Nx) + 1), dtype=np.float64)
    saved_v = np.empty((n_saved, int(Nx) + 1), dtype=np.float64)

    save_pos = 0
    if int(save_indices[0]) != 0:
        raise RuntimeError("Internal error: expected first saved index to be 0.")

    saved_times[save_pos] = 0.0
    saved_u[save_pos, :] = u_current
    saved_v[save_pos, :] = v_current
    save_pos += 1

    # Time integration
    stop_step: Optional[int] = None
    stop_reason = "t_max"
    for n in tqdm(range(Nt), desc="Progress..."):
        # rk4 steps
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

        # Update
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

        if config.stop_on_nonfinite == "yes":
            if not (np.isfinite(u_current).all() and np.isfinite(v_current).all()):
                stop_step = int(n) + 1
                stop_reason = "nonfinite"
                break

        now_step = int(n) + 1
        while save_pos < n_saved and now_step == int(save_indices[save_pos]):
            saved_times[save_pos] = float(now_step) * float(dt)
            saved_u[save_pos, :] = u_current
            saved_v[save_pos, :] = v_current
            save_pos += 1

    if stop_reason == "t_max" and save_pos != n_saved:
        raise RuntimeError(f"Internal error: saved {save_pos} frames but expected {n_saved}.")

    saved_times = saved_times[:save_pos]
    saved_u = saved_u[:save_pos, :]
    saved_v = saved_v[:save_pos, :]

    u_num = saved_u.T
    v_num = saved_v.T
    t_values = saved_times

    # Setup description for the title
    SetupDes = rf"""
    $a$ = {a}, $b$ = {b}, $c$ = {c}, $\alpha$ = {alpha};
    $m$ = {m}, $\beta$ = {beta}, $\chi_0$ = {chi};
    $\mu$ = {mu}, $\nu$ = {nu}, $\gamma$ = {gamma}; $N$ = {Nx}, $T$ = {T};
    $u^*$ = {uStar}, $\epsilon$ = {epsilon}, $\epsilon_2$ = {epsilon2}, $n_0$ = {config.eigen_mode_n_resolved}.
    """

    # Create static plots
    if config.save_static_plots == "yes":
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
        stop_time=float((T if stop_step is None else stop_step * dt)),
        stop_reason=stop_reason,
        dt=float(dt),
        setup_description=SetupDes,
    )


def RK4_until_converged(
    config: SimulationConfig, FileBaseName="Simulation"
) -> SimulationResult:
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

    # Match the legacy dt choice derived from `Nt = 2*(int(4*T*Nx^2/L^2)+1)`,
    # but allow an optional refinement factor.
    if float(config.dt_factor) <= 0:
        raise ValueError(f"dt_factor must be positive, got {config.dt_factor!r}")
    Nt_max = max(1, int(float(config.dt_factor) * (int(4 * T_max * Nx * Nx / L**2) + 1)))
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
        ok = (du <= config.convergence_tol * u_scale) and (
            dv <= config.convergence_tol * v_scale
        )
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

        if config.stop_on_nonfinite == "yes":
            if not (np.isfinite(u_current).all() and np.isfinite(v_current).all()):
                stop_step = now_step
                stop_reason = "nonfinite"
                if config.verbose == "yes":
                    print(f"[stop] nonfinite encountered at step={stop_step} t={now_time:.6g}")
                break

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
    elif stop_reason == "nonfinite":
        print(f"Stopped early at T_stop={stop_time:.6g} (non-finite values encountered).")
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
    $u^*$ = {config.uStar}, $\epsilon$ = {config.epsilon}, $\epsilon_2$ = {config.epsilon2}, $n_0$ = {config.eigen_mode_n_resolved}.
    """

    if config.save_static_plots == "yes":
        create_static_plots(
            t_values,
            x_values,
            u_num,
            v_num,
            config.uStar,
            config.vStar,
            SetupDes,
            FileBaseName,
        )
    if config.generate_video == "yes":
        create_animation(
            t_values, u_num, v_num, config.uStar, config.vStar, SetupDes, FileBaseName
        )

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
    argv = sys.argv[1:]

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Load default parameter values from a YAML file (CLI flags override)",
    )
    pre_parser.add_argument(
        "--config_warn_unknown",
        choices=["yes", "no"],
        default="no",
        help="Warn about unknown YAML keys (default: no)",
    )
    pre_args, _ = pre_parser.parse_known_args(argv)

    config_overrides: dict[str, Any] = {}
    if pre_args.config:
        config_overrides = _load_yaml_config_as_overrides(pre_args.config)

    parser = _build_arg_parser()
    _apply_parser_defaults_from_config(
        parser, config_overrides, warn_unknown=(pre_args.config_warn_unknown == "yes")
    )

    try:
        import argcomplete  # type: ignore
    except ModuleNotFoundError:
        argcomplete = None
    if argcomplete is not None:
        argcomplete.autocomplete(parser)

    args = parser.parse_args(argv)
    args_dict = vars(args)
    args_dict.pop("config", None)
    args_dict.pop("config_warn_unknown", None)
    return SimulationConfig(**args_dict)


def _flatten_yaml_mapping(data: Any) -> dict[str, Any]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError("YAML config must be a mapping (dict-like) at the top level.")
    out: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise TypeError("YAML config keys must be strings.")
        if isinstance(value, dict):
            out.update(_flatten_yaml_mapping(value))
        else:
            out[key] = value
    return out


def _yaml_yes_no(value: Any) -> Optional[str]:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if value is None:
        return None
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ("yes", "y", "true", "1", "on"):
            return "yes"
        if lowered in ("no", "n", "false", "0", "off"):
            return "no"
    return None


def _load_yaml_config_as_overrides(path: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency: PyYAML is required for `--config`. "
            "Install it with `pip install pyyaml`."
        ) from exc

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return _flatten_yaml_mapping(raw)


def _apply_parser_defaults_from_config(
    parser: argparse.ArgumentParser,
    config: dict[str, Any],
    *,
    warn_unknown: bool = False,
) -> None:
    if not config:
        return

    by_dest = {
        action.dest: action for action in parser._actions if action.dest
    }  # pylint: disable=protected-access

    if warn_unknown:
        unknown_keys = [k for k in config.keys() if k not in by_dest]
        for k in unknown_keys:
            print(f"Warning: unknown config key ignored: {k}", file=sys.stderr)

    coerced: dict[str, Any] = {}
    for key, value in config.items():
        action = by_dest.get(key)
        if action is None:
            continue

        if action.choices and list(action.choices) == ["yes", "no"]:
            yn = _yaml_yes_no(value)
            if yn is None:
                raise ValueError(
                    f"Invalid value for `{key}`: {value!r} (expected yes/no or boolean)"
                )
            coerced[key] = yn
            continue

        if value is None:
            coerced[key] = None
            continue

        if action.type is not None:
            try:
                coerced[key] = action.type(value)  # pylint: disable=not-callable
            except (TypeError, ValueError):
                coerced[key] = action.type(str(value))  # pylint: disable=not-callable
        else:
            coerced[key] = value

        if action.choices and coerced[key] not in action.choices:
            raise ValueError(
                f"Invalid value for `{key}`: {coerced[key]!r} (choices: {sorted(action.choices)})"
            )

    parser.set_defaults(**coerced)


def _build_arg_parser() -> argparse.ArgumentParser:
    examples = """Examples:
  # Run from a YAML config (CLI flags override YAML values)
  chemotaxis-sim --config config.example.yaml

  # Paper-II eigenmode indexing (0-based): n=2 -> cos(2πx/L)
  chemotaxis-sim --chi 30 --meshsize 100 --time 5 --eigen_mode_n 2 --epsilon 0.5

  # Batch workflow: keep only summary6 + saved data (skip heavy 3D plots)
  chemotaxis-sim --config config.example.yaml --save_data yes --save_summary6 yes --save_static_plots no

  # Post-process heavy plots later from the saved .npz
  chemotaxis-plot images/branch_capture/some_run.npz

  # Legacy eigen indexing (backward compatible): k=n+1 (k=2 -> n=1 -> cos(πx/L))
  chemotaxis-sim --chi 30 --meshsize 100 --time 5 --eigen_index 2 --epsilon 0.5
"""
    parser = argparse.ArgumentParser(
        description="Run 1D chemotaxis simulations (parabolic-elliptic).",
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Load default parameter values from a YAML file (CLI flags override)",
    )
    parser.add_argument(
        "--config_warn_unknown",
        choices=["yes", "no"],
        default="no",
        help="Warn about unknown YAML keys (default: no)",
    )

    io_group = parser.add_argument_group("Output / UI")
    io_group.add_argument(
        "--confirm",
        choices=["yes", "no"],
        default="yes",
        help="Skip confirmation prompt if set to yes (default: no)",
    )
    io_group.add_argument(
        "--generate_video",
        choices=["yes", "no"],
        default="no",
        help="Generate MP4 animation (default: no)",
    )
    io_group.add_argument(
        "--verbose",
        choices=["yes", "no"],
        default="no",
        help="Enable verbose output (default: no)",
    )

    model_group = parser.add_argument_group("Model parameters")
    model_group.add_argument(
        "--m", type=float, default=1.0, help="Parameter m (default: 1.0)"
    )
    model_group.add_argument(
        "--beta", type=float, default=1.0, help="Parameter beta (default: 1.0)"
    )
    model_group.add_argument(
        "--alpha", type=float, default=1.0, help="Parameter alpha (default: 1.0)"
    )
    model_group.add_argument(
        "--chi", type=float, default=-1.0, help="Parameter chi (default: -1.0)"
    )
    model_group.add_argument(
        "--a", type=float, default=1.0, help="Parameter a (default: 1.0)"
    )
    model_group.add_argument(
        "--b", type=float, default=1.0, help="Parameter b (default: 1.0)"
    )
    model_group.add_argument(
        "--c", type=float, default=1.0, help="Parameter c (default: 1.0)"
    )
    model_group.add_argument(
        "--mu", type=float, default=1.0, help="Parameter mu (default: 1.0)"
    )
    model_group.add_argument(
        "--nu", type=float, default=1.0, help="Parameter nu (default: 1.0)"
    )
    model_group.add_argument(
        "--gamma", type=float, default=1.0, help="Parameter gamma (default: 1.0)"
    )

    grid_group = parser.add_argument_group("Grid / initial condition")
    grid_group.add_argument(
        "--L",
        type=float,
        default=1.0,
        help="Domain length L (default: 1.0)",
    )
    grid_group.add_argument(
        "--meshsize",
        type=int,
        default=50,
        help="Parameter for spatial mesh size (default: 50)",
    )
    grid_group.add_argument(
        "--time",
        type=float,
        default=2.5,
        help="Parameter for time to lapse (default: 2.5)",
    )
    grid_group.add_argument(
        "--eigen_index",
        type=int,
        default=0,
        help=(
            "Legacy eigen index for the initial cosine perturbation (kept for backward compatibility). "
            "Interpretation: k=n+1 with paper mode n>=0, and k=0 means auto-select "
            "(constant if stable; otherwise the first nonconstant). "
            "Examples: k=1 -> n=0 (constant), k=2 -> n=1 (first nonconstant). "
            "Prefer `--eigen_mode_n` for a paper-0-based mode index."
        ),
    )
    grid_group.add_argument(
        "--eigen_mode_n",
        type=int,
        default=None,
        help=(
            "Paper mode index n>=0 for the initial cosine perturbation "
            "(n=0 constant, n=1 first nonconstant cos(pi x/L), etc). "
            "If provided, this overrides `--eigen_index`."
        ),
    )
    grid_group.add_argument(
        "--epsilon",
        type=float,
        default=0.001,
        help="Parameter perturbation epsilon (default: 0.001)",
    )
    grid_group.add_argument(
        "--epsilon2",
        type=float,
        default=0.0,
        help="Parameter perturbation epsilon2 (default: 0.0)",
    )
    grid_group.add_argument(
        "--restart_from",
        type=str,
        default="",
        help="Restart from a saved .npz (uses the last saved u as u0; epsilon terms are added on top; default: none)",
    )

    stopping_group = parser.add_argument_group("Stopping criteria")
    stopping_group.add_argument(
        "--until_converged",
        choices=["yes", "no"],
        default="no",
        help="Stop early once the solution is approximately steady (default: no)",
    )
    stopping_group.add_argument(
        "--convergence_tol",
        type=float,
        default=1e-4,
        help="Convergence tolerance (default: 1e-4)",
    )
    stopping_group.add_argument(
        "--convergence_window_time",
        type=float,
        default=5.0,
        help="Time window length used for convergence checks (default: 5.0)",
    )
    stopping_group.add_argument(
        "--convergence_min_time",
        type=float,
        default=10.0,
        help="Minimum time before convergence checks can stop the run (default: 10.0)",
    )
    stopping_group.add_argument(
        "--max_saved_frames",
        type=int,
        default=2000,
        help="Maximum number of time snapshots saved when --until_converged=yes (default: 2000)",
    )

    output_group = parser.add_argument_group("Saved outputs")
    output_group.add_argument(
        "--save_data",
        choices=["yes", "no"],
        default="yes",
        help="Save numerical simulation data to disk (default: yes)",
    )
    output_group.add_argument(
        "--data_format",
        choices=["npz"],
        default="npz",
        help="Format for saved numerical data (default: npz)",
    )
    output_group.add_argument(
        "--save_max_frames",
        type=int,
        default=2000,
        help="Maximum number of frames stored in the saved data (default: 2000)",
    )
    output_group.add_argument(
        "--save_summary6",
        choices=["yes", "no"],
        default="yes",
        help="Save a 6-frame (0,20,...,100 percent) summary figure (default: yes)",
    )
    output_group.add_argument(
        "--save_static_plots",
        choices=["yes", "no"],
        default="yes",
        help="Save the main 3D static plots (<basename>.png/.jpeg) (default: yes)",
    )
    output_group.add_argument(
        "--basename",
        type=str,
        default="",
        help="Override output filename basename (default: auto from parameters)",
    )
    output_group.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory for saved files (default: current working directory)",
    )

    numerics_group = parser.add_argument_group("Numerics / stability")
    numerics_group.add_argument(
        "--dt_factor",
        type=float,
        default=2.0,
        help=(
            "Multiply the default time-step refinement factor (default: 2.0). "
            "Larger values reduce dt and may improve stability at higher cost."
        ),
    )
    numerics_group.add_argument(
        "--stop_on_nonfinite",
        choices=["yes", "no"],
        default="yes",
        help="Stop early if NaN/Inf occurs in u or v (default: yes)",
    )

    return parser


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
    if not sys.argv[1:] and os.environ.get("_ARGCOMPLETE") != "1":
        print("chemotaxis-sim (short help)")
        print()
        print("Usage:")
        print("  chemotaxis-sim --config <config.yaml> [overrides...]")
        print("  chemotaxis-sim --chi <chi0> --meshsize <N> --time <T> --eigen_index <k> [options]")
        print()
        print("Common options:")
        print("  --config <yaml>            Load defaults from YAML")
        print("  --output_dir <dir>         Where to write outputs")
        print("  --basename <name>          Override output basename")
        print("  --eigen_mode_n <n>         Paper mode index n>=0 (overrides --eigen_index)")
        print("  --save_data yes|no         Save .npz (default: yes)")
        print("  --save_summary6 yes|no     Save *_summary6.{png,jpeg} (default: yes)")
        print("  --save_static_plots yes|no Save <basename>.{png,jpeg} (default: yes)")
        print("  --generate_video yes|no    Save <basename>.mp4 (default: no)")
        print("  --until_converged yes|no   Stop early if steady (default: no)")
        print()
        print("Run `chemotaxis-sim --help` for full help.")
        return

    # Get immutable config
    config: Final[SimulationConfig] = parse_args()

    # # Display the parsed arguments
    config.display_parameters()

    output_dir = (config.output_dir or "").strip() or "."
    os.makedirs(output_dir, exist_ok=True)

    user_basename = (config.basename or "").strip()
    if user_basename and (
        os.sep in user_basename or (os.altsep and os.altsep in user_basename)
    ):
        raise ValueError(
            "`--basename` must not contain path separators; use `--output_dir` for directories."
        )

    # Using the above parameters to generate a file base name string
    auto_basename = f"a={config.a}_b={config.b}_c={config.c}_alpha={config.alpha}_m={config.m}_beta={config.beta}_chi={config.chi}_mu={config.mu}_nu={config.nu}_gamma={config.gamma}_meshsize={config.meshsize}_time={config.time}_epsilon={config.epsilon}_epsilon2={config.epsilon2}_eigen_index={config.eigen_index}"
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
            chi_star = chi_star_threshold_discrete(config)
            create_six_frame_summary(
                x_values=result.x_values,
                t_values=result.t_values,
                u_num=result.u_num,
                uStar=config.uStar,
                chi0=float(config.chi),
                chi_star=float(chi_star),
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
