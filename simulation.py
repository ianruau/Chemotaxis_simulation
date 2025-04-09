#!/usr/bin/env python3
#

import argparse
import math

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import questionary
import termplotlib as tpl
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
from scipy.linalg import solve_banded
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from tabulate import tabulate
from tqdm import tqdm  # Import tqdm for progress bar

config = {}  # Global dictionary for parameters

# Enable LaTeX rendering
rc("text", usetex=True)
# TO LIST:
# 1. Write CLI interface


def solve_v_1(L=1, Nx=50, vector_u=np.zeros(50), diagnostic=False):
    """
    Solves a linear system to compute the vector `v` based on the given parameters.

    Parameters:
    - L (int): Length of the domain (default is 1).
    - Nx (int): Number of grid points (default is 50).
    - vector_u (np.ndarray): Input vector `u` of size `Nx` (default is a zero vector).
    - diagnostic (bool): Flag for enabling diagnostic output (default is False).

    Returns:
    - np.ndarray: Solution vector `v` of size `Nx + 1`.

    Notes:
    - The function constructs a sparse matrix `A` using finite difference discretization.
    - Neumann boundary conditions are applied by modifying the first and last off-diagonal elements.
    - The right-hand side vector `b` is computed based on the input vector `u` and parameters.
    - The system `A * v = b` is solved using a sparse solver.
    """
    mu = config["mu"]
    nu = config["nu"]
    gamma = config["gamma"]
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
    return v


def first_derivative_NBC(L, Nx, vector_f):
    """
    Create a sparse n x n square matrix where:
    - Lower diagonal is -1
    - Upper diagonal is 1
    - First and last rows are all zeros
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


def laplacian_NBC(L, Nx, vector_f):
    """
    Create a sparse Nx x Nx square matrix where:
    - Main diagonal is -2
    - Lower diagonal is 1
    - Upper diagonal is 1
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


# Right-hand side function
def rhs(L, Nx, u, v):
    m = config["m"]
    beta = config["beta"]
    alpha = config["alpha"]
    chi = config["chi"]
    a = config["a"]
    b = config["b"]
    mu = config["mu"]
    nu = config["nu"]
    gamma = config["gamma"]

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
    # return u_xx + chemotaxis + logistic


def RK4(L=1, Nx=50, T=5, Epsilon=0.001, EigenIndex=2, FileBaseName="Simulation"):
    Nt = 10 * (
        int(4 * T * Nx * Nx / L**2) + 1
    )  # Here we make sure that Delta t/Delta x^2 is small by letting it equal to 1/4.
    # dx = L / Nx
    dt = T / Nt

    # Initialize solutions
    u_num = np.zeros((Nt + 1, Nx + 1))
    v_num = np.zeros((Nt + 1, Nx + 1))

    m = config["m"]
    beta = config["beta"]
    alpha = config["alpha"]
    chi = config["chi"]
    a = config["a"]
    b = config["b"]
    mu = config["mu"]
    nu = config["nu"]
    gamma = config["gamma"]

    x_values = np.linspace(0, L, int(Nx) + 1, dtype=np.float64)
    positive_sigmas, uStar = Display_Parameters(L)

    # Initial condition for u
    print("\n# Initial value u_0\n")
    if EigenIndex == 0:
        if len(positive_sigmas) > 0:
            EigenIndex = 2
            print("Second (first nonconstant) eigenfunction is chosen.\n")
        else:
            EigenIndex = 1
            print("first (constant) eigenfunction is chosen.\n")

    u_num[0, :] = (
        uStar + Epsilon * np.cos(((EigenIndex - 1) * np.pi / L) * x_values)
    ).astype(np.float64)

    # Initial condition (must match exactly)
    v_num[0, :] = solve_v_1(L, Nx, u_num[0, :], False)
    # print("v_num=", v_num)

    # Time integration
    for n in tqdm(range(Nt), desc="Progress..."):
        # rk4 steps
        k1 = rhs(L, Nx, u_num[n, :], v_num[n, :])
        v1 = solve_v_1(L, Nx, vector_u=u_num[n, :] + 0.5 * dt * k1)

        k2 = rhs(L, Nx, u_num[n, :] + 0.5 * dt * k1, v1)
        v2 = solve_v_1(L, Nx, vector_u=u_num[n, :] + 0.5 * dt * k2)

        k3 = rhs(L, Nx, u_num[n, :] + 0.5 * dt * k2, v2)
        v3 = solve_v_1(L, Nx, vector_u=u_num[n, :] + dt * k3)

        k4 = rhs(L, Nx, u_num[n, :] + dt * k3, v3)

        # Update
        u_num[n + 1, :] = u_num[n, :] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        # print("u_numk1k2k3k4=", u_num)
        v_num[n + 1, :] = solve_v(L, Nx, vector_u=u_num[n + 1, :])

    # Convert lists to numpy arrays
    u_num = np.array(u_num).T  # Convert list to numpy array and transpose
    # time_data = np.arange(Nt + 1) * dt
    t_values = np.linspace(0, T, Nt + 1)

    # Setup description for the title
    SetupDes = rf"""
    $a$ = {a}, $b$ = {b}, $\alpha$ = {alpha};
    $m$ = {m}, $\beta$ = {beta}, $\chi_0$ = {chi};
    $\mu$ = {mu}, $\nu$ = {nu}, $\gamma$ = {gamma}; $N$ = {Nx}, $T$ = {T};
    $u^*$ = {uStar}, $\epsilon$ = {Epsilon}, $n$ = {EigenIndex}.
    """

    # Create static plots
    create_static_plots(x_values, u_num, t_values, uStar, SetupDes, FileBaseName)

    # Create animation if requested
    if config.get("generate_video", "yes") == "yes":
        create_animation(u_num, t_values, uStar, SetupDes, FileBaseName)

    return x_values, u_num, v_num


def solve_v(L=1, Nx=50, vector_u=np.zeros(50), diagnostic=False):
    mu = config["mu"]
    nu = config["nu"]
    gamma = config["gamma"]
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
    b = -(dx**2) * nu * vector_u**gamma

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


def solve_pde_system(
    L=1, Nx=50, T=5, Epsilon=0.001, EigenIndex=2, FileBaseName="Simulation"
):
    Nt = (
        int(4 * T * Nx * Nx / L**2) + 1
    )  # Here we make sure that Delta t/Delta x^2 is small by letting it equal to 1/4.
    dx = L / Nx
    dt = T / Nt

    m = config["m"]
    beta = config["beta"]
    alpha = config["alpha"]
    chi = config["chi"]
    a = config["a"]
    b = config["b"]
    mu = config["mu"]
    nu = config["nu"]
    gamma = config["gamma"]

    x = np.linspace(0, L, int(Nx) + 1, dtype=np.float64)
    positive_sigmas, uStar = Display_Parameters(L)

    x = np.linspace(0, L, int(Nx) + 1, dtype=np.float64)

    # Initial condition for u
    print("\n# Initial value u_0\n")
    if EigenIndex == 0:
        if len(positive_sigmas) > 0:
            EigenIndex = 2
            print("Second (first nonconstant) eigenfunction is chosen.\n")
        else:
            EigenIndex = 1
            print("first (constant) eigenfunction is chosen.\n")

    u = (uStar + Epsilon * np.cos(((EigenIndex - 1) * np.pi / L) * x)).astype(
        np.float64
    )
    # print(f"Initial vector of u: \n{' '.join(map(str, u))}\n")
    print(f"Initial vector of u: \n{' '.join(f'{x:.3f}' for x in u)}\n")
    fig = tpl.figure()
    fig.plot(range(len(u)), u, label="u_0", width=100, height=36)
    fig.show()

    times_to_plot = np.arange(0, T + dt, 0.01)
    current_time = 0
    # plt.figure()

    fig, ax = plt.subplots()
    (line,) = ax.plot(x, u, label=rf"$t$={current_time:.2f}")
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u$")
    ax.set_title("Evolution of u over time")
    ax.set_ylim(0, 2)
    # Add the asymptotic solution
    ax.axhline(y=uStar, color="r", linestyle="--", label=r"$u^*$")
    # ax.legend()

    u_data = []
    time_data = []

    break_outer_loop = False

    # for n in range(1000):
    print("\n# Simulations now ...\n")
    for n in tqdm(range(Nt), desc="Progress..."):
        # print('n=',n)
        u_new = np.copy(u).astype(np.float64)

        # Solve v first
        # v = solve_v(L=L, Nx=Nx, vector_u=u_new, diagnostic=True)
        v = solve_v(L=L, Nx=Nx, vector_u=u_new)

        for i in range(1, Nx):  # Loop for u
            v_x = (v[i + 1] - v[i - 1]) / (2 * dx)
            # print('vx=',v_x)
            # v_xx = (v[i + 1] - 2 * v[i] + v[i - 1]) / dx**2
            # print('vxx=',v_xx)
            u_x = (u[i + 1] - u[i - 1]) / (2 * dx)
            # print('u_x=',u_x)
            u_xx = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx**2
            # print('uxx=',u_x)

            term1 = ((beta * chi) / ((1 + v[i]) ** (beta + 1))) * (v_x**2) * (u[i] ** m)
            # print("term1=", term1)
            term2 = ((m * chi) / (1 + v[i]) ** beta) * (u[i] ** (m - 1)) * u_x * v_x
            # print("term2=", term2)
            term3 = (
                (chi / ((1 + v[i]) ** beta))
                * (u[i] ** m)
                * (mu * v[i] - nu * u[i] ** gamma)
            )
            # print("term3=", term3)
            logistic = a * u[i] - b * u[i] ** (1 + alpha)
            # print("logistic=", logistic)
            u_new[i] = u[i] + dt * (u_xx + term1 - term2 - term3 + logistic)
            # u_new[i] = u[i] + dt * (u_xx + logistic)
            # print(
            #     f"u_new={u_new[i]} and v={v[i]}",
            # )

            if (
                math.isnan(term1)
                or math.isnan(term2)
                or math.isnan(term3)
                or math.isnan(logistic)
            ):
                break_outer_loop = True
                print("Nan detected, breaking the loop")
                break

        if break_outer_loop:
            break

        # Neumann boundary conditions for u
        # u_new[0] = u_new[1]
        # u_new[-1] = u_new[-2]
        u_new[0] = (4 * u_new[1] - u_new[2]) / 3  # Left boundary
        u_new[-1] = (4 * u_new[-2] - u_new[-3]) / 3  # Right boundary

        u = u_new  # Update u

        # Check if current_time is close to any time in times_to_plot
        if np.any(np.abs(times_to_plot - current_time) < dt / 2):
            u_data.append(np.copy(u))
            time_data.append(current_time)

        current_time += dt

    if not u_data or not time_data:
        raise ValueError(
            "No data was collected for plotting. Check time-stepping alignment."
        )

    # Convert lists to numpy arrays
    u_data = np.array(u_data).T  # Convert list to numpy array and transpose
    time_data = np.array(time_data)  # Convert list to numpy array
    # print('u_data = ', u_data)
    # print('time_data=', time_data)

    # Setup description for the title
    SetupDes = rf"""
    $a$ = {a}, $b$ = {b}, $\alpha$ = {alpha};
    $m$ = {m}, $\beta$ = {beta}, $\chi_0$ = {chi};
    $\mu$ = {mu}, $\nu$ = {nu}, $\gamma$ = {gamma}; $N$ = {Nx}, $T$ = {T};
    $u^*$ = {uStar}, $\epsilon$ = {Epsilon}, $n$ = {EigenIndex}.
    """

    # Create static plots
    create_static_plots(x, u_data, time_data, uStar, SetupDes, FileBaseName)

    # Create animation if requested
    if config.get("generate_video", "yes") == "yes":
        create_animation(u_data, time_data, uStar, SetupDes, FileBaseName)

    # print(
    #     f"""
    # Output files saved:
    # - Image: {FileBaseName}.png
    # - Image: {FileBaseName}.jpeg
    # {f'- Video: {FileBaseName}.mp4' if config.get('generate_video', 'yes') == 'yes' else ''}
    # """
    # )

    return x, u, v


def Display_Parameters(L):
    m = config["m"]
    beta = config["beta"]
    alpha = config["alpha"]
    chi = config["chi"]
    a = config["a"]
    b = config["b"]
    mu = config["mu"]
    nu = config["nu"]
    gamma = config["gamma"]
    # Compute the asymptotic solution
    print("\n# Computing the asymptotic solutions and related constants")
    uStar = (a / b) ** (1 / alpha)
    vStar = nu / mu * (a / b) ** (gamma / alpha)
    print(f"Asymptotic solutions: u^* = {uStar:.2f} and v^* = {vStar:.2f}")
    ChiStar = (
        (1 + vStar) ** beta
        * (np.sqrt(a * alpha) + np.sqrt(mu)) ** 2
        / (nu * gamma * uStar ** (m + gamma - 1))
    )
    print(f"A lower bound for Chi* is {ChiStar:.2f}")
    betaTilde = 0
    if beta >= 0.5:
        betaTilde = min(1, 2 * beta - 1)
    ChiDStar = np.sqrt(
        b * 16 * (1 + betaTilde * vStar) * mu / (nu**2 * uStar ** (2 - alpha))
    )
    print(f"Chi** = {ChiDStar:.2f} and beta tilde = {betaTilde:.2f}")

    # Compute the value of Chi*
    print("\n# Computing Chi*\n")
    Lambdas = np.zeros(6)
    Chi_vector = np.zeros(6)
    for n in range(6):
        Lambdas[n] = -(((n + 1) * np.pi / L) ** 2)
        Chi_vector[n] = (
            ((a * alpha - Lambdas[n]) / (nu * gamma))
            * (((1 + vStar) ** beta) / ((uStar) ** (m + gamma - 1)))
            * ((Lambdas[n] - mu) / Lambdas[n])
        )
    # print(*(f"Lambda_{i + 2}^* = {lam}\n" for i, lam in enumerate(Lambdas)))
    # print(*(f"Chi_{0,i + 2}^* = {chi}\n" for i, chi in enumerate(Chi_vector)))
    data = [
        [f"Lambda_{i + 2}^*", f"{lam:.3f}", f"Chi_{0,i + 2}^*", f"{chi:.3f}"]
        for i, (lam, chi) in enumerate(zip(Lambdas, Chi_vector))
    ]
    headers = ["Lambda", "Value", "Chi", "Value"]
    print(tabulate(data, headers=headers, tablefmt="grid"))
    ChiStar = min(Chi_vector)
    print(f"\nChi* = {ChiStar:.3f} and the choice of chi = {chi:.3f}")

    # Computation of the eigenvalues lambda_n and sigma_n
    print("\n# Computing sigma_n\n")
    positive_sigmas = []  # List to store positive sigma values
    if chi >= ChiStar:
        n = 0
        sigma_n = 1.0
        while sigma_n > 0:
            n += 1
            lambda_n = -((n * np.pi / L) ** 2)
            sigma_n = (
                lambda_n
                + chi
                * nu
                * gamma
                * ((uStar ** (m + gamma - 1)) / ((1 + vStar) ** beta))
                * (1 - mu / (mu - lambda_n))
                - a * alpha
            )
            print(f"sigma_{n+1}=", sigma_n)
            if sigma_n > 0:
                positive_sigmas.append(sigma_n)  # Store positive sigma value

    return positive_sigmas, uStar


def solve_pde_RK(
    L=1, Nx=50, T=5, Epsilon=0.001, EigenIndex=2, FileBaseName="Simulation"
):
    Nt = (
        int(4 * T * Nx * Nx / L**2) + 1
    )  # Here we make sure that Delta t/Delta x^2 is small by letting it equal to 1/4.
    # dx = L / Nx
    dt = T / Nt

    m = config["m"]
    beta = config["beta"]
    alpha = config["alpha"]
    chi = config["chi"]
    a = config["a"]
    b = config["b"]
    mu = config["mu"]
    nu = config["nu"]
    gamma = config["gamma"]

    x = np.linspace(0, L, int(Nx) + 1, dtype=np.float64)
    positive_sigmas, uStar = Display_Parameters(L)

    # Initial condition for u
    print("\n# Initial value u_0\n")
    if EigenIndex == 0:
        if len(positive_sigmas) > 0:
            EigenIndex = 2
            print("Second (first nonconstant) eigenfunction is chosen.\n")
        else:
            EigenIndex = 1
            print("first (constant) eigenfunction is chosen.\n")

    u = (uStar + Epsilon * np.cos(((EigenIndex - 1) * np.pi / L) * x)).astype(
        np.float64
    )

    # print(f"Initial vector of u: \n{' '.join(map(str, u))}\n")
    print(f"Initial vector of u: \n{' '.join(f'{x:.3f}' for x in u)}\n")

    # Add terminal plot of initial condition
    fig = tpl.figure()
    fig.plot(range(len(u)), u, label="u_0", width=100, height=36)
    fig.show()

    print("\n# Simulations now ...\n")
    u_data = []
    u_data.append(np.copy(u))
    for n in tqdm(range(Nt), desc="Progress..."):
        # RK4 Stage 1: k1 = F(u, v) at t_n
        k1 = F(u, L, Nx)

        # RK4 Stage 2: k2 = F(u + dt/2*k1, v_new) at t_n + dt/2
        u_temp = u + 0.5 * dt * k1
        k2 = F(u_temp, L, Nx)

        # RK4 Stage 3: k3 = F(u + dt/2*k2, v_new) at t_n + dt/2
        u_temp = u + 0.5 * dt * k2
        k3 = F(u_temp, L, Nx)

        # RK4 Stage 4: k4 = F(u + dt*k3, v_new) at t_n + dt
        u_temp = u + dt * k3
        k4 = F(u_temp, L, Nx)

        # Update u
        u += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # Enforce Neumann BCs (optional, but safe)
        # u[0] = u[1]
        # u[-1] = u[-2]
        u[0] = (4 * u[1] - u[2]) / 3  # Left boundary
        u[-1] = (4 * u[-2] - u[-3]) / 3  # Right boundary
        u_data.append(np.copy(u))

        if config.get("verbose", "no") == "yes" and n % (Nt // 5) == 0:
            print(f"Step {n}, u[middle] = {u[len(u)//2]:.6f}")

    # Convert lists to numpy arrays
    u_data = np.array(u_data).T  # Convert list to numpy array and transpose
    time_data = np.arange(Nt + 1) * dt

    # Setup description for the title
    SetupDes = rf"""
    $a$ = {a}, $b$ = {b}, $\alpha$ = {alpha};
    $m$ = {m}, $\beta$ = {beta}, $\chi_0$ = {chi};
    $\mu$ = {mu}, $\nu$ = {nu}, $\gamma$ = {gamma}; $N$ = {Nx}, $T$ = {T};
    $u^*$ = {uStar}, $\epsilon$ = {Epsilon}, $n$ = {EigenIndex}.
    """

    # Create static plots
    create_static_plots(x, u_data, time_data, uStar, SetupDes, FileBaseName)

    # Create animation if requested
    if config.get("generate_video", "yes") == "yes":
        create_animation(u_data, time_data, uStar, SetupDes, FileBaseName)

    return x, u


# Right-hand side function F(u, v)
def F(u, L, Nx):

    m = config["m"]
    beta = config["beta"]
    alpha = config["alpha"]
    chi = config["chi"]
    a = config["a"]
    b = config["b"]
    dx = L / Nx
    # mu = config["mu"]
    # nu = config["nu"]
    # gamma = config["gamma"]

    v = solve_v(L=L, Nx=Nx, vector_u=u)
    u_x, u_xx = compute_derivatives(u, dx)
    v_x, v_xx = compute_derivatives(v, dx)

    term1 = u_xx  # Diffusion

    term2 = -chi * m * (u ** (m - 1) / (1 + v) ** beta) * u_x * v_x  # Cross-term

    term3 = (
        beta * chi * (u**m / (1 + v) ** (beta + 1)) * (v_x) ** 2
    )  # Nonlinear gradient

    term4 = -chi * (u**m / (1 + v) ** beta) * v_xx  # Coupling to v_xx

    term5 = a * u - b * u ** (1 + alpha)  # Reaction

    return term1 + term2 + term3 + term4 + term5


# Function to compute derivatives with Neumann BCs
def compute_derivatives(f, dx):
    f_x = np.zeros_like(f)
    f_xx = np.zeros_like(f)
    # Central differences for interior
    f_x[1:-1] = (f[2:] - f[:-2]) / (2 * dx)
    f_xx[1:-1] = (f[2:] - 2 * f[1:-1] + f[:-2]) / dx**2
    # Neumann BCs (f_x=0 at boundaries)
    f_x[0] = (f[1] - f[0]) / dx
    f_x[-1] = (f[-1] - f[-2]) / dx
    f_xx[0] = (f[1] - f[0]) / dx**2
    f_xx[-1] = (f[-2] - f[-1]) / dx**2
    return f_x, f_xx


def create_static_plots(x, u_data, time_data, uStar, SetupDes, FileBaseName):
    """Create and save static plots (2D and 3D)"""
    # 3D Plot
    fig_3d = plt.figure(dpi=300)
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    T_grid, X_grid = np.meshgrid(time_data, x, indexing="xy")
    ax_3d.plot_surface(T_grid, X_grid, u_data, cmap="viridis", alpha=0.8)

    # Create a constant plane at height uStar
    U_grid = np.full_like(T_grid, uStar)
    Zero_grid = np.full_like(T_grid, 0)

    ax_3d.set_xlabel(r"Time $t$")
    ax_3d.set_ylabel(r"Space $x$")
    ax_3d.set_zlabel(r"$u(t,x)$")
    ax_3d.set_zlim(-0.05, u_data.max())
    # ax_3d.set_zlim(uStar, u_data.max())
    ax_3d.set_zticks(np.linspace(uStar, u_data.max(), 5))
    ax_3d.zaxis.set_major_formatter(FormatStrFormatter("%.3f"))

    # Plot reference planes with muted colors and increased transparency
    ax_3d.plot_surface(
        T_grid,
        X_grid,
        U_grid,
        alpha=0.5,  # More transparent
        rstride=100,
        cstride=100,
        color="r",  # Muted color
    )

    ax_3d.plot_surface(
        T_grid,
        X_grid,
        Zero_grid,
        alpha=0.2,  # More transparent
        rstride=100,
        cstride=100,
        color="lightgray",  # Very light gray
    )

    ax_3d.set_title(SetupDes, fontsize=10, pad=-80)
    fig_3d.subplots_adjust(top=0.80)
    plt.tight_layout()

    # Save the plot as PNG and JPEG
    fig_3d.savefig(f"{FileBaseName}.png")
    fig_3d.savefig(f"{FileBaseName}.jpeg")
    print(
        f"""
    Output files saved:
    - Image: {FileBaseName}.png
    - Image: {FileBaseName}.jpeg
    """
    )


def create_animation(u_data, time_data, uStar, SetupDes, FileBaseName):
    """Create and save animation"""
    # Reduce DPI for faster rendering
    fig, ax = plt.subplots(dpi=300)
    fig.subplots_adjust(top=0.80)

    (line,) = ax.plot(u_data[:, 0], label="u(t)")
    ax.set_ylim(u_data.min() - 0.1, u_data.max() + 0.1)
    ax.axhline(y=uStar, color="r", linestyle="--", label=r"$u^*$")
    ax.legend(loc="upper right")

    def update(frame):
        line.set_ydata(u_data[:, frame])
        ax.set_title(SetupDes, fontsize=10, pad=-40)
        return (line,)

    # Take every nth frame to reduce total frames
    frame_stride = max(1, len(time_data) // 200)  # Aim for ~200 frames total
    frames = range(0, len(time_data), frame_stride)

    ani = animation.FuncAnimation(
        fig, update, frames=frames, interval=50, blit=True  # Enable blit for speed
    )

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


def inverse_tridiagonal(diagonal, offdiagonal):
    """Computes A⁻¹ for a tridiagonal matrix A."""
    n = len(diagonal)
    A_inv = np.zeros((n, n))

    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1  # Solve for each column of A⁻¹
        A_inv[:, i] = solve_banded(
            (1, 1),
            [np.append([0], offdiagonal), diagonal, np.append(offdiagonal, [0])],
            e_i,
        )

    # Print the inverse matrix to check the results
    # print("Inverse matrix A_inv:")
    # print(A_inv)

    return A_inv


def parse_args():
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
    parser.add_argument("--m", type=float, default=1, help="Parameter m (default: 1)")
    parser.add_argument(
        "--beta", type=float, default=1, help="Parameter beta (default: 1)"
    )
    parser.add_argument(
        "--alpha", type=float, default=1, help="Parameter alpha (default: 1)"
    )
    parser.add_argument(
        "--chi", type=float, default=-1, help="Parameter chi (default: -1)"
    )
    parser.add_argument("--a", type=float, default=1, help="Parameter a (default: 1)")
    parser.add_argument("--b", type=float, default=1, help="Parameter b (default: 1)")
    parser.add_argument("--mu", type=float, default=1, help="Parameter mu (default: 1)")
    parser.add_argument("--nu", type=float, default=1, help="Parameter nu (default: 1)")
    parser.add_argument(
        "--gamma", type=float, default=1, help="Parameter gamma (default: 1)"
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
        default=2.3,
        help="Parameter for time to lapse (default: 2.3)",
    )
    parser.add_argument(
        "--EigenIndex",
        type=int,
        default=0,
        help="Parameter eigen index (default: 0, letting system to choose)",
    )
    parser.add_argument(
        "--Epsilon",
        type=float,
        default=0.001,
        help="Parameter perturbation epsilon (default: 0.001)",
    )

    return parser.parse_args()


def main():
    """
    Main function to parse arguments, display simulation parameters, and run the simulation.

    Global Variables:
    - config: A dictionary containing parsed command-line arguments.

    Steps:
    1. Parse command-line arguments and store them in the global `config` variable.
    2. Display the parsed model and simulation parameters.
    3. Generate a base name for output files based on the parameters.
    4. Prompt the user for confirmation to proceed with the simulation.
    5. Run the simulation using the specified parameters or exit if declined.
    """
    global config
    args = parse_args()
    config = vars(args)

    # # Display the parsed arguments
    # print("Parameters:")

    # Now access them via config['m'], config['beta'], etc.
    print("Model Parameters:")
    print("1. Logistic term: ")
    print(f"\ta = {config['a']}, b = {config['b']}, alpha = {config['alpha']}")
    print("2. Reaction term: ")
    print(f"\tm = {config['m']}, beta = {config['beta']}, chi = {config['chi']}")
    print("3. The v equation: ")
    print(f"\tmu = {config['mu']}, nu = {config['nu']}, gamma = {config['gamma']}")
    print("4. Initial condition: ")
    print(f"\tEpsilon = {config['Epsilon']}, EigenIndex = {config['EigenIndex']}")
    # Run the solver

    print("Simulation Parameters:")
    print(f"\tMeshSize = {config['meshsize']}, time = {config['time']}")

    # Using the above parameters to generate a file base name string
    basename = f"a={config['a']}_b={config['b']}_alpha={config['alpha']}_m={config['m']}_beta={config['beta']}_chi={config['chi']}_mu={config['mu']}_nu={config['nu']}_gamma={config['gamma']}_meshsize={config['meshsize']}_time={config['time']}_Epsilon={config['Epsilon']}_EigenIndex={config['EigenIndex']}".replace(
        ".", "-"
    )
    print(f"Output files will be saved with the basename:\n\t {basename}\n")

    if (
        config["confirm"] == "no"
        or questionary.confirm("Do you want to continue the simulation?").ask()
    ):
        print("Continuing simulation...")
        # x, u, v = solve_pde_system(
        x, u = RK4(
            Nx=config["meshsize"],
            T=config["time"],
            Epsilon=config["Epsilon"],
            EigenIndex=config["EigenIndex"],
            FileBaseName=basename,
        )
    else:
        print("Exiting simulation.")
        exit()


if __name__ == "__main__":
    main()
