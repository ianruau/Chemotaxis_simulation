from math import floor
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm  # Import tqdm for progress bar
from matplotlib.gridspec import GridSpec

# Parameters
L = 1
Nx = 50  # Spatial resolution
dx = L / Nx
x_values = np.linspace(0, L, Nx + 1)
# print("xlinspace=", x_values)

# Time parameters
T = 1
Nt = 1000000  # Temporal resolution
dt = T / Nt
t_values = np.linspace(0, T, Nt + 1)
# print("tlinspace=", t_values)

# Physical parameters
pi = np.pi
lmbda = pi**2 - 1 / (1 + pi**2)  # Exact decay rate from analytical solution


def solve_v(L=1, Nx=50, vector_u=np.zeros(Nx), diagnostic=False):
    mu = 1
    nu = 1
    gamma = 1
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


# Initialize solutions
u_num = np.zeros((Nt + 1, Nx + 1))
# print("u_num=", u_num)
u_exact = np.zeros((Nt + 1, Nx + 1))
v_num = np.zeros((Nt + 1, Nx + 1))
v_exact = np.zeros((Nt + 1, Nx + 1))
v_exact_from_u = np.zeros((Nt + 1, Nx + 1))

# Exact solution
for n in range(Nt + 1):
    u_exact[n, :] = np.exp(-lmbda * t_values[n]) * np.cos(pi * x_values)
    v_exact[n, :] = np.exp(-lmbda * t_values[n]) * np.cos(pi * x_values) / (1 + pi**2)
# print("u_exact=", u_exact[1, :])
# print("u_exact=", u_exact)
# print("size of u exact=", np.shape(u_exact))

# Initial condition (must match exactly)
u_num[0, :] = np.cos(pi * x_values).copy()  # Ensure exact match at t=0

# Exact solution
for n in range(Nt + 1):
    v_exact_from_u[n, :] = solve_v(L, Nx, u_exact[n, :], False)
# print("v_exact=", v_exact[1, :])

# Initial condition (must match exactly)
v_num[0, :] = solve_v(L, Nx, u_num[0, :], False)
# print("v_num=", v_num)


# Right-hand side function
def rhs(L, Nx, u, v):
    u_xx = laplacian_NBC(L, Nx, u)
    v_x = first_derivative_NBC(L, Nx, v)
    # Exact coefficient from analytical solution
    coeff = pi**2 - 1 / (1 + pi**2) - lmbda  # Should be zero!
    return u_xx + v_x + coeff * v


# # Verification at t=0
# v_initial = solve_v(u_num[0, :])
# print("Initial RHS norm:", np.linalg.norm(rhs(u_num[0, :], v_initial)))
#
# Time integration
for n in tqdm(range(Nt), desc="Progress..."):
    # rk4 steps
    k1 = rhs(L, Nx, u_num[n, :], v_num[n, :])
    v1 = solve_v(L, Nx, vector_u=u_num[n, :] + 0.5 * dt * k1)

    k2 = rhs(L, Nx, u_num[n, :] + 0.5 * dt * k1, v1)
    v2 = solve_v(L, Nx, vector_u=u_num[n, :] + 0.5 * dt * k2)

    k3 = rhs(L, Nx, u_num[n, :] + 0.5 * dt * k2, v2)
    v3 = solve_v(L, Nx, vector_u=u_num[n, :] + dt * k3)

    k4 = rhs(L, Nx, u_num[n, :] + dt * k3, v3)

    # Update
    u_num[n + 1, :] = u_num[n, :] + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    # print("u_numk1k2k3k4=", u_num)
    v_num[n + 1, :] = solve_v(L, Nx, vector_u=u_num[n + 1, :])


# Compute L² error at each time step
u_l2_error_per_time = np.sqrt(dx * np.sum((u_num - u_exact) ** 2, axis=1))

# Compute L∞ error at each time step
u_linf_error_per_time = np.max(np.abs(u_num - u_exact), axis=1)

# Compute L² error at each time step
v_l2_error_per_time = np.sqrt(dx * np.sum((v_num - v_exact) ** 2, axis=1))

# Compute L∞ error at each time step
v_linf_error_per_time = np.max(np.abs(v_num - v_exact), axis=1)

# Global L² error
u_global_l2_error = np.sqrt(dt * dx * np.sum((u_num - u_exact) ** 2))
print("u_L2_Error=", u_global_l2_error)

# Global L∞ error
u_global_linf_error = np.max(np.abs(u_num - u_exact))
print("u_Linf_Error=", u_global_linf_error)

# Global L² error
v_global_l2_error = np.sqrt(dt * dx * np.sum((u_num - u_exact) ** 2))
print("v_L2_Error=", v_global_l2_error)

# Global L∞ error
v_global_linf_error = np.max(np.abs(u_num - u_exact))
print("v_Linf_Error=", v_global_linf_error)


n = floor(Nt / 2)  # Row to plot (0-based index)
# print("value of n=", n)
# print(f"u_num[{n}]=", u_num[n, :])
# print(f"u_exact[{n}]=", u_exact[n, :])
# print(f"v_num[{n}]=", v_num[n, :])
# print(f"v_exact[{n}]=", v_exact[n, :])

# # Generate 3D mesh
t_grid, x_grid = np.meshgrid(t_values, x_values)

# Create a 3x2 grid using GridSpec
fig = plt.figure(figsize=(12, 16))
gs = GridSpec(4, 2, height_ratios=[1, 1, 2, 1])  # 4 rows, 2 columns

# --- Row 1: First set of 2D plots (u and v in a middle time step)---
ax1 = fig.add_subplot(gs[0, 0])  # u plot (top-left)
ax2 = fig.add_subplot(gs[0, 1])  # v plot (top-right)

# --- Row 2: Second set of 2D plots (u and v at the final step) ---
ax3 = fig.add_subplot(gs[1, 0])  # New 2D plot (middle-left)
ax4 = fig.add_subplot(gs[1, 1])  # New 2D plot (middle-right)

# --- Row 3: 3D Plots ---
ax5 = fig.add_subplot(gs[2, 0], projection="3d")  # u 3D (bottom-middle-left)
ax6 = fig.add_subplot(gs[2, 1], projection="3d")  # v 3D (bottom-middle-right)

# --- Row 4: Error Plots ---
ax7 = fig.add_subplot(gs[3, 0])  # L2 error (bottom-left)
ax8 = fig.add_subplot(gs[3, 1])  # L∞ error (bottom-right)

# --- Plot Data ---
# Row 1: Original 2D plots
ax1.plot(x_values, u_num[n, :], "b", label=f"u_num[{n}]")
ax1.plot(x_values, u_exact[n, :], "r", label=f"u_exact[{n}]")
ax1.set_xlabel("x")
ax1.set_ylabel(f"u({n},x)")
ax1.set_title("u plot (2D)")
ax1.legend()
ax1.grid(True)

ax2.plot(x_values, v_num[n, :], "b", label=f"v_num[{n}]")
ax2.plot(x_values, v_exact[n, :], "r", linewidth=3.0, label=f"v_exact[{n}]")
ax2.plot(x_values, v_exact_from_u[n, :], "g--", label=f"v_exact_from_u[{n}]")
ax2.set_xlabel("x")
ax2.set_ylabel(f"v({n},x)")
ax2.set_title("v plot (2D)")
ax2.legend()
ax2.grid(True)

# Row 2: New 2D plots (customize as needed)
ax3.plot(x_values, u_num[Nt, :], "b", label=f"u_num[{Nt}]")
ax3.plot(x_values, u_exact[Nt, :], "r", label=f"u_exact[{Nt}]")
ax3.set_xlabel("x")
ax3.set_ylabel(f"u({Nt},x)")
ax3.set_title("New 2D Plot 1")
ax3.legend()
ax3.grid(True)

ax4.plot(x_values, v_num[Nt, :], "b", label=f"v_num[{Nt}]")
ax4.plot(x_values, v_exact[Nt, :], "r", linewidth=3.0, label=f"v_exact[{Nt}]")
ax4.plot(x_values, v_exact_from_u[Nt, :], "g--", label=f"v_exact_from_u[{Nt}]")
ax4.set_xlabel("x")
ax4.set_ylabel(f"v({Nt},x)")
ax4.set_title("New 2D Plot 2")
ax4.legend()
ax4.grid(True)

# Row 3: 3D plots
ax5.plot_surface(t_grid, x_grid, u_num.T, cmap="viridis")
ax5.plot_surface(t_grid, x_grid, u_exact.T, cmap="plasma")
ax5.set_title("u(t,x) (3D)")
ax5.set_xlabel("t")
ax5.set_ylabel("x")
ax5.set_zlabel("u")

ax6.plot_surface(t_grid, x_grid, v_num.T, cmap="viridis")
ax6.plot_surface(t_grid, x_grid, v_exact.T, cmap="plasma")
ax6.set_title("v(t,x) (3D)")
ax6.set_xlabel("t")
ax6.set_ylabel("x")
ax6.set_zlabel("v")

# Row 4: Error plots
ax7.plot(t_values, u_l2_error_per_time, label="u L² Error")
ax7.plot(t_values, v_l2_error_per_time, "r", label="v L² Error")
ax7.set_xlabel("Time")
ax7.set_ylabel("Error")
ax7.legend()

ax8.plot(t_values, u_linf_error_per_time, label="u L∞ Error")
ax8.plot(t_values, v_linf_error_per_time, "r", label="v L∞ Error")
ax8.set_xlabel("Time")
ax8.set_ylabel("Error")
ax8.legend()

# Adjust layout
plt.tight_layout()
fig.savefig(f"subplots_Nt={Nt}_Nx={Nx}.png", bbox_inches="tight")
plt.show()
