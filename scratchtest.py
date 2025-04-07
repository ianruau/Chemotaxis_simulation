import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, lil_matrix
from scipy.sparse.linalg import spsolve
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm  # Import tqdm for progress bar

# Parameters
L = 1
Nx = 5  # Spatial resolution
dx = L / Nx
x_values = np.linspace(0, L, Nx + 1)
# print("xlinspace=", x_values)

# Time parameters
T = 1
Nt = 1000  # Temporal resolution
dt = T / Nt
t_values = np.linspace(0, T, Nt + 1)
# print("tlinspace=", t_values)

# Physical parameters
pi = np.pi
alpha = 0.1  # Reduced coupling
# lmbda = pi**2 - 1 / (1 + pi**2)  # Exact decay rate from analytical solution
lmbda = 1


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

# Exact solution
for n in range(Nt + 1):
    u_exact[n, :] = np.exp(-lmbda * t_values[n]) * np.cos(pi * x_values)
# print("u_exact=", u_exact[1, :])

# Initial condition (must match exactly)
u_num[0, :] = np.cos(pi * x_values).copy()  # Ensure exact match at t=0

# Exact solution
for n in range(Nt + 1):
    v_exact[n, :] = solve_v(L, Nx, u_exact[n, :], False)
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

# # Error analysis
# initial_error = np.max(np.abs(u_num[0] - u_exact[0]))
# final_error = np.max(np.abs(u_num[-1] - u_exact[-1]))
# print(f"Initial error: {initial_error:.2e}")
# print(f"Final error: {final_error:.2e}")
#
# # Visualization
# fig = plt.figure(figsize=(18, 6))
#
# # Numerical solution
# ax1 = fig.add_subplot(121, projection="3d")
# X, T_mesh = np.meshgrid(x, t_values)
# ax1.plot_surface(X, T_mesh, u_num, cmap="viridis")
# ax1.set_title("Numerical Solution")
# ax1.set_xlabel("x")
# ax1.set_ylabel("t")
# ax1.set_zlim(-1, 1)
#
# # Exact solution
# ax2 = fig.add_subplot(122, projection="3d")
# ax2.plot_surface(X, T_mesh, u_exact, cmap="plasma")
# ax2.set_title("Exact Solution")
# ax2.set_xlabel("x")
# ax2.set_ylabel("t")
# ax2.set_zlim(-1, 1)
#
# plt.tight_layout()
# plt.show()
#

# # Plot the n-th row
# plt.plot(x_values, u_num[n, :], marker="o", label=f"u_num[{n}]")
# plt.plot(x_values, u_exact[n, :], "r--", label=f"u_exact[{n}]")  # Red dashed line
#
# # Add labels and legend
# plt.xlabel("X Values")
# plt.ylabel("Y Values")
# plt.title(f"Plot of Row {n}")
# plt.legend()
# plt.grid(True)
# plt.show()

# Create a figure with 1 row and 2 columns of subplots
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))  # 1 row, 2 columns

# --- Left Plot (First Subplot) ---
# ax1.plot(x_values, u_num[n, :], "b-o", label=f"u_num[{n}]")
# ax1.plot(x_values, u_exact[n, :], "r", label=f"u_exact[{n}]")  # Red line
# ax1.set_xlabel("x")
# ax1.set_ylabel(f"u({n},x)")
# ax1.set_title("u plot")
# ax1.legend()
# ax1.grid(True)

# --- Right Plot (Second Subplot) ---
# ax2.plot(x_values, v_num[n, :], "b-o", label=f"v_num[{n}]")
# ax2.plot(x_values, v_exact[n, :], "r", label=f"v_exact[{n}]")  # Red line
# ax2.set_xlabel("x")
# ax2.set_ylabel(f"v({n},x)")
# ax2.set_title("v plot")
# ax2.legend()
# ax2.grid(True)
#
# # Adjust layout to prevent overlap
# plt.tight_layout()
# fig.savefig(f"subplots_Nt={Nt}_Nx={Nx}.png", bbox_inches="tight")
# plt.show()
#
# Create a 2x2 grid of subplots
# print("u_num=", u_num)
# print("size_u=", len(u_num))
n = 200  # Row to plot (0-based index)
fig, axs = plt.subplots(2, 2, figsize=(12, 10))  # 2 rows, 2 columns

# --- Top Row: 2D Plots ---
# Left 2D plot (axs[0, 0])
axs[0, 0].plot(x_values, u_num[n, :], "b-o", label=f"u_num[{n}]")
axs[0, 0].plot(x_values, u_exact[n, :], "r", label=f"u_exact[{n}]")  # Red line
axs[0, 0].set_xlabel("x")
axs[0, 0].set_ylabel(f"u({n},x)")
axs[0, 0].set_title("u plot")
axs[0, 0].legend()
axs[0, 0].grid(True)

# Right 2D plot (axs[0, 1])
axs[0, 1].plot(x_values, v_num[n, :], "b-o", label=f"v_num[{n}]")
axs[0, 1].plot(x_values, v_exact[n, :], "r", label=f"v_exact[{n}]")  # Red line
axs[0, 1].set_xlabel("x")
axs[0, 1].set_ylabel(f"v({n},x)")
axs[0, 1].set_title("v plot")
axs[0, 1].legend()
axs[0, 1].grid(True)

# --- Bottom Row: 3D Plots ---
# Create 3D axes by setting `projection='3d'`
axs[1, 0] = fig.add_subplot(2, 2, 3, projection="3d")  # Left 3D plot
axs[1, 1] = fig.add_subplot(2, 2, 4, projection="3d")  # Right 3D plot

# Generate 3D mesh
t_grid, x_grid = np.meshgrid(t_values, x_values)
# u_exact = np.exp(-lmbda * t_grid) * np.cos(pi * x_grid)
# print("uexact=", u_exact)
# print("lenuexact=", len(u_exact))

# Left 3D plot (axs[1, 0])
axs[1, 0].plot_surface(t_grid, x_grid, u_num.T, cmap="viridis")
# axs[1, 0].plot_surface(t_grid, x_grid, u_exact.T, cmap="plasma")
axs[1, 0].set_title("u(t,x)")
axs[1, 0].set_xlabel("t")
axs[1, 0].set_ylabel("x")
axs[1, 0].set_zlabel("u")

# Right 3D plot (axs[1, 1])
axs[1, 1].plot_surface(t_grid, x_grid, v_num.T, cmap="viridis")
axs[1, 1].set_title("v(t,x)")
axs[1, 1].set_xlabel("t")
axs[1, 1].set_ylabel("Y")
axs[1, 1].set_zlabel("v")

# Adjust layout and save
plt.tight_layout()
# plt.savefig("2D_and_3D_subplots.png", dpi=300)  # Save high-resolution image
plt.show()
