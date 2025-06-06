#!/usr/bin/env python3
#

import argparse
import math

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import questionary
from matplotlib import rc
from scipy.linalg import solve_banded

config = {}  # Global dictionary for parameters

# Enable LaTeX rendering
rc("text", usetex=True)
# TO LIST:
# 1. Write CLI interface


def solve_pde_system(L=1, Nx=50, T=5, FileBaseName="Simulation"):
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

    # Compute the asymptotic solution
    uStar = (a / b) ** (1 / alpha)
    vStar = nu / mu * (a / b) ** (gamma / alpha)
    print(f"Asymptotic solutions: u* = {uStar:.2f} and v* = {vStar:.2f}")
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

    x = np.linspace(0, L, int(Nx) + 1, dtype=np.float64)

    # Initial condition for u
    # u = np.ones_like(x) * 0.5
    u = (1 + 0.5 * np.cos(np.pi * x)).astype(np.float64)
    print(f"Initial vector of u = {u}")

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
    for n in range(Nt):
        # print('n=',n)
        u_new = np.copy(u).astype(np.float64)

        diagonal = np.ones(int(Nx) + 1) * (2 / dx**2 + mu)
        diagonal[0] -= 1 / dx**2
        diagonal[-1] -= 1 / dx**2
        offdiagonal = np.ones(int(Nx)) * (-1 / dx**2)
        A_inv = inverse_tridiagonal(diagonal, offdiagonal)

        # Check the stability of the inverse
        condition_number = np.linalg.cond(A_inv)
        threshold = 1e10

        # if condition_number > threshold:
        #     print(
        #         "Warning: The matrix inverse may not be stable. Condition number:",
        #         condition_number,
        #     )
        # else:
        #     print(
        #         "The matrix inverse is stable. Condition number:",
        #         condition_number)

        v = np.dot(A_inv, nu * u**gamma)

        # v = np.zeros(int(Nx) + 1)
        # for i in range(1, Nx):  # Loop for v
        #     # v_xx = (v[i + 1] - 2*v[i] + v[i-1]) / dx**2
        #     # print('vxx=',v_xx)
        #     # # v[i] = v_xx - v[i] + u[i]
        #     # v[i] = v_xx + u[i]**gamma
        #     v[i] = (v[i + 1] + v[i - 1] + nu * dx**2 * u[i] ** gamma) / (
        #         2 + mu * dx**2
        #     )
        # print(f"vector = {u**gamma}")

        # Neumann boundary conditions for v
        v[0] = v[1]
        v[-1] = v[-2]

        for i in range(1, Nx):  # Loop for u
            v_x = (v[i + 1] - v[i - 1]) / (2 * dx)
            # print('vx=',v_x)
            v_xx = (v[i + 1] - 2 * v[i] + v[i - 1]) / dx**2
            # print('vxx=',v_xx)
            u_x = (u[i + 1] - u[i - 1]) / (2 * dx)
            # print('u_x=',u_x)
            u_xx = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx**2
            # print('uxx=',u_x)

            term1 = ((beta * chi) /
                     (v[i] ** (beta + 1))) * (v_x**2) * (u[i] ** m)
            # print("term1=", term1)
            term2 = ((m * chi) / (v[i]) ** beta) * \
                (u[i] ** (m - 1)) * u_x * v_x
            # print("term2=", term2)
            term3 = (
                (chi / ((v[i]) ** beta))
                * (u[i] ** m)
                * (mu * v[i] - nu * u[i] ** gamma)
            )
            # print("term3=", term3)
            logistic = a * u[i] - b * u[i] ** (1 + alpha)
            # print("logistic=", logistic)
            u_new[i] = max(u[i] + dt * (u_xx + term1 -
                                        term2 - term3 + logistic), 0)
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
        u_new[0] = u_new[1]
        u_new[-1] = u_new[-2]

        u = u_new  # Update u

        # Plot every 0.01 seconds
        # if np.isclose(current_time, times_to_plot, atol=dt).any():
        #     # plt.plot(x, u, label=f't={current_time:.2f}')
        #     u_data.append(np.copy(u))

        # Check if current_time is close to any time in times_to_plot
        if np.any(np.abs(times_to_plot - current_time) < dt / 2):
            u_data.append(np.copy(u))
            time_data.append(current_time)

        current_time += dt
        # print('u=',u)
        # print('v=',v)

    if not u_data or not time_data:
        raise ValueError(
            "No data was collected for plotting. Check time-stepping alignment."
        )

    u_data = np.array(u_data).T  # Convert list to numpy array and transpose
    time_data = np.array(time_data)  # Convert list to numpy array

    def update(frame):
        line.set_ydata(u_data[:, frame])
        # ax.set_title(f"Evolution of u over time (t={time_data[frame]:.2f}s)")
        ax.set_title(
            rf"""
            $a$ = {a}, $b$ = {b}, $\alpha$ = {alpha}; $m$ = {m}, $\beta$ = {beta}, $\chi_0$ = {chi};
            $\mu$ = {mu}, $\nu$ = {nu}, $\gamma$ = {gamma}; $N$ = {Nx}, $T$ = {T}.
            """,
            fontsize=10,
        )
        return (line,)

    ani = animation.FuncAnimation(
        fig, update, frames=len(time_data), interval=50, blit=True
    )

    # Save the animation as an MP4 file
    writer = animation.FFMpegWriter(
        fps=5, metadata=dict(
            artist="Me"), bitrate=1800)
    ani.save(f"{FileBaseName}.mp4", writer=writer)

    # plt.show()

    # 3D Plot
    fig_3d = plt.figure()
    ax_3d = fig_3d.add_subplot(111, projection="3d")
    T_grid, X_grid = np.meshgrid(
        time_data, x, indexing="xy")  # Ensure consistent shape
    ax_3d.plot_surface(
        T_grid,
        X_grid,
        u_data,
        cmap="viridis")  # Ensure correct shape

    # Create a constant plane at height uStar
    U_grid = np.full_like(T_grid, uStar)

    ax_3d.set_xlabel(r"Time $f$")
    ax_3d.set_ylabel(r"Space $x$")
    ax_3d.set_zlabel(r"$u(t,x)$")
    # ax_3d.set_title("3D Plot of u over Time and Space")
    ax_3d.plot_surface(
        T_grid, X_grid, U_grid, alpha=0.5, rstride=100, cstride=100, color="r"
    )
    ax_3d.set_title(
        rf"""
        $a$ = {a}, $b$ = {b}, $\alpha$ = {alpha}; $m$ = {m}, $\beta$ = {beta}, $\chi_0$ = {chi};
        $\mu$ = {mu}, $\nu$ = {nu}, $\gamma$ = {gamma}; $N$ = {Nx}, $T$ = {T}.
        """,
        fontsize=10,
    )

    # Save the plot as PNG and JPEG
    fig_3d.savefig(f"{FileBaseName}.png")
    fig_3d.savefig(f"{FileBaseName}.jpeg")

    print(
        f"""
    Output files saved:
    - Video: {FileBaseName}.mp4
    - Image:  {FileBaseName}.png
    - Image: {FileBaseName}.jpeg
    """
    )

    return x, u, v


def inverse_tridiagonal(diagonal, offdiagonal):
    """Computes A⁻¹ for a tridiagonal matrix A."""
    n = len(diagonal)
    A_inv = np.zeros((n, n))

    for i in range(n):
        e_i = np.zeros(n)
        e_i[i] = 1  # Solve for each column of A⁻¹
        A_inv[:, i] = solve_banded(
            (1, 1),
            [np.append([0], offdiagonal), diagonal,
             np.append(offdiagonal, [0])],
            e_i,
        )

    # Print the inverse matrix to check the results
    print("Inverse matrix A_inv:")
    print(A_inv)

    return A_inv


def parse_args():
    parser = argparse.ArgumentParser(
        description="A CLI tool for configuring parameters"
    )
    parser.add_argument(
        "--m",
        type=float,
        default=3,
        help="Parameter m (default: 3)")
    parser.add_argument(
        "--beta", type=float, default=1, help="Parameter beta (default: 1)"
    )
    parser.add_argument(
        "--alpha", type=float, default=1, help="Parameter alpha (default: 1)"
    )
    parser.add_argument(
        "--chi", type=float, default=-1, help="Parameter chi (default: -1)"
    )
    parser.add_argument(
        "--a",
        type=float,
        default=1,
        help="Parameter a (default: 1)")
    parser.add_argument(
        "--b",
        type=float,
        default=2,
        help="Parameter b (default: 2)")
    parser.add_argument(
        "--mu",
        type=float,
        default=1,
        help="Parameter mu (default: 1)")
    parser.add_argument(
        "--nu",
        type=float,
        default=1,
        help="Parameter nu (default: 1)")
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

    return parser.parse_args()


def main():
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
    print(
        f"\tm = {config['m']}, beta = {config['beta']}, chi = {config['chi']}")
    print("3. The v equation: ")
    print(
        f"\tmu = {config['mu']}, nu = {config['nu']}, gamma = {config['gamma']}")
    # Run the solver

    print("Simulation Parameters:")
    print(f"\tMeshSize = {config['meshsize']}, time = {config['time']}")

    # Using the above parameters to generate a file base name string
    basename = f"a={config['a']}_b={config['b']}_alpha={config['alpha']}_m={config['m']}_beta={config['beta']}_chi={config['chi']}_mu={config['mu']}_nu={config['nu']}_gamma={config['gamma']}_meshsize={config['meshsize']}_time={config['time']}".replace(
        ".", "-"
    )
    print(f"Output files will be saved with the basename:\n\t {basename}\n")

    if questionary.confirm("Do you want to continue the simulation?").ask():
        print("Continuing simulation...")
        x, u, v = solve_pde_system(
            Nx=config["meshsize"], T=config["time"], FileBaseName=basename
        )
    else:
        print("Exiting simulation.")
        exit()


if __name__ == "__main__":
    main()
