#!/usr/bin/env python3
#

import argparse

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import questionary

config = {}  # Global dictionary for parameters

# TO LIST:
# 1. Write CLI interface


def solve_pde_system(L=1, Nx=50, T=5, FileBaseName="Simulation"):
    Nt = int(2 * T * Nx * Nx / L**2) + 1
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

    x = np.linspace(0, L, Nx + 1, dtype=np.float64)
    u = (
        np.cos(
            2 *
            np.pi *
            x) +
        1.5).astype(
            np.float64)  # Initial condition for u
    # v = (np.sin(np.pi * x) + 1.5).astype(np.float64)  # Initial condition for v
    # u = np.ones_like(x) * 0.5
    # v = np.ones_like(x) * 2.0
    v = (
        np.cos(
            2 *
            np.pi *
            x) +
        1.5).astype(
            np.float64)  # Initial condition for u
    # print('u=', u)
    # print('v=', v)

    times_to_plot = np.arange(0, T + dt, 0.01)
    current_time = 0
    # plt.figure()

    fig, ax = plt.subplots()
    (line,) = ax.plot(x, u, label=f"t={current_time:.2f}")
    ax.set_xlabel("x")
    ax.set_ylabel("u")
    ax.set_title("Evolution of u over time")
    ax.set_ylim(0, 2)
    ax.legend()

    u_data = []
    time_data = []

    # for n in range(1000):
    for n in range(Nt):
        # print('n=',n)
        u_new = np.copy(u).astype(np.float64)
        v_new = np.copy(v).astype(np.float64)
        # print('v_newnew=',v_new)

        for i in range(1, Nx):  # Loop for v
            # v_xx = (v[i + 1] - 2*v[i] + v[i-1]) / dx**2
            # print('vxx=',v_xx)
            # # v_new[i] = v_xx - v[i] + u[i]
            # v_new[i] = v_xx + u[i]**gamma
            v_new[i] = (v[i + 1] + v[i - 1] + nu * dx**2 * u[i] ** gamma) / (
                2 + mu * dx**2
            )

        # Neumann boundary conditions for v
        v_new[0] = v_new[1]
        v_new[-1] = v_new[-2]
        # print('v_new=',v_new)

        for i in range(1, Nx):  # Loop for u
            v_x = (v[i + 1] - v[i - 1]) / (2 * dx)
            # print('vx=',v_x)
            v_xx = (v[i + 1] - 2 * v[i] + v[i - 1]) / dx**2
            # print('vxx=',v_xx)
            u_x = (u[i + 1] - u[i - 1]) / (2 * dx)
            # print('u_x=',u_x)
            u_xx = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx**2
            # print('uxx=',u_x)

            term1 = (
                ((beta * chi) / (1 + v_new[i]) **
                 (beta + 1)) * (v_x**2) * (u[i] ** m)
            )
            term2 = ((m * chi) / (1 + v_new[i]) **
                     beta) * (u[i] ** (m - 1)) * u_x * v_x
            term3 = (
                (chi / (1 + v_new[i]) ** beta)
                * (u[i] ** m)
                * (v_new[i] - u[i] ** gamma)
            )
            logistic = a * u[i] - b * u[i] ** (1 + alpha)

            u_new[i] = u[i] + dt * (u_xx + term1 - term2 - term3 + logistic)
            # u_new[i] = u[i] + dt * (u_xx + logistic)

        # Neumann boundary conditions for u
        u_new[0] = u_new[1]
        u_new[-1] = u_new[-2]

        u = u_new  # Update u
        v = v_new  # Update v

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
        ax.set_title(f"Evolution of u over time (t={time_data[frame]:.2f}s)")
        return (line,)

    ani = animation.FuncAnimation(
        fig, update, frames=len(time_data), interval=200, blit=True
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

    ax_3d.set_xlabel("Time (t)")
    ax_3d.set_ylabel("Space (x)")
    ax_3d.set_zlabel("u")
    ax_3d.set_title("3D Plot of u over Time and Space")

    # Save the plot as PNG and JPEG
    fig_3d.savefig(f"{FileBaseName}.png")
    fig_3d.savefig(f"{FileBaseName}.jpeg")

    print(f"""
    Output files saved:
    - Video: {FileBaseName}.mp4
    - Image:  {FileBaseName}.png
    - Image: {FileBaseName}.jpeg
    """)

    return x, u, v


def parse_args():
    parser = argparse.ArgumentParser(
        description="A CLI tool for configuring parameters"
    )
    parser.add_argument(
        "--m",
        type=int,
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
        type=float,
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
    print(f"Output files will be saved with the basename:\n\t {basename}")

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
