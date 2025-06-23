#!/usr/bin/env python3
#

import numpy as np
import matplotlib.pyplot as plt


def heat_equation_neumann_1D(Nx=101, dt=1e-4, T=0.1):
    """
    Solve the 1D heat equation u_t = u_xx
    on x in [0, 2π] with Neumann boundary conditions:
      u'(0,t) = 0, u'(2π, t) = 0
    and initial condition: u(x, 0) = cos(x) + 2.

    Parameters:
    -----------
    Nx : int
        Number of spatial grid points.
    dt : float
        Time step size.
    T : float
        Final time for the simulation.

    Returns:
    --------
    x : numpy.ndarray
        Spatial grid points.
    u : numpy.ndarray
        Solution at final time.
    """
    # 1) Set up the spatial grid
    L = 2.0 * np.pi
    x = np.linspace(0, L, Nx)
    dx = x[1] - x[0]

    # 2) Define the initial condition
    u = np.cos(x) + 2.0

    # 3) Number of time steps needed
    nt = int(np.ceil(T/dt))

    # Create a copy to store the updated solution each time step
    u_new = np.copy(u)

    for n in range(nt):
        # Update interior points (i = 1 to Nx-2)
        for i in range(1, Nx-1):
            u_xx = (u[i+1] - 2*u[i] + u[i-1]) / dx**2
            u_new[i] = u[i] + dt * u_xx

        # Neumann BC at the left boundary i=0: u'(0,t) = 0
        # => second derivative approx: 2*(u[1] - u[0]) / dx^2
        u_new[0] = u[0] + dt * (2.0*(u[1] - u[0]) / dx**2)

        # Neumann BC at the right boundary i = Nx-1: u'(2π,t) = 0
        # => second derivative approx: 2*(u[Nx-2] - u[Nx-1]) / dx^2
        u_new[Nx-1] = u[Nx-1] + dt * (2.0*(u[Nx-2] - u[Nx-1]) / dx**2)

        # Swap references: now u_new becomes the "old" solution for the next step
        u, u_new = u_new, u

    # Return the final solution
    return x, u


# --- Run the solver and visualize ---
if __name__ == "__main__":
    # Set parameters
    Nx = 101         # number of spatial points
    dt = 1e-4        # time step (should be small for stability)
    T  = 0.1         # final time

    # Solve
    x, u_final = heat_equation_neumann_1D(Nx=Nx, dt=dt, T=T)

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(x, u_final, label=f"t={T}")
    plt.xlabel('x')
    plt.ylabel('u(x,t)')
    plt.title('1D Heat Equation with Neumann BC')
    plt.grid(True)
    plt.legend()
    plt.show()
