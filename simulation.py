#!/usr/bin/env python3
#

import numpy as np
import matplotlib.pyplot as plt


def solve_pde_system(L=1, Nx=50, T=10):
    Nt = int(2 * T * Nx * Nx / L**2) + 1
    dx = L / Nx
    dt = T / Nt
    x = np.linspace(0, L, Nx + 1, dtype=np.float64)
    # u = (np.sin(np.pi * x) + 1.5).astype(np.float64)  # Initial condition for u
    # v = (np.sin(np.pi * x) + 1.5).astype(np.float64)  # Initial condition for v
    u = np.ones_like(x) * 1.0
    v = np.ones_like(x) * 1.0
    print('u=', u)
    print('v=', v)

    times_to_plot = np.arange(0, T, 1)
    current_time = 0
    plt.figure()

    for n in range(Nt):
        u_new = np.copy(u).astype(np.float64)
        v_new = np.copy(v).astype(np.float64)

        for i in range(1, Nx):
            v_xx = (v[i + 1] - 2*v[i] + v[i-1]) / dx**2
            v_new[i] = v_xx + v[i] - u[i]

        # Neumann boundary conditions for v
        v_new[0] = v_new[1]
        v_new[-1] = v_new[-2]

        for i in range(1, Nx):
            v_x = (v_new[i+1] - v_new[i-1]) / (2*dx)
            v_xx = (v_new[i+1] - 2*v_new[i] + v_new[i-1]) / dx**2
            u_x = (u[i+1] - u[i-1]) / (2*dx)
            u_xx = (u[i+1] - 2*u[i] + u[i-1]) / dx**2

            term1 = (2 / (1 + v_new[i])**3) * (v_x**2) * (u[i]**3)
            term2 = (3 / (1 + v_new[i])**2) * (u[i]**2) * u_x * v_x
            term3 = (1 / (1 + v_new[i])**2) * (u[i]**3) * (v_new[i] - u[i])
            reaction = u[i] - u[i]**2

            u_new[i] = u[i] + dt * (u_xx - term1 + term2 + term3 + reaction)

        # Neumann boundary conditions for u
        u_new[0] = u_new[1]
        u_new[-1] = u_new[-2]

        u = u_new  # Update u
        v = v_new  # Update v

        # Plot every 0.01 seconds
        if np.isclose(current_time, times_to_plot, atol=dt).any():
            plt.plot(x, u, label=f't={current_time:.2f}')
        current_time += dt
        print('n=',n)
        print('u=',u)
        print('v=',v)

    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('Evolution of u over time')
    plt.legend()
    plt.show()

    return x, u, v


# Run the solver
x, u, v = solve_pde_system()
