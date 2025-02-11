#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 2 * np.pi          # Domain length
N = 100                # Number of spatial points
dx = L / (N - 1)       # Spatial step size
x = np.linspace(0, L, N)  # Spatial grid

alpha = 1.0            # Thermal diffusivity
T_total = 2.0          # Total simulation time
dt = 0.001             # Time step
nt = int(T_total / dt) # Number of time steps

# Check stability condition
max_dt = dx**2 / (2 * alpha)
if dt > max_dt:
    raise ValueError(f"Time step dt={dt} too large; should be <= {max_dt:.3f}")

# Initial condition
u = np.cos(x) + 2
u_initial = u.copy()

# Time-stepping loop
for n in range(nt):
    d2u = np.zeros_like(u)
    # Compute second derivative
    d2u[1:-1] = (u[2:] - 2*u[1:-1] + u[:-2]) / dx**2  # Interior points
    d2u[0] = 2 * (u[1] - u[0]) / dx**2                # Left boundary
    d2u[-1] = 2 * (u[-2] - u[-1]) / dx**2              # Right boundary

    # Update solution
    u += alpha * dt * d2u

# Plot initial and final solutions
plt.figure(figsize=(8, 4))
plt.plot(x, u_initial, label='Initial')
plt.plot(x, u, label=f'Final (t={T_total})')
plt.xlabel('x')
plt.ylabel('u(x, t)')
plt.title('1D Heat Equation with Neumann Boundary Conditions')
plt.legend()
plt.grid(True)
plt.show()

# Verify conservation of the integral (approximately)
integral_initial = np.sum(u_initial) * dx
integral_final = np.sum(u) * dx
print(f"Initial integral: {integral_initial:.5f}")
print(f"Final integral:   {integral_final:.5f}")
