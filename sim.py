#!/usr/bin/env python3
#

import numpy as np
import matplotlib.pyplot as plt

# Set the constants of the system

gamma = 1

L = 1 #Length of \Omega
T = 10 # Maximum time
n = 50 #Amount of dx
m = 1000 # Amount of dt
h = L / 50 # Space step dx
k = T / 1000 # Time step dt

# Set the grids for u and v
u = np.zeros((m,n))
v = np.zeros((m,n))

print(u.shape)
#Set the initial conditions in the first column of u and v
for j in range(0,n):
    u[0,j] = np.sin(np.pi * j * h)
    v[0,j] = np.sin(np.pi * j * h)

print('u=',u)
print('v=',v)

#for i in range(1,m): #We run through time
i = 1
for j in range(1,n-2): #We run through space
    u_x[i,j] =
    u[i+1,j] = (k / h ** 2) * (u[i,j+1] - 2 * u[i,j] + u[i,j-1]) +
    v[i,j] = (v[i,j+1] + v[i,j-1])/(2 + h**2) + h**2/(2 + h**2) * (u[i,j])**gamma
# def solve_pde_system(L=1, Nx=50, Nt=1000, T=10):
#     dx = L / Nx
#     dt = T / Nt
#     x = np.linspace(0, L, Nx + 1, dtype=np.float64)
#     #u = (np.sin(np.pi * x) + 1.5).astype(np.float64)  # Initial condition for u
#     #v = (np.sin(np.pi * x) + 1.5).astype(np.float64)  # Initial condition for v
#     u = np.ones_like(x) * 1.0
#     v = np.ones_like(x) * 1.0
#     print('u=',u)
#     print('v=',v)
#
#     times_to_plot = np.arange(0, T, 1)
#     current_time = 0
#     plt.figure()
#
#     for n in range(Nt):
#         u_new = np.copy(u).astype(np.float64)
#         v_new = np.copy(v).astype(np.float64)
#
#         for i in range(1, Nx):
#             v_xx = (v[i+1] - 2*v[i] + v[i-1]) / dx**2
#             v_new[i] = v_xx + v[i] - u[i]
#
#         # Neumann boundary conditions for v
#         v_new[0] = v_new[1]
#         v_new[-1] = v_new[-2]
#
#         for i in range(1, Nx):
#             v_x = (v_new[i+1] - v_new[i-1]) / (2*dx)
#             v_xx = (v_new[i+1] - 2*v_new[i] + v_new[i-1]) / dx**2
#             u_x = (u[i+1] - u[i-1]) / (2*dx)
#             u_xx = (u[i+1] - 2*u[i] + u[i-1]) / dx**2
#
#             term1 = (2 / (1 + v_new[i])**3) * (v_x**2) * (u[i]**3)
#             term2 = (3 / (1 + v_new[i])**2) * (u[i]**2) * u_x * v_x
#             term3 = (1 / (1 + v_new[i])**2) * (u[i]**3) * (v_new[i] - u[i])
#             reaction = u[i] - u[i]**2
#
#             u_new[i] = u[i] + dt * (u_xx - term1 + term2 + term3 + reaction)
#
#         # Neumann boundary conditions for u
#         u_new[0] = u_new[1]
#         u_new[-1] = u_new[-2]
#
#         u = u_new  # Update u
#         v = v_new  # Update v
#
#         # Plot every 0.01 seconds
#         if np.isclose(current_time, times_to_plot, atol=dt).any():
#             plt.plot(x, u, label=f't={current_time:.2f}')
#         current_time += dt
#         print('n=',n)
#         print('u=',u)
#         print('v=',v)
#
#
#     plt.xlabel('x')
#     plt.ylabel('u')
#     plt.title('Evolution of u over time')
#     plt.legend()
#     plt.show()
#
#     return x, u, v
#
# # Run the solver
# x, u, v = solve_pde_system()
