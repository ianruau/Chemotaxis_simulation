#!/usr/bin/env python3
from __future__ import annotations

from typing import Tuple

import numpy as np


def chi_star_threshold_continuum_1d(
    *,
    u_star: float,
    v_star: float,
    c: float,
    a: float,
    alpha: float,
    mu: float,
    nu: float,
    gamma: float,
    m: float,
    beta: float,
    L: float,
    n_max: int = 5000,
) -> float:
    """
    Continuum 1D threshold scan for Paper II Eq. (1.12) on (0,L) with Neumann
    eigenvalues lambda_n = (n*pi/L)^2, n>=1.

    This mirrors the threshold used by `chemotaxis-sim` in summary plots.
    """
    if n_max < 1:
        raise ValueError("n_max must be >= 1")
    if u_star <= 0:
        raise ValueError("u_star must be positive to compute chi^*.")
    if nu == 0 or gamma == 0:
        raise ValueError("nu and gamma must be nonzero to compute chi^*.")
    if L <= 0:
        raise ValueError("L must be positive to compute chi^*.")
    if c + v_star <= 0:
        raise ValueError("Expected c + v_star > 0 for sensitivity (c+v)^(-beta).")

    prefactor = ((c + v_star) ** beta) / (
        nu * gamma * (u_star ** (m + gamma - 1.0))
    )
    n = np.arange(1, int(n_max) + 1, dtype=np.float64)
    lam = (n * np.pi / float(L)) ** 2
    values = prefactor * (((lam + a * alpha) * (mu + lam)) / lam)
    return float(np.min(values))


def neumann_eigenvalue_1d_discrete(*, n: int, L: float, meshsize: int) -> float:
    """
    Discrete Neumann eigenvalues for the standard 3-point Laplacian on a uniform
    grid of (0, L) with N=meshsize subintervals (N+1 grid points).

      h = L / N
      lambda_n^disc = 4/h^2 * sin^2(n*pi/(2N)),  n = 0,1,...,N.
    """
    if meshsize <= 0:
        raise ValueError("meshsize must be >= 1")
    if n < 0:
        raise ValueError("n must be >= 0")
    if n > meshsize:
        raise ValueError("n must be <= meshsize for the discrete Laplacian spectrum")
    if L <= 0:
        raise ValueError("L must be > 0")
    h = float(L) / float(meshsize)
    return float((4.0 / (h * h)) * (np.sin(n * np.pi / (2.0 * meshsize)) ** 2))


def chi_star_disc_fd(
    *,
    u_star: float,
    v_star: float,
    c: float,
    a: float,
    alpha: float,
    mu: float,
    nu: float,
    gamma: float,
    m: float,
    beta: float,
    L: float,
    meshsize: int,
) -> Tuple[float, int, float]:
    """
    Mesh-dependent discrete threshold chi^{*,disc} based on the eigenvalues of the
    discrete Neumann Laplacian on N=meshsize subintervals.

    Returns (chi_star_disc, n_min_disc, lambda_n_min_disc).
    """
    if u_star <= 0:
        raise ValueError("u_star must be > 0")
    if nu == 0 or gamma == 0:
        raise ValueError("nu and gamma must be nonzero")
    if L <= 0:
        raise ValueError("L must be > 0")
    if meshsize < 1:
        raise ValueError("meshsize must be >= 1")
    if c + v_star <= 0:
        raise ValueError("Eq. (1.12) requires c + v_star > 0.")

    prefactor = ((c + v_star) ** beta) / (
        nu * gamma * (u_star ** (m + gamma - 1.0))
    )

    best_val = float("inf")
    best_n = 1
    best_lambda = neumann_eigenvalue_1d_discrete(n=1, L=L, meshsize=meshsize)
    for n in range(1, int(meshsize) + 1):
        lam = neumann_eigenvalue_1d_discrete(n=n, L=L, meshsize=meshsize)
        val = prefactor * (((lam + a * alpha) * (mu + lam)) / lam)
        if val < best_val:
            best_val = float(val)
            best_n = int(n)
            best_lambda = float(lam)

    return best_val, best_n, best_lambda

