#!/usr/bin/env python3
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np


def equilibrium_v_star_from_u(*, u_star: float, mu: float, nu: float, gamma: float) -> float:
    if u_star <= 0:
        raise ValueError("u_star must be positive.")
    if mu == 0:
        raise ValueError("mu must be nonzero.")
    return float((nu / mu) * (u_star ** gamma))


def resolve_equilibrium_u_v_star(
    *,
    a: float,
    b: float,
    alpha: float,
    mu: float,
    nu: float,
    gamma: float,
    equilibrium_mode: str = "logistic",
    u_star_fixed: Optional[float] = None,
) -> Tuple[float, float]:
    mode = str(equilibrium_mode or "logistic").strip().lower()
    if mode not in {"logistic", "fixed"}:
        raise ValueError("equilibrium_mode must be 'logistic' or 'fixed'.")

    if mode == "fixed":
        if u_star_fixed is None:
            raise ValueError("u_star_fixed is required when equilibrium_mode='fixed'.")
        u_star = float(u_star_fixed)
        return u_star, equilibrium_v_star_from_u(u_star=u_star, mu=mu, nu=nu, gamma=gamma)

    if a <= 0 or b <= 0:
        raise ValueError("logistic equilibrium requires a>0 and b>0.")
    if alpha == 0:
        raise ValueError("alpha must be nonzero.")
    if mu == 0:
        raise ValueError("mu must be nonzero.")

    u_star = float((a / b) ** (1.0 / alpha))
    return u_star, equilibrium_v_star_from_u(u_star=u_star, mu=mu, nu=nu, gamma=gamma)


def chi_mode_threshold_1d(
    *,
    lambda_n: float,
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
    equilibrium_mode: str = "logistic",
) -> float:
    if lambda_n <= 0:
        raise ValueError("lambda_n must be positive.")
    if u_star <= 0:
        raise ValueError("u_star must be positive to compute chi^*.")
    if nu == 0 or gamma == 0:
        raise ValueError("nu and gamma must be nonzero to compute chi^*.")
    if c + v_star <= 0:
        raise ValueError("Expected c + v_star > 0 for sensitivity (c+v)^(-beta).")

    prefactor = ((c + v_star) ** beta) / (
        nu * gamma * (u_star ** (m + gamma - 1.0))
    )
    mode = str(equilibrium_mode or "logistic").strip().lower()
    if mode == "fixed":
        return float(prefactor * (mu + lambda_n))
    if mode != "logistic":
        raise ValueError("equilibrium_mode must be 'logistic' or 'fixed'.")
    return float(prefactor * (((lambda_n + a * alpha) * (mu + lambda_n)) / lambda_n))


def sigma_mode_1d(
    *,
    lambda_n: float,
    chi: float,
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
    equilibrium_mode: str = "logistic",
) -> float:
    if lambda_n <= 0:
        raise ValueError("lambda_n must be positive.")
    if u_star <= 0:
        raise ValueError("u_star must be positive to compute sigma.")
    if nu == 0 or gamma == 0:
        raise ValueError("nu and gamma must be nonzero to compute sigma.")
    if c + v_star <= 0:
        raise ValueError("Expected c + v_star > 0 for sensitivity (c+v)^(-beta).")

    chem = chi * nu * gamma * (u_star ** (m + gamma - 1.0))
    chem /= (c + v_star) ** beta
    chem *= lambda_n / (mu + lambda_n)
    mode = str(equilibrium_mode or "logistic").strip().lower()
    if mode == "fixed":
        return float(-lambda_n + chem)
    if mode != "logistic":
        raise ValueError("equilibrium_mode must be 'logistic' or 'fixed'.")
    return float(-lambda_n - a * alpha + chem)


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
    equilibrium_mode: str = "logistic",
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

    n = np.arange(1, int(n_max) + 1, dtype=np.float64)
    lam = (n * np.pi / float(L)) ** 2
    values = np.array(
        [
            chi_mode_threshold_1d(
                lambda_n=float(lam_i),
                u_star=u_star,
                v_star=v_star,
                c=c,
                a=a,
                alpha=alpha,
                mu=mu,
                nu=nu,
                gamma=gamma,
                m=m,
                beta=beta,
                equilibrium_mode=equilibrium_mode,
            )
            for lam_i in lam
        ],
        dtype=np.float64,
    )
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
    equilibrium_mode: str = "logistic",
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

    best_val = float("inf")
    best_n = 1
    best_lambda = neumann_eigenvalue_1d_discrete(n=1, L=L, meshsize=meshsize)
    for n in range(1, int(meshsize) + 1):
        lam = neumann_eigenvalue_1d_discrete(n=n, L=L, meshsize=meshsize)
        val = chi_mode_threshold_1d(
            lambda_n=float(lam),
            u_star=u_star,
            v_star=v_star,
            c=c,
            a=a,
            alpha=alpha,
            mu=mu,
            nu=nu,
            gamma=gamma,
            m=m,
            beta=beta,
            equilibrium_mode=equilibrium_mode,
        )
        if val < best_val:
            best_val = float(val)
            best_n = int(n)
            best_lambda = float(lam)

    return best_val, best_n, best_lambda
