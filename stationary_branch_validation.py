#!/usr/bin/env python3
"""Validate the local stationary branches used in Paper III.

The continuation parameter is the signed first-cosine amplitude, not chi.
This removes the trivial constant branch from the nonlinear solve.  The
stationary residual, elliptic solve, discrete threshold, and normal-form
coefficients are the public production operators from
``Chemotaxis_simulation``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import platform
import subprocess
import zipfile
from copy import copy
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib
import numpy as np
import scipy
from scipy.optimize import root

import simulation as simulator_module
from implied_constants import compute_bifurcation_coefficients
from simulation import SimulationConfig, laplacian_NBC, rhs, solve_v
from thresholds import (
    chi_mode_threshold_1d,
    chi_star_disc_fd,
    neumann_eigenvalue_1d_discrete,
)


DEFAULT_MESHES = (40, 80, 160)
DEFAULT_AMPLITUDES = (-0.02, -0.01, -0.005, -0.0025, 0.0025, 0.005, 0.01, 0.02)
INTERCEPT_ABSOLUTE_TOLERANCE = 1e-9
MINIMUM_OBSERVED_ORDER = 1.8
CONTINUUM_MODE_SCAN_MAX = 5000


@dataclass(frozen=True)
class ValidationCase:
    """One parameter set from the Paper III numerical validation."""

    name: str
    title: str
    minimal: bool
    a: float
    b: float
    alpha: float
    beta: float
    m: float
    gamma: float
    nu: float = 1.0
    mu: float = 1.0
    c: float = 1.0
    length: float = 1.0
    u_star: float = 1.0
    mode: int = 1

    @property
    def equilibrium_mode(self) -> str:
        return "fixed" if self.minimal else "logistic"

    def coefficient_parameters(self) -> dict[str, float | int | str]:
        parameters: dict[str, float | int | str] = {
            "equilibrium_mode": self.equilibrium_mode,
            "a": self.a,
            "b": self.b,
            "c": self.c,
            "alpha": self.alpha,
            "beta": self.beta,
            "m": self.m,
            "gamma": self.gamma,
            "nu": self.nu,
            "mu": self.mu,
            "L": self.length,
            "n0": self.mode,
        }
        if self.minimal:
            parameters["u_star_fixed"] = self.u_star
        return parameters


CASES: dict[str, ValidationCase] = {
    case.name: case
    for case in (
        ValidationCase(
            name="nonminimal-a10-beta0",
            title="nonminimal: a=10, beta=0 (supercritical)",
            minimal=False,
            a=10.0,
            b=1.0,
            alpha=1.0,
            beta=0.0,
            m=1.0,
            gamma=1.0,
            u_star=10.0,
        ),
        ValidationCase(
            name="nonminimal-beta3",
            title="nonminimal: beta=3 (subcritical)",
            minimal=False,
            a=1.0,
            b=1.0,
            alpha=1.0,
            beta=3.0,
            m=1.0,
            gamma=1.0,
        ),
        ValidationCase(
            name="minimal-m1-g1",
            title="minimal: m=1, gamma=1 (supercritical)",
            minimal=True,
            a=0.0,
            b=0.0,
            alpha=1.0,
            beta=1.0,
            m=1.0,
            gamma=1.0,
        ),
        ValidationCase(
            name="minimal-m2-g2",
            title="minimal: m=2, gamma=2 (subcritical)",
            minimal=True,
            a=0.0,
            b=0.0,
            alpha=1.0,
            beta=1.0,
            m=2.0,
            gamma=2.0,
        ),
    )
}


@dataclass(frozen=True)
class DiscreteThreshold:
    chi_star: float
    minimizing_mode: int
    eigenvalue: float


@dataclass(frozen=True)
class ContinuumThreshold:
    chi_star: float
    minimizing_mode: int
    runner_up_gap: float


@dataclass(frozen=True)
class BranchPoint:
    state_id: str
    reflection_partner_id: str
    case_name: str
    mesh: int
    requested_amplitude: float
    measured_amplitude: float
    chi: float
    chi_star_disc: float
    chi_star_continuum: float
    alpha_n0: float
    beta_n0: float
    theory_c2: float
    point_c2: float
    solver_success: bool
    solver_message: str
    function_evaluations: int
    system_residual_max: float
    full_stationary_residual_max: float
    elliptic_residual_max: float
    amplitude_error: float
    mass_constraint_error: float | None
    u_min: float
    v_min: float
    sensitivity_denominator_min: float
    x: np.ndarray
    u: np.ndarray
    v: np.ndarray


@dataclass(frozen=True)
class BranchFit:
    case_name: str
    mesh: int
    chi_star_disc: float
    minimizing_mode: int
    discrete_eigenvalue: float
    c2: float
    c4: float
    theory_c2: float
    c2_relative_error: float
    fit_residual_max: float
    unconstrained_intercept: float
    intercept_error: float
    unconstrained_c2: float
    unconstrained_c4: float
    unconstrained_fit_residual_max: float
    observed_order: float | None
    reflection_chi_error_max: float | None
    reflection_u_error_max: float | None
    reflection_v_error_max: float | None


@dataclass(frozen=True)
class Gate:
    name: str
    passed: bool
    value: float | int | str
    limit: float | int | str


def trapezoidal_integral(values: np.ndarray, length: float) -> float:
    """Integrate nodal values on an equispaced grid."""

    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size < 2:
        raise ValueError("trapezoidal_integral expects at least two nodal values")
    weights = np.ones(array.size, dtype=np.float64)
    weights[[0, -1]] = 0.5
    return float((float(length) / (array.size - 1)) * np.dot(weights, array))


def critical_amplitude(
    u: np.ndarray,
    *,
    u_star: float,
    x: np.ndarray,
    length: float,
    mode: int,
) -> float:
    """Return the cosine coefficient in the manuscript normalization."""

    phi = np.cos(float(mode) * np.pi * np.asarray(x, dtype=np.float64) / float(length))
    return (2.0 / float(length)) * trapezoidal_integral(
        (np.asarray(u, dtype=np.float64) - float(u_star)) * phi,
        float(length),
    )


@lru_cache(maxsize=None)
def bifurcation_coefficients(case_name: str) -> dict[str, float]:
    """Evaluate coefficients through the simulator's public API."""

    public_result = compute_bifurcation_coefficients(
        CASES[case_name].coefficient_parameters()
    )
    return {
        **public_result,
        "chi_star": float(public_result["chi_star_mode_n0"]),
    }


@lru_cache(maxsize=None)
def continuum_threshold(case_name: str) -> ContinuumThreshold:
    """Return the continuum minimum and its separation from the runner-up."""

    case = CASES[case_name]
    v_star = (case.nu / case.mu) * case.u_star**case.gamma
    values: list[tuple[float, int]] = []
    for mode in range(1, CONTINUUM_MODE_SCAN_MAX + 1):
        eigenvalue = (mode * math.pi / case.length) ** 2
        value = chi_mode_threshold_1d(
            lambda_n=eigenvalue,
            u_star=case.u_star,
            v_star=v_star,
            c=case.c,
            a=case.a,
            alpha=case.alpha,
            mu=case.mu,
            nu=case.nu,
            gamma=case.gamma,
            m=case.m,
            beta=case.beta,
            equilibrium_mode=case.equilibrium_mode,
        )
        values.append((float(value), mode))
    values.sort()
    minimum, minimizing_mode = values[0]
    runner_up_gap = values[1][0] - minimum
    if minimizing_mode != case.mode:
        raise ValueError(
            f"{case.name}: continuum critical mode is {minimizing_mode}, "
            f"not requested mode {case.mode}"
        )
    if runner_up_gap <= 1e-12 * max(1.0, abs(minimum)):
        raise ValueError(
            f"{case.name}: continuum critical mode is not numerically unique; "
            f"runner-up gap={runner_up_gap:.3e}"
        )
    coefficient_threshold = bifurcation_coefficients(case.name)["chi_star"]
    if not math.isclose(minimum, coefficient_threshold, rel_tol=1e-12, abs_tol=1e-12):
        raise RuntimeError(
            f"{case.name}: continuum scan and coefficient threshold disagree"
        )
    return ContinuumThreshold(
        chi_star=minimum,
        minimizing_mode=minimizing_mode,
        runner_up_gap=runner_up_gap,
    )


def discrete_threshold(case: ValidationCase, mesh: int) -> DiscreteThreshold:
    """Compute the finite-difference instability threshold directly."""

    v_star = (case.nu / case.mu) * case.u_star**case.gamma
    chi_star, minimizing_mode, eigenvalue = chi_star_disc_fd(
        u_star=case.u_star,
        v_star=v_star,
        c=case.c,
        a=case.a,
        alpha=case.alpha,
        mu=case.mu,
        nu=case.nu,
        gamma=case.gamma,
        m=case.m,
        beta=case.beta,
        L=case.length,
        meshsize=int(mesh),
        equilibrium_mode=case.equilibrium_mode,
    )
    expected_eigenvalue = neumann_eigenvalue_1d_discrete(
        n=case.mode,
        L=case.length,
        meshsize=int(mesh),
    )
    if minimizing_mode != case.mode:
        raise ValueError(
            f"{case.name}: discrete critical mode is {minimizing_mode}, "
            f"not requested mode {case.mode} on mesh {mesh}"
        )
    if not math.isclose(eigenvalue, expected_eigenvalue, rel_tol=1e-13, abs_tol=1e-13):
        raise RuntimeError("discrete threshold returned an inconsistent eigenvalue")
    return DiscreteThreshold(
        chi_star=float(chi_star),
        minimizing_mode=int(minimizing_mode),
        eigenvalue=float(eigenvalue),
    )


def _base_config(case: ValidationCase, mesh: int, chi: float) -> SimulationConfig:
    return SimulationConfig(
        a=case.a,
        b=case.b,
        alpha=case.alpha,
        beta=case.beta,
        m=case.m,
        gamma=case.gamma,
        nu=case.nu,
        mu=case.mu,
        c=case.c,
        L=case.length,
        meshsize=int(mesh),
        chi=float(chi),
        equilibrium_mode=case.equilibrium_mode,
        u_star_fixed=case.u_star if case.minimal else None,
        eigen_mode_n=case.mode,
        diagnostic=False,
    )


def _config_at_chi(base: SimulationConfig, chi: float) -> SimulationConfig:
    """Copy a fully initialized config without rerunning its threshold scan."""

    config = copy(base)
    object.__setattr__(config, "chi", float(chi))
    return config


def _elliptic_solution(u: np.ndarray, case: ValidationCase, mesh: int) -> np.ndarray:
    return solve_v(
        np.asarray(u, dtype=np.float64),
        case.length,
        int(mesh),
        case.mu,
        case.nu,
        case.gamma,
        diagnostic=False,
    )


def state_identifier(case_name: str, mesh: int, amplitude: float) -> str:
    """Return a path-safe identifier stable across runs and output formats."""

    if amplitude == 0.0:
        raise ValueError("a branch state identifier requires nonzero amplitude")
    sign = "plus" if amplitude > 0.0 else "minus"
    magnitude = format(abs(float(amplitude)), ".12g")
    magnitude_token = magnitude.replace("+", "p").replace("-", "m").replace(".", "p")
    return f"{case_name}__N{int(mesh):04d}__A{sign}_{magnitude_token}"


def solve_branch_point(
    case: ValidationCase,
    *,
    mesh: int,
    amplitude: float,
    threshold: DiscreteThreshold | None = None,
    previous: BranchPoint | None = None,
    solver_xtol: float = 1e-11,
) -> BranchPoint:
    """Solve one amplitude-constrained stationary problem."""

    if mesh < 4:
        raise ValueError("mesh must be at least 4")
    if amplitude == 0.0:
        raise ValueError("amplitude must be nonzero; A=0 leaves chi undetermined")
    threshold = threshold or discrete_threshold(case, mesh)
    coefficients = bifurcation_coefficients(case.name)
    theory_c2 = coefficients["beta_n0"] / coefficients["alpha_n0"]
    x = np.linspace(0.0, case.length, int(mesh) + 1, dtype=np.float64)
    phi = np.cos(case.mode * np.pi * x / case.length)

    if previous is None:
        u_guess = case.u_star + float(amplitude) * phi
        chi_guess = threshold.chi_star + theory_c2 * float(amplitude) ** 2
    else:
        if previous.case_name != case.name or previous.mesh != mesh:
            raise ValueError("continuation seed must come from the same case and mesh")
        u_guess = previous.u + (float(amplitude) - previous.requested_amplitude) * phi
        chi_guess = previous.chi + theory_c2 * (
            float(amplitude) ** 2 - previous.requested_amplitude**2
        )

    base = _base_config(case, mesh, threshold.chi_star)
    mass_target = case.u_star * case.length

    def system(unknown: np.ndarray) -> np.ndarray:
        u = np.asarray(unknown[:-1], dtype=np.float64)
        chi = float(unknown[-1])
        v = _elliptic_solution(u, case, mesh)
        stationary = rhs(u, v, _config_at_chi(base, chi))
        amplitude_residual = critical_amplitude(
            u,
            u_star=case.u_star,
            x=x,
            length=case.length,
            mode=case.mode,
        ) - float(amplitude)
        if case.minimal:
            mass_residual = trapezoidal_integral(u, case.length) - mass_target
            # The conservative flux discretization has one weighted linear
            # dependence.  Drop the last stationary equation and check it a
            # posteriori through ``full_stationary_residual_max``.
            return np.concatenate(
                (stationary[:-1], np.array([mass_residual, amplitude_residual]))
            )
        return np.concatenate((stationary, np.array([amplitude_residual])))

    initial = np.concatenate((u_guess, np.array([chi_guess], dtype=np.float64)))
    solution = root(
        system,
        initial,
        method="hybr",
        options={"xtol": float(solver_xtol), "maxfev": 100 * (int(mesh) + 2)},
    )
    unknown = np.asarray(solution.x, dtype=np.float64)
    if not np.all(np.isfinite(unknown)):
        raise RuntimeError(
            f"{case.name}, N={mesh}, A={amplitude}: nonfinite Newton iterate"
        )

    u = unknown[:-1]
    chi = float(unknown[-1])
    v = _elliptic_solution(u, case, mesh)
    full_stationary = rhs(u, v, _config_at_chi(base, chi))
    elliptic_residual = (
        laplacian_NBC(case.length, int(mesh), v)
        - case.mu * v
        + case.nu * np.power(u, case.gamma)
    )
    system_residual = system(unknown)
    measured_amplitude = critical_amplitude(
        u,
        u_star=case.u_star,
        x=x,
        length=case.length,
        mode=case.mode,
    )
    mass_error = (
        trapezoidal_integral(u, case.length) - mass_target if case.minimal else None
    )
    state_id = state_identifier(case.name, mesh, amplitude)

    point = BranchPoint(
        state_id=state_id,
        reflection_partner_id=state_identifier(case.name, mesh, -amplitude),
        case_name=case.name,
        mesh=int(mesh),
        requested_amplitude=float(amplitude),
        measured_amplitude=float(measured_amplitude),
        chi=chi,
        chi_star_disc=threshold.chi_star,
        chi_star_continuum=float(coefficients["chi_star"]),
        alpha_n0=float(coefficients["alpha_n0"]),
        beta_n0=float(coefficients["beta_n0"]),
        theory_c2=float(theory_c2),
        point_c2=float((chi - threshold.chi_star) / float(amplitude) ** 2),
        solver_success=bool(solution.success),
        solver_message=str(solution.message),
        function_evaluations=int(solution.nfev),
        system_residual_max=float(np.max(np.abs(system_residual))),
        full_stationary_residual_max=float(np.max(np.abs(full_stationary))),
        elliptic_residual_max=float(np.max(np.abs(elliptic_residual))),
        amplitude_error=float(measured_amplitude - float(amplitude)),
        mass_constraint_error=float(mass_error) if mass_error is not None else None,
        u_min=float(np.min(u)),
        v_min=float(np.min(v)),
        sensitivity_denominator_min=float(np.min(case.c + v)),
        x=x,
        u=np.array(u, dtype=np.float64, copy=True),
        v=np.array(v, dtype=np.float64, copy=True),
    )
    if not point.solver_success and point.system_residual_max > 1e-8:
        raise RuntimeError(
            f"{case.name}, N={mesh}, A={amplitude}: Newton failed: "
            f"{point.solver_message}; residual={point.system_residual_max:.3e}"
        )
    return point


def fit_quadratic_quartic(
    amplitudes: Sequence[float], delta_chi: Sequence[float]
) -> tuple[float, float, float]:
    """Fit delta-chi = c2*A^2 + c4*A^4 with the intercept fixed at zero."""

    amplitude_array = np.asarray(amplitudes, dtype=np.float64)
    delta_array = np.asarray(delta_chi, dtype=np.float64)
    if amplitude_array.shape != delta_array.shape or amplitude_array.ndim != 1:
        raise ValueError(
            "amplitudes and delta_chi must be one-dimensional arrays of equal size"
        )
    if len({round(abs(float(value)), 15) for value in amplitude_array}) < 2:
        raise ValueError(
            "at least two distinct absolute amplitudes are required for the fit"
        )
    squared = amplitude_array**2
    scale = float(np.max(squared))
    design = np.column_stack((squared / scale, (squared / scale) ** 2))
    scaled_coefficients, _, rank, _ = np.linalg.lstsq(design, delta_array, rcond=None)
    if rank != 2:
        raise RuntimeError("quadratic/quartic branch fit is rank deficient")
    c2 = float(scaled_coefficients[0] / scale)
    c4 = float(scaled_coefficients[1] / scale**2)
    residual = delta_array - (c2 * squared + c4 * squared**2)
    return c2, c4, float(np.max(np.abs(residual)))


def fit_with_intercept(
    amplitudes: Sequence[float], chi_values: Sequence[float]
) -> tuple[float, float, float, float]:
    """Fit chi = intercept + c2*A^2 + c4*A^4 as a threshold diagnostic."""

    amplitude_array = np.asarray(amplitudes, dtype=np.float64)
    chi_array = np.asarray(chi_values, dtype=np.float64)
    if amplitude_array.shape != chi_array.shape or amplitude_array.ndim != 1:
        raise ValueError(
            "amplitudes and chi_values must be one-dimensional arrays of equal size"
        )
    if len({round(abs(float(value)), 15) for value in amplitude_array}) < 3:
        raise ValueError(
            "at least three distinct absolute amplitudes are required for the intercept fit"
        )
    squared = amplitude_array**2
    scale = float(np.max(squared))
    scaled_squared = squared / scale
    design = np.column_stack(
        (np.ones(amplitude_array.size), scaled_squared, scaled_squared**2)
    )
    scaled_coefficients, _, rank, _ = np.linalg.lstsq(design, chi_array, rcond=None)
    if rank != 3:
        raise RuntimeError("intercept/quadratic/quartic branch fit is rank deficient")
    intercept = float(scaled_coefficients[0])
    c2 = float(scaled_coefficients[1] / scale)
    c4 = float(scaled_coefficients[2] / scale**2)
    residual = chi_array - (intercept + c2 * squared + c4 * squared**2)
    return intercept, c2, c4, float(np.max(np.abs(residual)))


def observed_convergence_order(
    coarse_error: float,
    fine_error: float,
    coarse_mesh: int,
    fine_mesh: int,
) -> float:
    """Return the empirical order for an error measured on two meshes."""

    if coarse_error <= 0.0 or fine_error <= 0.0:
        raise ValueError("convergence errors must be positive")
    if coarse_mesh <= 0 or fine_mesh <= coarse_mesh:
        raise ValueError("fine_mesh must be larger than a positive coarse_mesh")
    return float(
        math.log(float(coarse_error) / float(fine_error))
        / math.log(float(fine_mesh) / float(coarse_mesh))
    )


def _reflection_errors(
    points: Sequence[BranchPoint],
) -> tuple[float | None, float | None, float | None]:
    by_amplitude = {round(point.requested_amplitude, 15): point for point in points}
    chi_errors: list[float] = []
    u_errors: list[float] = []
    v_errors: list[float] = []
    for amplitude, positive in sorted(by_amplitude.items()):
        if amplitude <= 0.0:
            continue
        negative = by_amplitude.get(round(-amplitude, 15))
        if negative is None:
            continue
        chi_errors.append(abs(positive.chi - negative.chi))
        u_errors.append(float(np.max(np.abs(positive.u - negative.u[::-1]))))
        v_errors.append(float(np.max(np.abs(positive.v - negative.v[::-1]))))
    if not chi_errors:
        return None, None, None
    return max(chi_errors), max(u_errors), max(v_errors)


def fit_branch(
    case: ValidationCase,
    threshold: DiscreteThreshold,
    points: Sequence[BranchPoint],
) -> BranchFit:
    coefficients = bifurcation_coefficients(case.name)
    theory_c2 = coefficients["beta_n0"] / coefficients["alpha_n0"]
    amplitudes = [point.requested_amplitude for point in points]
    delta_chi = [point.chi - threshold.chi_star for point in points]
    c2, c4, fit_residual = fit_quadratic_quartic(amplitudes, delta_chi)
    intercept, free_c2, free_c4, free_fit_residual = fit_with_intercept(
        amplitudes, [point.chi for point in points]
    )
    chi_error, u_error, v_error = _reflection_errors(points)
    return BranchFit(
        case_name=case.name,
        mesh=points[0].mesh,
        chi_star_disc=threshold.chi_star,
        minimizing_mode=threshold.minimizing_mode,
        discrete_eigenvalue=threshold.eigenvalue,
        c2=c2,
        c4=c4,
        theory_c2=float(theory_c2),
        c2_relative_error=float(abs(c2 - theory_c2) / abs(theory_c2)),
        fit_residual_max=fit_residual,
        unconstrained_intercept=intercept,
        intercept_error=float(intercept - threshold.chi_star),
        unconstrained_c2=free_c2,
        unconstrained_c4=free_c4,
        unconstrained_fit_residual_max=free_fit_residual,
        observed_order=None,
        reflection_chi_error_max=chi_error,
        reflection_u_error_max=u_error,
        reflection_v_error_max=v_error,
    )


def _continuation_order(amplitudes: Sequence[float]) -> Iterable[list[float]]:
    positive = sorted((float(value) for value in amplitudes if value > 0.0), key=abs)
    negative = sorted((float(value) for value in amplitudes if value < 0.0), key=abs)
    if positive:
        yield positive
    if negative:
        yield negative


def run_validation(
    cases: Sequence[ValidationCase],
    *,
    meshes: Sequence[int],
    amplitudes: Sequence[float],
    solver_xtol: float = 1e-11,
) -> tuple[list[BranchPoint], list[BranchFit]]:
    """Run all requested cases and return branch points and meshwise fits."""

    case_names = [case.name for case in cases]
    if not case_names:
        raise ValueError("provide at least one validation case")
    if len(case_names) != len(set(case_names)):
        raise ValueError("validation cases must not be repeated")
    normalized_meshes = sorted({int(mesh) for mesh in meshes})
    normalized_amplitudes = sorted({float(value) for value in amplitudes})
    if not normalized_meshes:
        raise ValueError("provide at least one mesh")
    if any(mesh < 4 for mesh in normalized_meshes):
        raise ValueError("all meshes must be at least 4")
    if not normalized_amplitudes:
        raise ValueError("provide at least one amplitude")
    if not all(math.isfinite(value) for value in normalized_amplitudes):
        raise ValueError("amplitudes must be finite")
    if 0.0 in normalized_amplitudes:
        raise ValueError("amplitudes must not contain zero")
    if len({round(abs(value), 15) for value in normalized_amplitudes}) < 3:
        raise ValueError("provide at least three distinct absolute amplitudes")
    amplitude_keys = {round(value, 15) for value in normalized_amplitudes}
    if any(round(-value, 15) not in amplitude_keys for value in normalized_amplitudes):
        raise ValueError("amplitudes must contain an exact positive/negative pair")
    state_ids = {
        state_identifier(case_name, mesh, amplitude)
        for case_name in case_names
        for mesh in normalized_meshes
        for amplitude in normalized_amplitudes
    }
    expected_state_count = (
        len(case_names) * len(normalized_meshes) * len(normalized_amplitudes)
    )
    if len(state_ids) != expected_state_count:
        raise ValueError("amplitudes produce colliding state identifiers")
    if not math.isfinite(solver_xtol) or solver_xtol <= 0.0:
        raise ValueError("solver_xtol must be finite and positive")

    all_points: list[BranchPoint] = []
    all_fits: list[BranchFit] = []
    for case in cases:
        bifurcation_coefficients(case.name)
        continuum_threshold(case.name)
        for mesh in normalized_meshes:
            threshold = discrete_threshold(case, mesh)
            mesh_points: list[BranchPoint] = []
            for signed_group in _continuation_order(normalized_amplitudes):
                previous: BranchPoint | None = None
                for amplitude in signed_group:
                    point = solve_branch_point(
                        case,
                        mesh=mesh,
                        amplitude=amplitude,
                        threshold=threshold,
                        previous=previous,
                        solver_xtol=solver_xtol,
                    )
                    mesh_points.append(point)
                    previous = point
            mesh_points.sort(key=lambda point: point.requested_amplitude)
            all_points.extend(mesh_points)
            current_fit = fit_branch(case, threshold, mesh_points)
            previous_fits = [fit for fit in all_fits if fit.case_name == case.name]
            if previous_fits:
                previous_fit = previous_fits[-1]
                current_fit = replace(
                    current_fit,
                    observed_order=observed_convergence_order(
                        previous_fit.c2_relative_error,
                        current_fit.c2_relative_error,
                        previous_fit.mesh,
                        current_fit.mesh,
                    ),
                )
            all_fits.append(current_fit)
    return all_points, all_fits


def acceptance_gates(
    cases: Sequence[ValidationCase],
    points: Sequence[BranchPoint],
    fits: Sequence[BranchFit],
) -> list[Gate]:
    """Evaluate the submission-facing numerical acceptance gates."""

    gates: list[Gate] = []
    for point in points:
        prefix = f"{point.case_name}/N{point.mesh}/A{point.requested_amplitude:+g}"
        gates.extend(
            (
                Gate(
                    f"{prefix}/solver",
                    point.solver_success,
                    int(point.solver_success),
                    1,
                ),
                Gate(
                    f"{prefix}/full-residual",
                    point.full_stationary_residual_max <= 1e-8,
                    point.full_stationary_residual_max,
                    1e-8,
                ),
                Gate(
                    f"{prefix}/elliptic-residual",
                    point.elliptic_residual_max <= 1e-8,
                    point.elliptic_residual_max,
                    1e-8,
                ),
                Gate(
                    f"{prefix}/amplitude",
                    abs(point.amplitude_error) <= 5e-11,
                    abs(point.amplitude_error),
                    5e-11,
                ),
                Gate(f"{prefix}/u-positive", point.u_min > 0.0, point.u_min, "> 0"),
                Gate(f"{prefix}/v-positive", point.v_min > 0.0, point.v_min, "> 0"),
                Gate(
                    f"{prefix}/sensitivity-positive",
                    point.sensitivity_denominator_min > 0.0,
                    point.sensitivity_denominator_min,
                    "> 0",
                ),
            )
        )
        if point.mass_constraint_error is not None:
            gates.append(
                Gate(
                    f"{prefix}/mass",
                    abs(point.mass_constraint_error) <= 5e-12,
                    abs(point.mass_constraint_error),
                    5e-12,
                )
            )

    for case in cases:
        case_fits = sorted(
            (fit for fit in fits if fit.case_name == case.name),
            key=lambda fit: fit.mesh,
        )
        for fit in case_fits:
            gates.extend(
                (
                    Gate(
                        f"{case.name}/N{fit.mesh}/c2-sign",
                        fit.c2 * fit.theory_c2 > 0.0,
                        fit.c2,
                        f"same sign as {fit.theory_c2:+.12g}",
                    ),
                    Gate(
                        f"{case.name}/N{fit.mesh}/intercept",
                        abs(fit.intercept_error) <= INTERCEPT_ABSOLUTE_TOLERANCE,
                        abs(fit.intercept_error),
                        INTERCEPT_ABSOLUTE_TOLERANCE,
                    ),
                )
            )
            if fit.observed_order is not None:
                gates.append(
                    Gate(
                        f"{case.name}/N{fit.mesh}/c2-observed-order",
                        fit.observed_order >= MINIMUM_OBSERVED_ORDER,
                        fit.observed_order,
                        f">= {MINIMUM_OBSERVED_ORDER}",
                    )
                )
        for coarse, fine in zip(case_fits, case_fits[1:]):
            error_ratio = fine.c2_relative_error / coarse.c2_relative_error
            gates.append(
                Gate(
                    f"{case.name}/c2-error-monotone-N{coarse.mesh}-N{fine.mesh}",
                    error_ratio < 1.0,
                    error_ratio,
                    "< 1",
                )
            )
        finest = max(case_fits, key=lambda fit: fit.mesh)
        gates.append(
            Gate(
                f"{case.name}/finest-c2-relative-error",
                finest.c2_relative_error <= 5e-3,
                finest.c2_relative_error,
                5e-3,
            )
        )
        reflection_present = all(
            value is not None
            for value in (
                finest.reflection_chi_error_max,
                finest.reflection_u_error_max,
                finest.reflection_v_error_max,
            )
        )
        gates.append(
            Gate(
                f"{case.name}/finest-reflection-pairs-complete",
                reflection_present,
                int(reflection_present),
                1,
            )
        )
        if reflection_present:
            gates.extend(
                (
                    Gate(
                        f"{case.name}/finest-reflection-chi",
                        float(finest.reflection_chi_error_max) <= 1e-8,
                        float(finest.reflection_chi_error_max),
                        1e-8,
                    ),
                    Gate(
                        f"{case.name}/finest-reflection-u",
                        float(finest.reflection_u_error_max) <= 1e-9,
                        float(finest.reflection_u_error_max),
                        1e-9,
                    ),
                    Gate(
                        f"{case.name}/finest-reflection-v",
                        float(finest.reflection_v_error_max) <= 1e-9,
                        float(finest.reflection_v_error_max),
                        1e-9,
                    ),
                )
            )
    return gates


def _point_row(point: BranchPoint) -> dict[str, Any]:
    return {
        "state_id": point.state_id,
        "reflection_partner_id": point.reflection_partner_id,
        "case": point.case_name,
        "mesh": point.mesh,
        "requested_amplitude": point.requested_amplitude,
        "measured_amplitude": point.measured_amplitude,
        "chi": point.chi,
        "chi_star_disc": point.chi_star_disc,
        "chi_star_continuum": point.chi_star_continuum,
        "alpha_n0": point.alpha_n0,
        "beta_n0": point.beta_n0,
        "theory_c2": point.theory_c2,
        "point_c2": point.point_c2,
        "delta_chi": point.chi - point.chi_star_disc,
        "delta_chi_over_amplitude_squared": (
            (point.chi - point.chi_star_disc) / point.requested_amplitude**2
        ),
        "solver_success": point.solver_success,
        "function_evaluations": point.function_evaluations,
        "system_residual_max": point.system_residual_max,
        "full_stationary_residual_max": point.full_stationary_residual_max,
        "elliptic_residual_max": point.elliptic_residual_max,
        "amplitude_error": point.amplitude_error,
        "mass_constraint_error": point.mass_constraint_error,
        "u_min": point.u_min,
        "v_min": point.v_min,
        "sensitivity_denominator_min": point.sensitivity_denominator_min,
    }


def _write_csv(
    path: Path, rows: Sequence[dict[str, Any]], fieldnames: Sequence[str]
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _ordered_points(points: Sequence[BranchPoint]) -> list[BranchPoint]:
    case_order = {name: index for index, name in enumerate(CASES)}
    return sorted(
        points,
        key=lambda point: (
            case_order[point.case_name],
            point.mesh,
            point.requested_amplitude,
        ),
    )


def _write_deterministic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write an uncompressed NPZ with fixed member order and timestamps."""

    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            buffer = io.BytesIO()
            np.lib.format.write_array(
                buffer,
                np.asarray(arrays[name], dtype=np.float64),
                version=(1, 0),
                allow_pickle=False,
            )
            member = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            member.compress_type = zipfile.ZIP_STORED
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            archive.writestr(member, buffer.getvalue())


def _write_state_archive(
    archive_path: Path,
    index_path: Path,
    points: Sequence[BranchPoint],
) -> None:
    ordered = _ordered_points(points)
    present_ids = {point.state_id for point in ordered}
    arrays: dict[str, np.ndarray] = {}
    states: list[dict[str, Any]] = []
    for point in ordered:
        array_names = {field: f"{point.state_id}__{field}" for field in ("x", "u", "v")}
        arrays[array_names["x"]] = point.x
        arrays[array_names["u"]] = point.u
        arrays[array_names["v"]] = point.v
        states.append(
            {
                "state_id": point.state_id,
                "reflection_partner_id": point.reflection_partner_id,
                "reflection_partner_present": point.reflection_partner_id
                in present_ids,
                "case": point.case_name,
                "mesh": point.mesh,
                "requested_amplitude": point.requested_amplitude,
                "measured_amplitude": point.measured_amplitude,
                "chi": point.chi,
                "array_names": array_names,
                "array_shape": [point.mesh + 1],
            }
        )
    _write_deterministic_npz(archive_path, arrays)
    index = {
        "schema_version": 1,
        "archive": archive_path.name,
        "array_dtype": "float64",
        "array_naming": "<state_id>__<x|u|v>",
        "state_order": "case declaration, mesh ascending, amplitude ascending",
        "state_count": len(states),
        "states": states,
    }
    index_path.write_text(
        json.dumps(index, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_provenance(source_path: Path) -> dict[str, str | bool | None]:
    """Return path-free Git provenance for a source file when available."""

    try:
        root_process = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=source_path.resolve().parent,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        root = Path(root_process.stdout.strip())
        revision_process = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        status_process = subprocess.run(
            ["git", "status", "--short", "--untracked-files=all"],
            cwd=root,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        relative_path = source_path.resolve().relative_to(root).as_posix()
    except (OSError, subprocess.CalledProcessError, ValueError):
        return {"path": source_path.name, "revision": None, "dirty": None}
    return {
        "path": relative_path,
        "revision": revision_process.stdout.strip(),
        "dirty": bool(status_process.stdout.strip()),
    }


def software_provenance() -> dict[str, Any]:
    """Describe the generator, simulator, and deterministic runtime versions."""

    return {
        "generator": _git_provenance(Path(__file__)),
        "simulator": _git_provenance(Path(simulator_module.__file__)),
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "matplotlib": matplotlib.__version__,
        },
    }


def _write_sha256sums(path: Path, artifacts: Sequence[Path]) -> None:
    lines = [f"{_sha256(artifact)}  {artifact.name}" for artifact in sorted(artifacts)]
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _write_plot(
    path_pdf: Path,
    path_png: Path,
    cases: Sequence[ValidationCase],
    points: Sequence[BranchPoint],
    fits: Sequence[BranchFit],
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "text.usetex": False,
        }
    )
    from matplotlib import pyplot as plt

    column_count = min(2, len(cases))
    row_count = int(math.ceil(len(cases) / column_count))
    figure, axes = plt.subplots(
        row_count,
        column_count,
        figsize=(5.2 * column_count, 3.9 * row_count),
        squeeze=False,
    )
    colors = ("#0072B2", "#E69F00", "#009E73", "#CC79A7")
    flat_axes = list(axes.flat)
    for axis, case in zip(flat_axes, cases):
        case_fits = sorted(
            (fit for fit in fits if fit.case_name == case.name),
            key=lambda item: item.mesh,
        )
        max_squared = 0.0
        for color, fit in zip(colors, case_fits):
            fit_points = [
                point
                for point in points
                if point.case_name == case.name and point.mesh == fit.mesh
            ]
            squared = np.array([point.requested_amplitude**2 for point in fit_points])
            delta = np.array([point.chi - point.chi_star_disc for point in fit_points])
            max_squared = max(max_squared, float(np.max(squared)))
            axis.scatter(
                squared, delta, s=20, color=color, alpha=0.8, label=f"N={fit.mesh}"
            )
            grid = np.linspace(0.0, float(np.max(squared)), 200)
            axis.plot(
                grid, fit.c2 * grid + fit.c4 * grid**2, color=color, linewidth=1.2
            )
        theory_c2 = case_fits[-1].theory_c2
        theory_grid = np.linspace(0.0, max_squared, 200)
        axis.plot(
            theory_grid,
            theory_c2 * theory_grid,
            color="black",
            linestyle="--",
            linewidth=1.2,
            label="normal form",
        )
        axis.axhline(0.0, color="0.75", linewidth=0.7)
        axis.set_title(case.title)
        axis.set_xlabel("prescribed amplitude squared")
        axis.set_ylabel("chi minus discrete threshold")
        axis.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2))
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
    for axis in flat_axes[len(cases) :]:
        axis.set_visible(False)
    figure.suptitle("Amplitude-constrained stationary continuation", fontsize=12)
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure.savefig(
        path_pdf,
        bbox_inches="tight",
        metadata={
            "Title": "Paper III stationary branch validation",
            "Author": "Paper III reproducibility pipeline",
            "Creator": "codes/stationary_branch_validation.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    figure.savefig(path_png, bbox_inches="tight", dpi=200, metadata={"Software": None})
    plt.close(figure)


def write_outputs(
    output_dir: Path,
    cases: Sequence[ValidationCase],
    meshes: Sequence[int],
    amplitudes: Sequence[float],
    points: Sequence[BranchPoint],
    fits: Sequence[BranchFit],
    gates: Sequence[Gate],
) -> dict[str, Path]:
    """Write the deterministic machine-readable and graphical evidence."""

    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory must be empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    ordered_points = _ordered_points(points)
    point_rows = [_point_row(point) for point in ordered_points]
    points_path = output_dir / "branch-points.csv"
    _write_csv(points_path, point_rows, list(point_rows[0]))

    profile_rows: list[dict[str, Any]] = []
    for point in ordered_points:
        for index, (x_value, u_value, v_value) in enumerate(
            zip(point.x, point.u, point.v)
        ):
            profile_rows.append(
                {
                    "state_id": point.state_id,
                    "reflection_partner_id": point.reflection_partner_id,
                    "case": point.case_name,
                    "mesh": point.mesh,
                    "requested_amplitude": point.requested_amplitude,
                    "node": index,
                    "x": float(x_value),
                    "u": float(u_value),
                    "v": float(v_value),
                }
            )
    profiles_path = output_dir / "stationary-profiles.csv"
    _write_csv(profiles_path, profile_rows, list(profile_rows[0]))

    archive_path = output_dir / "stationary-states.npz"
    index_path = output_dir / "states-index.json"
    _write_state_archive(archive_path, index_path, ordered_points)

    summary_path = output_dir / "fit-summary.json"
    pdf_path = output_dir / "stationary-continuation.pdf"
    png_path = output_dir / "stationary-continuation.png"
    checksums_path = output_dir / "SHA256SUMS"
    summary = {
        "schema_version": 2,
        "method": {
            "continuation_parameter": "signed first-cosine amplitude",
            "primary_fit": "chi-chi_star_disc = c2*A^2 + c4*A^4",
            "diagnostic_fit": "chi = intercept + c2*A^2 + c4*A^4",
            "intercept_absolute_tolerance": INTERCEPT_ABSOLUTE_TOLERANCE,
            "minimum_observed_order": MINIMUM_OBSERVED_ORDER,
            "stationary_operator": "Chemotaxis_simulation.simulation.rhs",
            "u_residual_scope": (
                "all nodal u equations, including the dependent endpoint "
                "equation omitted from each minimal-model Newton system"
            ),
            "elliptic_operator": "Chemotaxis_simulation.simulation.solve_v",
            "elliptic_residual_scope": (
                "independent reconstruction of laplacian(v)-mu*v+nu*u^gamma"
            ),
            "discrete_threshold": "Chemotaxis_simulation.thresholds.chi_star_disc_fd",
            "coefficient_source": (
                "Chemotaxis_simulation.implied_constants."
                "compute_bifurcation_coefficients"
            ),
        },
        "artifacts": {
            "branch_points": points_path.name,
            "fit_summary": summary_path.name,
            "profiles": profiles_path.name,
            "states": archive_path.name,
            "states_index": index_path.name,
            "figure_pdf": pdf_path.name,
            "figure_png": png_path.name,
            "checksums": checksums_path.name,
        },
        "meshes": sorted({int(mesh) for mesh in meshes}),
        "amplitudes": sorted({float(value) for value in amplitudes}),
        "provenance": software_provenance(),
        "cases": [
            {
                "parameters": asdict(case),
                "continuum_threshold": asdict(continuum_threshold(case.name)),
                "bifurcation_coefficients": bifurcation_coefficients(case.name),
                "fits": [asdict(fit) for fit in fits if fit.case_name == case.name],
            }
            for case in cases
        ],
        "point_count": len(points),
        "acceptance": {
            "passed": all(gate.passed for gate in gates),
            "gates": [asdict(gate) for gate in gates],
        },
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )

    _write_plot(pdf_path, png_path, cases, points, fits)
    artifacts = (
        points_path,
        summary_path,
        profiles_path,
        archive_path,
        index_path,
        pdf_path,
        png_path,
    )
    _write_sha256sums(checksums_path, artifacts)
    return {
        "summary": summary_path,
        "points": points_path,
        "profiles": profiles_path,
        "states": archive_path,
        "states_index": index_path,
        "pdf": pdf_path,
        "png": png_path,
        "checksums": checksums_path,
    }


def _parse_csv_integers(raw: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "expected a comma-separated list of integers"
        ) from exc
    if not values:
        raise argparse.ArgumentTypeError("list must not be empty")
    return values


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(CASES),
        dest="case_names",
        help="case to run; repeat the option (default: all four)",
    )
    parser.add_argument(
        "--meshes",
        type=_parse_csv_integers,
        default=list(DEFAULT_MESHES),
        help="comma-separated subinterval counts (default: 40,80,160)",
    )
    parser.add_argument(
        "--amplitude",
        action="append",
        type=float,
        dest="amplitudes",
        help="signed prescribed amplitude; repeat (default: eight symmetric amplitudes)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("stationary_branch_validation_output"),
        help="directory for the complete publication artifact bundle",
    )
    parser.add_argument(
        "--solver-xtol",
        type=float,
        default=1e-11,
        help="MINPACK relative iterate tolerance (default: 1e-11)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit nonzero unless all residual, mass, symmetry, and c2 gates pass",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    case_names = args.case_names or list(CASES)
    cases = [CASES[name] for name in case_names]
    amplitudes = args.amplitudes or list(DEFAULT_AMPLITUDES)
    points, fits = run_validation(
        cases,
        meshes=args.meshes,
        amplitudes=amplitudes,
        solver_xtol=args.solver_xtol,
    )
    gates = acceptance_gates(cases, points, fits)
    paths = write_outputs(
        args.output_dir,
        cases,
        args.meshes,
        amplitudes,
        points,
        fits,
        gates,
    )

    for fit in fits:
        order_text = "--" if fit.observed_order is None else f"{fit.observed_order:.3f}"
        print(
            f"{fit.case_name:24s} N={fit.mesh:3d} "
            f"chi*_disc={fit.chi_star_disc:.12g} "
            f"c2={fit.c2:+.12g} theory={fit.theory_c2:+.12g} "
            f"rel.err={fit.c2_relative_error:.3e} "
            f"intercept.err={fit.intercept_error:+.3e} order={order_text}"
        )
    failed = [gate for gate in gates if not gate.passed]
    print(
        f"acceptance: {'PASS' if not failed else 'FAIL'} ({len(gates) - len(failed)}/{len(gates)} gates)"
    )
    for gate in failed:
        print(f"  FAIL {gate.name}: value={gate.value!r}, limit={gate.limit!r}")
    for label, path in paths.items():
        print(f"{label}: {path}")
    return 1 if args.check and failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
