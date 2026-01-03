#!/usr/bin/env python3
"""
Paper II helper utilities (11-25-2025-CRS-2).

This module provides small numeric helpers to compute:
- the equilibrium (u^*, v^*) from Eq. (1.8) [label: E:equilibrium]
- the discrete stability threshold chi_a^*(u^*) from Eq. (1.12) [label: E:chi-star]

The definition of chi_a^*(u^*) involves the Neumann Laplace eigenvalues
{lambda_n}_{n>=0} of -Delta on the spatial domain. For the 1D Neumann interval
(0, L), these eigenvalues are:

  lambda_n = (n*pi/L)^2,  n = 0,1,2,...

We evaluate the infimum over n>=1 by scanning n and taking the minimum.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import sys
from typing import Any, Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class Paper2Constants:
    u_star: float
    v_star: float
    chi_a_star: float
    n_min: int
    lambda_min: float


def equilibrium_u_v_star(*, a: float, b: float, alpha: float, mu: float, nu: float, gamma: float) -> Tuple[float, float]:
    """
    Return (u^*, v^*) from Paper II Eq. (1.8) [E:equilibrium] for a,b>0.
    """
    if a <= 0 or b <= 0:
        raise ValueError("Eq. (1.8) requires a>0 and b>0.")
    if alpha == 0:
        raise ValueError("alpha must be nonzero.")
    if mu == 0:
        raise ValueError("mu must be nonzero.")
    u_star = (a / b) ** (1.0 / alpha)
    v_star = (nu / mu) * (u_star**gamma)
    return float(u_star), float(v_star)


def neumann_eigenvalue_1d(*, n: int, L: float = 1.0) -> float:
    """
    Neumann Laplace eigenvalues for the 1D interval (0, L):
      lambda_n = (n*pi/L)^2.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    if L <= 0:
        raise ValueError("L must be > 0")
    return float((n * np.pi / L) ** 2)


def chi_a_star_eq112_discrete(
    *,
    u_star: float,
    v_star: float,
    c: float = 1.0,
    a: float,
    alpha: float,
    mu: float,
    nu: float,
    gamma: float,
    m: float,
    beta: float,
    L: float = 1.0,
    n_max: int = 200000,
    early_stop_patience: int = 2000,
) -> Tuple[float, int, float]:
    """
    Compute chi_a^*(u^*) from Paper II Eq. (1.12) [E:chi-star] on (0,L) (1D Neumann).

    Returns (chi_a_star, n_min, lambda_n_min).
    """
    if u_star <= 0:
        raise ValueError("u_star must be > 0")
    if nu == 0 or gamma == 0:
        raise ValueError("nu and gamma must be nonzero")
    if L <= 0:
        raise ValueError("L must be > 0")
    if n_max < 1:
        raise ValueError("n_max must be >= 1")

    if c + v_star <= 0:
        raise ValueError("Eq. (1.12) requires c + v_star > 0.")

    prefactor = ((c + v_star) ** beta) / (nu * gamma * (u_star ** (m + gamma - 1.0)))

    best_val = float("inf")
    best_n = 1
    best_lambda = neumann_eigenvalue_1d(n=1, L=L)

    increasing_streak = 0
    last_val: Optional[float] = None

    for n in range(1, n_max + 1):
        lam = neumann_eigenvalue_1d(n=n, L=L)
        val = prefactor * (((lam + a * alpha) * (mu + lam)) / lam)

        if val < best_val:
            best_val = float(val)
            best_n = n
            best_lambda = float(lam)
            increasing_streak = 0
        else:
            increasing_streak += 1

        if last_val is not None and val < last_val:
            increasing_streak = 0
        last_val = float(val)

        if early_stop_patience > 0 and n > best_n and increasing_streak >= early_stop_patience:
            break

    return best_val, best_n, best_lambda


def paper2_eq112_constants(
    *,
    a: float,
    b: float,
    c: float = 1.0,
    alpha: float,
    mu: float,
    nu: float,
    gamma: float,
    m: float,
    beta: float,
    L: float = 1.0,
    n_max: int = 200000,
    early_stop_patience: int = 2000,
) -> Paper2Constants:
    """
    Convenience wrapper: compute (u^*, v^*) [Eq. (1.8)] and chi_a^*(u^*) [Eq. (1.12)].
    """
    u_star, v_star = equilibrium_u_v_star(a=a, b=b, alpha=alpha, mu=mu, nu=nu, gamma=gamma)
    chi_a_star, n_min, lambda_min = chi_a_star_eq112_discrete(
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
        L=L,
        n_max=n_max,
        early_stop_patience=early_stop_patience,
    )
    return Paper2Constants(
        u_star=u_star,
        v_star=v_star,
        chi_a_star=chi_a_star,
        n_min=n_min,
        lambda_min=lambda_min,
    )


def _flatten_yaml_mapping(data: Any) -> dict[str, Any]:
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError("YAML config must be a mapping (dict-like) at the top level.")
    out: dict[str, Any] = {}
    for key, value in data.items():
        if not isinstance(key, str):
            raise TypeError("YAML config keys must be strings.")
        if isinstance(value, dict):
            out.update(_flatten_yaml_mapping(value))
        else:
            out[key] = value
    return out


def _load_yaml_config_as_overrides(path: str) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "Missing dependency: PyYAML is required for `--config`. "
            "Install it with `pip install pyyaml`."
        ) from exc

    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return _flatten_yaml_mapping(raw)


def _apply_parser_defaults_from_config(parser, config: dict[str, Any]) -> None:
    _apply_parser_defaults_from_config_impl(parser, config, warn_unknown=False)


def _apply_parser_defaults_from_config_impl(
    parser, config: dict[str, Any], *, warn_unknown: bool = False
) -> None:
    if not config:
        return

    by_dest = {action.dest: action for action in parser._actions if action.dest}  # pylint: disable=protected-access

    if warn_unknown:
        unknown_keys = [k for k in config.keys() if k not in by_dest]
        for k in unknown_keys:
            print(f"Warning: unknown config key ignored: {k}", file=sys.stderr)

    coerced: dict[str, Any] = {}
    for key, value in config.items():
        action = by_dest.get(key)
        if action is None:
            continue
        if value is None:
            coerced[key] = None
            continue
        if action.type is not None:
            try:
                coerced[key] = action.type(value)  # pylint: disable=not-callable
            except (TypeError, ValueError):
                coerced[key] = action.type(str(value))  # pylint: disable=not-callable
        else:
            coerced[key] = value

        if action.choices and coerced[key] not in action.choices:
            raise ValueError(
                f"Invalid value for `{key}`: {coerced[key]!r} (choices: {sorted(action.choices)})"
            )

    parser.set_defaults(**coerced)


def main() -> None:
    import argparse

    argv = sys.argv[1:]
    if not argv:
        parser = argparse.ArgumentParser(
            description="Compute Paper II Eq. (1.8) and Eq. (1.12) constants."
        )
        parser.add_argument(
            "--config",
            type=str,
            default="",
            help="Load default parameter values from a YAML file (CLI flags override)",
        )
        parser.add_argument("--a", type=float, default=1.0, help="Parameter a (default: 1.0)")
        parser.add_argument("--b", type=float, default=1.0, help="Parameter b (default: 1.0)")
        parser.add_argument("--alpha", type=float, default=1.0, help="Parameter alpha (default: 1.0)")
        parser.add_argument("--mu", type=float, default=1.0, help="Parameter mu (default: 1.0)")
        parser.add_argument("--nu", type=float, default=1.0, help="Parameter nu (default: 1.0)")
        parser.add_argument("--gamma", type=float, default=1.0, help="Parameter gamma (default: 1.0)")
        parser.add_argument("--m", type=float, default=1.0, help="Parameter m (default: 1.0)")
        parser.add_argument("--beta", type=float, default=1.0, help="Parameter beta (default: 1.0)")
        parser.add_argument("--L", type=float, default=1.0, help="Domain length L (default: 1.0)")
        parser.add_argument("--n_max", type=int, default=200000)
        parser.add_argument("--early_stop_patience", type=int, default=2000)
        parser.add_argument(
            "--config_warn_unknown",
            choices=["yes", "no"],
            default="no",
            help="Warn about unknown YAML keys (default: no)",
        )
        parser.print_help(sys.stderr)
        return

    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Load default parameter values from a YAML file (CLI flags override)",
    )
    pre_parser.add_argument(
        "--config_warn_unknown",
        choices=["yes", "no"],
        default="no",
        help="Warn about unknown YAML keys (default: no)",
    )
    pre_args, _ = pre_parser.parse_known_args(argv)
    config_overrides: dict[str, Any] = {}
    if pre_args.config:
        config_overrides = _load_yaml_config_as_overrides(pre_args.config)

    parser = argparse.ArgumentParser(
        description="Compute Paper II Eq. (1.8) and Eq. (1.12) constants."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="",
        help="Load default parameter values from a YAML file (CLI flags override)",
    )
    parser.add_argument(
        "--config_warn_unknown",
        choices=["yes", "no"],
        default="no",
        help="Warn about unknown YAML keys (default: no)",
    )
    parser.add_argument("--a", type=float, default=1.0, help="Parameter a (default: 1.0)")
    parser.add_argument("--b", type=float, default=1.0, help="Parameter b (default: 1.0)")
    parser.add_argument("--c", type=float, default=1.0, help="Sensitivity shift c (default: 1.0)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Parameter alpha (default: 1.0)")
    parser.add_argument("--mu", type=float, default=1.0, help="Parameter mu (default: 1.0)")
    parser.add_argument("--nu", type=float, default=1.0, help="Parameter nu (default: 1.0)")
    parser.add_argument("--gamma", type=float, default=1.0, help="Parameter gamma (default: 1.0)")
    parser.add_argument("--m", type=float, default=1.0, help="Parameter m (default: 1.0)")
    parser.add_argument("--beta", type=float, default=1.0, help="Parameter beta (default: 1.0)")
    parser.add_argument("--L", type=float, default=1.0, help="Domain length L (default: 1.0)")
    parser.add_argument("--n_max", type=int, default=200000)
    parser.add_argument("--early_stop_patience", type=int, default=2000)
    _apply_parser_defaults_from_config_impl(
        parser, config_overrides, warn_unknown=(pre_args.config_warn_unknown == "yes")
    )

    args = parser.parse_args(argv)

    c = paper2_eq112_constants(
        a=args.a,
        b=args.b,
        c=args.c,
        alpha=args.alpha,
        mu=args.mu,
        nu=args.nu,
        gamma=args.gamma,
        m=args.m,
        beta=args.beta,
        L=args.L,
        n_max=args.n_max,
        early_stop_patience=args.early_stop_patience,
    )

    print(f"u* = {c.u_star}")
    print(f"v* = {c.v_star}")
    print(f"chi_a^*(u*) [Eq. (1.12)] = {c.chi_a_star}")
    print(f"argmin n = {c.n_min} (lambda_n = {c.lambda_min})")


if __name__ == "__main__":
    main()
