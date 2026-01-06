#!/usr/bin/env python3
"""
Implied constants / bifurcation diagnostics for Paper II.

This CLI is intended to unify:
- the equilibrium + discrete threshold chi_a^*(u^*) from Paper II Eq. (1.8)/(1.12), and
- the bifurcation coefficients (alpha_{n0}, beta_{n0}, Gamma terms) used to classify the pitchfork.

It is designed for batch workflows where parameters live in a YAML config used by `chemotaxis-sim`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import sympy as sp

from paper2_constants import equilibrium_u_v_star, paper2_eq112_constants


PI = sp.pi


def _to_sympy(value: Any) -> sp.Expr:
    if isinstance(value, sp.Basic):
        return value
    return sp.nsimplify(value, rational=True)


def _laplace_eigenvalue(index: int, length: sp.Expr) -> sp.Expr:
    return (index * PI / length) ** 2


def _chi_star_mode(params: Dict[str, sp.Expr], lambda_n0: sp.Expr) -> sp.Expr:
    u_star = params["u_star"]
    v_star = params["v_star"]
    c = params["c"]
    numerator = (c + v_star) ** params["beta"]
    denominator = (
        params["nu"]
        * params["gamma"]
        * u_star ** (params["m"] + params["gamma"] - 1)
    )
    factor = (lambda_n0 + params["a"] * params["alpha"])
    factor *= (params["mu"] + lambda_n0) / lambda_n0
    return numerator / denominator * factor


def _sigma(lambda_n: sp.Expr, chi_value: sp.Expr, params: Dict[str, sp.Expr]) -> sp.Expr:
    u_star = params["u_star"]
    v_star = params["v_star"]
    c = params["c"]
    prefactor = chi_value
    prefactor *= params["nu"] * params["gamma"] * u_star ** (params["m"] + params["gamma"] - 1)
    prefactor /= (c + v_star) ** params["beta"]
    prefactor *= lambda_n / (params["mu"] + lambda_n)
    return -lambda_n + prefactor - params["a"] * params["alpha"]


def _C1(params: Dict[str, sp.Expr], lambda_k: sp.Expr) -> sp.Expr:
    return (
        params["nu"]
        * params["gamma"]
        * params["u_star"] ** (params["gamma"] - 1)
        / (params["mu"] + lambda_k)
    )


def _C2(params: Dict[str, sp.Expr], lambda_k: sp.Expr) -> sp.Expr:
    return (
        params["nu"]
        * params["gamma"]
        * (params["gamma"] - 1)
        * params["u_star"] ** (params["gamma"] - 2)
        / (4 * (params["mu"] + lambda_k))
    )


def _C3(params: Dict[str, sp.Expr], lambda_k: sp.Expr) -> sp.Expr:
    return (
        params["nu"]
        * params["gamma"]
        * (params["gamma"] - 1)
        * (params["gamma"] - 2)
        * params["u_star"] ** (params["gamma"] - 3)
        / (24 * (params["mu"] + lambda_k))
    )


def _gamma_n0_cubic(params: Dict[str, sp.Expr], lambda_n0: sp.Expr) -> sp.Expr:
    u_star = params["u_star"]
    v_star = params["v_star"]
    c = params["c"]
    L = params["L"]
    n0 = params["n0"]

    lambda_0 = _laplace_eigenvalue(0, L)
    lambda_2n0 = _laplace_eigenvalue(2 * n0, L)

    C1_n0 = _C1(params, lambda_n0)
    C2_0 = _C2(params, lambda_0)
    C2_2n0 = _C2(params, lambda_2n0)
    C3_n0 = _C3(params, lambda_n0)

    pref1 = (PI ** 2 * n0 ** 2) / (2 * L ** 2)
    pref2 = (PI ** 2 * n0 ** 2) / (4 * L ** 2)

    leading = (
        2 * u_star ** params["m"] / (c + v_star) ** params["beta"] * (3 * C3_n0)
        + 2
        * params["m"]
        * u_star ** (params["m"] - 1)
        / (c + v_star) ** params["beta"]
        * C2_2n0
        - params["beta"]
        * u_star ** params["m"]
        / (c + v_star) ** (params["beta"] + 1)
        * C1_n0
        * (2 * C2_0 + C2_2n0)
    )

    tail = (
        params["m"]
        * (params["m"] - 1)
        * u_star ** (params["m"] - 2)
        / (c + v_star) ** params["beta"]
        * C1_n0
        - 2
        * params["m"]
        * params["beta"]
        * u_star ** (params["m"] - 1)
        / (c + v_star) ** (params["beta"] + 1)
        * C1_n0 ** 2
        + params["beta"]
        * (params["beta"] + 1)
        * u_star ** params["m"]
        / (c + v_star) ** (params["beta"] + 2)
        * C1_n0 ** 3
    )

    return pref1 * leading + pref2 * tail


def _gamma_2n0(params: Dict[str, sp.Expr], chi_star: sp.Expr) -> sp.Expr:
    x = sp.symbols("x", real=True)
    amp = sp.symbols("A", real=True)
    chi0 = sp.symbols("chi0", real=True)

    n0 = params["n0"]
    L = params["L"]
    u_star = params["u_star"]
    v_star = params["v_star"]
    c = params["c"]

    kappa = n0 * PI / L
    u_sum = amp * sp.cos(n0 * PI * x / L)
    v_sum = params["C_n0"] * amp * sp.cos(n0 * PI * x / L)
    grad_sum = kappa * params["C_n0"] * amp * sp.sin(n0 * PI * x / L)

    B0 = u_star ** params["m"] / (c + v_star) ** params["beta"]
    B1 = params["m"] * u_star ** (params["m"] - 1) / (c + v_star) ** params["beta"]
    B2 = (
        -params["beta"]
        * u_star ** params["m"]
        / (c + v_star) ** (params["beta"] + 1)
    )
    coef2 = (
        params["m"]
        * (params["m"] - 1)
        * u_star ** (params["m"] - 2)
        / (2 * (c + v_star) ** params["beta"])
    )
    coef3 = (
        params["m"]
        * u_star ** (params["m"] - 1)
        * params["beta"]
        / ((c + v_star) ** (params["beta"] + 1))
    )
    coef4 = (
        params["beta"]
        * (params["beta"] + 1)
        * u_star ** params["m"]
        / (2 * (c + v_star) ** (params["beta"] + 2))
    )

    I1 = chi0 * sp.diff((B0 + B1 * u_sum + B2 * v_sum) * grad_sum, x)
    I2 = chi0 * sp.diff(coef2 * u_sum ** 2 * grad_sum, x)
    I3 = -chi0 * sp.diff(coef3 * u_sum * v_sum * grad_sum, x)
    I4 = chi0 * sp.diff(coef4 * v_sum ** 2 * grad_sum, x)

    integrand = I1 + I2 + I3 + I4
    projection = sp.integrate(
        integrand * sp.cos(2 * n0 * PI * x / L),
        (x, 0, L),
    )
    G_expr = (2 / (L * chi0)) * projection
    coeff = sp.expand(G_expr).coeff(amp, 2)
    return sp.simplify(2 * coeff.subs(chi0, chi_star))


def _prepare_params(raw: Dict[str, Any]) -> Dict[str, sp.Expr]:
    params: Dict[str, sp.Expr] = {}
    for key, value in raw.items():
        params[key] = _to_sympy(value)
    params.setdefault("alpha", _to_sympy(1))
    params.setdefault("beta", _to_sympy(1))
    params.setdefault("m", _to_sympy(1))
    params.setdefault("gamma", _to_sympy(1))
    params.setdefault("nu", _to_sympy(1))
    params.setdefault("mu", _to_sympy(1))
    params.setdefault("L", _to_sympy(1))
    params.setdefault("n0", _to_sympy(1))
    params.setdefault("c", _to_sympy(1))

    params["u_star"] = (params["a"] / params["b"]) ** (1 / params["alpha"])
    params["v_star"] = (params["nu"] / params["mu"]) * params["u_star"] ** params["gamma"]
    params["n0"] = int(params["n0"])
    return params


def _classify(beta_value: float, tol: float = 1e-9) -> str:
    if abs(beta_value) <= tol:
        return "degenerate"
    return "supercritical" if beta_value > 0 else "subcritical"


def compute_bifurcation_coefficients(raw_params: Dict[str, Any]) -> Dict[str, float]:
    params = _prepare_params(raw_params)
    n0 = params["n0"]
    lambda_n0 = _laplace_eigenvalue(n0, params["L"])
    lambda_2n0 = _laplace_eigenvalue(2 * n0, params["L"])
    lambda_0 = _laplace_eigenvalue(0, params["L"])

    C1_n0 = _C1(params, lambda_n0)
    C2_0 = _C2(params, lambda_0)
    C2_2n0 = _C2(params, lambda_2n0)
    C3_n0 = _C3(params, lambda_n0)
    params["C_n0"] = C1_n0

    chi_star = _chi_star_mode(params, lambda_n0)
    alpha_n0 = (
        params["nu"]
        * params["gamma"]
        * params["u_star"] ** (params["m"] + params["gamma"] - 1)
        / (params["c"] + params["v_star"]) ** params["beta"]
        * lambda_n0
        / (params["mu"] + lambda_n0)
    )

    gamma_n0_cubic = _gamma_n0_cubic(params, lambda_n0)
    gamma_2n0 = _gamma_2n0(params, chi_star)

    a01 = (
        (1 + params["alpha"])
        * params["alpha"]
        * params["b"]
        * params["u_star"] ** (params["alpha"] - 1)
        / (
            4
            * (params["a"] - (1 + params["alpha"]) * params["b"] * params["u_star"] ** params["alpha"])
        )
    )

    sigma_2n0 = _sigma(lambda_2n0, chi_star, params)
    a2n0 = (
        (1 + params["alpha"]) * params["alpha"] * params["b"] * params["u_star"] ** (params["alpha"] - 1) / 4
        - chi_star * gamma_2n0
    ) / sigma_2n0

    beta_n0 = (
        (1 + params["alpha"]) * params["alpha"] * params["b"] * params["u_star"] ** (params["alpha"] - 1) / 4
        * (4 * a01 + 2 * a2n0)
        + chi_star * gamma_n0_cubic
        + (1 + params["alpha"]) * params["alpha"] * (params["alpha"] - 1) * params["b"] * params["u_star"] ** (params["alpha"] - 2) / 8
    )

    beta_float = float(sp.N(beta_n0))
    return {
        "chi_star_mode_n0": float(sp.N(chi_star)),
        "lambda_n0": float(sp.N(lambda_n0)),
        "lambda_2n0": float(sp.N(lambda_2n0)),
        "alpha_n0": float(sp.N(alpha_n0)),
        "beta_n0": beta_float,
        "gamma_cubic": float(sp.N(gamma_n0_cubic)),
        "gamma_2n0": float(sp.N(gamma_2n0)),
        "sigma_2n0": float(sp.N(sigma_2n0)),
        "a01": float(sp.N(a01)),
        "a2n0": float(sp.N(a2n0)),
        "C1_n0": float(sp.N(C1_n0)),
        "C2_0": float(sp.N(C2_0)),
        "C2_2n0": float(sp.N(C2_2n0)),
        "C3_n0": float(sp.N(C3_n0)),
    }


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


@dataclass(frozen=True)
class Inputs:
    a: float
    b: float
    c: float
    alpha: float
    beta: float
    m: float
    mu: float
    nu: float
    gamma: float
    L: float
    meshsize: Optional[float]
    meshsize_abs: Optional[int]
    n0: Optional[int]
    n_max: int
    early_stop_patience: int


def _build_parser() -> argparse.ArgumentParser:
    examples = """Examples:
  # Full report from a YAML config
  chemotaxis-constants --config config.example.yaml report

  # Equivalent form (supported): --config after the subcommand
  chemotaxis-constants report --config config.example.yaml

  # Global threshold + equilibrium only
  chemotaxis-constants --config config.example.yaml threshold

  # Bifurcation coefficients at the default mode n0 (argmin of chi_a^* scan)
  chemotaxis-constants --config config.example.yaml bifurcation

  # Override n0 explicitly (e.g., compare modes)
  chemotaxis-constants --config config.example.yaml bifurcation --n0 1

  # JSON output (useful for scripts)
  chemotaxis-constants --config config.example.yaml report --format json

	  # Also compute the mesh-dependent threshold chi^{*,disc} using N=ceil(meshsize*L) subintervals
	  chemotaxis-constants --config config.example.yaml report --meshsize 50
	  # Or specify N directly
	  chemotaxis-constants --config config.example.yaml report --meshsize_abs 50
"""
    common = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    common.add_argument(
        "--config",
        type=str,
        default="",
        help="Load defaults from YAML (CLI overrides; accepted before/after the subcommand)",
    )
    common.add_argument(
        "--format",
        choices=["text", "json"],
        default="text",
        help="Output format (default: text; accepted before/after the subcommand)",
    )
    common.add_argument("--a", type=float, default=1.0)
    common.add_argument("--b", type=float, default=1.0)
    common.add_argument("--c", type=float, default=1.0)
    common.add_argument("--alpha", type=float, default=1.0)
    common.add_argument("--beta", type=float, default=1.0)
    common.add_argument("--m", type=float, default=1.0)
    common.add_argument("--mu", type=float, default=1.0)
    common.add_argument("--nu", type=float, default=1.0)
    common.add_argument("--gamma", type=float, default=1.0)
    common.add_argument("--L", type=float, default=1.0)
    common.add_argument(
        "--meshsize",
        type=float,
        default=None,
        help=(
            "If set, also compute mesh-dependent chi^{*,disc}. "
            "Interpreted as mesh density per unit length (effective N=ceil(meshsize*L))."
        ),
    )
    common.add_argument(
        "--meshsize_abs",
        type=int,
        default=None,
        help="Absolute number of subintervals N (overrides --meshsize if set).",
    )
    common.add_argument(
        "--n0",
        type=int,
        default=None,
        help="Mode index for bifurcation coefficients (default: argmin mode from the chi_a^* scan)",
    )
    common.add_argument(
        "--n_max",
        type=int,
        default=200000,
        help="Max n for discrete chi* scan (default: 200000)",
    )
    common.add_argument("--early_stop_patience", type=int, default=2000)

    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[common],
        allow_abbrev=False,
    )

    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser(
        "threshold",
        parents=[common],
        help="Compute Eq. (1.8) and global discrete chi_a^*(u^*)",
    )
    sub.add_parser(
        "bifurcation",
        parents=[common],
        help="Compute bifurcation coefficients at mode n0",
    )
    sub.add_parser(
        "report",
        parents=[common],
        help="Compute both threshold and bifurcation, plus consistency checks",
    )
    return parser


def _apply_config_defaults(parser: argparse.ArgumentParser, cfg: dict[str, Any]) -> None:
    if not cfg:
        return
    by_dest = {a.dest: a for a in parser._actions if getattr(a, "dest", None)}  # pylint: disable=protected-access
    defaults: dict[str, Any] = {}
    for key, value in cfg.items():
        action = by_dest.get(key)
        if action is None:
            continue
        if value is None:
            continue
        if action.type is not None:
            try:
                defaults[key] = action.type(value)  # pylint: disable=not-callable
            except (TypeError, ValueError):
                defaults[key] = action.type(str(value))  # pylint: disable=not-callable
        else:
            defaults[key] = value
    parser.set_defaults(**defaults)


def _parse_inputs(argv: list[str]) -> tuple[argparse.Namespace, Inputs]:
    pre = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    pre.add_argument("--config", type=str, default="")
    pre_args, _ = pre.parse_known_args(argv)

    parser = _build_parser()
    if pre_args.config:
        _apply_config_defaults(parser, _load_yaml_config_as_overrides(pre_args.config))

    try:
        import argcomplete  # type: ignore
    except ModuleNotFoundError:
        argcomplete = None
    if argcomplete is not None:
        argcomplete.autocomplete(parser)

    args = parser.parse_args(argv)
    inputs = Inputs(
        a=float(args.a),
        b=float(args.b),
        c=float(args.c),
        alpha=float(args.alpha),
        beta=float(args.beta),
        m=float(args.m),
        mu=float(args.mu),
        nu=float(args.nu),
        gamma=float(args.gamma),
        L=float(args.L),
        meshsize=(None if args.meshsize is None else float(args.meshsize)),
        meshsize_abs=(None if args.meshsize_abs is None else int(args.meshsize_abs)),
        n0=(None if args.n0 is None else int(args.n0)),
        n_max=int(args.n_max),
        early_stop_patience=int(args.early_stop_patience),
    )
    return args, inputs


def _as_jsonable(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            out[k] = v.item()
        else:
            out[k] = v
    return out


def main() -> None:
    args, inp = _parse_inputs(sys.argv[1:])

    u_star, v_star = equilibrium_u_v_star(
        a=inp.a, b=inp.b, alpha=inp.alpha, mu=inp.mu, nu=inp.nu, gamma=inp.gamma
    )

    effective_n0: Optional[int] = inp.n0
    threshold_constants = None
    if args.cmd in ("threshold", "report") or (args.cmd == "bifurcation" and effective_n0 is None):
        threshold_constants = paper2_eq112_constants(
            a=inp.a,
            b=inp.b,
            c=inp.c,
            alpha=inp.alpha,
            mu=inp.mu,
            nu=inp.nu,
            gamma=inp.gamma,
            m=inp.m,
            beta=inp.beta,
            L=inp.L,
            n_max=inp.n_max,
            early_stop_patience=inp.early_stop_patience,
            meshsize=inp.meshsize,
            meshsize_abs=inp.meshsize_abs,
        )
        if effective_n0 is None:
            effective_n0 = int(threshold_constants.n_min)

    if args.cmd in ("bifurcation", "report") and effective_n0 is None:
        raise RuntimeError("Internal error: n0 was not resolved.")

    report: dict[str, Any] = {
        "inputs": {
            "a": inp.a,
            "b": inp.b,
            "c": inp.c,
            "alpha": inp.alpha,
            "beta": inp.beta,
            "m": inp.m,
            "mu": inp.mu,
            "nu": inp.nu,
            "gamma": inp.gamma,
            "L": inp.L,
            "meshsize": inp.meshsize,
            "meshsize_abs": inp.meshsize_abs,
            "n0": effective_n0,
            "n_max": inp.n_max,
            "early_stop_patience": inp.early_stop_patience,
        },
        "equilibrium": {"u_star": u_star, "v_star": v_star},
    }

    if args.cmd in ("threshold", "report"):
        if threshold_constants is None:
            raise RuntimeError("Internal error: threshold constants missing.")
        cst = threshold_constants
        report["threshold"] = {
            "chi_a_star": cst.chi_a_star,
            "n_min": int(cst.n_min),
            "lambda_min": float(cst.lambda_min),
            "chi_star_disc": cst.chi_star_disc,
            "n_min_disc": cst.n_min_disc,
            "lambda_min_disc": cst.lambda_min_disc,
            "meshsize": cst.meshsize,
            "mesh_per_unit": cst.mesh_per_unit,
        }

    if args.cmd in ("bifurcation", "report"):
        coeffs = compute_bifurcation_coefficients(
            {
                "a": inp.a,
                "b": inp.b,
                "c": inp.c,
                "alpha": inp.alpha,
                "beta": inp.beta,
                "m": inp.m,
                "mu": inp.mu,
                "nu": inp.nu,
                "gamma": inp.gamma,
                "L": inp.L,
                "n0": int(effective_n0),
            }
        )
        beta_n0 = float(coeffs["beta_n0"])
        coeffs_out: dict[str, Any] = dict(coeffs)
        coeffs_out["classification"] = _classify(beta_n0)
        report["bifurcation"] = coeffs_out

    if args.cmd == "report":
        chi_global = float(report["threshold"]["chi_a_star"])
        chi_n0 = float(report["bifurcation"]["chi_star_mode_n0"])
        n_min = int(report["threshold"]["n_min"])
        report["consistency"] = {
            "n0_matches_argmin": bool(int(effective_n0) == n_min),
            "chi_a_star_minus_chi_star_n0": chi_global - chi_n0,
        }

    if args.format == "json":
        print(json.dumps(_as_jsonable(report), indent=2, sort_keys=True))
        return

    print("Inputs:")
    for k, v in report["inputs"].items():
        print(f"  {k}: {v}")
    print()
    print("Equilibrium:")
    print(f"  u*: {report['equilibrium']['u_star']}")
    print(f"  v*: {report['equilibrium']['v_star']}")
    print()

    if "threshold" in report:
        thr = report["threshold"]
        print("Threshold (Eq. 1.12 discrete scan):")
        print(f"  chi_a^*(u*): {thr['chi_a_star']}")
        print(f"  argmin n:    {thr['n_min']}")
        print(f"  lambda_min:  {thr['lambda_min']}")
        print()
        if thr.get("chi_star_disc") is not None:
            print("Threshold (finite-difference mesh):")
            print(f"  meshsize N:      {thr['meshsize']}")
            if thr.get("mesh_per_unit") is not None:
                print(f"  mesh_per_unit:   {thr['mesh_per_unit']}")
            print(f"  chi^{{*,disc}}:  {thr['chi_star_disc']}")
            print(f"  argmin n:      {thr['n_min_disc']}")
            print(f"  lambda_min:    {thr['lambda_min_disc']}")
            print()

    if "bifurcation" in report:
        bf = report["bifurcation"]
        print("Bifurcation (mode n0):")
        print(f"  lambda_n0:         {bf['lambda_n0']}")
        print(f"  chi*(n0):          {bf['chi_star_mode_n0']}")
        print(f"  alpha_n0:          {bf['alpha_n0']}")
        print(f"  beta_n0:           {bf['beta_n0']}")
        print(f"  classification:    {bf['classification']}")
        print(f"  Gamma_cubic:       {bf['gamma_cubic']}")
        print(f"  Gamma_2n0:         {bf['gamma_2n0']}")
        print()

    if "consistency" in report:
        cs = report["consistency"]
        print("Consistency:")
        print(f"  n0 matches argmin: {cs['n0_matches_argmin']}")
        print(f"  chi_a^* - chi*(n0): {cs['chi_a_star_minus_chi_star_n0']}")


if __name__ == "__main__":
    main()
