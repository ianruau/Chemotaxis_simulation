#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from npz_io import load_simulation_data_npz
from plots import create_six_frame_summary, create_static_plots
from thresholds import chi_star_disc_fd, chi_star_threshold_continuum_1d


@dataclass(frozen=True)
class PlotConfig:
    npz_file: str
    out_base: str
    overwrite: bool
    summary6: bool
    chi_star_n_max: int
    summary6_beta_n0: bool


def _build_arg_parser() -> argparse.ArgumentParser:
    examples = """Examples:
  # Render <basename>.{png,jpeg} next to the .npz
  chemotaxis-plot images/branch_capture/some_run.npz

  # Put outputs in a specific directory with a new basename
  chemotaxis-plot images/branch_capture/some_run.npz --output_dir images/plots --basename some_run_3d

  # Also render the lightweight 6-slice summary figure (<basename>_summary6.{png,jpeg})
  chemotaxis-plot images/branch_capture/some_run.npz --summary6 yes

  # Include the bifurcation coefficient beta_{n0} in the summary title (super/subcritical)
  chemotaxis-plot images/branch_capture/some_run.npz --summary6 yes --summary6_beta_n0 yes
"""
    parser = argparse.ArgumentParser(
        description=(
            "Generate the main 3D static plots (<basename>.png/.jpeg) from a saved .npz.\n\n"
            "This is intended for batch workflows: run `chemotaxis-sim --save_static_plots no --save_data yes`,\n"
            "then render heavy plots later from the saved data."
        ),
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "npz_file",
        type=str,
        help="Input .npz created by chemotaxis-sim",
    )
    parser.add_argument(
        "--out_base",
        type=str,
        default="",
        help=(
            "Output base path (without extension). "
            "Default: input path with '.npz' stripped."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="",
        help="Output directory (used with --basename; overrides directory of --out_base if provided)",
    )
    parser.add_argument(
        "--basename",
        type=str,
        default="",
        help="Output basename (used with --output_dir)",
    )
    parser.add_argument(
        "--overwrite",
        choices=["yes", "no"],
        default="yes",
        help="Overwrite existing output images (default: yes)",
    )
    parser.add_argument(
        "--summary6",
        choices=["yes", "no"],
        default="no",
        help="Also render <out_base>_summary6.{png,jpeg} (default: no)",
    )
    parser.add_argument(
        "--chi_star_n_max",
        type=int,
        default=5000,
        help="Max n for chi^* scan used by --summary6 (default: 5000)",
    )
    parser.add_argument(
        "--summary6_beta_n0",
        choices=["yes", "no"],
        default="yes",
        help="Include beta_{n0} and classification in the summary6 title (default: yes)",
    )
    return parser


def _parse_args() -> PlotConfig:
    parser = _build_arg_parser()
    try:
        import argcomplete  # type: ignore
    except ModuleNotFoundError:
        argcomplete = None
    if argcomplete is not None:
        argcomplete.autocomplete(parser)
    args = parser.parse_args()

    npz_file = args.npz_file
    if not npz_file.endswith(".npz"):
        raise ValueError(f"Expected a .npz file, got: {npz_file!r}")
    if not os.path.exists(npz_file):
        raise FileNotFoundError(npz_file)

    out_base = (args.out_base or "").strip()
    output_dir = (args.output_dir or "").strip()
    basename = (args.basename or "").strip()

    if output_dir or basename:
        if not output_dir or not basename:
            raise ValueError("Use --output_dir and --basename together (or neither).")
        if os.sep in basename or (os.altsep and os.altsep in basename):
            raise ValueError("`--basename` must not contain path separators; use `--output_dir`.")
        os.makedirs(output_dir, exist_ok=True)
        out_base = os.path.join(output_dir, basename)

    if not out_base:
        out_base = os.path.splitext(npz_file)[0]

    overwrite = args.overwrite == "yes"
    summary6 = args.summary6 == "yes"
    chi_star_n_max = int(args.chi_star_n_max)
    summary6_beta_n0 = args.summary6_beta_n0 == "yes"
    return PlotConfig(
        npz_file=npz_file,
        out_base=out_base,
        overwrite=overwrite,
        summary6=summary6,
        chi_star_n_max=chi_star_n_max,
        summary6_beta_n0=summary6_beta_n0,
    )


def _maybe_remove_existing(out_base: str, *, overwrite: bool) -> None:
    png_path = f"{out_base}.png"
    jpeg_path = f"{out_base}.jpeg"
    if overwrite:
        return
    existing = [p for p in (png_path, jpeg_path) if os.path.exists(p)]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite existing outputs (use --overwrite yes): "
            + ", ".join(existing)
        )


def main() -> None:
    cfg = _parse_args()

    data = load_simulation_data_npz(cfg.npz_file)
    config = data.get("config", {})

    x_values = np.asarray(data["x_values"], dtype=np.float64)
    t_values = np.asarray(data["t_values"], dtype=np.float64)
    u_num = np.asarray(data["u_num"], dtype=np.float64)
    v_num = np.asarray(data["v_num"], dtype=np.float64)
    setup_description = str(data.get("setup_description", ""))

    u_star = float(config.get("uStar", np.nan))
    v_star = float(config.get("vStar", np.nan))
    if not np.isfinite(u_star) or not np.isfinite(v_star):
        raise ValueError("Missing uStar/vStar in npz config metadata; re-save with a newer chemotaxis-sim.")

    _maybe_remove_existing(cfg.out_base, overwrite=cfg.overwrite)

    create_static_plots(
        t_values,
        x_values,
        u_num,
        v_num,
        u_star,
        v_star,
        setup_description,
        cfg.out_base,
    )

    print(f"wrote: {cfg.out_base}.png")
    print(f"wrote: {cfg.out_base}.jpeg")

    if cfg.summary6:
        beta_n0 = None
        mode_n0 = None
        chi_star_disc = None
        meshsize = config.get("meshsize", None)
        chi0 = float(config.get("chi", np.nan))
        try:
            chi_star = chi_star_threshold_continuum_1d(
                u_star=u_star,
                v_star=v_star,
                c=float(config.get("c", 1.0)),
                a=float(config.get("a", 1.0)),
                alpha=float(config.get("alpha", 1.0)),
                mu=float(config.get("mu", 1.0)),
                nu=float(config.get("nu", 1.0)),
                gamma=float(config.get("gamma", 1.0)),
                m=float(config.get("m", 1.0)),
                beta=float(config.get("beta", 1.0)),
                L=float(config.get("L", 1.0)),
                n_max=int(cfg.chi_star_n_max),
            )
        except Exception:
            chi_star = float("nan")

        try:
            if meshsize is not None:
                chi_star_disc, _, _ = chi_star_disc_fd(
                    u_star=u_star,
                    v_star=v_star,
                    c=float(config.get("c", 1.0)),
                    a=float(config.get("a", 1.0)),
                    alpha=float(config.get("alpha", 1.0)),
                    mu=float(config.get("mu", 1.0)),
                    nu=float(config.get("nu", 1.0)),
                    gamma=float(config.get("gamma", 1.0)),
                    m=float(config.get("m", 1.0)),
                    beta=float(config.get("beta", 1.0)),
                    L=float(config.get("L", 1.0)),
                    meshsize=int(meshsize),
                )
        except Exception:
            chi_star_disc = None

        if cfg.summary6_beta_n0:
            mode_n0 = config.get("eigen_mode_n_resolved", None)
            if mode_n0 is None:
                mode_n0 = config.get("eigen_mode_n", None)
            if mode_n0 is None and config.get("eigen_index", None) is not None:
                try:
                    mode_n0 = int(config.get("eigen_index")) - 1
                except Exception:
                    mode_n0 = None
            try:
                from implied_constants import compute_bifurcation_coefficients

                if mode_n0 is not None:
                    coeffs = compute_bifurcation_coefficients(
                        {
                            "a": float(config.get("a", 1.0)),
                            "b": float(config.get("b", 1.0)),
                            "c": float(config.get("c", 1.0)),
                            "alpha": float(config.get("alpha", 1.0)),
                            "beta": float(config.get("beta", 1.0)),
                            "m": float(config.get("m", 1.0)),
                            "mu": float(config.get("mu", 1.0)),
                            "nu": float(config.get("nu", 1.0)),
                            "gamma": float(config.get("gamma", 1.0)),
                            "L": float(config.get("L", 1.0)),
                            "n0": int(mode_n0),
                        }
                    )
                    beta_n0 = float(coeffs.get("beta_n0"))
            except Exception:
                beta_n0 = None

        create_six_frame_summary(
            x_values=x_values,
            t_values=t_values,
            u_num=u_num,
            uStar=u_star,
            chi0=chi0,
            chi_star=float(chi_star),
            file_base_name=cfg.out_base,
            beta_n0=beta_n0,
            mode_n0=mode_n0,
            chi_star_disc=chi_star_disc,
            meshsize=(None if meshsize is None else int(meshsize)),
        )
        print(f"wrote: {cfg.out_base}_summary6.png")
        print(f"wrote: {cfg.out_base}_summary6.jpeg")


if __name__ == "__main__":
    main()
