#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np

from simulation import create_static_plots, load_simulation_data_npz


@dataclass(frozen=True)
class PlotConfig:
    npz_file: str
    out_base: str
    overwrite: bool


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Generate the main 3D static plots (<basename>.png/.jpeg) from a saved .npz.\n\n"
            "This is intended for batch workflows: run `chemotaxis-sim --save_static_plots no --save_data yes`,\n"
            "then render heavy plots later from the saved data."
        )
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
    return PlotConfig(npz_file=npz_file, out_base=out_base, overwrite=overwrite)


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


if __name__ == "__main__":
    main()
