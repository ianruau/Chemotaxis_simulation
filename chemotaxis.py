#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import sys
from contextlib import contextmanager
from typing import Iterable, Optional


def _distribution_version() -> str:
    try:
        from importlib.metadata import PackageNotFoundError, version  # type: ignore
    except ImportError:  # pragma: no cover
        return ""
    try:
        return str(version("chemotaxis-sim"))
    except PackageNotFoundError:  # pragma: no cover
        return ""


def _build_parser() -> argparse.ArgumentParser:
    version_text = _distribution_version()
    version_line = f" (v{version_text})" if version_text else ""
    examples = """Examples:
  # Overview
  chemotaxis --help

  # Run a simulation (same as chemotaxis-sim)
  chemotaxis sim --config config.example.yaml --chi 2.19 --meshsize 50 --time 100 --eigen_mode_n 1 --epsilon 0.001

  # Compute Paper II thresholds / bifurcation coefficients (same as chemotaxis-constants)
  chemotaxis constants --config config.example.yaml report
  chemotaxis constants --config config.example.yaml report --meshsize 50
  chemotaxis constants --config config.example.yaml report --meshsize_abs 50

  # Render static plots from a saved .npz (same as chemotaxis-plot)
  chemotaxis plot images/branch_capture/some_run.npz
"""
    parser = argparse.ArgumentParser(
        prog="chemotaxis",
        description=(
            "Chemotaxis_simulation toolbox" + version_line + ".\n\n"
            "This command provides a quick overview and a single entry point to:\n"
            "- run simulations,\n"
            "- compute Paper II thresholds / implied constants,\n"
            "- render plots from saved .npz outputs.\n\n"
            "All legacy commands remain available:\n"
            "  chemotaxis-sim, chemotaxis-constants, chemotaxis-plot"
        ),
        epilog=examples,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        allow_abbrev=False,
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Print installed package version and exit",
    )
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser(
        "sim",
        add_help=False,
        help="Run simulations (alias for `chemotaxis-sim`)",
    ).add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser(
        "constants",
        add_help=False,
        help="Compute thresholds / bifurcation coefficients (alias for `chemotaxis-constants`)",
    ).add_argument("args", nargs=argparse.REMAINDER)
    sub.add_parser(
        "plot",
        add_help=False,
        help="Render plots from a saved .npz (alias for `chemotaxis-plot`)",
    ).add_argument("args", nargs=argparse.REMAINDER)
    return parser


@contextmanager
def _patched_argv(program: str, argv: Iterable[str]):
    old_argv = sys.argv
    sys.argv = [program, *list(argv)]
    try:
        yield
    finally:
        sys.argv = old_argv


def _delegate(module_name: str, program: str, argv: list[str]) -> None:
    module = importlib.import_module(module_name)
    main_func = getattr(module, "main", None)
    if not callable(main_func):
        raise RuntimeError(f"Expected {module_name}.main() to exist")
    with _patched_argv(program, argv):
        main_func()


def main(argv: Optional[list[str]] = None) -> None:
    parser = _build_parser()
    try:
        import argcomplete  # type: ignore
    except ModuleNotFoundError:
        argcomplete = None
    if argcomplete is not None:
        argcomplete.autocomplete(parser)

    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    if args.version:
        v = _distribution_version() or "unknown"
        print(f"chemotaxis-sim {v}")
        return

    cmd = getattr(args, "cmd", None)
    if not cmd:
        parser.print_help(sys.stderr)
        return

    forwarded = list(getattr(args, "args", []) or [])
    if cmd in ("constants", "plot") and not forwarded and os.environ.get("_ARGCOMPLETE") != "1":
        forwarded = ["--help"]

    if cmd == "sim":
        _delegate("simulation", "chemotaxis-sim", forwarded)
        return
    if cmd == "constants":
        _delegate("implied_constants", "chemotaxis-constants", forwarded)
        return
    if cmd == "plot":
        _delegate("plot_from_npz", "chemotaxis-plot", forwarded)
        return

    raise RuntimeError(f"Unhandled command: {cmd}")


if __name__ == "__main__":
    main()
