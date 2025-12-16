# Repository Guidelines

## Project Structure

- `simulation.py`: main module and CLI entry point (`chemotaxis-sim=simulation:main`).
- `docs/`: Sphinx documentation sources (`conf.py`, `index.rst`).
- `homepage/`: images used in the README/docs site.
- `Reproducing_Previous_Results/`: notes/scripts for reproducing paper results.
- `Mathematica/`, `documents/`: ancillary research material (not required to run the CLI).
- Generated outputs (`*.png`, `*.jpeg`, `*.mp4`, `*.joblib`) are intentionally ignored via `.gitignore`.

## Build, Test, and Development Commands

- Create an environment: `python -m venv venv && source venv/bin/activate`
- Install for development (includes runtime deps from `setup.py`): `pip install -e .`
- Install dev tooling (lint-related): `pip install -r requirements.txt`
- Run the CLI: `chemotaxis-sim --help` or `python simulation.py --help`
- Example run: `chemotaxis-sim --chi 30 --meshsize 100 --time 5 --eigen_index 2 --epsilon 0.5 --generate_video yes`
- Lint (matches CI intent): `python -m pylint $(find . -name '*.py')`
- Build docs: `pip install -r docs/requirements.txt && sphinx-build -b html docs docs/_build/html`

## Coding Style & Naming Conventions

- Python: 4-space indentation, type hints where practical, and keep functions small/typed.
- Formatting: `./Format.sh simulation.py` (expects `black`, `isort`, `autopep8` installed).
- Keep CLI options backward compatible; if flags/outputs change, update `README.md` and `docs/`.
- Dependency hygiene: keep imports in sync with `setup.py:install_requires` and `requirements.txt`.

## Testing Guidelines

- No dedicated test suite is currently present. For changes, add a minimal smoke check and run a short simulation locally (e.g., `--time 0.1`) to validate output generation.
- If adding tests, use `tests/test_*.py` and keep them fast and deterministic.

## Commit & Pull Request Guidelines

- Prefer imperative, scoped subjects aligned with existing history when possible (e.g., `feat(cli): ...`, `docs: ...`, `fix: ...`).
- PRs should include: a brief rationale, how to reproduce/verify, and (when relevant) a screenshot/GIF of generated plots; do not commit generated media artifacts.
