# Chemotaxis Simulations CLI Tool

[![PyPI version](https://img.shields.io/pypi/v/chemotaxis-sim.svg)](https://pypi.org/project/chemotaxis-sim/)
[![Python versions](https://img.shields.io/pypi/pyversions/chemotaxis-sim.svg)](https://pypi.org/project/chemotaxis-sim/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

A command-line interface (CLI) tool to simulate chemotaxis processes based on a parabolic-elliptic PDE model.

## Table of Contents

- [About](#about)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

## About

This repository contains a program that simulates a chemotaxis process from a parabolic-elliptic system of partial differential equations. It is based on the research paper *Global Existence and Asymptotic Behaviour of Classical Solutions of Chemotaxis Models with Signal-dependent Sensitivity and Logistic Source* by Dr. Wenxian Shen, Dr. Le Chen, and their PhD student Ian Ruau. Here is a link to the research paper **[link to paper]**

Simulations for special sets of parameters are consistent with proven theoretical results. The advantage of this package is that it helps us get insights about behaviours outside of the sets of parameters considered in the research paper.

## Installation

To get started with these simulations, you can install the package using pip:

```bash
pip install chemotaxis-sim
```

## Features

- Run chemotaxis simulations with customizable parameters.
- Terminal plots of initial conditions via termplotlib.
- 3D static surface plots (PNG/JPEG) and optional MP4 animations.
- Progress bar and verbose logging.
- Easy CLI usage with help prompts.

## Usage

Once the package is installed, the user can access the help prompt by simply
running the following command

```bash
chemotaxis-sim --help
```

### Quick examples

```bash
# Run from a YAML config (CLI flags override YAML values)
chemotaxis-sim --config config.example.yaml

# Paper-II eigenmode indexing (0-based): n=2 -> cos(2πx/L)
chemotaxis-sim --chi 30 --meshsize 100 --time 5 --eigen_mode_n 2 --epsilon 0.5

# Domain length (overrides YAML if provided)
chemotaxis-sim --config config.example.yaml --L 5.3

# Batch workflow: keep only summary6 + saved data (skip heavy 3D plots)
chemotaxis-sim --config config.example.yaml --save_data yes --save_summary6 yes --save_static_plots no

# Post-process heavy 3D plots later from the saved .npz
chemotaxis-plot images/branch_capture/some_run.npz

# Implied constants / bifurcation diagnostics (Paper II)
chemotaxis-constants --config config.example.yaml report
```

Notes:
- `--meshsize` is a mesh density (subintervals per unit length). The solver uses an effective uniform mesh with `N = ceil(meshsize * L)` subintervals (so there are `N+1` grid points).
- Use `--meshsize_abs N` to force an absolute number of subintervals independent of `L`.

### Shell completion (zsh)
This package supports tab completion via `argcomplete`, but your shell must be configured.

In `~/.zshrc`:

```bash
autoload -Uz compinit && compinit
eval "$(register-python-argcomplete --shell zsh chemotaxis-sim)"
eval "$(register-python-argcomplete --shell zsh chemotaxis-plot)"
eval "$(register-python-argcomplete --shell zsh chemotaxis-constants)"
```

Restart your shell (or run `exec zsh`).

Here is an example of a simulation

```bash
chemotaxis-sim --chi 30 --meshsize 100 --time 5 --eigen_index 2 --epsilon 0.5 --generate_video yes
```

Saved data controls (optional):

```bash
chemotaxis-sim --save_data yes --save_max_frames 2000 --save_summary6 yes
```

Notes:
- `--save_max_frames` controls the maximum number of time snapshots kept for outputs (plots/summary/video) and written to the saved `.npz` (default: `2000`).
- When using `--until_converged yes`, the in-run snapshot cap is instead `--max_saved_frames` (default: `2000`).

Numerics / stability controls (optional):

```bash
# Reduce dt (more stable, slower): larger values => smaller dt
chemotaxis-sim --dt_factor 4

# Stop early if u or v becomes NaN/Inf (default: yes)
chemotaxis-sim --stop_on_nonfinite yes
```

Notes:
- If `--stop_on_nonfinite yes` triggers, the run stops early with `stop_reason=nonfinite` saved in the `.npz`.
- If you see `RuntimeWarning: overflow` during time stepping, increase `--dt_factor` (and/or reduce `--time`) and rerun.

Batch-run output control (optional):

```bash
# Keep only the 6-frame summary images (*_summary6.{png,jpeg}) and any saved data
chemotaxis-sim --save_static_plots no --save_summary6 yes
```

Post-processing heavy plots from saved data:

```bash
# Later, regenerate the main 3D plots (<basename>.png/.jpeg) from the saved .npz
chemotaxis-plot images/branch_capture/some_run.npz

# Optionally also regenerate the lightweight 6-slice summary (<basename>_summary6.{png,jpeg})
chemotaxis-plot images/branch_capture/some_run.npz --summary6 yes

# Generate only the lightweight summary6 (skip heavy 3D static plots)
chemotaxis-plot images/branch_capture/some_run.npz --summary6 yes --save_static_plots no

# Optionally overlay the leading-order bifurcation prediction u_pred^± (Paper II, Local pitchfork bifurcation)
chemotaxis-plot images/branch_capture/some_run.npz --summary6 yes --summary6_bifurcation_curve yes
```

Notes:
- The overlay is labeled `stable` for supercritical cases (`beta_{n0}>0`) and `unstable` for subcritical cases (`beta_{n0}<0`).

Implied constants / bifurcation diagnostics:

```bash
# Global threshold (Eq. (1.12)) + equilibrium (Eq. (1.8))
chemotaxis-constants threshold --config config.example.yaml

# `--config` can be passed before or after the subcommand
chemotaxis-constants --config config.example.yaml threshold

# Bifurcation coefficients (defaults to n0 = argmin mode from the chi_a^* scan)
chemotaxis-constants bifurcation --config config.example.yaml

# Override the mode if desired
chemotaxis-constants bifurcation --config config.example.yaml --n0 1

# Full report + consistency check (is n0 the argmin mode for chi_a^*?)
chemotaxis-constants report --config config.example.yaml --format json

# Sensitivity shift c is supported (defaults to c=1)
chemotaxis-constants report --config config.example.yaml --c 1
```

Why override `--n0`?
- Multi-mode diagnostics: compare higher modes even if `n_min` destabilizes first.
- Near-degenerate minima: inspect competing modes when `chi^*(n)` is very close for multiple `n`.
- Reproducibility: match tables/figures that report coefficients at a fixed mode (often `n0=1`).
- Sensitivity studies: track how the cubic coefficient `beta_{n0}` varies with `n0`.
- Domain effects: compare the same `n0` across changes in `L` even when the argmin mode shifts.

Output naming controls (optional):

```bash
chemotaxis-sim --output_dir images/branch_capture --basename run_beta0_chi2185_epsP
```

### YAML config files
For long parameter lists, you can load defaults from a YAML file via `--config`.
CLI flags always override YAML values, and any missing parameter falls back to
the CLI default.

Example:

```bash
chemotaxis-sim --config config.example.yaml
chemotaxis-sim --config config.example.yaml --chi 2.185 --meshsize 100
```

See `config.example.yaml` for a grouped template.

### Eigenmode indexing (`--eigen_index`)
The initial condition uses a cosine perturbation of the form:

`u_0(x) = u^* + \epsilon cos(n\pi x/L) + \epsilon_2 cos((n+1)\pi x/L)`.

Paper II indexes Neumann eigenmodes by `n>=0`, with `n=0` the constant mode.

Preferred: use the paper-0-based flag `--eigen_mode_n n`:
- `--eigen_mode_n 0`: constant mode (`\lambda_0=0`)
- `--eigen_mode_n 1`: first nonconstant mode (`\cos(\pi x/L)`)
- `--eigen_mode_n 2`: second nonconstant mode (`\cos(2\pi x/L)`), etc.

Backward compatible: the legacy flag `--eigen_index k` is interpreted as `k=n+1`, with `k=0` reserved for auto:
- `--eigen_index 0`: auto-select (`k=1` if stable, otherwise `k=2`)
- `--eigen_index 1`: constant mode (`n=0`)
- `--eigen_index 2`: first nonconstant mode (`n=1`)
- `--eigen_index k` with `k>=1`: mode `n=k-1`

If `--eigen_mode_n` is provided, it overrides `--eigen_index`.

The CLI prints the resolved mode as `mode n = ...` when it starts.

### Paper II constants (Eq. (1.8) and Eq. (1.12))
The module `paper2_constants.py` computes the equilibrium and the discrete
threshold \(\chi_a^*(u^*)\) from Paper II:

```bash
python paper2_constants.py --a 10 --b 1 --alpha 1 --mu 1 --nu 1 --gamma 1 --m 1 --beta 0 --L 1
```

Load parameters from a YAML file (same key names as the simulator; missing keys
fall back to defaults; CLI overrides YAML):

```bash
python paper2_constants.py --config config.example.yaml
python paper2_constants.py --config config.example.yaml --beta 0 --L 1
```

If you also want the mesh-dependent discrete threshold \(\chi^{*,{\rm disc}}\)
(using the eigenvalues of the finite-difference Neumann Laplacian on `N` subintervals),
pass `--meshsize` (mesh density per unit length; effective `N=ceil(meshsize*L)`) or `--meshsize_abs N`:

```bash
python paper2_constants.py --config config.example.yaml --meshsize 50
```

Or import it in Python:

```python
from paper2_constants import paper2_eq112_constants

c = paper2_eq112_constants(a=10, b=1, alpha=1, mu=1, nu=1, gamma=1, m=1, beta=0)
print(c.u_star, c.v_star, c.chi_a_star)
```

The CLI `chemotaxis-constants` reports the same values; add `--meshsize 50` (or
`--meshsize_abs N`) to include \(\chi^{*,{\rm disc}}\) in the output.

### Stopping automatically when converged
If you do not want to guess a sufficiently large final time, you can run with
`--until_converged yes`.
In this mode, `--time` is interpreted as a *maximum* time horizon and the solver
will stop early once the solution changes by at most `--convergence_tol` over a
time window of length `--convergence_window_time` (after an initial
`--convergence_min_time` warm-up).

Example:

```bash
chemotaxis-sim --chi 2.19 --meshsize 50 --time 200 --eigen_index 2 --epsilon 0.001 --until_converged yes
```

The command above will show first some characteristic constants of the system of
partial differential equations that depend on the parameters input by the user:

![constants](./homepage/chi_table.png)

A terminal plot of the initial functions u(0,x) and v(0,x) is also shown on the
terminal:

![initial_plots](./homepage/u_v_terminal_plots.png)

Once the simulation is complete, a picture of the results is saved in both .png
and .jpeg formats (disable with `--save_static_plots no`):

![images_saved](./homepage/images_saved.png)

![Plots](./homepage/a=1-0_b=1-0_alpha=1-0_m=1-0_beta=1-0_chi=30-0_mu=1-0_nu=1_gamma=1-0_meshsize=100_time=5-0_epsilon=0-5_eigen_index=2.png)

In addition, the numerical data are saved for later post-processing in a
compressed NumPy file:

- `a=..._b=..._c=..._alpha=..._m=..._beta=..._chi=..._mu=..._nu=..._gamma=..._meshsize=..._time=..._epsilon=..._epsilon2=..._eigen_index=....npz`

This `.npz` contains (downsampled) `x_values`, `t_values`, `u_num`, `v_num`,
plus a JSON-encoded copy of the run configuration and some metadata.

### Continuing from a saved `.npz`
If you want to run longer without redoing the early time interval, use `--continue_from`.
In this mode, `--time` is interpreted as the desired *final* time \(T_{\rm final}\) (and must be larger than the saved `stop_time`).
The output `.npz` contains the concatenated (downsampled) time history.

Example:

```bash
# First run (writes <out>.npz in the current directory)
chemotaxis-sim --chi 2.19 --meshsize 50 --time 100 --eigen_mode_n 2 --epsilon 0.001 --save_data yes --save_summary6 yes

# Continue to T_final=200 into a new output folder/basename (keeps the original run for comparison)
chemotaxis-sim --continue_from <out>.npz --chi 2.19 --meshsize 50 --time 200 --output_dir images/continued --basename run_T200 --save_data yes --save_summary6 yes
```

This is different from `--restart_from`, which starts a new run at \(t=0\) using the last saved \(u\) as the initial condition (optionally adding the epsilon cosine perturbations).

The CLI also writes a quick diagnostic figure with 6 time slices of $u(x,t)$
(0%, 20%, …, 100%). The title shows $\chi_0$ and the critical threshold
$\chi^*(u^*)$:

- `..._summary6.png` and `..._summary6.jpeg`

If the user chose to save the animation of the chemotaxis process, an .mp4 video
will be saved as well:

![Video](./homepage/video.gif)

## Documentation

For detailed information about the package and its functionalities, visit the [documentation webpage](https://chemotaxis-simulation.readthedocs.io/en/latest/).

For a quick terminal overview of the available tools, run `chemotaxis --help`.

## TODO (performance / workflow)

- Expose an explicit time-step/CFL knob (current `Nt ~ O(T N^2)` can be expensive for `T=100`).
- Add a pure-Python (optional Numba) tridiagonal solver for `solve_v` as a non-SciPy fallback / speed path.
- Generate heavy 3D surface plots from saved `.npz` in post-processing for large batch runs.
- Add a lightweight smoke-test script (short run + basic invariants) since there is no dedicated test suite yet.

## Reproducing previous results

In the following page, we reproduce the results of the research paper by ....

[Reproducing_Results](./Reproducing_Previous_Results/README.md)


## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature/my-feature`).
3. Commit your changes (`git commit -am 'Add my feature'`).
4. Push to the branch (`git push origin feature/my-feature`).
5. Open a pull request.

## License

This project is licensed under the MIT License – see the [LICENSE](LICENSE) file for details.

## Contact

For any queries or further discussion, feel free to contact us at

- Le Chen: [chenle02@gmail.com] or [le.chen@auburn.edu] or [Homepage](https://webhome.auburn.edu/~lzc0090/index.html).
- Ian Ruau: [ian.ruau@auburn.edu].
