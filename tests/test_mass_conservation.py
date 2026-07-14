import numpy as np

from simulation import RK4, SimulationConfig, rhs


def _minimal_config(**overrides: object) -> SimulationConfig:
    values: dict[str, object] = {
        "equilibrium_mode": "fixed",
        "u_star_fixed": 1.0,
        "a": 0.0,
        "b": 0.0,
        "c": 1.0,
        "alpha": 1.0,
        "beta": 1.0,
        "m": 2.0,
        "chi": 2.05,
        "mu": 1.0,
        "nu": 1.0,
        "gamma": 2.0,
        "L": 1.0,
        "meshsize": 20,
        "time": 0.02,
        "eigen_mode_n": 1,
        "epsilon": 0.05,
        "confirm": "yes",
        "generate_video": "no",
        "save_data": "no",
        "save_summary6": "no",
        "save_static_plots": "no",
        "save_max_frames": 1000,
    }
    values.update(overrides)
    return SimulationConfig(**values)


def _trapezoidal_mass(values: np.ndarray, *, dx: float) -> np.ndarray:
    return dx * (0.5 * values[0] + np.sum(values[1:-1], axis=0) + 0.5 * values[-1])


def test_minimal_rhs_has_zero_discrete_mass_rate() -> None:
    config = _minimal_config(meshsize=32)
    rng = np.random.default_rng(20260713)
    varying_u = 0.75 + rng.random(config.meshsize + 1)
    varying_v = 0.50 + rng.random(config.meshsize + 1)
    states = (
        (varying_u, varying_v, 0.0),
        (np.ones_like(varying_u), varying_v, 2.05),
        (varying_u, varying_v, 2.05),
    )

    for u, v, chi in states:
        case = _minimal_config(meshsize=32, chi=chi)
        mass_rate = float(_trapezoidal_mass(rhs(u, v, case), dx=case.L / case.meshsize))

        assert abs(mass_rate) <= 5.0e-13


def test_minimal_rk4_mass_drift_stays_at_roundoff() -> None:
    config = _minimal_config()

    result = RK4(config)
    masses = _trapezoidal_mass(result.u_num, dx=config.L / config.meshsize)

    assert np.max(np.abs(masses - masses[0])) <= 5.0e-13
