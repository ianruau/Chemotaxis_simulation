import math
import unittest

from implied_constants import compute_bifurcation_coefficients
from paper2_constants import equilibrium_u_v_star, paper2_eq112_constants
from simulation import SimulationConfig


class TestFixedEquilibriumSupport(unittest.TestCase):
    def test_equilibrium_u_v_star_fixed_mode(self) -> None:
        u_star, v_star = equilibrium_u_v_star(
            a=0.0,
            b=0.0,
            alpha=1.0,
            mu=1.0,
            nu=1.0,
            gamma=2.0,
            equilibrium_mode="fixed",
            u_star_fixed=1.0,
        )
        self.assertAlmostEqual(u_star, 1.0)
        self.assertAlmostEqual(v_star, 1.0)

    def test_threshold_fixed_mode(self) -> None:
        cst = paper2_eq112_constants(
            a=0.0,
            b=0.0,
            c=1.0,
            alpha=1.0,
            mu=1.0,
            nu=1.0,
            gamma=2.0,
            m=2.0,
            beta=1.0,
            L=1.0,
            equilibrium_mode="fixed",
            u_star_fixed=1.0,
            n_max=5000,
        )
        self.assertAlmostEqual(cst.u_star, 1.0)
        self.assertAlmostEqual(cst.v_star, 1.0)
        self.assertAlmostEqual(cst.chi_a_star, 1.0 + math.pi**2, places=6)
        self.assertEqual(cst.n_min, 1)

    def test_bifurcation_coefficients_fixed_mode(self) -> None:
        supercritical = compute_bifurcation_coefficients(
            {
                "equilibrium_mode": "fixed",
                "u_star_fixed": 1.0,
                "a": 0.0,
                "b": 0.0,
                "c": 1.0,
                "alpha": 1.0,
                "beta": 1.0,
                "m": 2.0,
                "mu": 1.0,
                "nu": 1.0,
                "gamma": 2.0,
                "L": 1.0,
                "n0": 1,
            }
        )
        subcritical = compute_bifurcation_coefficients(
            {
                "equilibrium_mode": "fixed",
                "u_star_fixed": 1.0,
                "a": 0.0,
                "b": 0.0,
                "c": 1.0,
                "alpha": 1.0,
                "beta": 1.0,
                "m": 1.0,
                "mu": 1.0,
                "nu": 1.0,
                "gamma": 1.0,
                "L": 1.0,
                "n0": 1,
            }
        )
        self.assertGreater(supercritical["beta_n0"], 0.0)
        self.assertLess(subcritical["beta_n0"], 0.0)
        self.assertIsNone(supercritical["a01"])
        self.assertIsNone(supercritical["a2n0"])

    def test_simulation_config_fixed_mode(self) -> None:
        cfg = SimulationConfig(
            equilibrium_mode="fixed",
            u_star_fixed=1.0,
            a=0.0,
            b=0.0,
            c=1.0,
            alpha=1.0,
            beta=1.0,
            m=2.0,
            chi=2.05,
            mu=1.0,
            nu=1.0,
            gamma=2.0,
            L=1.0,
            meshsize=5,
            mesh_per_unit=5.0,
            time=0.1,
            eigen_mode_n=1,
            confirm="yes",
            save_data="no",
            save_summary6="no",
            save_static_plots="no",
        )
        self.assertAlmostEqual(cfg.uStar, 1.0)
        self.assertAlmostEqual(cfg.vStar, 1.0)
        self.assertTrue(math.isfinite(cfg.ChiStar))
        self.assertTrue(math.isnan(cfg.ChiDStar))


if __name__ == "__main__":
    unittest.main()
