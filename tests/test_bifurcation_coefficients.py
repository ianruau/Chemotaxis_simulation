import unittest

from implied_constants import compute_bifurcation_coefficients


class TestBifurcationCoefficientRegressions(unittest.TestCase):
    def test_nonminimal_center_graph_regressions(self) -> None:
        cases = (
            (
                "base",
                {},
                3.06443516403513,
            ),
            (
                "zero_sensitivity_exponent",
                {"a": 10.0, "beta": 0.0},
                0.0765030805152031,
            ),
            (
                "nonlinear_mobility_and_production",
                {"m": 2.0, "gamma": 2.0},
                9.50888036338459,
            ),
            (
                "paper_iii_preferred_case",
                {"m": 2.0, "beta": 0.5, "gamma": 3.0},
                8.46192115072218,
            ),
        )

        defaults = {
            "equilibrium_mode": "logistic",
            "a": 1.0,
            "b": 1.0,
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

        for name, overrides, expected_beta in cases:
            with self.subTest(name=name):
                params = {**defaults, **overrides}
                result = compute_bifurcation_coefficients(params)
                self.assertAlmostEqual(result["beta_n0"], expected_beta, places=12)
                self.assertGreater(result["beta_n0"], 0.0)

    def test_reduced_coefficient_uses_negative_chemotactic_projection(self) -> None:
        params = {
            "equilibrium_mode": "logistic",
            "a": 1.0,
            "b": 1.0,
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
        result = compute_bifurcation_coefficients(params)
        logistic_quadratic = 0.5
        expected = (
            logistic_quadratic * (4 * result["a01"] + 2 * result["a2n0"])
            - result["chi_star_mode_n0"] * result["gamma_cubic"]
        )
        self.assertAlmostEqual(result["beta_n0"], expected, places=12)


if __name__ == "__main__":
    unittest.main()
