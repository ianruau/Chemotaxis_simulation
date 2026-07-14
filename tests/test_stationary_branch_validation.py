from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from stationary_branch_validation import (
    CASES,
    bifurcation_coefficients,
    critical_amplitude,
    discrete_threshold,
    fit_quadratic_quartic,
    fit_with_intercept,
    observed_convergence_order,
    run_validation,
    solve_branch_point,
    state_identifier,
    trapezoidal_integral,
    write_outputs,
)


class StationaryBranchValidationTests(unittest.TestCase):
    def test_trapezoidal_mass_and_cosine_amplitude_are_exact(self) -> None:
        mesh = 20
        x = np.linspace(0.0, 1.0, mesh + 1)
        amplitude = -0.0375
        u = 1.0 + amplitude * np.cos(np.pi * x)
        self.assertAlmostEqual(trapezoidal_integral(u, 1.0), 1.0, places=14)
        self.assertAlmostEqual(
            critical_amplitude(u, u_star=1.0, x=x, length=1.0, mode=1),
            amplitude,
            places=14,
        )

    def test_anchored_and_free_intercept_fits_recover_coefficients(self) -> None:
        amplitudes = np.array([-0.02, -0.01, -0.005, 0.005, 0.01, 0.02])
        intercept = 2.125
        expected_c2 = -1.75
        expected_c4 = 3.25
        delta = expected_c2 * amplitudes**2 + expected_c4 * amplitudes**4
        c2, c4, anchored_residual = fit_quadratic_quartic(amplitudes, delta)
        free_intercept, free_c2, free_c4, free_residual = fit_with_intercept(
            amplitudes, intercept + delta
        )
        self.assertAlmostEqual(c2, expected_c2, places=12)
        self.assertAlmostEqual(c4, expected_c4, places=8)
        self.assertAlmostEqual(free_intercept, intercept, places=12)
        self.assertAlmostEqual(free_c2, expected_c2, places=9)
        self.assertAlmostEqual(free_c4, expected_c4, places=6)
        self.assertLess(anchored_residual, 1e-15)
        self.assertLess(free_residual, 1e-14)

    def test_observed_order_reports_second_order_refinement(self) -> None:
        self.assertAlmostEqual(
            observed_convergence_order(0.04, 0.01, 40, 80), 2.0, places=14
        )

    def test_public_simulator_coefficients_cover_a10_supercritical_case(self) -> None:
        case = CASES["nonminimal-a10-beta0"]
        coefficients = bifurcation_coefficients(case.name)
        self.assertEqual(case.u_star, 10.0)
        self.assertAlmostEqual(coefficients["chi_star"], 2.188281623751274, places=12)
        self.assertAlmostEqual(coefficients["alpha_n0"], 9.080003316496247, places=12)
        self.assertAlmostEqual(coefficients["beta_n0"], 0.07650308051520315, places=12)
        self.assertAlmostEqual(
            coefficients["beta_n0"] / coefficients["alpha_n0"],
            0.00842544631852886,
            places=14,
        )

    def test_discrete_threshold_uses_the_first_discrete_mode(self) -> None:
        threshold = discrete_threshold(CASES["minimal-m1-g1"], 20)
        self.assertEqual(threshold.minimizing_mode, 1)
        self.assertLess(
            threshold.chi_star, bifurcation_coefficients("minimal-m1-g1")["chi_star"]
        )

    def test_validation_rejects_unpaired_amplitudes(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive/negative pair"):
            run_validation(
                [CASES["minimal-m1-g1"]],
                meshes=[12],
                amplitudes=(-0.02, -0.01, -0.005, 0.005, 0.01),
            )

    def test_minimal_stationary_solve_closes_full_residual_and_mass(self) -> None:
        case = CASES["minimal-m1-g1"]
        point = solve_branch_point(case, mesh=20, amplitude=0.01)
        self.assertTrue(point.solver_success, point.solver_message)
        self.assertLess(point.full_stationary_residual_max, 1e-8)
        self.assertLess(point.elliptic_residual_max, 1e-8)
        self.assertLess(abs(point.amplitude_error), 5e-11)
        self.assertIsNotNone(point.mass_constraint_error)
        self.assertLess(abs(float(point.mass_constraint_error)), 5e-12)
        self.assertGreater(point.u_min, 0.0)
        self.assertGreater(point.v_min, 0.0)
        self.assertEqual(
            point.reflection_partner_id,
            state_identifier(case.name, point.mesh, -point.requested_amplitude),
        )

    def test_output_bundle_indexes_states_and_checksums_every_artifact(self) -> None:
        case = CASES["minimal-m1-g1"]
        amplitudes = (-0.02, -0.01, -0.005, 0.005, 0.01, 0.02)
        points, fits = run_validation([case], meshes=[12], amplitudes=amplitudes)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_dir = Path(temporary_directory) / "first"
            paths = write_outputs(
                output_dir,
                [case],
                [12],
                amplitudes,
                points,
                fits,
                [],
            )
            second_output_dir = Path(temporary_directory) / "second"
            second_paths = write_outputs(
                second_output_dir,
                [case],
                [12],
                amplitudes,
                points,
                fits,
                [],
            )
            expected_names = {
                "branch-points.csv",
                "fit-summary.json",
                "stationary-profiles.csv",
                "stationary-states.npz",
                "states-index.json",
                "stationary-continuation.pdf",
                "stationary-continuation.png",
                "SHA256SUMS",
            }
            self.assertEqual({path.name for path in paths.values()}, expected_names)
            self.assertEqual(
                {path.name for path in second_paths.values()}, expected_names
            )
            for name in expected_names:
                self.assertEqual(
                    (output_dir / name).read_bytes(),
                    (second_output_dir / name).read_bytes(),
                )

            index = json.loads(paths["states_index"].read_text(encoding="utf-8"))
            self.assertEqual(index["state_count"], len(points))
            self.assertTrue(
                all(state["reflection_partner_present"] for state in index["states"])
            )
            with np.load(paths["states"], allow_pickle=False) as archive:
                expected_arrays = {
                    array_name
                    for state in index["states"]
                    for array_name in state["array_names"].values()
                }
                self.assertEqual(set(archive.files), expected_arrays)

            checksum_entries = {}
            for line in paths["checksums"].read_text(encoding="ascii").splitlines():
                digest, name = line.split("  ", 1)
                checksum_entries[name] = digest
            checksummed_names = expected_names - {"SHA256SUMS"}
            self.assertEqual(set(checksum_entries), checksummed_names)
            for name, expected_digest in checksum_entries.items():
                actual_digest = hashlib.sha256(
                    (output_dir / name).read_bytes()
                ).hexdigest()
                self.assertEqual(actual_digest, expected_digest)


if __name__ == "__main__":
    unittest.main()
