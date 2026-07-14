import sys
import unittest
from unittest.mock import patch

from simulation import (
    SimulationConfig,
    chi_star_threshold_discrete,
    parse_args,
)


class TestThresholdReporting(unittest.TestCase):
    def test_fixed_mass_summary_uses_mesh_dependent_threshold(self) -> None:
        config = SimulationConfig(
            equilibrium_mode="fixed",
            u_star_fixed=1.0,
            a=0.0,
            b=0.0,
            c=1.0,
            alpha=1.0,
            beta=1.0,
            m=1.0,
            mu=1.0,
            nu=1.0,
            gamma=1.0,
            L=1.0,
            meshsize=20,
        )

        threshold = chi_star_threshold_discrete(config)

        self.assertAlmostEqual(threshold, 21.698655047779635, places=12)
        self.assertNotAlmostEqual(threshold, 21.739208802178716, places=6)

    def test_explicit_c_flag_is_not_abbreviated_to_config(self) -> None:
        argv = ["chemotaxis-sim", "--c", "1", "--meshsize_abs", "20"]

        with patch.object(sys, "argv", argv):
            config = parse_args()

        self.assertEqual(config.c, 1.0)
        self.assertEqual(config.meshsize, 20)


if __name__ == "__main__":
    unittest.main()
