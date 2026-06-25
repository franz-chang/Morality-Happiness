import unittest

from morality_happiness import SimulationParameters, run_batch


class SmokeTest(unittest.TestCase):
    def test_all_scenarios_smoke(self) -> None:
        result = run_batch(parameters=SimulationParameters(population_size=32, ticks=8, seed=3))
        self.assertEqual(len(result["results"]), 6)
        for scenario_result in result["results"]:
            metrics = scenario_result["society_metrics"]
            self.assertGreaterEqual(metrics["avg_wellbeing"], 0)
            self.assertLessEqual(metrics["avg_wellbeing"], 100)
            self.assertGreaterEqual(metrics["avg_food_allocated"], 0)
            self.assertGreaterEqual(metrics["social_gini_index"], 0)
            self.assertLessEqual(metrics["social_gini_index"], 1)
            self.assertGreaterEqual(metrics["minimum_survival_rate"], 0)
            self.assertLessEqual(metrics["minimum_survival_rate"], 1)


if __name__ == "__main__":
    unittest.main()
