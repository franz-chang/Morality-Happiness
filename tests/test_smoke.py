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

    def test_trade_and_hierarchy_outputs(self) -> None:
        params = SimulationParameters(
            population_size=120,
            ticks=28,
            seed=11,
            trade_enabled=True,
            rank_enabled=True,
            promotion_enabled=True,
        )
        result = run_batch(["S2"], params)
        scenario = result["results"][0]
        self.assertEqual(scenario["role_metrics"]["chief"]["population"], 1.0)
        self.assertEqual(scenario["role_metrics"]["chieftain"]["population"], 7.0)
        self.assertEqual(scenario["role_metrics"]["hunter"]["population"], 112.0)
        self.assertGreaterEqual(scenario["trade_metrics"]["total_trade_volume_food"], 0.0)
        self.assertGreaterEqual(len(scenario["promotion_events"]), 0)
        self.assertAlmostEqual(
            scenario["society_metrics"]["avg_shell_balance"],
            params.initial_shells,
            places=6,
        )

    def test_baseline_can_disable_new_modules(self) -> None:
        params = SimulationParameters(
            population_size=24,
            ticks=4,
            seed=5,
            trade_enabled=False,
            rank_enabled=False,
            rank_split_enabled=False,
            promotion_enabled=False,
        )
        result = run_batch(["S1", "S2"], params)
        for scenario in result["results"]:
            self.assertEqual(scenario["trade_metrics"]["total_trade_volume_food"], 0.0)
            self.assertEqual(scenario["promotion_events"], [])


if __name__ == "__main__":
    unittest.main()
