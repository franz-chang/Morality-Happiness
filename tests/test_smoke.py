import unittest

from morality_happiness import SimulationParameters, run_batch
from morality_happiness.simulation import Simulation


class SmokeTest(unittest.TestCase):
    def test_wellbeing_observers_and_missing_values(self) -> None:
        params = SimulationParameters(population_size=8, ticks=2, replications=1, burn_in_ticks=0, trade_enabled=False)
        individual = run_batch(["S3"], params)["results"][0]
        self.assertIsNone(individual["society_metrics"]["avg_peer_perceived_wellbeing"])
        self.assertTrue(all(a["peer_perceived_wellbeing"] is None for a in individual["final_agents"]))
        organized = run_batch(["S1"], params)["results"][0]
        self.assertTrue(organized["wellbeing_evaluations"])
        for agent in organized["final_agents"]:
            self.assertGreaterEqual(agent["objective_wellbeing"], 0)
            self.assertLessEqual(agent["objective_wellbeing"], 100)
            self.assertAlmostEqual(agent["wellbeing_perception_gap"], agent["peer_perceived_wellbeing"] - agent["objective_wellbeing"])

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
            scenario["shell_accounting"]["initial_shell_total"] / params.population_size,
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

    def test_food_shell_ledgers_and_no_trade_price(self) -> None:
        params = SimulationParameters(
            population_size=24,
            ticks=8,
            seed=13,
            trade_enabled=False,
            relief_source="public",
            public_food_reserve=12.0,
        )
        result = run_batch(["S1"], params)["results"][0]
        self.assertAlmostEqual(result["food_accounting"]["balance_error"], 0.0, places=6)
        self.assertLess(result["food_accounting"]["max_abs_tick_error"], 1e-6)
        self.assertAlmostEqual(result["shell_accounting"]["balance_error"], 0.0, places=6)
        self.assertIsNone(result["trade_metrics"]["avg_market_clearing_price"])
        self.assertTrue(all(record["clearing_price"] is None for record in result["trade_records"]))
        self.assertGreaterEqual(result["food_accounting"]["public_relief_total"], 0.0)

    def test_replicated_stream_honors_replications_without_agent_log_bloat(self) -> None:
        params = SimulationParameters(
            population_size=16,
            ticks=2,
            replications=2,
            seed=17,
            trade_enabled=False,
        )
        events = []
        result = run_batch(["S3"], params, event_callback=events.append)
        scenario = result["results"][0]
        self.assertEqual(result["replication_count"], 2)
        self.assertEqual(len(scenario["replicate_summaries"]), 2)
        self.assertFalse(scenario["experiment_protocol"]["detailed_agent_logs_retained"])
        self.assertEqual(scenario["agent_logs"], [])
        self.assertTrue(any(event["type"] == "complete" for event in events))

    def test_config_defaults_follow_long_run_protocol(self) -> None:
        params = SimulationParameters.from_dict({})
        self.assertEqual(params.ticks, 560)
        self.assertEqual(params.replications, 10)
        self.assertEqual(params.initial_shells_distribution, "uniform")
        self.assertEqual(params.initial_shells_min, 20.0)
        self.assertEqual(params.initial_shells_max, 40.0)
        self.assertEqual(params.burn_in_ticks, 28)
        self.assertAlmostEqual(params.chief_leadership_share, 4.0 / 130.0)
        self.assertEqual(params.trade_safety_buffer, 1.5)

    def test_minimum_tenure_locks_recent_role_changes(self) -> None:
        params = SimulationParameters(
            population_size=120,
            ticks=56,
            replications=1,
            seed=19,
            min_tenure_periods=1,
        )
        result = run_batch(["S2"], params)["results"][0]
        first_period_moves = {
            event["agent_id"] for event in result["promotion_events"] if event["tick"] == 28
        }
        second_period_moves = {
            event["agent_id"] for event in result["promotion_events"] if event["tick"] == 56
        }
        self.assertTrue(first_period_moves.isdisjoint(second_period_moves))
        self.assertEqual(result["role_metrics"]["chief"]["population"], 1.0)
        self.assertEqual(result["role_metrics"]["chieftain"]["population"], 7.0)
        self.assertEqual(result["role_metrics"]["hunter"]["population"], 112.0)

    def test_revised_market_burn_in_and_zlog_protocol(self) -> None:
        params = SimulationParameters(population_size=48, ticks=56, replications=1, seed=29)
        scenario = run_batch(["S2"], params)["results"][0]
        records = scenario["trade_records"]
        prices = [record["clearing_price"] for record in records if record["clearing_price"] is not None]
        self.assertTrue(all(params.shell_min_price <= price <= params.shell_max_price for price in prices))
        self.assertEqual(scenario["experiment_protocol"]["burn_in_ticks"], 28)
        self.assertEqual(scenario["promotion_reviews"][0]["tick"], 28)
        self.assertEqual(scenario["promotion_reviews"][0]["score_mode"], "zlog")
        initial_total = scenario["shell_accounting"]["initial_shell_total"]
        self.assertGreater(initial_total, params.initial_shells_min * params.population_size)
        self.assertLess(initial_total, params.initial_shells_max * params.population_size)
        self.assertAlmostEqual(scenario["shell_accounting"]["balance_error"], 0.0, places=6)

    def test_idiosyncratic_allocation_preserves_team_pool(self) -> None:
        simulation = Simulation("S1", SimulationParameters(population_size=8, idio_sigma=0.3, seed=31))
        members = simulation.agents
        baseline = {agent.id: 1.0 for agent in members}
        realized = simulation._apply_idiosyncratic_allocation_noise(baseline, members)
        self.assertAlmostEqual(sum(realized.values()), sum(baseline.values()), places=9)
        self.assertGreater(len({round(value, 8) for value in realized.values()}), 1)


if __name__ == "__main__":
    unittest.main()
