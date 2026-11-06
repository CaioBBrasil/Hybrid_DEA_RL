from __future__ import annotations

import unittest

import numpy as np

from rl.common.baselines import EnergyAwarePolicy, HighestMapPolicy, LowestEnergyPolicy, TopsisStaticPolicy
from rl.common.dea import dea_bcc_input_scores
from rl.common.model_selection import pareto_frontier


PROFILES = {
    "economical": {"benchmark_map": .50, "latency_s": 1.0, "incremental_energy_j": 1.0},
    "accurate": {"benchmark_map": .90, "latency_s": 10.0, "incremental_energy_j": 10.0},
}


class FakeEnergy:
    usable_energy_j = 100.0
    reserve_j = 0.0
    cell_distance_m = 1.0
    def return_energy(self, distance): return distance


class FakeEnv:
    def __init__(self, models=None, battery=100.0, viable=True):
        self.models = models or list(PROFILES); self.return_action = len(self.models)
        self.battery_j = battery; self.processed = 1; self.energy = FakeEnergy(); self.viable = viable
    def can_process_next(self, model): return self.viable
    def action_for_model(self, model): return self.models.index(model)


class BaselineTests(unittest.TestCase):
    def test_highest_map(self):
        policy = HighestMapPolicy(PROFILES, list(PROFILES))
        self.assertEqual(policy.model, "accurate")

    def test_lowest_energy(self):
        policy = LowestEnergyPolicy(PROFILES, list(PROFILES))
        self.assertEqual(policy.model, "economical")

    def test_topsis_ranking(self):
        dominant = {"a": {"benchmark_map": .9, "latency_s": 1, "incremental_energy_j": 1},
                    "b": {"benchmark_map": .5, "latency_s": 2, "incremental_energy_j": 2}}
        policy = TopsisStaticPolicy(dominant, ["a", "b"], [1, 1, 1])
        self.assertEqual(policy.model, "a"); self.assertGreater(policy.ranking[0][1], policy.ranking[1][1])

    def test_energy_aware_reacts_to_battery(self):
        parameters = {"accuracy_weight": 1, "base_energy_weight": .1, "energy_pressure_weight": 1.0,
                      "base_latency_weight": .05, "latency_pressure_weight": .5}
        policy = EnergyAwarePolicy(PROFILES, list(PROFILES), parameters); obs = np.zeros(5)
        self.assertEqual(policy.select(obs, FakeEnv(battery=100)), 1)
        self.assertEqual(policy.select(obs, FakeEnv(battery=2)), 0)

    def test_return_when_no_model_is_viable(self):
        policies = [HighestMapPolicy(PROFILES, list(PROFILES)), LowestEnergyPolicy(PROFILES, list(PROFILES)),
                    TopsisStaticPolicy(PROFILES, list(PROFILES), [1, 1, 1]),
                    EnergyAwarePolicy(PROFILES, list(PROFILES), {"accuracy_weight": 1,
                        "base_energy_weight": .1, "energy_pressure_weight": 1,
                        "base_latency_weight": .1, "latency_pressure_weight": 1})]
        env = FakeEnv(viable=False)
        self.assertTrue(all(policy.select(np.zeros(5), env) == env.return_action for policy in policies))

    def test_action_encoding_for_rl_all_and_dea_subset(self):
        policy = HighestMapPolicy(PROFILES, list(PROFILES)); obs = np.zeros(5)
        self.assertEqual(policy.select(obs, FakeEnv(["economical", "accurate"])), 1)
        subset_policy = HighestMapPolicy(PROFILES, ["accurate"])
        self.assertEqual(subset_policy.select(obs, FakeEnv(["accurate"])), 0)

    def test_dea_assigns_zero_efficiency_to_zero_quality(self):
        profiles = {"broken": {"benchmark_map": 0, "latency_s": 1, "incremental_energy_j": 1},
                    "working": {"benchmark_map": .5, "latency_s": 2, "incremental_energy_j": 2}}
        scores = dea_bcc_input_scores(profiles, list(profiles))
        self.assertEqual(scores["broken"], 0.0)
        self.assertGreater(scores["working"], 0.0)

    def test_pareto_frontier_reports_dominators(self):
        profiles = {
            "fast": {"quality": .5, "latency": 1, "energy": 1},
            "strong": {"quality": .9, "latency": 2, "energy": 2},
            "dominated": {"quality": .4, "latency": 3, "energy": 3},
        }
        frontier, dominators = pareto_frontier(
            profiles, list(profiles), minimize=("latency", "energy"), maximize=("quality",)
        )
        self.assertEqual(frontier, ["fast", "strong"])
        self.assertEqual(set(dominators["dominated"]), {"fast", "strong"})


if __name__ == "__main__": unittest.main()
