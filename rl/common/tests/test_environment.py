from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np

from rl.common.environment import UAVModelSelectionEnv
from rl.common.map_generator import generate_map
from rl.common.map_generator import generate_synthetic_map
from rl.common.quality import DetectionCounts, QualityProvider


class PerfectProvider(QualityProvider):
    def evaluate(self, model, cell):
        return DetectionCounts(cell.ground_truth_count, 0, 0)


def fixture(root: Path, count: int = 6) -> dict:
    for split in ("train", "valid", "test"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "labels").mkdir()
        for index in range(count):
            (root / split / "images" / f"{split}_{index}.jpg").touch()
            label = "0 0.5 0.5 0.2 0.2\n" if index % 2 else ""
            (root / split / "labels" / f"{split}_{index}.txt").write_text(label)
    return {
        "dataset": {"root": str(root)},
        "physical": {"battery_capacity_j": 100000, "speed_m_s": 10,
                     "movement_power_w": 100, "hover_power_w": 100,
                     "cell_distance_m": 20, "reserve_j": 0},
        "models": {"fast": {"benchmark_map": .7, "latency_s": .1, "incremental_energy_j": 2,
                              "synthetic_precision": .8, "synthetic_recall": .8}},
        "quality_provider": {"type": "synthetic", "confidence_threshold": .25, "iou_threshold": .5},
        "reward": {"alpha_fp": .5, "beta_fn": 1, "quality_weight": 1,
                   "coverage_weight": 2, "safe_return_reward": 1,
                   "energy_weight": .25, "failure_penalty": 10},
        "scenarios": {"rl_all": ["fast"], "dea_rl": []},
    }


class EnvironmentTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.cfg = fixture(Path(self.temp.name))

    def tearDown(self): self.temp.cleanup()

    def env(self, cfg=None): return UAVModelSelectionEnv(cfg or self.cfg, quality_provider=PerfectProvider())

    def test_map_uses_every_image_once(self):
        cells = generate_map(Path(self.temp.name), "train", 7)
        self.assertEqual(len(cells), len({x.image_id for x in cells}))
        self.assertEqual(len(cells), 6)

    def test_map_is_reproducible(self):
        a = [x.image_id for x in generate_map(Path(self.temp.name), "train", 7)]
        b = [x.image_id for x in generate_map(Path(self.temp.name), "train", 7)]
        self.assertEqual(a, b)

    def test_map_can_standardize_episode_length(self):
        cells = generate_map(Path(self.temp.name), "train", 7, cells_per_episode=4)
        self.assertEqual(len(cells), 4)
        with self.assertRaises(ValueError):
            generate_map(Path(self.temp.name), "train", 7, cells_per_episode=7)

    def test_splits_are_disjoint(self):
        sets = [{x.image_id for x in generate_map(Path(self.temp.name), split, 1)} for split in ("train", "valid", "test")]
        self.assertFalse(sets[0] & sets[1] | sets[0] & sets[2] | sets[1] & sets[2])

    def test_synthetic_map_requires_no_files(self):
        cells = generate_synthetic_map(10, "train", 7)
        self.assertEqual(len(cells), 10)
        self.assertTrue(all(cell.ground_truth_count == 1 and not cell.image_path for cell in cells))

    def test_observation_is_normalized(self):
        env = self.env(); obs, _ = env.reset(seed=1)
        self.assertTrue(np.all(obs >= 0) and np.all(obs <= 1))
        obs, *_ = env.step(0); self.assertTrue(np.all(obs >= 0) and np.all(obs <= 1))

    def test_sequential_progress_has_no_skip(self):
        env = self.env(); env.reset(seed=1)
        for expected in (1, 2, 3): env.step(0); self.assertEqual(env.processed, expected)

    def test_energy_accounting(self):
        env = self.env(); env.reset(seed=1); before = env.battery_j; env.step(0)
        expected = 100 * 2 + 100 * .1 + 2
        self.assertAlmostEqual(before - env.battery_j, expected)

    def test_return_is_always_an_action(self):
        env = self.env(); env.reset(seed=1)
        self.assertEqual(env.return_action, len(env.models)); self.assertTrue(env.action_space.contains(env.return_action))

    def test_model_only_action_space_uses_common_shield(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["physical"]["battery_capacity_j"] = 400
        cfg["safety"] = {
            "shield_infeasible_actions": True,
            "agent_return_action": False,
        }
        cfg["reward"]["safety_override_penalty"] = 0
        env = self.env(cfg); env.reset(seed=1)
        self.assertEqual(env.action_space.n, len(env.models))
        self.assertFalse(env.action_space.contains(env.return_action))
        _, _, terminated, _, info = env.step(0)
        self.assertTrue(terminated and info["safe_return"] and info["is_success"])
        self.assertTrue(info["shield_return"])
        self.assertFalse(info["autonomous_safe_return"])

    def test_return_terminates_safely(self):
        env = self.env(); env.reset(seed=1); env.step(0)
        _, _, terminated, _, info = env.step(env.return_action)
        self.assertTrue(terminated and info["safe_return"] and info["is_success"] and not info["failed"])

    def test_safe_return_reward_can_scale_with_coverage(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["reward"].update({"safe_return_reward": 12, "scale_safe_return_by_coverage": True})
        env = self.env(cfg); env.reset(seed=1); env.step(0)
        _, reward, *_ = env.step(env.return_action)
        energy_fraction = env.energy_used_j / env.energy.battery_capacity_j
        self.assertAlmostEqual(reward, 12 / 6 - .25 * energy_fraction)

    def test_unsafe_analysis_fails(self):
        cfg = copy.deepcopy(self.cfg); cfg["physical"]["battery_capacity_j"] = 100
        env = self.env(cfg); env.reset(seed=1)
        _, reward, terminated, _, info = env.step(0)
        self.assertTrue(terminated and info["failed"]); self.assertEqual(reward, -10)

    def test_safety_shield_converts_unsafe_analysis_into_return(self):
        cfg = copy.deepcopy(self.cfg)
        cfg["physical"]["battery_capacity_j"] = 400
        cfg["safety"] = {"shield_infeasible_actions": True}
        cfg["reward"]["safety_override_penalty"] = 5
        env = self.env(cfg); env.reset(seed=1)
        _, reward, terminated, _, info = env.step(0)
        self.assertTrue(terminated and info["safe_return"] and not info["failed"])
        self.assertFalse(info["is_success"])
        self.assertTrue(info["safety_override"])
        self.assertEqual(info["safety_overrides"], 1)
        self.assertEqual(info["return_count"], 1)
        self.assertAlmostEqual(reward, -4)

    def test_full_route_auto_returns(self):
        env = self.env(); env.reset(seed=1); terminated = False
        while not terminated: _, _, terminated, _, info = env.step(0)
        self.assertEqual(info["coverage"], 1); self.assertTrue(info["safe_return"])

if __name__ == "__main__": unittest.main()
