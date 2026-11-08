import copy
import csv
import json
import io
import zipfile
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

from dspigcn_candidate.interface import Predictor
from dspigcn_candidate.protocol import compare_manifests
from dspigcn_candidate.data import digest, read_tracks
from dspigcn_candidate.runner import default_config, train, evaluate, benchmark
from dspigcn_candidate.audit import audit_run


class IntegrationTests(unittest.TestCase):
    def test_six_columns_ignore_provided_velocities(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tracks.txt"
            np.savetxt(path, [[0, 1, 2, 3, 999, 999], [10, 1, 4, 5, 999, 999]])
            np.testing.assert_array_equal(read_tracks(path), [[0, 1, 2, 3], [10, 1, 4, 5]])

    def test_protocol_rejects_same_count_different_population(self):
        manifest = {"held_out": "eth", "split": "test", "files_sha256": {"x": "abc"},
                    "config": {}, "sample_ids": ["a", "b"], "sample_ids_sha256": digest(["a", "b"])}
        self.assertTrue(compare_manifests(manifest, manifest)["compatible"])
        other = copy.deepcopy(manifest)
        other["sample_ids"] = ["a", "c"]
        with self.assertRaises(ValueError):
            compare_manifests(manifest, other)

    def test_serialized_default_matches_cli(self):
        path = Path(__file__).resolve().parents[1] / "configs/default.json"
        self.assertEqual(json.loads(path.read_text()), default_config())

    def test_end_to_end_cpu_checkpoint_interface_and_exports(self):
        from dspigcn_candidate.data import SCENES
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            rng = np.random.default_rng(7)
            for names in SCENES.values():
                for name in names:
                    positions = np.cumsum(rng.normal(.1, .02, size=(250, 2, 2)), axis=0)
                    np.savetxt(data / name, [[t*10, p, *positions[t, p]] for t in range(250) for p in range(2)])
            cfg = default_config()
            cfg["data"]["stride"] = 100
            cfg["anchor_components"] = 2
            cfg["model"].update(hidden=16, heads=2, prediction_layers=1)
            cfg["training"].update(epochs=1, batch_size=4)
            checkpoint = train(data, "eth", 0, cfg, root / "run", "cpu")
            results = evaluate(data, checkpoint, root / "run/evaluation", "cpu")
            self.assertEqual(len(results), 6)
            self.assertEqual(results[0]["trajectories"], 6)
            with zipfile.ZipFile(root / "run/evaluation/clean/predictions.zip") as archive:
                with np.load(io.BytesIO(archive.read("000000.npz")), allow_pickle=False) as loaded:
                    export = dict(loaded)
            self.assertEqual(export["predictions"].shape, (20, 12, 2, 2))
            caller_rng = torch.random.get_rng_state().clone()
            predictor = Predictor(checkpoint)
            torch.testing.assert_close(torch.random.get_rng_state(), caller_rng)
            obs = torch.from_numpy(export["observations"])
            first = predictor.predict(obs, seq_start_end=[(0, 1), (1, 2)], seed=2)
            torch.testing.assert_close(torch.random.get_rng_state(), caller_rng)
            np.testing.assert_array_equal(obs.numpy(), export["observations"])
            modified = obs.clone()
            modified[:, 1] += 1000
            second = predictor.predict(modified, seq_start_end=[(0, 1), (1, 2)], seed=2)
            torch.testing.assert_close(first[:, :, 0], second[:, :, 0])
            self.assertEqual(first.shape, (20, 12, 2, 2))
            with self.assertRaises(ValueError):
                predictor.predict(obs, seq_start_end=[(0, 1)])
            runtime = benchmark(checkpoint, root / "runtime.json", "cpu", pedestrians=1, repeats=2)
            self.assertGreater(runtime["prediction_ms"]["median"], 0)
            audited = audit_run(root / "run", data_root=data, expected_scene="eth", expected_seed=0)
            self.assertTrue(audited["source_data_verified"])
            with self.assertRaisesRegex(ValueError, "wrong training seed"):
                audit_run(root / "run", expected_seed=1)
            table = root / "run/evaluation/clean/per_trajectory.csv"
            with table.open() as f:
                rows = list(csv.DictReader(f))
            rows[0]["meanADE"] = "999"
            with table.open("w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            with self.assertRaisesRegex(ValueError, "Per-trajectory metric mismatch"):
                audit_run(root / "run")


if __name__ == "__main__":
    unittest.main()
