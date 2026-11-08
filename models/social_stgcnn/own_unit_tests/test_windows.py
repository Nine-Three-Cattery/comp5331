"""Causal inputs, manifest order and the neutral forecast bundle on a tiny scene."""

import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from li_junhao_trajectronpp.independent_tpp.data import SceneStore, WindowRecord
from li_junhao_trajectronpp.independent_tpp.protocol import CONDITIONS, corruption_seed, perturb_history
from models.social_stgcnn.export import export_condition, raw_for_batch, target_history
from models.social_stgcnn.model import future_displacements, sample_world
from models.social_stgcnn.windows import GraphWindows


SCENE = ("eth", "tiny")


def track(x, y, end=22, dx=0.0, dy=0.0):
    return {t: np.array([x + dx * t, y + dy * t, 17., 19., 23., 29.]) for t in range(end + 1)}


def store(tracks, targets=(("1", 7),)):
    records = [WindowRecord({
        "window_id": f"eth:tiny:{target}:{anchor}", "scene_name": "eth__tiny__test",
        "scene_id": "eth", "source_file": "tiny", "scene_t": anchor, "anchor_t": anchor,
        "target_id": int(target), "fold": "eth", "split": "test", "dt": 0.4,
        "coordinate_unit": "meter",
    }) for target, anchor in targets]
    return SceneStore({SCENE: tracks}, records, {
        "fold": "eth", "split": "test", "version": "causal_v2",
        "manifest_hash": "synthetic-manifest", "input_hash": "synthetic-input",
    })


def moving_scene():
    return store({
        "1": track(18.123456789, -0.983216789, dx=0.371234567, dy=-0.13),
        "2": track(16.234567891, 1.294817321, dx=0.52, dy=-0.03),
    }, targets=(("1", 9), ("1", 7), ("2", 8)))


class GraphWindowTest(unittest.TestCase):
    def test_complete_history_pool_keeps_all_neighbors_in_numeric_order(self):
        tracks = {"1": track(0., 0., end=35)}
        offsets = {"2": -1, "10": 12, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6, "8": 7, "9": 8}
        for person, offset in offsets.items():
            tracks[person] = track(offset, 0., end=7)  # No future is available for these agents.
        tracks["11"] = track(0.02, 0., end=6)
        tracks["12"] = track(0.01, 0., end=9)
        del tracks["12"][4]
        dataset = GraphWindows(store(tracks, targets=(("1", 18), ("1", 7))))
        batch = dataset.batch([0, 1])
        self.assertEqual([r.anchor for r in dataset.records], [18, 7])
        self.assertEqual(dataset.available_neighbors.tolist(), [0, 9])
        self.assertEqual(tuple(batch["positions"].shape), (2, 8, 10, 2))
        self.assertEqual(batch["positions"].dtype, torch.float64)
        self.assertEqual(batch["target_index"].tolist(), [0, 0])
        solo, dense = batch["graph_index"].tolist()
        self.assertEqual(batch["agent_mask"][[solo, dense]].sum(1).tolist(), [1, 10])
        self.assertTrue(torch.all(batch["positions"][solo, :, 1:] == 0))
        self.assertEqual(batch["positions"][dense, -1, :, 0].tolist(), [0., -1., 2., 3., 4., 5., 6., 7., 8., 12.])

    def test_perturbation_uses_original_index_and_preserves_other_tracks(self):
        scene = moving_scene()
        dataset = GraphWindows(scene)
        indices = [2, 0]
        for tag, kind, level in CONDITIONS:
            with self.subTest(tag=tag):
                batch = dataset.batch(indices, kind, level, seed=2)
                for j, original in enumerate(indices):
                    row = scene.manifests[original]
                    frames = range(row.anchor - 7, row.anchor + 1)
                    person = scene.tracks[SCENE][row.target_id]
                    original_xy = np.array([person[t][:2] for t in frames])
                    expected, mask = perturb_history(original_xy, kind, level, corruption_seed(2, original))
                    g, slot = batch["graph_index"][j], batch["target_index"][j]
                    self.assertEqual(slot.item(), int(row.target_id) - 1)
                    np.testing.assert_array_equal(batch["positions"][g, :, slot], expected)
                    np.testing.assert_array_equal(batch["observed_mask"][j], mask)
                    other = scene.tracks[SCENE]["1" if row.target_id == "2" else "2"]
                    np.testing.assert_array_equal(batch["positions"][g, :, 1 - slot],
                                                  np.array([other[t][:2] for t in frames], dtype=np.float64))

    def test_missing_suffix_uses_filled_target_and_keeps_the_neighbor_order(self):
        target = track(0., 0., dx=1.)
        target[7][0] = 12.
        scene = store({"1": target, "2": track(7., 0.), "3": track(12., 0.)})
        windows = GraphWindows(scene)
        self.assertEqual(windows.batch([0])["positions"][0, -1, :, 0].tolist(), [12., 7., 12.])
        missing = windows.batch([0], "missing", 1)
        self.assertEqual(missing["positions"][0, -1, :, 0].tolist(), [7., 7., 12.])
        target[7][:2] = [-100., 400.]
        changed = GraphWindows(scene).batch([0], "missing", 1)
        self.assertTrue(torch.equal(missing["positions"], changed["positions"]))

    def test_target_and_neighbor_future_invariance_separately(self):
        original = GraphWindows(moving_scene()).batch([1], "noise", 0.1, seed=1)
        for changed_person in ("1", "2"):
            with self.subTest(changed_person=changed_person):
                altered = moving_scene()
                for t, state in altered.tracks[SCENE][changed_person].items():
                    if t > 7:
                        state[:2] += [317., -690.]
                    state[2:] = -8888.  # Precomputed whole-track derivatives are never features.
                dataset = GraphWindows(altered)
                self.assertEqual(torch.equal(original["future"], torch.from_numpy(dataset.future[[1]])),
                                 changed_person == "2")
                dataset.future = None  # Inference-only batches need no labels.
                actual = dataset.batch([1], "noise", 0.1, seed=1, include_future=False)
                self.assertNotIn("future", actual)
                for key in ("positions", "agent_mask", "observed_mask", "graph_index", "target_index"):
                    self.assertTrue(torch.equal(original[key], actual[key]), key)

    def test_metric_displacements_round_trip_through_the_visible_anchor(self):
        dataset = GraphWindows(moving_scene())
        for _, kind, level in CONDITIONS:
            batch = dataset.batch([0, 2], kind, level, seed=2)
            origin = target_history(batch)[:, -1].float()
            raw = torch.zeros(2, 12, 5)
            raw[..., :2] = future_displacements(batch["future"], origin)
            restored = sample_world(raw, origin, torch.zeros(2, 1, 12, 2))
            torch.testing.assert_close(restored[:, 0], batch["future"], rtol=0, atol=3e-6)

    def test_clean_scenes_supervise_numeric_slots_once_at_source_precision(self):
        tracks = {"10": track(100., 2., end=20), "2": track(2., 1., end=40), "7": track(7., 0., end=7)}
        # Rounding at four decimals gives different increments if prematurely cast to float32.
        tracks["10"][7][0] = 100.00004999
        tracks["10"][8][0] = 100.00015001
        scene = store(tracks, targets=(("10", 7), ("2", 21), ("2", 7)))
        ds = GraphWindows(scene)
        dense, solo = list(ds.clean_scenes())
        self.assertEqual((dense.anchor, solo.anchor), (7, 21))
        self.assertEqual(dense.target_slots.tolist(), [0, 2])
        self.assertEqual(dense.manifest_indices.tolist(), [2, 0])
        self.assertEqual(solo.manifest_indices.tolist(), [1])
        self.assertEqual(tuple(dense.positions.shape), (8, 3, 2))
        self.assertEqual(dense.positions[-1, :, 0].tolist(), [2., 7., 100.00004999])
        self.assertEqual(dense.positions.dtype, torch.float64)
        self.assertEqual(dense.future_steps.dtype, torch.float32)
        path = np.stack([tracks["10"][t][:2] for t in range(7, 20)])
        expected = np.diff(path.round(4), axis=0).astype(np.float32)
        np.testing.assert_array_equal(dense.future_steps[1], expected)
        premature = np.diff(path.astype(np.float32).astype(np.float64).round(4), axis=0).astype(np.float32)
        self.assertNotEqual(expected[0, 0], premature[0, 0])
        np.testing.assert_array_equal(ds.future[0], path[1:].astype(np.float32))

    def test_inference_groups_exact_widths_and_gathers_canonical_slots(self):
        class SlotReadout:
            def __init__(self):
                self.shapes = []

            def full_scene(self, increments, laplacian):
                count, _, width, _ = increments.shape
                self.shapes.append((count, width))
                assert laplacian.shape == (count, 8, width, width)
                out = increments.new_zeros((count, 12, width, 5))
                out[..., 0] = width * 100 + torch.arange(width)
                out[..., 1] = increments[:, -1, :, 0][:, None]
                return out

        tracks = {"10": track(10., 2., end=20, dx=.2), "2": track(2., 1., end=40, dx=.1), "7": track(7., 0., end=7)}
        ds = GraphWindows(store(tracks, targets=(("10", 7), ("2", 21), ("2", 7))))
        for kind, level, expected_shapes in (("clean", 0., [(1, 1), (1, 3)]), ("noise", .2, [(1, 1), (2, 3)])):
            batch = ds.batch([0, 1, 2], kind, level, seed=2, include_future=False)
            model = SlotReadout()
            raw = raw_for_batch(model, batch, "cpu")
            self.assertEqual(model.shapes, expected_shapes)
            self.assertEqual(raw[:, 0, 0].tolist(), [302., 100., 300.])
            xy = target_history(batch).numpy()
            expected = np.diff(xy.round(4), axis=1).astype(np.float32)[:, -1, 0]
            np.testing.assert_array_equal(raw[:, 0, 1], expected)

    def test_bundles_pass_shared_validator_and_replay_recorded_draws(self):
        # The benchmark CLI imports its sibling modules from this directory.
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "benchmark"))
        from validate import positions, verify_bundle

        class UnitGaussian(torch.nn.Module):
            def full_scene(self, increments, laplacian):
                return increments.new_zeros((len(increments), 12, increments.shape[2], 5))

        scene = moving_scene()
        dataset = GraphWindows(scene)
        clean = positions(scene, range(-7, 1))
        future = positions(scene, range(1, 13)).astype(np.float32)
        identity = {"checkpoint": "fixture.pt", "config_hash": "fixture-config", "checkpoint_sha256": "fixture-checkpoint"}
        with TemporaryDirectory() as tmp:
            run = Path(tmp) / "eth" / "seed_2"
            for tag, kind, level in CONDITIONS:
                with self.subTest(tag=tag):
                    path = export_condition(UnitGaussian(), dataset, run, seed=2, tag=tag, batch_size=2,
                                            device="cpu", run_hashes=identity)
                    self.assertEqual(path, run / "evaluation" / tag / f"predictions_test_{tag}.npz")
                    verify_bundle(path, scene, clean, future, 2, 20, kind, level, identity, None)
                    with np.load(path, allow_pickle=False) as bundle:
                        seeds = [json.loads(row)["sampling_seed"] for row in bundle["records"]]
                        self.assertEqual(seeds, [200006, 200006, 200007])
                        for start, stop in ((0, 2), (2, 3)):
                            generator = torch.Generator().manual_seed(seeds[start])
                            noise = torch.randn((stop - start, 20, 12, 2), generator=generator)
                            origin = bundle["observed"][start:stop, -1, None, None]
                            np.testing.assert_array_equal(bundle["samples"][start:stop],
                                                          origin + noise.cumsum(2).numpy())


if __name__ == "__main__":
    unittest.main()
