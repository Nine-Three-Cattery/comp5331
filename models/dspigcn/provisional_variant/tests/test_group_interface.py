import copy
from dataclasses import asdict
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

from dspigcn_candidate.group_interface import GroupPredictor
from dspigcn_candidate.model import DSPIGCN, ModelConfig
from dspigcn_candidate.sampling import GaussianMixtureAnchors


ROOT = Path(__file__).resolve().parents[4]


@unittest.skipUnless((ROOT / "benchmark/contract.py").is_file(), "Requires group repository")
class GroupInterfaceTests(unittest.TestCase):
    def test_actual_group_loader_future_invariance_and_batching(self):
        # Import the existing owner's modules alongside the candidate namespace.
        sys.path[:0] = [str(ROOT / "li_junhao_trajectronpp"), str(ROOT / "models/dspigcn")]
        from independent_tpp.data import SceneStore, WindowRecord
        from dspigcn.data import GraphDataset, collate
        from dspigcn.model import DSPIGCN as RegisteredDSPIGCN
        self.assertIsNot(RegisteredDSPIGCN, DSPIGCN)
        records = [WindowRecord(dict(scene_id="synthetic", source_file="toy",
                    target_id=str(i), scene_t=7, anchor_t=7, window_id=f"toy:{i}"))
                   for i in range(2)]
        tracks = {str(i): {t: np.array([t * .4, i * .8, 999., 999., 999., 999.])
                          for t in range(20 if i < 2 else 8)} for i in range(3)}
        store = SceneStore({records[0].scene_key: tracks}, records)
        changed = copy.deepcopy(store)
        for scene_tracks in changed.tracks.values():
            scene_tracks["future_only"] = {t: np.full(6, 888.) for t in range(8, 20)}
            for track in scene_tracks.values():
                for t in range(8, 20):
                    track[t] = np.full(6, -1e6)
                for state in track.values():
                    state[2:] = -999.
        config = ModelConfig(hidden=16, heads=2, prediction_layers=1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(7)
            model = DSPIGCN(config)
        anchors = GaussianMixtureAnchors(1)
        anchors.fitted.fill_(True)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "test.pt"
            torch.save(dict(model_config=asdict(config), model=model.state_dict(),
                            anchor_components=1, anchors=anchors.state_dict()), checkpoint)
            caller_rng = torch.random.get_rng_state().clone()
            predictor = GroupPredictor(checkpoint)
            for condition, level in [("clean", 0), ("noise", .2), ("missing", 1), ("missing", 2)]:
                dataset = GraphDataset(store, 2, condition, level)
                batch = collate([dataset.inputs(i) for i in range(2)])
                original = {key: value.clone() for key, value in batch.items() if isinstance(value, torch.Tensor)}
                expected = predictor.predict(batch, seed=7)
                other = GraphDataset(changed, 2, condition, level)
                changed_batch = collate([other.inputs(i) for i in range(2)])
                changed_batch["future"] = torch.full((2, 12, 2), float("nan"))
                torch.testing.assert_close(expected, predictor.predict(changed_batch, seed=7), rtol=0, atol=0)
                reversed_batch = collate([dataset.inputs(i) for i in [1, 0]])
                torch.testing.assert_close(expected.flip(0), predictor.predict(reversed_batch, seed=7), rtol=0, atol=0)
                batch["positions"] = torch.cat((batch["positions"], torch.full((2, 8, 1, 2), float("nan"))), 2)
                batch["mask"] = torch.cat((batch["mask"], torch.zeros(2, 1, dtype=torch.bool)), 1)
                torch.testing.assert_close(expected, predictor.predict(batch, seed=7), rtol=0, atol=0)
                self.assertEqual(expected.shape, (2, 20, 12, 2))
                for key, value in original.items():
                    if key not in ("positions", "mask"):
                        torch.testing.assert_close(value, batch[key], rtol=0, atol=0)
            torch.testing.assert_close(caller_rng, torch.random.get_rng_state(), rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
