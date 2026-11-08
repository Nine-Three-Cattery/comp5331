import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from dspigcn_candidate.data import DataConfig, SCENES, load_fold, collate, read_tracks, windows
from dspigcn_candidate.evaluation import metrics, motion, perturb
from dspigcn_candidate.losses import gaussian_nll, polynomial_motion, training_loss
from dspigcn_candidate.model import DSPIGCN, ModelConfig, differences
from dspigcn_candidate.sampling import GaussianMixtureAnchors, anchor_sample


class MetricTests(unittest.TestCase):
    def test_known_sample_minima_are_not_framewise(self):
        target = np.zeros((2, 1, 2))
        pred = np.zeros((2, 2, 1, 2))
        pred[0, :, 0, 0] = [0, 4]
        pred[1, :, 0, 0] = [3, 0]
        result = metrics(pred, target)
        self.assertEqual(result["minADE"][0], 1.5)
        self.assertEqual(result["minFDE"][0], 0)
        self.assertEqual(result["meanADE"][0], 1.75)
        self.assertEqual(result["meanFDE"][0], 2)
        self.assertEqual(result["energy_score"][0], 2.25)

    def test_perfect_deterministic(self):
        target = np.ones((12, 3, 2))
        for v in metrics(target[None], target).values():
            np.testing.assert_equal(v, 0)

    def test_bad_inputs(self):
        with self.assertRaises(ValueError):
            metrics(np.zeros((2, 12, 1, 2)), np.zeros((12, 2, 2)))

    def test_corruption_causal_and_order_independent(self):
        obs = np.arange(32, dtype=np.float32).reshape(8, 2, 2)
        actual = perturb(obs, ["a", "b"], sigma=.2, seed=7)
        swapped = perturb(obs[:, ::-1], ["b", "a"], sigma=.2, seed=7)
        np.testing.assert_array_equal(actual, swapped[:, ::-1])
        dirty = obs.copy()
        dirty[-2:] = 1000
        np.testing.assert_equal(perturb(dirty, ["a", "b"], missing=2), obs)
        np.testing.assert_equal(perturb(obs, ["a", "b"]), obs)

    def test_motion_units_and_pairs(self):
        times = np.arange(20) * .4
        trajectories = np.zeros((20, 2, 2))
        trajectories[:, :, 0] = times[:, None] * 2
        trajectories[:, 1, 1] = .1
        values, pairs = motion(trajectories[None, 8:], trajectories[:8], .4)
        np.testing.assert_allclose(values["speed_mean"], 2)
        np.testing.assert_allclose(values["acceleration_mean"], 0, atol=1e-12)
        self.assertEqual(pairs["close_pairs"], 12)
        self.assertEqual(pairs["pair_timesteps"], 12)


class ModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        self.model = DSPIGCN(ModelConfig(hidden=16, heads=2, prediction_layers=2)).eval()

    def test_shapes_translation_permutation_and_padding(self):
        obs = torch.randn(1, 8, 3, 2)
        output = self.model(obs)
        self.assertEqual(output["displacement"].shape, (1, 12, 3, 5))
        translated = self.model(obs + torch.tensor([10., -5.]))
        torch.testing.assert_close(translated["displacement"], output["displacement"], atol=2e-5, rtol=2e-5)
        order = [2, 0, 1]
        torch.testing.assert_close(self.model(obs[:, :, order])["displacement"], output["displacement"][:, :, order], atol=2e-5, rtol=2e-5)
        padded = torch.cat((obs, torch.full((1, 8, 2, 2), 999.)), 2)
        mask = torch.tensor([[True, True, True, False, False]])
        torch.testing.assert_close(self.model(padded, mask)["displacement"][:, :, :3], output["displacement"], atol=2e-5, rtol=2e-5)
        self.assertEqual(self.model.predict(obs[:, :, :1]).shape, (20, 1, 12, 1, 2))

    def test_all_components_receive_finite_gradients(self):
        self.model.train()
        output = self.model(torch.randn(2, 8, 3, 2))
        loss = training_loss(output, torch.randn(2, 12, 3, 2))["total"]
        loss.backward()
        for name, param in self.model.named_parameters():
            self.assertIsNotNone(param.grad, name)
            self.assertTrue(torch.isfinite(param.grad).all(), name)
        for module in (self.model.displacement_stream, self.model.velocity_stream, self.model.physics):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0)

    def test_derivatives_and_units(self):
        coefficients = torch.zeros(1, 1, 2, 4)
        # x(t)=2t+3t^2; coefficients refer to normalized time t/4.8.
        coefficients[0, 0, 0, :2] = torch.tensor([2 * 4.8, 3 * 4.8**2])
        pos, vel = polynomial_motion(coefficients, 12, .4)
        time = torch.arange(1, 13) * .4
        torch.testing.assert_close(pos[0, :, 0, 0], 2*time + 3*time**2)
        torch.testing.assert_close(vel[0, :, 0, 0], 2 + 6*time)
        x = torch.arange(8)[None, :, None, None].float() * .8
        torch.testing.assert_close(differences(x, .4), torch.ones_like(x) * 2)

    def test_nll_matches_torch_distribution(self):
        raw = torch.tensor([[[[.2, -.1, .3, -.2, .4]]]])
        target = torch.tensor([[[[1., 2.]]]])
        sx, sy = raw[..., 2:4].exp().flatten()
        rho = raw[..., 4].tanh().item() * .999
        cov = torch.tensor([[sx*sx, rho*sx*sy], [rho*sx*sy, sy*sy]])
        dist = torch.distributions.MultivariateNormal(raw[..., :2].flatten(), cov)
        torch.testing.assert_close(gaussian_nll(raw, target).squeeze(), -dist.log_prob(target.flatten()))

    def test_sampler_reproducibility_and_moments(self):
        raw = torch.zeros(1, 12, 2, 5)
        a = anchor_sample(raw, 10000, torch.Generator().manual_seed(3))
        b = anchor_sample(raw, 10000, torch.Generator().manual_seed(3))
        torch.testing.assert_close(a, b)
        torch.testing.assert_close(a.mean((0, 1, 2, 3)), torch.zeros(2), atol=.01, rtol=0)
        torch.testing.assert_close(a.var((0, 1, 2, 3)), torch.ones(2), atol=.02, rtol=0)
        self.assertFalse(torch.equal(a[:, :, :, 0], a[:, :, :, 1]))

    def test_train_only_gmm_persistence(self):
        rng = np.random.default_rng(2)
        dataset = [{"observations": rng.normal(size=(8, 10, 2)), "future": rng.normal(size=(12, 10, 2))} for _ in range(3)]
        mixture = GaussianMixtureAnchors(2)
        mixture.fit(dataset)
        weights = mixture.weights[:, None, None]
        torch.testing.assert_close((weights * mixture.means).sum(0), torch.zeros(12, 2), atol=1e-6, rtol=0)
        torch.testing.assert_close((weights * (mixture.scales**2 + mixture.means**2)).sum(0), torch.ones(12, 2))
        copy = GaussianMixtureAnchors(2)
        copy.load_state_dict(mixture.state_dict())
        raw = torch.zeros(1, 12, 1, 5)
        torch.testing.assert_close(mixture.sample(raw, generator=torch.Generator().manual_seed(4)),
                                   copy.sample(raw, generator=torch.Generator().manual_seed(4)))


class DataTests(unittest.TestCase):
    def test_splits_population_and_identifiers(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [[t*10, p, t*.4, p*.5] for t in range(250) for p in range(2)]
            for names in SCENES.values():
                for name in names:
                    np.savetxt(root / name, rows)
            train = load_fold(root, "eth", "train")
            val = load_fold(root, "eth", "val")
            test = load_fold(root, "eth", "test")
            self.assertEqual(len(test), 231)
            self.assertEqual(test.manifest["trajectories"], 462)
            self.assertEqual(test[0]["ids"], ["biwi_eth.txt:0:0", "biwi_eth.txt:0:1"])
            train_rows = {(x["source"], f) for x in train for f in x["frames"]}
            val_rows = {(x["source"], f) for x in val for f in x["frames"]}
            self.assertFalse(train_rows & val_rows)
            self.assertNotIn("biwi_eth.txt", train.manifest["files_sha256"])
            self.assertEqual(collate([test[0], test[1]])["observations"].shape, (2, 8, 2, 2))

    def test_missing_frames_and_duplicate_rows(self):
        data = np.array([[t*10, 1, t, 0] for t in range(30) if t != 15], float)
        self.assertEqual(windows(data, "x", np.unique(data[:, 0]), DataConfig()), [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.txt"
            np.savetxt(path, [[0, 1, 0, 0], [0, 1, 0, 0]])
            with self.assertRaises(ValueError):
                read_tracks(path)


if __name__ == "__main__":
    unittest.main()
