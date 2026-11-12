"""Check the distance units and the separate ADE/FDE winners in the owner analysis."""
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze import displacement_scores


def test_distinct_ade_and_endpoint_winners_with_two_targets():
    truth = np.zeros((2, 12, 2))
    paths = np.zeros((2, 20, 12, 2))
    paths[0, :5, :-1, 1] = 3.0
    paths[0, 5:, -1, 0] = 8.4
    paths[1] = [0.6, 0.8]
    result = displacement_scores(paths, truth)
    for metric, expected in dict(min_ade_mean=0.85, min_fde_mean=0.5,
                                 mean_ade_mean=1.10625, mean_fde_mean=3.65).items():
        np.testing.assert_allclose(result[metric], expected, rtol=0, atol=1e-12)
