from __future__ import print_function

import numpy as np


def sequence_distance(first, second):
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    return float(np.sqrt(np.mean(np.sum((first - second) ** 2, axis=-1))))


def distribution_metrics(samples, target):
    errors = np.sqrt(np.mean(np.sum((samples - target) ** 2, axis=-1), axis=-1))
    count = len(samples)
    if count < 2:
        return float(errors.mean()), 0.0
    first, second = np.triu_indices(count, 1)
    difference = samples[first] - samples[second]
    distances = np.sqrt(np.mean(np.sum(difference ** 2, axis=-1), axis=-1))
    diversity = float(distances.mean())
    return float(errors.mean() - 0.5 * diversity), diversity


def energy_score(samples, target):
    return distribution_metrics(np.asarray(samples), np.asarray(target))[0]


def mean_pairwise_distance(samples):
    samples = np.asarray(samples)
    return distribution_metrics(samples, samples[0])[1]


def _motion_arrays(samples, observed_last, dt):
    samples = np.asarray(samples, dtype=np.float64)
    start = np.asarray(observed_last, dtype=np.float64).reshape(1, 1, 2)
    full = np.concatenate([np.repeat(start, samples.shape[0], axis=0), samples], axis=1)
    velocity = np.diff(full, axis=1) / dt
    acceleration = np.diff(velocity, axis=1) / dt
    jerk = np.diff(acceleration, axis=1) / dt
    speed = np.linalg.norm(velocity, axis=-1)
    acceleration_norm = np.linalg.norm(acceleration, axis=-1)
    jerk_norm = np.linalg.norm(jerk, axis=-1)
    return speed, acceleration_norm, jerk_norm


def compute_window_metrics(samples, target, observed_last, dt):
    samples = np.asarray(samples, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if samples.ndim != 3 or samples.shape[-1] != 2:
        raise ValueError("samples must have shape [K,T,2]")
    if target.shape != samples.shape[1:]:
        raise ValueError("target shape does not match samples")
    ade = np.linalg.norm(samples - target[None, :, :], axis=-1).mean(axis=-1)
    fde = np.linalg.norm(samples[:, -1, :] - target[-1, :], axis=-1)
    speed, acceleration, jerk = _motion_arrays(samples, observed_last, dt)

    def summary(values, prefix):
        return {
            prefix + "_mean": float(np.mean(values)),
            prefix + "_p95": float(np.percentile(values, 95)),
            prefix + "_p99": float(np.percentile(values, 99)),
            prefix + "_max": float(np.max(values)),
        }

    score, diversity = distribution_metrics(samples, target)
    result = {
        "min_ade": float(np.min(ade)),
        "mean_ade": float(np.mean(ade)),
        "min_fde": float(np.min(fde)),
        "mean_fde": float(np.mean(fde)),
        "energy_score": score,
        "sample_diversity": diversity,
        "sample_count": int(samples.shape[0]),
    }
    result.update(summary(speed, "speed"))
    result.update(summary(acceleration, "acceleration"))
    result.update(summary(jerk, "jerk"))
    result["endpoint_spread"] = float(
        np.trace(np.cov(samples[:, -1, :].T)) if samples.shape[0] > 1 else 0.0
    )
    return result


def add_threshold_flags(metrics, thresholds):
    for quantity in ("speed", "acceleration", "jerk"):
        threshold = thresholds.get(quantity)
        if threshold is None:
            continue
        metrics[quantity + "_violation_rate"] = float(
            metrics[quantity + "_p99"] > float(threshold)
        )
    return metrics


def aggregate_rows(rows):
    if not rows:
        return {}
    numeric = {}
    keys = set()
    for row in rows:
        keys.update(row.keys())
    for key in sorted(keys):
        values = []
        for row in rows:
            value = row.get(key)
            if isinstance(value, (int, float, np.integer, np.floating)):
                values.append(float(value))
        if values:
            numeric[key + "_mean"] = float(np.mean(values))
            numeric[key + "_std"] = float(np.std(values))
            numeric[key + "_count"] = int(len(values))
    return numeric
