"""Independent metrics and deterministic, identity-addressed perturbations."""
import hashlib
import numpy as np


CONDITIONS = {"clean": (0., 0), "noise_0.05": (0.05, 0), "noise_0.10": (0.10, 0),
              "noise_0.20": (0.20, 0), "missing_1": (0., 1), "missing_2": (0., 2)}


def stable_seed(identity, seed):
    return int.from_bytes(hashlib.sha256(f"{seed}:{identity}".encode()).digest()[:8], "little")


def perturb(observations, ids, sigma=0., missing=0, seed=2026):
    if sigma < 0 or missing not in (0, 1, 2) or observations.shape[0] - missing < 2:
        raise ValueError("Invalid corruption parameters")
    result = np.array(observations, copy=True)
    if sigma:
        for person, identity in enumerate(ids):
            rng = np.random.Generator(np.random.PCG64(stable_seed(identity, seed)))
            result[:, person] += rng.normal(0, sigma, result[:, person].shape)
    if missing:
        last = len(result) - missing - 1
        velocity = result[last] - result[last - 1]
        for offset in range(1, missing + 1):
            result[last + offset] = result[last] + offset * velocity
    return result


def metrics(predictions, target):
    """[K,T,N,2] vs [T,N,2] -> one value per pedestrian.

    Minima select a whole sample per pedestrian, separately for ADE and FDE.
    Energy score uses Euclidean norm of the flattened 2T coordinate vector and
    the empirical distribution (V-statistic, including diagonal pairs).
    """
    predictions, target = np.asarray(predictions), np.asarray(target)
    if predictions.ndim != 4 or target.shape != predictions.shape[1:] or predictions.shape[-1] != 2:
        raise ValueError("Expected predictions [K,T,N,2], target [T,N,2]")
    if min(predictions.shape) == 0 or not np.isfinite(predictions).all() or not np.isfinite(target).all():
        raise ValueError("Metrics require nonempty, finite trajectories")
    error = np.linalg.norm(predictions - target[None], axis=-1)
    ade, fde = error.mean(1), error[:, -1]
    k = len(predictions)
    flat = predictions.transpose(0, 2, 1, 3).reshape(k, target.shape[1], -1)
    flat_target = target.transpose(1, 0, 2).reshape(target.shape[1], -1)
    pair_energy = np.zeros(target.shape[1], np.float64)
    diversity = np.zeros_like(pair_energy)
    for i in range(k):
        pair_energy += np.linalg.norm(flat[i] - flat, axis=-1).sum(0)
        diversity += np.linalg.norm(predictions[i] - predictions, axis=-1).mean(1).sum(0)
    return {"minADE": ade.min(0), "minFDE": fde.min(0), "meanADE": ade.mean(0),
            "meanFDE": fde.mean(0), "diversity": diversity / max(k * (k - 1), 1),
            "energy_score": np.linalg.norm(flat - flat_target, axis=-1).mean(0) - pair_energy / (2 * k * k)}


def motion(predictions, observations, dt=0.4, thresholds=None):
    thresholds = thresholds or {}
    history = np.broadcast_to(observations[-2:], (len(predictions), 2, *observations.shape[1:]))
    joined = np.concatenate((history, predictions), axis=1)
    velocity = np.diff(joined, axis=1) / dt
    acceleration = np.diff(velocity, axis=1) / dt
    speed = np.linalg.norm(velocity[:, 1:], axis=-1)
    accel = np.linalg.norm(acceleration, axis=-1)
    jerk = np.linalg.norm(np.diff(acceleration, axis=1) / dt, axis=-1)
    result = {"speed_mean": speed.mean((0, 1)), "acceleration_mean": accel.mean((0, 1)),
              "jerk_mean": jerk.mean((0, 1))}
    for name, values in (("speed", speed), ("acceleration", accel), ("jerk", jerk)):
        if name in thresholds:
            result[f"{name}_exceedance"] = (values > thresholds[name]).mean((0, 1))
    # Pair-level denominator excludes self pairs and other scenes/timestamps.
    n = predictions.shape[2]
    i, j = np.triu_indices(n, 1)
    distances = np.linalg.norm(predictions[:, :, i] - predictions[:, :, j], axis=-1)
    close = int((distances < thresholds.get("close_distance", 0.2)).sum())
    return result, {"close_pairs": close, "pair_timesteps": distances.size}


def training_thresholds(dataset, dt):
    values = {"speed": [], "acceleration": [], "jerk": [], "close_distance": []}
    for ex in dataset:
        trajectory = np.concatenate((ex["observations"], ex["future"]))
        v = np.diff(trajectory, axis=0) / dt
        a = np.diff(v, axis=0) / dt
        j = np.diff(a, axis=0) / dt
        for name, array in (("speed", v), ("acceleration", a), ("jerk", j)):
            values[name].append(np.linalg.norm(array, axis=-1).ravel())
        i, k = np.triu_indices(trajectory.shape[1], 1)
        if len(i):
            values["close_distance"].append(np.linalg.norm(trajectory[:, i] - trajectory[:, k], axis=-1).ravel())
    return {name: float(np.quantile(np.concatenate(parts), 0.01 if name == "close_distance" else 0.99))
            if parts else 0.2 for name, parts in values.items()}
