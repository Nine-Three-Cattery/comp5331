"""Replay 45 preselected formal paths with the authors' network and the students' trained weights."""

import argparse
from collections import Counter
import json
from pathlib import Path
import platform
import shutil
import sys
from tempfile import TemporaryDirectory

import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from li_junhao_trajectronpp.independent_tpp.protocol import (
    CONDITIONS, FOLDS, SEEDS, corruption_seed, digest_json, perturb_history, sha256, write_json,
)


ARTIFACTS = ROOT / "artifacts/social_stgcnn"
FIXTURES = ROOT / "benchmark/fixtures"
RUNTIME_INDEX = 64
TAGS = ("clean_0", "noise_020", "missing_2")
ABS_TOLERANCE_M = 1e-4


def read(path):
    return json.loads(path.read_text())


def environment():
    return {"python": platform.python_version(), "os": platform.system(), "architecture": platform.machine(),
            "torch": str(torch.__version__), "numpy": np.__version__, "threads": torch.get_num_threads(),
            "cpu_capability": torch.backends.cpu.get_cpu_capability()}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def prepare_fold(runs, data, directory, fold, window_id, lock, settings):
    from li_junhao_trajectronpp.independent_tpp.data import load_split

    store = load_split(fold, "test", str(data))
    require(store.identity["manifest_hash"] == lock["sha256"] and len(store.manifests) == lock["window_count"],
            f"{fold}: different test manifest")
    matches = [i for i, row in enumerate(store.manifests) if row.window_id == window_id]
    require(len(matches) == 1, f"{fold}: missing or duplicated runtime window {window_id}")
    index = matches[0]
    row = store.manifests[index]
    rosters = {}

    def roster(key, anchor):
        if (key, anchor) not in rosters:
            tracks = store.tracks[key]
            frames = range(anchor - 7, anchor + 1)
            rosters[key, anchor] = [person for person in sorted(tracks, key=int)
                                     if all(t in tracks[person] for t in frames)]
        return rosters[key, anchor]

    ids = roster(row.scene_key, row.anchor)
    slot = ids.index(row.target_id)
    clean = np.array([[store.state(row.scene_key, person, t)[:2] for person in ids]
                      for t in range(row.anchor - 7, row.anchor + 1)], dtype=np.float64)
    kinds = {tag: (kind, level) for tag, kind, level in CONDITIONS}
    cases = []
    for seed in SEEDS:
        run = runs / fold / f"seed_{seed}"
        folder = directory / fold / f"seed_{seed}"
        folder.mkdir(parents=True)
        config = read(run / "config.json")
        selection = read(run / "checkpoint_selection.json")
        expected = ("formal", fold, seed, "released_full_scene", settings["reference_revision"])
        require(tuple(config[k] for k in ("stage", "fold", "seed", "architecture", "reference_revision")) == expected,
                f"{fold}/{seed}: different formal run")
        production = config["environment"]
        require((production["torch"], production["os"], production["architecture"])
                == (str(torch.__version__), platform.system(), platform.machine()),
                "Prepare with the production Torch version, operating system and CPU architecture")
        torch.set_num_threads(production["threads"])
        require(selection["selection_split"] == "val" and not selection["test_used_for_selection"],
                f"{fold}/{seed}: selected weights have no validation decision")
        checkpoint = selection["selected"]["checkpoint"]
        weight_hash = sha256(run / checkpoint)
        config_hash = digest_json(config)
        shutil.copyfile(run / checkpoint, folder / "selected.pt")
        for tag in TAGS:
            kind, level = kinds[tag]
            bundle_path = run / "evaluation" / tag / f"predictions_test_{tag}.npz"
            with np.load(bundle_path, allow_pickle=False) as bundle:
                meta = json.loads(str(bundle["metadata"]))
                identity = {**store.identity, "config_hash": config_hash, "checkpoint": checkpoint,
                            "checkpoint_sha256": weight_hash, "seed": seed, "sample_count": 20,
                            "condition": kind, "level": level, "batch_size": config["export_batch_size"]}
                require(all(meta[k] == v for k, v in identity.items()), f"{fold}/{seed}/{tag}: bundle identity")
                record = json.loads(str(bundle["records"][index]))
                require((record["window_id"], record["target_id"], record["condition"], record["seed"])
                        == (window_id, row.target_id, kind, seed), f"{fold}/{seed}/{tag}: wrong row")
                scene = clean.copy()
                scene[:, slot], mask = perturb_history(clean[:, slot], kind, level, corruption_seed(seed, index))
                np.testing.assert_array_equal(bundle["observed"][index], scene[:, slot].astype(np.float32))
                np.testing.assert_array_equal(bundle["observed_mask"][index], mask)
                stored = bundle["samples"][index]
            require(stored.shape == (20, 12, 2) and stored.dtype == np.float32 and np.isfinite(stored).all(),
                    f"{fold}/{seed}/{tag}: invalid stored forecasts")
            batch_size = meta["batch_size"]
            ordinal, offset = divmod(index, batch_size)
            start = ordinal * batch_size
            batch_rows = store.manifests[start:start + batch_size]
            batch_keys = [(r.scene_key, r.anchor) for r in batch_rows]
            graph_keys = dict.fromkeys(batch_keys) if kind == "clean" else batch_keys
            width_counts = Counter(len(roster(*key)) for key in graph_keys)
            sampling_seed = record["sampling_seed"]
            require(sampling_seed == 100003 * seed + ordinal, f"{fold}/{seed}/{tag}: unexpected sampling seed")
            generator = torch.Generator(device="cpu").manual_seed(sampling_seed)
            noise_shape = (len(batch_rows), 20, 12, 2)
            noise = torch.randn(noise_shape, dtype=torch.float32, device="cpu", generator=generator)[offset].numpy()
            np.savez_compressed(folder / f"{tag}.npz", scene=scene, anchor=scene[-1, slot].astype(np.float32),
                                noise=noise, stored=stored)
            cases.append({"fold": fold, "seed": seed, "condition": tag, "window_id": window_id,
                          "manifest_index": index, "manifest_hash": store.identity["manifest_hash"],
                          "input_hash": store.identity["input_hash"], "agent_ids": ids, "target_id": row.target_id,
                          "target_slot": slot, "checkpoint_sha256": weight_hash, "config_hash": config_hash,
                          "selected_epoch": selection["selected"]["epoch"], "prediction_sha256": sha256(bundle_path),
                          "sampling_seed": sampling_seed, "noise_shape": noise_shape, "noise_batch_offset": offset,
                          "production_graphs_of_width": width_counts[len(ids)],
                          "production_environment": {k: production[k] for k in
                                                     ("device", "torch", "os", "architecture", "cpu", "threads")}})
    return cases


def prepare(args):
    fixed = read(FIXTURES / "runtime_windows.json")
    lock = read(FIXTURES / "manifests.json")["folds"]
    settings = read(Path(__file__).with_name("configs") / "experiment.json")
    args.pack.parent.mkdir(parents=True, exist_ok=True)
    require(not args.pack.exists(), f"Use a new input directory: {args.pack}")
    with TemporaryDirectory(dir=args.pack.parent) as temporary:
        directory = Path(temporary) / "inputs"
        cases = []
        for fold in FOLDS:
            cases.extend(prepare_fold(args.runs, args.data, directory, fold, fixed[fold][RUNTIME_INDEX],
                                      lock[fold]["test"], settings))
        write_json(directory / "manifest.json",
                   {"release": settings["reference_revision"], "runtime_index": RUNTIME_INDEX,
                    "preparation_environment": environment(), "cases": cases})
        directory.rename(args.pack)
    print(f"TRAINED_REFERENCE_INPUTS_OK cases={len(cases)} weights={len(FOLDS)*len(SEEDS)} directory={args.pack}")


def reference_graph(utils, scene):
    rounded = np.around(scene, decimals=4)
    motion = np.zeros_like(rounded)
    motion[1:] = np.diff(rounded, axis=0)
    positions = torch.from_numpy(rounded.transpose(1, 2, 0).copy()).float()
    increments = torch.from_numpy(motion.transpose(1, 2, 0).copy()).float()
    # All five fixed windows have at least three fully observed actors.
    require(len(positions) >= 2, "The fixed multi-actor input has an unexpected singleton roster")
    return utils.seq_to_graph(positions, increments)


def upstream_paths(reference, metrics, graph, laplacian, slot, anchor, noise, graph_count=1):
    scenes = graph.unsqueeze(0).expand(graph_count, -1, -1, -1).contiguous()
    features = scenes.permute(0, 3, 1, 2)
    parameters, _ = reference(features, laplacian)
    target = parameters.permute(0, 2, 3, 1)[0, :, slot]
    scales = target[:, 2:4].exp()
    covariance = torch.diag_embed(scales.square())
    covariance[:, 0, 1] = covariance[:, 1, 0] = target[:, 4].tanh() * scales.prod(-1)
    density = torch.distributions.MultivariateNormal(target[:, :2], covariance_matrix=covariance)
    draws = density.loc + (density.scale_tril @ noise.unsqueeze(-1)).squeeze(-1)
    return np.stack([metrics.nodes_rel_to_nodes_abs(draw.numpy()[:, None], anchor[None]).reshape(12, 2)
                     for draw in draws])


@torch.inference_mode()
def compare(args):
    from models.social_stgcnn.check_reference import INACTIVE, RELEASE, external_modules, reference_name
    import scipy

    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    manifest = read(args.pack / "manifest.json")
    require(manifest["release"] == RELEASE and manifest["runtime_index"] == RUNTIME_INDEX, "Wrong input scope")
    fixed = read(FIXTURES / "runtime_windows.json")
    expected = [(f, s, t, fixed[f][RUNTIME_INDEX]) for f in FOLDS for s in SEEDS for t in TAGS]
    actual = [(c["fold"], c["seed"], c["condition"], c["window_id"]) for c in manifest["cases"]]
    require(actual == expected, "The input set differs from the 45 registered cases")
    release, utils, metrics = external_modules(args.reference)
    models = {}
    rows = []
    for case in manifest["cases"]:
        fold, seed, tag = case["fold"], case["seed"], case["condition"]
        folder = args.pack / fold / f"seed_{seed}"
        if (fold, seed) not in models:
            require(sha256(folder / "selected.pt") == case["checkpoint_sha256"], f"{fold}/{seed}: wrong weights")
            selected = torch.load(folder / "selected.pt", map_location="cpu", weights_only=True)
            oracle = release.social_stgcnn(n_stgcnn=1, n_txpcnn=5)
            missing, extra = oracle.load_state_dict({reference_name(k): v for k, v in selected.items()}, strict=False)
            require(set(missing) == INACTIVE and not extra, f"{fold}/{seed}: incomplete mapping of trained weights")
            models[fold, seed] = oracle.eval()
        row = dict(case)
        try:
            with np.load(folder / f"{tag}.npz", allow_pickle=False) as pack:
                graph, laplacian = reference_graph(utils, pack["scene"])
                noise, anchor, stored = torch.from_numpy(pack["noise"]), pack["anchor"], pack["stored"]
                inputs = (models[fold, seed], metrics, graph, laplacian, case["target_slot"], anchor, noise)
                paths = upstream_paths(*inputs)
                require(np.isfinite(paths).all(), "External sampler produced non-finite paths")
                diff = np.abs(paths.astype(np.float64) - stored)
                row["max_abs_coordinate_error_m"] = float(diff.max())
                row["passed"] = row["max_abs_coordinate_error_m"] <= ABS_TOLERANCE_M
                if not row["passed"]:
                    where = np.unravel_index(diff.argmax(), diff.shape)
                    row["worst_sample_step_coordinate_zero_based"] = [int(i) for i in where]
                    groups = case["production_graphs_of_width"]
                    if groups > 1:
                        batched = upstream_paths(*inputs, graph_count=groups)
                        row["same_graph_repeated_to_production_width_batch_max_error_m"] = float(
                            np.abs(batched.astype(np.float64) - stored).max())
        except (ValueError, RuntimeError, FloatingPointError) as err:
            row.update(passed=False, error=str(err))
        rows.append(row)
    count = sum(r["passed"] for r in rows)
    receipt = {"passed": count == len(expected), "upstream_revision": RELEASE,
               "runtime_fixture_index_zero_based": RUNTIME_INDEX, "absolute_coordinate_tolerance_m": ABS_TOLERANCE_M,
               "reference_graphs_per_forward": 1, "samples_per_case": 20,
               "preparation_environment": manifest["preparation_environment"],
               "reference_environment": {**environment(), "networkx": utils.nx.__version__, "scipy": scipy.__version__},
               "summary": {"cases": len(rows), "passed": count, "failed": len(rows) - count,
                           "max_abs_coordinate_error_m": max((r["max_abs_coordinate_error_m"] for r in rows
                                                              if "max_abs_coordinate_error_m" in r), default=None)},
               "cases": rows}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, receipt)
    if not receipt["passed"]:
        raise SystemExit(f"Trained reference comparison failed for {len(rows)-count} cases. See {args.out}")
    print(f"TRAINED_REFERENCE_OK cases={count} max_abs_m={receipt['summary']['max_abs_coordinate_error_m']:.8g}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    production = phases.add_parser("prepare", help="Extract fixed cases and production RNG draws")
    production.add_argument("--runs", type=Path, default=ARTIFACTS / "formal/runs")
    production.add_argument("--data", type=Path, default=ROOT / "data/causal_v2")
    production.add_argument("--pack", type=Path, required=True)
    external = phases.add_parser("compare", help="Replay with the external release")
    external.add_argument("--pack", type=Path, required=True)
    external.add_argument("--reference", type=Path, required=True)
    external.add_argument("--out", type=Path, default=ARTIFACTS / "trained_reference.json")
    args = parser.parse_args()
    (prepare if args.phase == "prepare" else compare)(args)


if __name__ == "__main__":
    main()
