"""Numerically compare the owner model with an external checkout of the CVPR release."""

import argparse
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import torch
from torch import nn
import numpy as np
import scipy

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from models.social_stgcnn.model import SocialSTGCNN, bivariate_nll, sample_world, scene_geometry
from li_junhao_trajectronpp.independent_tpp.protocol import sha256


RELEASE = "333d3a57b4d2705e129b21aefefa09c79b2b9ae1"
MODULES = {
    "message_projection": "st_gcns.0.gcn.conv", "graph_norm": "st_gcns.0.tcn.0",
    "graph_activation": "st_gcns.0.tcn.1", "temporal_filter": "st_gcns.0.tcn.2",
    "temporal_norm": "st_gcns.0.tcn.3", "bypass_projection": "st_gcns.0.residual.0",
    "bypass_norm": "st_gcns.0.residual.1", "block_activation": "st_gcns.0.prelu",
    "expand": "tpcnns.0", "expand_activation": "prelus.0", "readout": "tpcnn_ouput",
    **{f"refinements.{i}": f"tpcnns.{i+1}" for i in range(3)},
    **{f"refinement_activations.{i}": f"prelus.{i+1}" for i in range(3)},
}
INACTIVE = {"tpcnns.4.weight", "tpcnns.4.bias", "prelus.4.weight"}


def external_modules(path):
    git = ["git", "-C", str(path)]
    if subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip() != RELEASE:
        raise ValueError(f"Reference checkout must be at {RELEASE}")
    subprocess.run([*git, "diff", "--quiet", "HEAD", "--", "model.py", "utils.py", "metrics.py",
                    "train.py", "test.py", "datasets/eth/test/biwi_eth.txt"], check=True)
    modules = []
    for name in ("model", "utils", "metrics"):
        spec = importlib.util.spec_from_file_location(f"release_{name}", path / f"{name}.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        modules.append(module)
    return modules


def source_fixture(utils):
    """Pass 20 metric frames through the release's own file reader and dataset."""
    t = np.arange(20, dtype=np.float64)
    first = np.stack([2.123471 + .13791*t + .00123*t*t, -1.398267 + .05273*t], axis=-1)
    other = np.stack([-.72819 - .09837*t, .23463 + .18314*t + .00218*t*t], axis=-1)
    turn = np.stack([1.791376 + .10953*t, .67147 + .01607*t*t - .07213*t], axis=-1)
    # Nodes 0 and 1 have identical rounded displacements despite distinct locations.
    world = np.stack([first, first + [1.2137, -.6124], other, turn], axis=1)
    identifiers = [2, 9, 15, 31]
    with TemporaryDirectory() as folder:
        path = Path(folder) / "scene.txt"
        with path.open("w") as stream:
            for frame, positions in enumerate(world):
                for i in (2, 0, 3, 1):
                    x, y = positions[i]
                    stream.write(f"{frame}\t{identifiers[i]}\t{x:.10f}\t{y:.10f}\n")
        dataset = utils.TrajectoryDataset(folder, obs_len=8, pred_len=12)
    if len(dataset) != 1:
        raise AssertionError("The external fixture must contain one 20-frame scene")
    return torch.from_numpy(world), dataset[0]


def recorded_eth_fixture(utils, reference):
    """First 20 consecutive release ETH frames with three complete trajectories."""
    rows = np.loadtxt(reference / "datasets/eth/test/biwi_eth.txt")
    rows = rows[(rows[:, 0] >= 2860) & (rows[:, 0] <= 3050)]
    with TemporaryDirectory() as folder:
        np.savetxt(Path(folder) / "eth.txt", rows, fmt="%.10f", delimiter="\t")
        dataset = utils.TrajectoryDataset(folder, obs_len=8, pred_len=12)
    ids = [51, 52, 56]
    world = np.stack([rows[rows[:, 1] == agent, 2:4] for agent in ids], axis=1)
    if len(dataset) != 1 or world.shape != (20, 3, 2):
        raise AssertionError("Unexpected released ETH fixture population")
    return torch.from_numpy(world), dataset[0]


def reference_name(name):
    module, field = name.rsplit(".", 1)
    return f"{MODULES[module]}.{field}"


def compare(a, b, *, atol=3e-6, rtol=3e-5):
    torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    return (a - b).abs().max().item()


def models_for_mode(release, training):
    torch.manual_seed(731)
    external = release.social_stgcnn(n_stgcnn=1, n_txpcnn=5)
    for module in external.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.running_mean.copy_(torch.linspace(-.2, .18, module.num_features))
            module.running_var.copy_(torch.linspace(.7, 1.3, module.num_features))
            module.num_batches_tracked.fill_(4)
    owner = SocialSTGCNN()
    state = external.state_dict()
    translated = {name: state[reference_name(name)] for name in owner.state_dict()}
    if set(state) - {reference_name(name) for name in translated} != INACTIVE:
        raise AssertionError("The weight map does not cover exactly the executed release network")
    owner.load_state_dict(translated)
    return owner.train(training), external.train(training)


def forward_check(release, increments, laplacian, training):
    model, oracle = models_for_mode(release, training)
    x = increments.clone().requires_grad_()
    a = laplacian.clone().requires_grad_()
    xr = increments.clone().requires_grad_()
    ar = laplacian.clone().requires_grad_()
    actual = model.full_scene(x, a)
    # These are the input and output permutations from the released test driver.
    prediction, _ = oracle(xr.permute(0, 3, 1, 2), ar[0])
    expected = prediction.permute(0, 2, 3, 1)
    errors = {"output": compare(actual, expected)}
    probe = torch.arange(actual.numel(), dtype=actual.dtype).reshape_as(actual).mul(.37).cos()
    (actual * probe).mean().backward()
    (expected * probe).mean().backward()
    errors["input_gradient"] = compare(x.grad, xr.grad)
    errors["laplacian_gradient"] = compare(a.grad, ar.grad)
    oracle_params = dict(oracle.named_parameters())
    errors["parameter_gradients"] = max(compare(parameter.grad, oracle_params[reference_name(name)].grad)
                                        for name, parameter in model.named_parameters())
    if any(oracle_params[name].grad is not None for name in INACTIVE):
        raise AssertionError("A presumed inactive release parameter received a gradient")
    buffers = dict(oracle.named_buffers())
    errors["batchnorm_buffers"] = max(compare(value, buffers[reference_name(name)])
                                      for name, value in model.named_buffers())
    return actual.detach(), expected.detach(), errors


def distribution_check(metrics, raw, reference_raw, actual_future, world_anchor, release_anchor):
    # An extreme label exercises the release's density floor and its zero gradient.
    target = actual_future.clone()
    target[0, 0] += 100.0
    ours = raw[0].clone().requires_grad_()
    theirs = reference_raw[0].clone().requires_grad_()
    loss = bivariate_nll(ours, target)
    ref_loss = metrics.bivariate_loss(theirs, target)
    errors = {"loss": compare(loss, ref_loss)}
    loss.backward()
    ref_loss.backward()
    errors["loss_gradient"] = compare(ours.grad, theirs.grad)

    # Drive the released evaluator's multivariate normal and the owner sampler
    # with the same standard normals. The release integrates each node in metres.
    params = reference_raw[0]
    scales = params[..., 2:4].exp()
    covariance = torch.diag_embed(scales.square())
    covariance[..., 0, 1] = covariance[..., 1, 0] = params[..., 4].tanh() * scales.prod(-1)
    distribution = torch.distributions.MultivariateNormal(params[..., :2], covariance)
    noise, paths, original_paths = [], [], []
    torch.manual_seed(281)
    for _ in range(20):
        initial_state = torch.get_rng_state()
        draw = distribution.sample()
        after_sample = torch.get_rng_state()
        torch.set_rng_state(initial_state)
        noise.append(torch.empty(draw.shape, dtype=draw.dtype).normal_())
        if not torch.equal(after_sample, torch.get_rng_state()):
            raise AssertionError("The sampler and fixed-noise replay consumed different draws")
        paths.append(metrics.nodes_rel_to_nodes_abs(draw.numpy(), world_anchor.numpy()).reshape(draw.shape))
        original_paths.append(metrics.nodes_rel_to_nodes_abs(draw.numpy(), release_anchor.numpy()).reshape(draw.shape))
    by_actor = torch.stack(noise).permute(2, 0, 1, 3)
    result = sample_world(raw[0].transpose(0, 1), world_anchor, by_actor)
    expected = torch.from_numpy(np.stack(paths)).permute(2, 0, 1, 3)
    original = torch.from_numpy(np.stack(original_paths)).permute(2, 0, 1, 3)
    errors["sample_paths_common_anchor_m"] = compare(result, expected, atol=5e-6)
    shifted = original + (world_anchor - release_anchor)[:, None, None]
    errors["sample_paths_after_known_anchor_shift_m"] = compare(result, shifted, atol=5e-6)
    return errors


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True, type=Path, help="Clean external Git checkout at the pinned release")
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/social_stgcnn/source_parity.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    release, utils, metrics = external_modules(args.reference)
    world, example = source_fixture(utils)
    absolute, _, relative = example[:3]
    # The released dataset requires at least two agents; its graph utility
    # squeezes away the node dimension for a singleton supplied directly.
    try:
        utils.seq_to_graph(absolute[:1], relative[:1])
    except IndexError:
        singleton_note = "Upstream singleton graph utility raises IndexError. A disconnected identical-motion pair supplies the one-node graph; the released neural network accepts the preserved singleton dimensions."
    else:
        raise AssertionError("Unexpected singleton support in the external utility")
    receipt = {"release": RELEASE, "model_sha256": sha256(Path(__file__).with_name("model.py")),
               "torch": str(torch.__version__), "numpy": np.__version__,
               "networkx": utils.nx.__version__, "scipy": scipy.__version__,
               "sample_count": 20, "comparison_tolerances": {"atol": 3e-6, "rtol": 3e-5,
               "increments_exact": True, "sample_atol_m": 5e-6},
               "upstream_registered_parameters": sum(p.numel() for p in release.social_stgcnn(n_txpcnn=5).parameters()),
               "owner_active_parameters": sum(p.numel() for p in SocialSTGCNN().parameters()),
               "singleton_scope": singleton_note, "cases": []}
    cases = {"full_numeric_order": [0, 1, 2, 3], "full_permuted_order": [2, 0, 3, 1],
             "moving_pair": [3, 1], "equal_motion_pair": [0, 1], "singleton": [0]}
    inputs = [(name, order, world, example) for name, order in cases.items()]
    real_world, real_example = recorded_eth_fixture(utils, args.reference)
    inputs.append(("recorded_eth_frames_2860_3050", [0, 1, 2], real_world, real_example))
    for name, order, world, example in inputs:
        absolute, _, relative, _, _, _, dataset_v, dataset_a, future, _ = example
        graph_order = order if len(order) > 1 else [order[0], order[0]]
        ref_v, ref_a = utils.seq_to_graph(absolute[graph_order], relative[graph_order])
        n = len(order)
        ref_v, ref_a = ref_v[:, :n], ref_a[:, :n, :n]
        if name in ("full_numeric_order", "recorded_eth_frames_2860_3050"):
            compare(ref_v, dataset_v, atol=0, rtol=0)
            compare(ref_a, dataset_a, atol=0, rtol=0)
        observed = world[:8, order].unsqueeze(0)
        v, a = scene_geometry(observed)
        errors = {"increments": compare(v[0], ref_v, atol=0, rtol=0), "laplacian": compare(a[0], ref_a)}
        next_steps = future[:, order]
        anchor, released_anchor = observed[0, -1].float(), absolute[order, :, -1]
        case = {"case": name, "node_order": order, "agents": n, "geometry_max_abs": errors}
        for training in (False, True):
            actual, expected, errors = forward_check(release, ref_v[None], ref_a[None], training)
            case["train_max_abs" if training else "eval_max_abs"] = errors
            if not training:
                case["distribution_max_abs"] = distribution_check(metrics, actual, expected, next_steps, anchor, released_anchor)
        receipt["cases"].append(case)
    receipt["passed"] = True
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n")
    print(f"REFERENCE_PARITY_OK cases={len(inputs)} release={RELEASE} output={args.out}")


if __name__ == "__main__":
    main()
