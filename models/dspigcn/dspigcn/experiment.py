"""Validation-only training, resumable checkpoints and neutral export."""
import json
import platform
import random
import resource
import time
from pathlib import Path

import numpy as np
import torch

from independent_tpp.data import load_split
from independent_tpp.protocol import CONDITIONS, digest_json, sha256, write_json
from .data import GraphDataset, batches
from .model import DSPIGCN, anchor_sample, objective


def environment(device):
    return dict(python=platform.python_version(), platform=platform.platform(),
                machine=platform.machine(), torch=torch.__version__, numpy=np.__version__,
                device=str(device), cuda=torch.version.cuda,
                threads=torch.get_num_threads())


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def forward(model, batch, device):
    return model(batch['positions'].to(device), batch['mask'].to(device))


def sampling_seed(seed, index):
    return 1700003 + seed*100003 + index*1009


@torch.no_grad()
def validation(model, items, device, batch_size, seed):
    model.eval()
    total = np.zeros(2)
    for batch in batches(items, batch_size):
        output = forward(model, batch, device)
        samples = anchor_sample(output, [sampling_seed(seed, i) for i in batch['indices']])
        truth = (batch['future']-batch['origin'][:, None]).to(device)
        error = samples-truth[:, None]
        sequence = error.square().sum(-1).mean(-1).sqrt().mean(-1)
        # Same U-statistic Energy Score as the common evaluator.
        flattened = samples.flatten(2)/np.sqrt(12)
        pair_sum = torch.cdist(flattened, flattened).triu(diagonal=1).sum((1, 2))
        energy = sequence-pair_sum/(20*19)
        ade = error.square().sum(-1).sqrt().mean((1, 2))
        total += np.array([energy.sum().item(), ade.sum().item()])
    return dict(energy_score=float(total[0]/len(items)), mean_ade=float(total[1]/len(items)))


def train(data_root, run, fold, seed, config, device):
    run = Path(run)
    run.mkdir(parents=True, exist_ok=True)
    seed_all(seed)
    stores = {s: load_split(fold, s, str(data_root)) for s in ('train', 'val')}
    frozen = {**config, 'fold': fold, 'seed': seed, 'model': 'DSPIGCN',
              **{f'{s}_identity': store.identity for s, store in stores.items()},
              'implementation_hashes': {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob('*.py'))},
              'environment': environment(device)}
    path = run/'config.json'
    if path.exists() and json.loads(path.read_text()) != frozen:
        raise ValueError('Existing run has a different config; use a new run directory')
    write_json(path, frozen)
    model = DSPIGCN(config['width'], config['dropout']).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=config['learning_rate'], momentum=0.9)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['lr_step'], gamma=0.5)
    begin, candidates, best = 0, [], (float('inf'), float('inf'), 0)
    resume = run/'last.pt'
    if resume.exists():
        state = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(state['model'])
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        begin, candidates, best = state['epoch'], state['candidates'], tuple(state['best'])
        torch.set_rng_state(state['torch_rng'].cpu())
        if device == 'cuda':
            torch.cuda.set_rng_state_all(state['cuda_rng'])
        if device == 'mps':
            torch.mps.set_rng_state(state['mps_rng'].cpu())
    if begin >= config['epochs']:
        return model
    started = time.perf_counter()
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    items = {}
    for split, store in stores.items():
        dataset = GraphDataset(store, radius=config['radius'])
        items[split] = [dataset[i] for i in range(len(dataset))]
        dataset.context.cache_clear()
        print(f'CACHE {fold} {split} {len(dataset)}', flush=True)
    for epoch in range(begin+1, config['epochs']+1):
        epoch_started = time.perf_counter()
        model.train()
        order = np.random.default_rng(seed*100003+epoch).permutation(len(items['train']))
        losses = np.zeros(4)
        weight = min(5., config['reconstruction_weight']*config['reconstruction_growth']**(epoch-1))
        for batch in batches(items['train'], config['batch_size'], order):
            optimizer.zero_grad(set_to_none=True)
            output = forward(model, batch, device)
            target = (batch['future']-batch['origin'][:, None]).to(device)
            loss, parts = objective(output, target, config['physics_weight'], weight)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at epoch {epoch}')
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
            optimizer.step()
            losses += len(batch['indices'])*np.array([loss.item(), *[x.item() for x in parts.values()]])
        scheduler.step()
        record = {'epoch': epoch, 'reconstruction_weight': weight,
                  'train': dict(zip(('total', 'nll', 'reconstruction', 'physics'), (losses/len(items['train'])).tolist()))}
        if epoch % config['validate_every'] == 0 or epoch == config['epochs']:
            record.update(validation(model, items['val'], device, config['batch_size'], seed))
            candidates.append(record)
            key = (record['energy_score'], record['mean_ade'], epoch)
            if key < best:
                best = key
                torch.save({'model': model.state_dict(), 'epoch': epoch, 'config_hash': digest_json(frozen)}, run/'best.pt')
            write_json(run/'checkpoint_selection.json', dict(selection_split='val', test_used_for_selection=False,
                       criterion='energy_score, mean_ade, earlier epoch', candidates=candidates,
                       selected=dict(checkpoint='best.pt', epoch=best[2], energy_score=best[0], mean_ade=best[1])))
        state = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                     epoch=epoch, candidates=candidates, best=best, torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device == 'cuda' else None,
                     mps_rng=torch.mps.get_rng_state() if device == 'mps' else None)
        torch.save(state, run/'last.tmp.pt')
        (run/'last.tmp.pt').replace(resume)
        with (run/'training.jsonl').open('a') as handle:
            handle.write(json.dumps({**record, 'elapsed_seconds': time.perf_counter()-started,
                'epoch_seconds': time.perf_counter()-epoch_started,
                'peak_cuda_bytes': torch.cuda.max_memory_allocated() if device == 'cuda' else None,
                'peak_host_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss *
                    (1 if platform.system() == 'Darwin' else 1024)})+'\n')
        print(f'{fold} seed={seed} {json.dumps(record)}', flush=True)
    return model


def load_model(run, device):
    run = Path(run)
    config = json.loads((run/'config.json').read_text())
    selection = json.loads((run/'checkpoint_selection.json').read_text())
    checkpoint = run/selection['selected']['checkpoint']
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    if state['config_hash'] != digest_json(config):
        raise ValueError('Checkpoint config hash mismatch')
    model = DSPIGCN(config['width'], config['dropout']).to(device)
    model.load_state_dict(state['model'])
    model.eval()
    return model, config, checkpoint


@torch.no_grad()
def predict(data_root, run, device):
    run = Path(run)
    model, config, checkpoint = load_model(run, device)
    seed, fold = config['seed'], config['fold']
    store = load_split(fold, 'test', str(data_root))
    for tag, kind, level in CONDITIONS:
        dataset = GraphDataset(store, seed, kind, level, radius=config['radius'])
        arrays = {k: [] for k in ('samples', 'ground_truth', 'observed', 'observed_mask')}
        records = []
        for batch in batches(dataset, config['batch_size']):
            seeds = [sampling_seed(seed, i) for i in batch['indices']]
            samples = anchor_sample(forward(model, batch, device), seeds).cpu()+batch['origin'][:, None, None]
            arrays['samples'].append(samples.numpy())
            for dest, source in [('ground_truth', 'future'), ('observed', 'observed'), ('observed_mask', 'observed_mask')]:
                arrays[dest].append(batch[source].numpy())
            for i, sample_seed in zip(batch['indices'], seeds):
                row = store.manifests[i]
                records.append(json.dumps({**row.values, 'target_id': row.target_id, 'seed': seed,
                    'fold': fold, 'split': 'test', 'condition': kind, 'sigma': level if kind == 'noise' else 0,
                    'missing': level if kind == 'missing' else 0, 'coordinate_unit': 'meter', 'dt': 0.4,
                    'sample_count': 20, 'sampling_seed': sample_seed}))
        metadata = {**store.identity, 'model_name': 'DSPIGCN', 'seed': seed, 'sample_count': 20,
                    'condition': kind, 'level': level, 'config_hash': digest_json(config),
                    'checkpoint': checkpoint.name, 'checkpoint_sha256': sha256(checkpoint)}
        output = run/'evaluation'/tag
        output.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output/f'predictions_test_{tag}.npz',
                            **{k: np.concatenate(v) for k, v in arrays.items()},
                            records=np.array(records), metadata=json.dumps(metadata))
        dataset.context.cache_clear()
        print(f'EXPORTED {fold} seed={seed} {tag} N={len(dataset)}', flush=True)
