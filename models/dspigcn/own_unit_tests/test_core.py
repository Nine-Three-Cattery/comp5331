import copy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'li_junhao_trajectronpp'), str(ROOT/'models/dspigcn')]

import numpy as np
import pytest
import torch
from independent_tpp.data import SceneStore, WindowRecord
from dspigcn.data import GraphDataset, collate
from dspigcn.model import DSPIGCN, anchor_sample, motion_inputs, normalized_graph, objective, polynomial

torch.set_num_threads(1)


def scene():
    tracks = {str(i): {t: np.array([t*0.4, i*0.8, 999., 999., 999., 999.])
                      for t in range(20 if i == 0 else 8)} for i in range(3)}
    record = WindowRecord(dict(scene_id='synthetic', source_file='toy', target_id='0',
                               scene_t=7, anchor_t=7, window_id='toy'))
    return SceneStore({record.scene_key: tracks}, [record])


@pytest.mark.parametrize('condition,level', [('clean',0),('noise',0.2),('missing',1),('missing',2)])
def test_target_and_neighbor_future_invariance(condition, level):
    store = scene()
    expected = GraphDataset(store, 2, condition, level).inputs(0)
    # Alter target futures; add a neighbor with ONLY future positions. Change all
    # precomputed derivatives; add future rows to history-visible neighbors.
    changed = copy.deepcopy(store)
    for tracks in changed.tracks.values():
        tracks['future_only'] = {t: np.full(6, -888.) for t in range(8,20)}
        for track in tracks.values():
            for t in range(8,20):
                track[t] = np.full(6, 1e6)
            for t, state in track.items():
                state[2:] = -1e9
    actual = GraphDataset(changed, 2, condition, level).inputs(0)
    for key in expected:
        np.testing.assert_array_equal(expected[key], actual[key])
    assert expected['positions'].shape[1] == 3  # Neighbors need no future.


def test_round_trip_and_derivative_units():
    item = GraphDataset(scene()).inputs(0)
    np.testing.assert_allclose(item['positions'][:,0]+item['origin'], item['observed'], atol=3e-7)
    _, velocity = motion_inputs(torch.tensor(item['positions'])[None])
    torch.testing.assert_close(velocity[0,1:,:,0], torch.ones(7,3))


def test_polynomial_matches_analytic_and_autograd():
    coeff = torch.tensor([[[1.,-2.],[2.,3.],[3.,0.],[0.,0.],[0.,0.]]], dtype=torch.float64)
    times = torch.tensor([0.,0.4,2.,4.8], dtype=torch.float64, requires_grad=True)
    position, velocity = polynomial(coeff, times)
    torch.testing.assert_close(position[0,:,0], 1+2*times+3*times**2)
    torch.testing.assert_close(velocity[0,:,0], 2+6*times)
    derivative = torch.autograd.grad(position[0,:,0].sum(), times)[0]
    torch.testing.assert_close(derivative, velocity[0,:,0])


def test_graph_known_weights_and_padding():
    points = torch.tensor([[[[0.,0.],[2.,0.],[999.,999.]]]])
    graph = normalized_graph(points, torch.tensor([[True,True,False]]))
    torch.testing.assert_close(graph[0,0,:2,:2], torch.tensor([[2/3,1/3],[1/3,2/3]]))
    assert graph[...,2,:].eq(0).all() and graph[...,:,2].eq(0).all()


def test_likelihood_against_torch_distribution():
    torch.manual_seed(4)
    mean = torch.randn(2,12,2, dtype=torch.float64)
    scale = torch.rand_like(mean)+0.2
    rho = torch.full((2,12),0.7, dtype=torch.float64)
    target = torch.randn_like(mean)
    covariance = torch.diag_embed(scale**2)
    covariance[...,0,1] = covariance[...,1,0] = rho*scale.prod(-1)
    output = dict(mean=mean, log_scale=scale.log(), rho=rho, velocity=mean,
                  physical_position=mean, physical_velocity=mean)
    loss, parts = objective(output,target,0,0)
    reference = -torch.distributions.MultivariateNormal(mean,covariance_matrix=covariance).log_prob(target).mean()
    torch.testing.assert_close(loss,reference)
    assert parts['physics'] == 0
    torch.testing.assert_close(parts['reconstruction'], (target-mean).square().sum(-1).mean())


def test_sampler_reproducible_batch_independent_and_uses_each_anchor_once():
    mean = torch.zeros(2,12,2)
    output = dict(mean=mean,log_scale=torch.zeros_like(mean),rho=torch.zeros(2,12))
    samples = anchor_sample(output,[8,9])
    assert samples.shape == (2,20,12,2)
    for i,seed in enumerate([8,9]):
        single = anchor_sample({k:v[i:i+1] for k,v in output.items()},[seed])
        torch.testing.assert_close(samples[i], single[0], rtol=0,atol=0)
        # With mu=0, sigma=1, rho=0, the pool is exactly the seeded normals.
        anchors = torch.randn(12,20,2,generator=torch.Generator().manual_seed(seed)).reshape(-1,2)
        np.testing.assert_array_equal(np.sort(samples[i].numpy().reshape(-1)),np.sort(anchors.numpy().reshape(-1)))


def test_forward_padding_permutation_translation_and_training():
    torch.manual_seed(1)
    model = DSPIGCN(width=16,dropout=0).eval()
    # Randomize zero initialization so invariance tests exercise graph branches.
    for head in (model.position_head, model.velocity_head):
        torch.nn.init.normal_(head[-1].weight,std=0.01)
    batch = collate([GraphDataset(scene())[0]])
    p,m = batch['positions'],batch['mask']
    expected = model(p,m)
    permuted = model(p[:,:,[0,2,1]],m[:,[0,2,1]])
    padded = model(torch.cat((p,torch.ones(1,8,2,2)*100),2),torch.cat((m,torch.zeros(1,2,dtype=torch.bool)),1))
    translated = model(p+13,m)
    for key in expected:
        torch.testing.assert_close(expected[key],permuted[key],rtol=1e-4,atol=1e-5)
        torch.testing.assert_close(expected[key],padded[key],rtol=1e-4,atol=1e-5)
        torch.testing.assert_close(expected[key],translated[key],rtol=1e-4,atol=1e-5)
    target = batch['future']-batch['origin'][:,None]
    optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
    before = objective(model(p,m),target)[0].item()
    for _ in range(30):
        optimizer.zero_grad()
        loss,_ = objective(model(p,m),target)
        loss.backward()
        assert all(torch.isfinite(x.grad).all() for x in model.parameters() if x.grad is not None)
        optimizer.step()
    assert objective(model(p,m),target)[0].item() < before
    assert all(any(x.grad is not None and x.grad.abs().sum()>0 for x in branch.parameters())
               for branch in [model.displacement,model.velocity,model.physics])


def test_interrupted_training_resumes_exactly_without_test_selection(tmp_path, monkeypatch):
    import json
    import dspigcn.experiment as experiment
    calls = []
    def load(fold, split, root):
        calls.append(split)
        assert split in ('train','val')
        store = scene()
        store.identity = {'manifest_hash':split, 'fold':fold, 'split':split}
        return store
    monkeypatch.setattr(experiment,'load_split',load)
    config = json.loads((ROOT/'models/dspigcn/configs/pilot.json').read_text())
    config.update(width=16,batch_size=1,dropout=0.2)
    continuous, resumed = tmp_path/'continuous', tmp_path/'resumed'
    experiment.train(tmp_path,continuous,'eth',0,config,'cpu')
    real_validation = experiment.validation
    count = 0
    def interrupt(*args,**kwargs):
        nonlocal count
        count += 1
        if count == 2:
            raise RuntimeError('simulated interruption')
        return real_validation(*args,**kwargs)
    monkeypatch.setattr(experiment,'validation',interrupt)
    with pytest.raises(RuntimeError,match='simulated interruption'):
        experiment.train(tmp_path,resumed,'eth',0,config,'cpu')
    monkeypatch.setattr(experiment,'validation',real_validation)
    experiment.train(tmp_path,resumed,'eth',0,config,'cpu')
    a = torch.load(continuous/'last.pt',weights_only=False)
    b = torch.load(resumed/'last.pt',weights_only=False)
    for key in a['model']:
        torch.testing.assert_close(a['model'][key],b['model'][key],rtol=0,atol=0)
    assert a['candidates']==b['candidates']
    selection = json.loads((resumed/'checkpoint_selection.json').read_text())
    assert selection['selection_split']=='val' and not selection['test_used_for_selection']
    assert 'test' not in calls
