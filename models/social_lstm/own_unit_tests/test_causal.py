"""Independent owner regression checks for the actual causal feature/model boundary."""
import copy,sys,unittest
from pathlib import Path
import numpy as np
import torch

MODEL_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(MODEL_ROOT));sys.path.insert(0,str(MODEL_ROOT.parents[1]/'li_junhao_trajectronpp'))
from causal.data import Windows
from causal.model import DeterministicSocialLSTM
from causal.run import training_batches
from independent_tpp.data import SceneStore,WindowRecord

def fixture():
    key=('synthetic','fixture');tracks={}
    for actor,y in [('1',0.),('2',.75),('3',1.5),('4',50.)]:
        tracks[actor]={t:np.array([t*.1,y,777.,-888.,999.,-999.],np.float64) for t in range(20 if actor!='3' else 8)}
    values={'window_id':'synthetic:fixture:1:7','scene_id':'synthetic','source_file':'fixture',
        'target_id':'1','scene_t':7,'anchor_t':7,'scene_name':'synthetic__fixture__test'}
    rec=WindowRecord(values);rec2=WindowRecord({**values,'window_id':'synthetic:fixture:2:7','target_id':'2'})
    return SceneStore({key:tracks},[rec,rec2],{'version':'causal_v2','fold':'eth','split':'test','manifest_hash':'synthetic','input_hash':'synthetic'})

class CausalModelTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(17)
    def model(self):
        m=DeterministicSocialLSTM(hidden=8,embedding=4,grid=2).eval()
        with torch.no_grad():m.head.weight.normal_(0,.05)
        return m
    def test_target_neighbor_future_and_stored_derivative_invariance(self):
        store=fixture();other=copy.deepcopy(store)
        for tracks in other.tracks.values():
            for actor,states in tracks.items():
                for t,state in states.items():
                    state[2:]=np.arange(4)*1e8
                    if t>7:state[:2]+=np.array([10000.,-10000.])
        a,b=Windows(store),Windows(other)
        self.assertFalse(np.array_equal(a.future,b.future))
        model=self.model()
        for condition,level in [('clean',0),('noise',.1),('missing',2)]:
            xa,va,ma=a.batch([0,1],seed=2,condition=condition,level=level)
            xb,vb,mb=b.batch([0,1],seed=2,condition=condition,level=level)
            torch.testing.assert_close(xa,xb,rtol=0,atol=0);torch.testing.assert_close(va,vb)
            np.testing.assert_array_equal(ma,mb)
            with torch.no_grad():torch.testing.assert_close(model(xa,va),model(xb,vb),rtol=0,atol=0)
    def test_neighbor_future_presence_does_not_filter_context(self):
        s=fixture();a=Windows(s);changed=copy.deepcopy(s)
        tr=changed.tracks[('synthetic','fixture')]['3']
        for t in range(8,20):tr[t]=np.array([123.,456.,0,0,0,0])
        x,mask=a.features(0);y,other=Windows(changed).features(0)
        self.assertEqual(len(x),3);np.testing.assert_array_equal(x,y);np.testing.assert_array_equal(mask,other)
    def test_only_target_is_corrupted_with_manifest_index_seed(self):
        data=Windows(fixture());clean,_=data.features(1)
        noisy,mask=data.features(1,seed=2,condition='noise',level=.1)
        expected=data.histories[1]+np.random.RandomState(2*100003+1009).normal(0,.1,(8,2))
        np.testing.assert_array_equal(noisy[0],expected.astype(np.float32))
        np.testing.assert_array_equal(noisy[1:],clean[1:]);self.assertTrue(mask.all())
    def test_missing_suffix_uses_only_visible_prefix(self):
        d=Windows(fixture());a,m=d.features(0,condition='missing',level=2)
        other=Windows(fixture());other.histories[0,-2:]=99999
        b,n=other.features(0,condition='missing',level=2)
        np.testing.assert_array_equal(a,b);np.testing.assert_array_equal(m,[True]*6+[False]*2);np.testing.assert_array_equal(m,n)
    def test_deterministic_k1_and_metric_translation_roundtrip(self):
        data=Windows(fixture());x,v,_=data.batch([0,1]);model=self.model();offset=torch.tensor([20.,-13.])
        with torch.no_grad():a=model(x,v);b=model(x,v);c=model(x+offset,v)-offset
        self.assertEqual(tuple(a.shape),(2,1,12,2));self.assertTrue(torch.isfinite(a).all())
        torch.testing.assert_close(a,b,rtol=0,atol=0);torch.testing.assert_close(a,c,rtol=3e-5,atol=3e-5)
    def test_batch_and_padding_do_not_influence_target(self):
        x,v,_=Windows(fixture()).batch([0,1]);model=self.model()
        padded=torch.cat([x,torch.ones(2,2,8,2)*999],dim=1);mask=torch.cat([v,torch.zeros(2,2,dtype=torch.bool)],dim=1)
        with torch.no_grad():a=model(x,v);b=model(padded,mask);one=model(x[:1],v[:1])
        torch.testing.assert_close(a,b,rtol=2e-5,atol=2e-5);torch.testing.assert_close(a[:1],one,rtol=2e-5,atol=2e-5)
    def test_social_grid_self_padding_boundaries_and_gradient(self):
        m=DeterministicSocialLSTM(hidden=2,embedding=4,grid=2,neighborhood=4)
        p=torch.tensor([[[0.,0.],[.5,.5],[.8,.8],[-2.,0.],[2.,0.],[.5,.5]]])
        h=torch.arange(12,dtype=torch.float32).reshape(1,6,2).requires_grad_();valid=torch.tensor([[1,1,1,1,1,0]],dtype=torch.bool)
        out=m.pool(p,h,valid).reshape(1,6,4,2)
        torch.testing.assert_close(out[0,0,3],h[0,1]+h[0,2]);torch.testing.assert_close(out[0,0,2],h[0,3]);torch.testing.assert_close(out[0,5],torch.zeros_like(out[0,5]))
        out.sum().backward();self.assertGreater(float(h.grad.abs().sum()),0)
    def test_untrained_head_is_fixed_observed_three_interval_anchor(self):
        data=Windows(fixture());x,v,_=data.batch([0]);m=DeterministicSocialLSTM(hidden=8,embedding=4,grid=2).eval()
        base=(x[:,0,-1]-x[:,0,-4])/3
        expected=x[:,0,-1,None]+torch.arange(1,13)[None,:,None]*base[:,None]
        with torch.no_grad():torch.testing.assert_close(m(x,v)[:,0],expected,rtol=1e-6,atol=1e-6)
    def test_closed_loop_backward_is_finite(self):
        data=Windows(fixture());x,v,_=data.batch([0,1]);m=self.model();truth=torch.from_numpy(data.future)
        loss=(m(x,v)[:,0]-truth).square().mean();loss.backward()
        self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(m.head.weight.grad).all())
    def test_bucket_sampler_preserves_every_window_and_seed(self):
        counts=np.arange(1031)%53
        np.random.seed(2);a=training_batches(counts);np.random.seed(2);b=training_batches(counts)
        self.assertEqual([x.tolist() for x in a],[x.tolist() for x in b]);self.assertTrue(all(len(x)<=32 for x in a))
        np.testing.assert_array_equal(np.sort(np.concatenate(a)),np.arange(len(counts)))

if __name__=='__main__':unittest.main(verbosity=2)
