import math,sys,unittest
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'legacy'))
from model import SocialLSTM,gaussian_nll
from metrics import per_trajectory
from data import perturb,visible_history,protocol_fingerprint

class LegacyUnitTests(unittest.TestCase):
    def setUp(self):torch.set_num_threads(1);torch.manual_seed(7)
    def test_analytic_metrics(self):
        y=np.zeros((1,12,2));p=np.zeros((2,1,12,2));p[0,:,:,0]=1;p[1,:,:,0]=3
        m=per_trajectory(p,y)
        for key,value in [('mean_ADE',2),('min_ADE',1),('mean_FDE',2),('min_FDE',1),('energy_score',1.5*math.sqrt(12)),('diversity_RMS',2)]:np.testing.assert_allclose(m[key],value)
    def test_whole_path_oracles(self):
        p=np.zeros((2,1,12,2));p[0,0,-1,0]=12;p[1,0,:,0]=2;m=per_trajectory(p,np.zeros((1,12,2)))
        self.assertAlmostEqual(m['min_ADE'][0],1);self.assertAlmostEqual(m['min_FDE'][0],2)
    def test_gaussian_nll(self):
        mu=torch.randn(5,2,dtype=torch.float64);ls=torch.randn(5,2,dtype=torch.float64);r=torch.linspace(-.8,.8,5,dtype=torch.float64);y=torch.randn(5,2,dtype=torch.float64)
        cov=torch.diag_embed(ls.exp().square());cov[:,0,1]=cov[:,1,0]=r*ls.sum(-1).exp()
        torch.testing.assert_close(gaussian_nll((mu,ls,r),y),-torch.distributions.MultivariateNormal(mu,covariance_matrix=cov).log_prob(y))
    def test_social_grid_isolation_and_gradient(self):
        m=SocialLSTM(hidden=2,embedding=4,grid=2,neighborhood=4)
        p=torch.tensor([[0.,0.],[.5,.5],[.8,.8],[-2.,0.],[2.,0.],[.5,.5]])
        h=torch.arange(12,dtype=torch.float32).reshape(6,2).requires_grad_();g=torch.tensor([0,0,0,0,0,1]);s=m.social_tensor(p,h,g)
        torch.testing.assert_close(s[0,3],h[1]+h[2]);torch.testing.assert_close(s[0,2],h[3]);torch.testing.assert_close(s[5],torch.zeros_like(s[5]))
        s.sum().backward();self.assertGreater(float(h.grad.abs().sum()),0)
    def test_missing_suffix_is_causal(self):
        x=np.arange(32,dtype=np.float32).reshape(8,2,2);w={'id':'synthetic:0'}
        before=perturb(x,w,'missing_2');x[-2:]=999;np.testing.assert_array_equal(before,perturb(x,w,'missing_2'))
    def test_future_target_and_neighbor_invariance(self):
        m=SocialLSTM(hidden=8,embedding=4,grid=2,dropout=0.).eval();xy=np.random.default_rng(7).normal(size=(20,3,2)).astype(np.float32)
        changed=xy.copy();changed[8:,0]+=1000;changed[8:,1:]-=1000
        a=visible_history(xy);b=visible_history(changed);np.testing.assert_array_equal(a,b)
        g=torch.zeros(3,dtype=torch.long)
        pa=m.predict(torch.from_numpy(a),g,20,torch.Generator().manual_seed(9));pb=m.predict(torch.from_numpy(b),g,20,torch.Generator().manual_seed(9))
        torch.testing.assert_close(pa,pb,rtol=0,atol=0)
    def test_metric_roundtrip_and_translation(self):
        # Legacy transform is identity metric coordinates; no fitted normalization.
        m=SocialLSTM(hidden=8,embedding=4,grid=2,dropout=0.).eval();obs=torch.randn(8,3,2);g=torch.tensor([0,0,1]);offset=torch.tensor([10.,-5.])
        torch.testing.assert_close(torch.from_numpy(obs.numpy().copy()),obs,rtol=0,atol=0)
        a=m.predict(obs,g,3,torch.Generator().manual_seed(2));b=m.predict(obs+offset,g,3,torch.Generator().manual_seed(2))
        torch.testing.assert_close(b-offset,a,atol=3e-5,rtol=3e-5)
    def test_fingerprint_excludes_runtime(self):
        a={'source':'x','stride':4,'preprocessing_seconds':1};b={'source':'x','stride':4,'preprocessing_seconds':8}
        self.assertEqual(protocol_fingerprint(a),protocol_fingerprint(b));b['stride']=1;self.assertNotEqual(protocol_fingerprint(a),protocol_fingerprint(b))
    def test_backward_and_twenty_sample_shape(self):
        m=SocialLSTM(hidden=8,embedding=4,grid=2,dropout=0.);xy=torch.randn(20,3,2);mask=torch.ones(3,dtype=torch.bool);g=torch.zeros(3,dtype=torch.long)
        loss=m.teacher_nll(xy,mask,g);loss.backward();self.assertTrue(torch.isfinite(loss));self.assertTrue(torch.isfinite(m.head.weight.grad).all())
        m.eval();p=m.predict(xy[:8],g,20,torch.Generator().manual_seed(3));self.assertEqual(p.shape,(20,3,12,2));self.assertTrue(torch.isfinite(p).all())

if __name__=='__main__':unittest.main(verbosity=2)
