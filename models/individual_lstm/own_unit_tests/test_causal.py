from pathlib import Path
import unittest, sys, copy
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from run import history, IndividualLSTM, constant_velocity
from independent_tpp.data import WindowRecord, SceneStore
from independent_tpp.protocol import perturb_history

class CausalTest(unittest.TestCase):
    def test_history_ignores_future_neighbors_and_derivative_columns(self):
        rec=WindowRecord({'scene_id':'eth','source_file':'abc','target_id':'7','scene_t':7})
        track={t:np.array([float(t),.5*t,0,0,0,0]) for t in range(20)}
        store=SceneStore({rec.scene_key:{'7':track,'8':copy.deepcopy(track)}},[rec])
        clean=history(store,rec)
        for t,state in track.items():
            if t>7: state[:2]=999
            state[2:]=-999
        for state in store.tracks[rec.scene_key]['8'].values(): state[:]=111
        np.testing.assert_array_equal(history(store,rec),clean)
        model=IndividualLSTM().eval()
        for cond,lev in [('clean',0),('missing',2),('noise',.2)]:
            a,mask=perturb_history(clean,cond,lev,123)
            b,_=perturb_history(history(store,rec),cond,lev,123)
            x=torch.tensor(a[None],dtype=torch.float32)
            with torch.no_grad():
                for f in (model,constant_velocity):
                    out=f(x)
                    torch.testing.assert_close(out,f(torch.tensor(b[None],dtype=x.dtype)),rtol=0,atol=0)
                    self.assertEqual(out.shape,(1,12,2))
                    torch.testing.assert_close(out+10,f(x+10),rtol=1e-5,atol=1e-5)
        d=np.diff(clean,axis=0)
        np.testing.assert_allclose(clean[0]+d.cumsum(axis=0),clean[1:])

if __name__=='__main__': unittest.main()
