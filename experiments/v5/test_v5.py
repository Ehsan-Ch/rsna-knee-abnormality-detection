import ast, csv, io, json, math, time, types, unittest
from pathlib import Path
import numpy as np
from phase_sampling import phase_rows, ensemble_passes, pool_probabilities
ROOT=Path(__file__).resolve().parent
trainer=ast.parse((ROOT/'candidate_sources/compact_train.py').read_text())
selected=[x for x in trainer.body if isinstance(x,ast.FunctionDef) and x.name in ('covering_centers','phys_step','window_rows')]
ns={'math':math}
exec(compile(ast.Module(body=selected,type_ignores=[]),'trainer_sampling','exec'),ns)
window_rows=ns['window_rows']

def metadata(n,spacing=3.0):
    m=np.arange(n*11,dtype=np.float32).reshape(n,11)/100
    m[:,5]=math.log1p(spacing)
    return m

class SamplingTests(unittest.TestCase):
    def test_baseline_exact_and_complete_coverage(self):
        for n in range(1,101):
            m=metadata(n)
            blocks=[{'n_slices':n,'views':[(0,0,0)]}]
            bags=[phase_rows(window_rows,blocks,lambda f,r:m[r],3,3,0.0,o) for o in range(3)]
            self.assertEqual(bags[0],window_rows(blocks,lambda f,r:m[r],'eval',3,3))
            self.assertEqual(set(r[1] for bag in bags for r in bag),set(range(n)))
            for bag in bags:
                self.assertTrue(all(0<=k<n for r in bag for k in r[1:]))
    def test_no_cross_series_or_view_rows(self):
        m=metadata(38)
        blocks=[{'n_slices':7,'views':[(0,0,0),(0,1,7)]},{'n_slices':12,'views':[(0,0,14),(0,1,26)]}]
        for offset in range(3):
            rows=phase_rows(window_rows,blocks,lambda f,r:m[r],3,3,0.0,offset)
            for r in rows:
                bounds=next((lo,hi) for lo,hi in [(0,7),(7,14),(14,26),(26,38)] if lo<=r[1]<hi)
                self.assertTrue(all(bounds[0]<=k<bounds[1] for k in r[2:]))
    def test_physical_neighbour_spacing_is_preserved(self):
        m=metadata(100,spacing=0.4)
        blocks=[{'n_slices':100,'views':[(0,0,0)]}]
        ps=ns['phys_step'](math.expm1(float(m[0,5])),3.0)
        for offset in range(3):
            rows=phase_rows(window_rows,blocks,lambda f,r:m[r],3,3,3.0,offset)
            for r in rows:
                self.assertEqual(r[2:],(max(0,r[1]-ps),r[1],min(99,r[1]+ps)))
    def test_pool_limits_identity_and_formula(self):
        p=np.array([0.0,0.2,0.8,1.0])
        np.testing.assert_array_equal(pool_probabilities([p,p,p],[.5,.25,.25]),p)
        q=np.array([[.2,.6],[.8,.9],[.4,.3]])
        expected=1/(1+np.exp(-((np.log(q)-np.log1p(-q))*np.array([.5,.25,.25])[:,None]).sum(0)))
        np.testing.assert_allclose(pool_probabilities(q,[.5,.25,.25]),expected,rtol=1e-14)
        for bad in [np.nan,np.inf,-0.1,1.1]:
            with self.assertRaises(ValueError): pool_probabilities([[bad]],[1])
    def test_extra_failures_are_renormalized(self):
        p=np.array([.1,.9]); q=np.array([.7,.6])
        def predict(i):
            if i==1: raise RuntimeError('extra pass failed')
            return p if i==0 else q
        got,base,errors=ensemble_passes(predict)
        np.testing.assert_array_equal(base,p)
        np.testing.assert_allclose(got,pool_probabilities([p,q],[.5,.25]))
        self.assertEqual(set(errors),{'1'})
        def only_base(i):
            if i: raise RuntimeError('failed')
            return p
        got,_,errors=ensemble_passes(only_base)
        np.testing.assert_array_equal(got,p)
        self.assertEqual(len(errors),2)
    def test_bad_baseline_is_never_hidden(self):
        with self.assertRaises(ValueError): ensemble_passes(lambda i: np.array([np.nan]) if i==0 else np.array([.5]))
    def test_actual_loader_phase_tensor_mapping(self):
        tree=ast.parse((ROOT/'candidate_sources/compact_infer.py').read_text())
        cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='CStudies')
        fn=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__getitem__')
        body=next(n for n in fn.body if isinstance(n,ast.Try)).body
        start=next(i for i,n in enumerate(body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='blocks' for t in n.targets))
        px=np.arange(14*4*4,dtype=np.uint8).reshape(14,4,4); meta=metadata(14)
        item={'blocks':[{'n_slices':7,'views':[(0,0),(1,7)]}]}
        env={'np':np,'torch':types.SimpleNamespace(from_numpy=lambda x:x),
            'self':types.SimpleNamespace(window_rows=window_rows,slices=3,eval_stride=3,phys_mm=0.,extent_meta=True),
            'px':px,'meta':meta,'item':item,'uid':'x','anchor_note':None,'side_note':None,'phase_rows':phase_rows}
        exec(compile(ast.Module(body=body[start:],type_ignores=[]),'loader','exec'),env)
        out=env['out']; counts=out['phase_window_counts']
        self.assertEqual(out['phase_unique_centres'],14)
        self.assertEqual(out['meta'].shape,(counts[0],9))
        blocks=[{'n_slices':7,'views':[(0,0,0),(0,1,7)]}]
        for offset in range(3):
            bag=out if offset==0 else out['phase_bags'][offset-1]
            rows=phase_rows(window_rows,blocks,lambda f,r:meta[r],3,3,0.,offset)
            expected=np.stack([px[np.array(rows)[:,2+d]] for d in range(3)],1)
            np.testing.assert_array_equal(bag['images'],expected)
    def test_actual_worker_applies_specialists_per_phase(self):
        class Tensor(np.ndarray):
            def double(self): return self.astype(np.float64).view(Tensor)
            def cpu(self): return self
            def numpy(self): return np.asarray(self)
        def tensor(x): return np.asarray(x).view(Tensor)
        tree=ast.parse((ROOT/'candidate_sources/compact_infer.py').read_text())
        main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
        handle=next(n for n in main.body if isinstance(n,ast.FunctionDef) and n.name=='handle')
        head_calls=[]
        def score(images,meta):
            z=tensor([float(images[0]),float(images[0])+.2])
            return z,z
        def head(uid,item,zs,p):
            head_calls.append(uid)
            p[0]=1/(1+np.exp(-(2*zs[0]+.3)))
        pred_stream,base_stream=io.StringIO(),io.StringIO()
        env={'np':np,'time':time,'json':json,'torch':types.SimpleNamespace(isfinite=np.isfinite,sigmoid=lambda x:tensor(1/(1+np.exp(-x)))),
            'EAGER':False,'score':score,'STK':object(),'SST':None,'apply_stacker':head,
            'ensemble_passes':ensemble_passes,'PHASE_WEIGHTS':(.5,.25,.25),
            'fallback':{},'phase_records':[],'wr':csv.writer(pred_stream),'bwr':csv.writer(base_stream),'fh':pred_stream,'bfh':base_stream,
            'stats':{k:0 for k in ('rows','decode_s','render_s','anchor_s','gpu_s','windows')},
            'side_counts':{'checked':0,'flipped':0},'anchor_methods':{},'per_study':[],
            'a':types.SimpleNamespace(dump_logits=False),'print':lambda *a,**k:None}
        exec(compile(ast.Module(body=[handle],type_ignores=[]),'handle','exec'),env)
        item={'uid':'study','error':None,'images':np.array([0.]),'meta':np.array([0.]),
              'phase_bags':[{'same_as_base':False,'images':np.array([1.]),'meta':np.array([0.])},
                            {'same_as_base':False,'images':np.array([-1.]),'meta':np.array([0.])}],
              'phase_window_counts':[1,1,1],'phase_unique_centres':3,'seconds':[0.,0.,0.], 'side':None,'anchor':None}
        env['handle'](item)
        self.assertEqual(head_calls,['study','study|phase1','study|phase2'])
        self.assertEqual(env['fallback'],{})
        got=np.array([float(x) for x in list(csv.reader(io.StringIO(pred_stream.getvalue())))[0][1:]])
        bp=np.array([1/(1+np.exp(-.3)),1/(1+np.exp(-.2))])
        base_got=np.array([float(x) for x in list(csv.reader(io.StringIO(base_stream.getvalue())))[0][1:]])
        np.testing.assert_allclose(base_got,bp,rtol=1e-8)
        probs=[[1/(1+np.exp(-(2*x+.3))),1/(1+np.exp(-(x+.2)))] for x in (0.,1.,-1.)]
        np.testing.assert_allclose(got,pool_probabilities(probs,[.5,.25,.25]),rtol=1e-8)
        self.assertEqual(env['stats']['rows'],1)
        self.assertFalse(env['phase_records'][0]['failed_phases'])

if __name__=='__main__': unittest.main(verbosity=2)
