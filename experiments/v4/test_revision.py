import contextlib
import io
import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
from io_source import OneStudySource, balanced_parts

ROOT = Path(__file__).resolve().parent
BASE = json.loads((ROOT / 'baseline.ipynb').read_text(encoding='utf-8'))
OLD = ''.join(BASE['cells'][13]['source'])
NEW = (ROOT / 'blend_v4.py').read_text(encoding='utf-8')
LABELS = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']

# Local runtime lacks SciPy. Spearman here is only a printed diagnostic and
# integrity check; use its definition (Pearson correlation of average ranks)
# for both baseline and candidate. Production Kaggle code stays untouched.
_series_corr = pd.Series.corr
def local_corr(self, other, method='pearson', min_periods=None):
    if method == 'spearman':
        return _series_corr(self.rank(), other.rank(), method='pearson', min_periods=min_periods)
    return _series_corr(self, other, method=method, min_periods=min_periods)

class RevisionTests(unittest.TestCase):
    def test_cache_byte_parity_and_reuse(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            for i in range(12):
                p = root / 'test_series' / 'study' / 'series' / f'{i:03d}.dcm'
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(bytes([i])*100)
            OneStudySource.current = 'study'
            src = OneStudySource(root, 'test')
            src.cache_limit = 2000
            items = src.infolist()
            expected = [(root / i.filename).read_bytes() for i in items]
            with patch.object(Path, 'read_bytes', side_effect=AssertionError('unneeded repeated disk read')):
                self.assertEqual([src.read(i) for i in items], expected)
                self.assertEqual([src.read(i.filename) for i in items], expected)
            self.assertLessEqual(src._bytes, src.cache_limit)
            src.close()
            self.assertEqual(src._bytes, 0)

    def test_oversized_input_and_lru_bound(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            base = root / 'test_series' / 'big' / 'series'
            base.mkdir(parents=True)
            for name, size in [('a',30),('b',30),('c',200)]:
                (base / (name+'.dcm')).write_bytes(name.encode()*size)
            OneStudySource.current = 'big'
            src = OneStudySource(root, 'test')
            src.cache_limit = 50
            for item in src.infolist()*3:
                self.assertEqual(src.read(item), (root / item.filename).read_bytes())
                self.assertLessEqual(src._bytes, 50)

    def test_prefetch_error_is_retried_on_demand(self):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            p = root / 'test_series' / 's' / 'series' / 'a.dcm'
            p.parent.mkdir(parents=True)
            p.write_bytes(b'unchanged')
            OneStudySource.current = 's'
            src = OneStudySource(root, 'test')
            with patch.object(Path, 'read_bytes', side_effect=OSError('temporary')):
                items = src.infolist()
            self.assertEqual(src.read(items[0]), b'unchanged')

    def test_scheduling_coverage_and_estimated_load(self):
        rng = np.random.default_rng(42)
        for n in (1,3,10,100):
            ids = [f'{i:04d}' for i in range(n)]
            costs = dict(zip(ids, rng.integers(0,1000,n)))
            ng = min(n,2)
            parts = balanced_parts(ids,costs,ng)
            self.assertEqual(sorted(sum(parts,[])), sorted(ids))
            ordered = sorted(ids,key=lambda u:(-costs[u],u))
            old = [ordered[i::ng] for i in range(ng)]
            load = lambda ps: max(sum(max(1,costs[u]) for u in p) for p in ps)
            self.assertLessEqual(load(parts),load(old))

    def run_blend(self, source, fail224=False, fail384=False, failr6=False, ties=False):
        with tempfile.TemporaryDirectory(dir=ROOT) as folder:
            root = Path(folder)
            work, data, pred = root/'work',root/'data',root/'pred'
            for p in (work,data,pred/'worker0'):
                p.mkdir(parents=True)
            n=100
            ids=[f'study_{i:03d}' for i in range(n)]
            rng=np.random.default_rng(817)
            a,b,c=[rng.uniform(.01,.99,(n,12)) for _ in range(3)]
            if ties:
                a,b,c=[x.round(1) for x in (a,b,c)]
            if fail384:
                a[0]=.5
            def frame(x):
                f=pd.DataFrame(x,columns=LABELS)
                f.insert(0,'StudyInstanceUID',ids)
                return f
            pd.DataFrame({'StudyInstanceUID':ids}).to_csv(data/'test.csv',index=False)
            frame(a).to_csv(data/'sample_submission.csv',index=False)
            frame(a).to_csv(work/'raptor_raw.csv',index=False)
            frame(pd.DataFrame(a).rank(pct=True).values).to_csv(work/'raptor_sub.csv',index=False)
            frame(pd.DataFrame(b).rank(pct=True).values).to_csv(work/'raptor224_sub.csv',index=False)
            status=np.ones(n,int)
            if fail224:
                status[1]=0
            pd.DataFrame({'StudyInstanceUID':ids,'scored':status}).to_csv(work/'raptor224_status.csv',index=False)
            frame(c).to_csv(work/'ours_sub.csv',index=False)
            valid=frame(c)
            if failr6:
                valid=valid.iloc[1:]
            valid.to_csv(pred/'worker0'/'predictions.csv',index=False)
            env={'DATA':data,'WORK':work,'pred':pred,'R224_OK':True,'time':time,'NB_T0':time.time(),'RAPTOR_SECONDS':1,'json':json,'display':lambda x:None}
            with contextlib.redirect_stdout(io.StringIO()), patch.object(pd.Series, 'corr', local_corr):
                exec(compile(source,'blend','exec'),env)
            return pd.read_csv(work/'submission.csv'),env

    def test_complete_predictions_exactly_preserved(self):
        for ties in (False, True):
            old,_=self.run_blend(OLD,ties=ties)
            new,_=self.run_blend(NEW,ties=ties)
            pd.testing.assert_frame_equal(old,new,check_exact=True)

    def test_independent_224_failure_uses_384(self):
        new,env=self.run_blend(NEW,fail224=True)
        row=1
        for c in LABELS:
            rr=env['pct'](env['raw'][c].values)[row]
            ro=env['pct'](env['ours'][c].values)[row]
            w=env['W_RAPTOR_C'].get(c,.5)
            self.assertAlmostEqual(new.loc[row,c],float(np.float32(w*rr+(1-w)*ro)),places=7)
        self.assertEqual(env['blend']['raptor224_failed'],1)

    def test_existing_384_and_r6_fallbacks_still_work(self):
        for f384,fr6 in ((True,False),(False,True),(True,True)):
            old,_=self.run_blend(OLD,fail384=f384,failr6=fr6)
            new,_=self.run_blend(NEW,fail384=f384,failr6=fr6)
            pd.testing.assert_frame_equal(old,new,check_exact=True)

if __name__=='__main__':
    unittest.main(verbosity=2)
