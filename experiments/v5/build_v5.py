import ast, base64, copy, hashlib, json, re, textwrap, zlib
from pathlib import Path
ROOT = Path(__file__).resolve().parent
nb = json.loads((ROOT/'baseline_v4.ipynb').read_text(encoding='utf-8-sig'))
base = copy.deepcopy(nb)
get = lambda i: ''.join(nb['cells'][i]['source'])
def put(i,s): nb['cells'][i]['source'] = s.splitlines(keepends=True)
def once(s, old, new):
    assert s.count(old)==1, (s.count(old),old[:120])
    return s.replace(old,new,1)
setup=get(10)
payload=re.search(r"_b64.b64decode\('([^']+)'\)",setup).group(1)
sources=json.loads(zlib.decompress(base64.b64decode(payload)))
old_sources=copy.deepcopy(sources)
sources['phase_sampling.py']=(ROOT/'phase_sampling.py').read_text(encoding='utf-8-sig')
s=sources['compact_infer.py']
s=once(s,'import numpy as np\n', 'import numpy as np\nfrom phase_sampling import phase_rows, ensemble_passes, PHASE_WEIGHTS\n')
# Preserve the base tensors exactly; extra bags use the same decoded/rendered slices.
anchor='                   "side": side_note}\n'
extra='''            out["phase_bags"] = []
            out["phase_window_counts"] = [len(cand)]
            all_centres = {(row[0], row[1]) for row in cand}
            for offset in (1, 2):
                pc = phase_rows(self.window_rows, blocks, lambda f, r: meta[r],
                                self.slices, self.eval_stride, self.phys_mm, offset)
                out["phase_window_counts"].append(len(pc))
                all_centres.update((row[0], row[1]) for row in pc)
                if pc == cand:
                    out["phase_bags"].append({"same_as_base": True})
                    continue
                pr = np.asarray(pc, dtype=np.int64)
                pi = np.stack([px[pr[:, 2 + d]] for d in range(self.slices)], 1)
                pm = meta[pr[:, 1]]
                pm6 = np.stack([pm[:, 1], pm[:, 3], pm[:, 4], pm[:, 5], pm[:, 6], pm[:, 7]], 1).astype(np.float32)
                if self.extent_meta:
                    pm6 = np.concatenate([pm6, np.stack([pm[:, 0], pm[:, 2], pm[:, 10]], 1).astype(np.float32)], 1)
                out["phase_bags"].append({"same_as_base": False,
                    "images": torch.from_numpy(pi), "meta": torch.from_numpy(pm6)})
            out["phase_unique_centres"] = len(all_centres)
'''
s=once(s,anchor,anchor+extra)
s=once(s,'    wr.writerow([UIDK] + LABELS)\n','''    wr.writerow([UIDK] + LABELS)
    bfh = open(out / "baseline_predictions.csv", "w", encoding="utf-8", newline="")
    bwr = csv.writer(bfh)
    bwr.writerow([UIDK] + LABELS)
    phase_records = []
''')
start=s.index('        tg = time.perf_counter()\n',s.index('    def handle(item):'))
end=s.index('        wr.writerow([uid]',start)
s=s[:start]+'''        tg = time.perf_counter()
        phase_cache = {}
        def predict_phase(index):
            if index == 0:
                images_, meta_ = item["images"], item["meta"]
            else:
                bag = item["phase_bags"][index - 1]
                if bag["same_as_base"]:
                    return phase_cache[0].copy()
                images_, meta_ = bag["images"], bag["meta"]
            zmean_, zs_ = score(images_, meta_)
            if not torch.isfinite(zmean_).all():
                raise ValueError("non-finite phase score")
            prob_ = torch.sigmoid(zmean_).double().cpu().numpy()
            # Each pass gets its OWN existing specialist heads before fusion.
            pass_uid = uid if index == 0 else uid + "|phase" + str(index)
            if STK is not None:
                apply_stacker(pass_uid, item, zs_, prob_)
            if SST is not None:
                apply_state_stacker(pass_uid, item, zs_, prob_)
            phase_cache[index] = prob_.copy()
            return prob_
        try:
            p, base_p, phase_errors = ensemble_passes(predict_phase)
        except Exception as exc:
            fallback[uid] = "phase baseline: " + repr(exc)[:200]
            return
        dt = time.perf_counter() - tg
        bwr.writerow([uid] + ["%.9g" % float(x) for x in base_p])
        bfh.flush()
        phase_record = {"uid": uid, "weights": list(PHASE_WEIGHTS),
            "windows_per_phase": item["phase_window_counts"],
            "unique_centres": item["phase_unique_centres"],
            "failed_phases": phase_errors,
            "max_probability_change": float(np.max(np.abs(p - base_p)))}
        phase_records.append(phase_record)
        print("PHASE_STUDY", json.dumps(phase_record), flush=True)
''' +s[end:]
# Do not mislabel last-pass logits as the fused result.
start=s.index('        if a.dump_logits:\n',s.index('    def handle(item):'))
end=s.index('\n    pos = 0\n',start)
s=s[:start]+'''        if a.dump_logits:
            dump[uid] = {"kind": "V5 post-stacker phase probabilities",
                         "baseline": base_p.tolist(), "ensemble": p.tolist(),
                         "passes": {str(k): v.tolist() for k, v in phase_cache.items()},
                         "phase_receipt": phase_record}
''' +s[end:]
s=once(s,'    fh.close()\n','    fh.close()\n    bfh.close()\n')
s=once(s,'    if FLIP is not None:                              # v4c\n', '''    receipt["phase_sampling"] = {"method": "three fixed training-sampler offsets",
        "weights": list(PHASE_WEIGHTS), "eval_stride": a.eval_stride,
        "physical_step_preserved": True, "specialist_heads_per_pass": True,
        "studies": phase_records,
        "failed_extra_studies": sum(bool(r["failed_phases"]) for r in phase_records),
        "sha256": hashlib.sha256((HERE / "phase_sampling.py").read_bytes()).hexdigest()}
    if FLIP is not None:                              # v4c
''')
sources['compact_infer.py']=s
# Keep original trainer, rendering modules and checkpoint pinning unchanged.
new_payload=base64.b64encode(zlib.compress(json.dumps(sources).encode('utf-8'),9)).decode('ascii')
setup=setup.replace(payload,new_payload)
sha={k:hashlib.sha256(v.encode('utf-8')).hexdigest() for k,v in sources.items()}
setup=re.sub(r'^SOURCE_SHA = .*$', 'SOURCE_SHA = '+repr(sha),setup,flags=re.M)
put(10,setup)
collect='''
# Preserve the baseline pass from these same studies for a controlled comparison.
_base_parts = [pd.read_csv(f, dtype={'StudyInstanceUID': str})
               for f in pred.glob('worker*/baseline_predictions.csv')]
assert _base_parts, 'No baseline phase predictions'
_base_all = pd.concat(_base_parts, ignore_index=True)
assert _base_all['StudyInstanceUID'].is_unique, 'Duplicate baseline study'
_base_ids = pd.read_csv(DATA / 'test.csv', dtype={'StudyInstanceUID': str})['StudyInstanceUID'].tolist()
_base_all = _base_all.set_index('StudyInstanceUID')
_base_all = _base_all.reindex(_base_ids)
_base_all = _base_all.fillna(_base_all.mean())
assert np.isfinite(_base_all.to_numpy()).all()
_base_all.reset_index().to_csv(WORK / 'ours_baseline.csv', index=False)
_phase_records = [p for w in receipt['workers'] for p in w.get('phase_sampling', {}).get('studies', [])]
assert len(_phase_records) == int(receipt['scored']), 'Missing phase receipts'
assert all(int(w.get('phase_sampling', {}).get('eval_stride', 0)) == 3 for w in receipt['workers']), 'V5 expects stride 3'
_phase_failures = sum(bool(p['failed_phases']) for p in _phase_records)
assert _phase_failures <= 0.03 * len(_base_ids), 'More than 3% of studies lost an extra phase'
assert int(receipt['fallback_studies']) <= 0.03 * len(_base_ids), 'More than 3% unscored studies'
(WORK / 'phase_sampling_receipt.json').write_text(json.dumps({
    'method': 'three fixed offsets of the training coverage sampler',
    'weights': [0.5, 0.25, 0.25], 'scored': len(_phase_records),
    'extra_phase_failure_studies': _phase_failures, 'studies': _phase_records,
    'validation': 'Sampling and artifact checks; no independent AUC measured'
}, indent=2))
'''
put(11,get(11)+'\nimport numpy as np\n'+collect)
blend=get(13)
base_blend=blend.replace("'ours_sub.csv'","'ours_baseline.csv'").replace("'submission.csv'","'baseline_submission.csv'").replace("'blend_receipt.json'","'baseline_blend_receipt.json'")
put(13,'# Run the identical blend for the base pass and for the phase ensemble.\n'+base_blend+'\n\n'+blend)
# New audit must never assert that changed predictions equal the original.
audit='''from pathlib import Path
import hashlib, json, numpy as np, pandas as pd, time
work = Path('/kaggle/working')
ids = pd.read_csv(DATA / 'test.csv', dtype={'StudyInstanceUID': str})['StudyInstanceUID'].tolist()
cols = pd.read_csv(DATA / 'sample_submission.csv', nrows=0).columns.tolist()
checks = {}
for filename in ('submission.csv', 'baseline_submission.csv'):
    frame = pd.read_csv(work / filename, dtype={'StudyInstanceUID': str})
    assert frame.columns.tolist() == cols
    assert frame['StudyInstanceUID'].tolist() == ids and frame['StudyInstanceUID'].is_unique
    values = frame[cols[1:]].to_numpy(dtype=np.float64)
    assert values.shape == (len(ids), 12) and len(ids) > 0
    assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all()
    checks[filename] = hashlib.sha256((work / filename).read_bytes()).hexdigest()
base_frame = pd.read_csv(work / 'baseline_submission.csv')
new_frame = pd.read_csv(work / 'submission.csv')
id_hash = hashlib.sha256('\\n'.join(ids).encode()).hexdigest()
preview_ids = id_hash == '9679026cd1dbad718cd9a8a7e717322d30f1fef5536d11685224a5cc26af3011'
preview_base_match = checks['baseline_submission.csv'] == '0df03e0fd8cf7f03d87f6a234f2417d2c31a6c98bfd952b122077965e4a53ca4' if preview_ids else None
if preview_ids:
    assert preview_base_match, 'Baseline pass differs from the verified V4 preview'
phase = json.loads((work / 'phase_sampling_receipt.json').read_text())
raw_delta = np.abs(pd.read_csv(work / 'ours_sub.csv')[cols[1:]].to_numpy() - pd.read_csv(work / 'ours_baseline.csv')[cols[1:]].to_numpy())
audit = {'candidate': 'Ehsan RSNA V5 three-offset sampling', 'status': 'PASS',
    'rows': len(ids), 'weights': [0.5, 0.25, 0.25], 'sha256': checks,
    'baseline_preview_byte_parity': preview_base_match,
    'raw_r6_max_change': float(raw_delta.max()),
    'raw_r6_mean_change': float(raw_delta.mean()),
    'submission_changed_cells': int((new_frame[cols[1:]].to_numpy() != base_frame[cols[1:]].to_numpy()).sum()),
    'mean_centres_base': float(np.mean([r['windows_per_phase'][0] for r in phase['studies']])),
    'mean_unique_centres_three_phases': float(np.mean([r['unique_centres'] for r in phase['studies']])),
    'extra_phase_failure_studies': phase['extra_phase_failure_studies'],
    'elapsed_seconds': round(time.time() - NB_T0, 1), 'official_score': None,
    'validation_limit': 'Preview parity verifies the control, not accuracy. No independent held-out AUC.'}
(work / 'v5_experiment_receipt.json').write_text(json.dumps(audit, indent=2))
print('EHSAN_V5_EXPERIMENT', json.dumps(audit), flush=True)
'''
put(15,audit)
put(0,'''# Ehsan RSNA V5 - three-offset slice sampling

**Experimental accuracy candidate; not scored or GPU-validated at creation.**
The last verified V4 public score was 0.958. The competition's listed final
submission deadline was 22 October 2026; this notebook was prepared on 23 October.

R6's stride-three evaluation uses one sampling offset. This experiment reuses
its original training coverage sampler at three fixed offsets, running each
bag through the existing models, anatomical mirror TTA and specialist heads.
The final R6 probability is a fixed logit pool: base pass 0.50, other offsets
0.25 each. These weights are conservative design choices, not fitted values.
For nonzero physical spacing, offsets preserve the training physical step;
three offsets do not necessarily cover every raw DICOM slice.

DICOM decoding and rendering occur once per study. The extra inference costs
up to about three times R6's GPU work; full runtime and memory remain to be measured.
The CoAtNet models and the outer per-finding rank blend are retained.

Outputs include `submission.csv`, `baseline_submission.csv`, and a phase audit.
The known three-study preview must reproduce V4 in the baseline output.
More than 3% missing studies or failed extra phases stops the notebook.
Run with **GPU T4 x2, Internet off**, and the inherited V4 input datasets.
''')
put(1,get(1).replace('Ehsan V4 changes only byte I/O, scheduling, 224-branch status tracking, and fallback handling. Checkpoints, DICOM transforms, image sizes, test-time augmentation, and fixed complete-data blend weights are retained.', 'Ehsan V5 adds fixed sampling offsets through the original trainer sampler and post-specialist logit pooling. It retains the V4 I/O changes, checkpoint sources and outer blend. Public model training and original architectures belong to the attributed authors.'))
put(9,'# Part A2 - Compact R6 with three sampling offsets and anatomical mirror TTA\n')
put(12,'# Part A3 - identical outer blending for the control and experimental predictions\n')
put(14,'# Final output contract, phase coverage and V4 control parity\n')
for cell in nb['cells']:
    if cell['cell_type']=='code':
        cell['execution_count']=None
        cell['outputs']=[]
# Validate syntax of every executable source before writing.
for name,src in sources.items(): ast.parse(src, filename=name)
for i,cell in enumerate(nb['cells']):
    if cell['cell_type']=='code':
        src=''.join(cell['source'])
        if src.startswith('%%writefile'): src=src.split('\n',1)[1]
        ast.parse(src,filename=f'cell_{i}')
assert sources['compact_train.py'] == old_sources['compact_train.py']
assert ''.join(nb['cells'][4]['source']) == ''.join(base['cells'][4]['source'])
assert ''.join(nb['cells'][7]['source']) == ''.join(base['cells'][7]['source'])
(ROOT/'candidate_sources').mkdir(exist_ok=True)
for name,src in sources.items(): (ROOT/'candidate_sources'/name).write_bytes(src.encode('utf-8'))
(ROOT/'Ehsan_RSNA_V5_Three_Offset_Experimental.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
print('BUILT V5; all cells and embedded sources parse; trainer and CoAtNet unchanged.')
