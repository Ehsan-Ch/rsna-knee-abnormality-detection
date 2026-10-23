import ast
import base64
import copy
import hashlib
import json
import re
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent
nb = json.loads((ROOT / 'baseline.ipynb').read_text(encoding='utf-8'))
original = copy.deepcopy(nb)

def get(i):
    return ''.join(nb['cells'][i]['source'])

def put(i, source):
    nb['cells'][i]['source'] = source.splitlines(keepends=True)

def replace_once(source, before, after):
    assert source.count(before) == 1, before[:100]
    return source.replace(before, after, 1)

setup = get(10)
payload = re.search(r"_b64.b64decode\('([^']+)'\)", setup).group(1)
sources = json.loads(zlib.decompress(base64.b64decode(payload)))
old_hashes = {k: hashlib.sha256(v.encode()).hexdigest() for k, v in sources.items()}
io = (ROOT / 'io_source.py').read_text(encoding='utf-8')
reader = io[:io.index('\n\ndef balanced_parts')]
start = sources['probe_infer.py'].index('class OneStudySource:')
end = sources['probe_infer.py'].index('\n\ndef quantize_series_slicewise', start)
sources['probe_infer.py'] = sources['probe_infer.py'][:start] + reader + sources['probe_infer.py'][end:]
controller = sources['probe_dual.py']
controller = controller.replace('\n\ndef main():', io[io.index('\n\ndef balanced_parts'):] + '\n\ndef main():', 1)
controller = replace_once(controller,
    '    order = sorted(ids, key=lambda u: (-costs[u], u))\n    parts = [order[i::n_gpu] for i in range(n_gpu)]\n    # inside a worker: keep the big studies from arriving back to back\n    parts = [p[::2] + p[1::2] for p in parts]',
    '    n_gpu = min(n_gpu, max(1, len(ids)))\n    parts = balanced_parts(ids, costs, n_gpu)\n    print("EHSAN_SCHEDULING", json.dumps({"loads": [sum(costs[u] for u in p) for p in parts], "studies": [len(p) for p in parts]}), flush=True)')
controller = replace_once(controller, '"--workers", str(a.loader_workers)', '"--workers", str(min(a.loader_workers, len(part)))')
sources['probe_dual.py'] = controller
new_payload = base64.b64encode(zlib.compress(json.dumps(sources).encode(), 9)).decode()
setup = setup.replace(payload, new_payload)
for name, old_hash in old_hashes.items():
    new_hash = hashlib.sha256(sources[name].encode()).hexdigest()
    if new_hash != old_hash:
        setup = replace_once(setup, old_hash, new_hash)
put(10, setup)

# Add an explicit success mask to the otherwise unchanged 224 inference.
r224 = get(7)
r224 = replace_once(r224, '    pos = {sid: i for i, sid in enumerate(test_ids)}',
    '    pos = {sid: i for i, sid in enumerate(test_ids)}\n    inference_ok = np.zeros(N, dtype=bool)  # Ehsan V4: branch-specific status')
r224 = replace_once(r224, '                        del xm\n            except Exception as e:',
    '                        del xm\n                inference_ok[i] = all(np.isfinite(p[i]).all() and ((p[i] >= 0) & (p[i] <= 1)).all() for p in arm_probs)\n            except Exception as e:')
r224 = replace_once(r224, '    sub.to_csv(out, index=False)',
    '    sub.to_csv(out, index=False)\n    pd.DataFrame({"StudyInstanceUID": test_ids, "scored": inference_ok.astype(int)}).to_csv("/kaggle/working/raptor224_status.csv", index=False)')
put(7, r224)

blend = get(13)
blend = replace_once(blend, 'r224 = None\nif R224_OK:',
    'r224 = None\nr224_ok = np.zeros(len(test_ids), dtype=bool)\nif R224_OK:')
blend = replace_once(blend, '        assert np.isfinite(r224.values).all()',
    '''        assert np.isfinite(r224.values).all()
        _status = pd.read_csv(WORK / 'raptor224_status.csv', dtype={'StudyInstanceUID': str})
        assert _status['StudyInstanceUID'].is_unique and sorted(_status['StudyInstanceUID']) == sorted(test_ids)
        _status = _status.set_index('StudyInstanceUID').loc[test_ids, 'scored']
        assert _status.isin([0, 1]).all()
        r224_ok = _status.to_numpy(dtype=bool)''')
blend = replace_once(blend,
    '''    if r224 is not None:   # [s5] mix the 224 model into the Raptor slot, over Raptor's non-failed rows
        rr = (1.0 - A224) * rr + A224 * pct_on(r224[c].values, ~rap_fail)''',
    '''    if r224 is not None:
        # Use the 224 branch only on its own successfully scored studies.
        # Complete-data arithmetic stays identical to V3.
        ok224 = (~rap_fail) & r224_ok
        rr224 = pct_on(r224[c].values, ok224)
        rr[ok224] = (1.0 - A224) * rr[ok224] + A224 * rr224[ok224]''')
blend = replace_once(blend, "'rank_basis': 'each model over its own non-failed rows'}",
    "'rank_basis': 'each model over its own non-failed rows', 'raptor224_available': r224 is not None, 'raptor224_failed': int((~r224_ok).sum())}")
put(13, blend)

# Replace stale first-person claims copied from the original public author.
history = get(1)
oa = history[history.index('*OAI acknowledgement'):history.index('## What is ours')]
put(1, '''## Model provenance and attribution

This notebook is Ehsan's engineering revision of an existing community ensemble.
It does not claim that Ehsan trained the public models or developed the original reader.

- [nartaa / Danial Zakaria](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror): CoAtNet 384/224 checkpoints and anatomical-mirror inference, based on [Dread Development's Raptor](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense).
- [heliosli](https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters): Compact R6, flip TTA, source bundle, and fixed per-finding blend weights.
- [rostigor](https://www.kaggle.com/datasets/rostigor/rsna-knee-r6-fe055406): pinned R6 dataset copy.
- [varshithakolupuri](https://www.kaggle.com/code/varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956): 224-pixel model contribution to the CoAtNet slot.
- [goodpjw2008](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-s6-2-5d-mil-lb-0-954): source combined notebook and historical 2.5D reader, omitted here.

The original authors' comments remain inside their source. Ehsan V4 changes only byte I/O, scheduling, 224-branch status tracking, and fallback handling. Checkpoints, DICOM transforms, image sizes, test-time augmentation, and fixed complete-data blend weights are retained.

''' + oa)
put(0, '''# Ehsan RSNA V4 - bounded I/O and reliable blending

Based on Ehsan's **official 0.958 V3 submission**, version 360371736.
This candidate is an engineering optimization; no increase in AUC is claimed before scoring.

Changes:
- Cache repeated immutable DICOM byte reads, bounded to 128 MiB per loader; at most eight concurrent reads.
- Allocate studies by estimated file-count workload when that reduces the predicted maximum GPU load.
- Avoid starting more loader processes than assigned studies.
- Track the 224 model's own success mask so an unsuccessful prediction cannot enter the blend unnoticed.
- Validate output identity, ordering, finite values, and the known preview checksum.

Run with **GPU T4 x2, Internet off**, using the inherited input versions.
The three visible test studies are a smoke test, not an accuracy validation set.
Use independent out-of-fold predictions before any future tuning of ensemble weights.
''')
put(9, get(9).replace('(unchanged)', '(model unchanged; I/O and scheduling optimized)'))
put(12, get(12).replace('This writes the submission that Part B nudges.', 'This is the final submission; the optional reader is omitted.'))
put(14, '## Final contract and preview parity audit\n\nThe final prediction file is the Stage 3 blend. No extra model or post-hoc weight tuning runs.\n')
audit = get(15)
audit = audit.replace('Ehsan R6 V3 no-2.5D reader ablation (based on official V2 0.958)', 'Ehsan R6 V4 bounded I/O and branch-specific fallback (V3 baseline 0.958)')
audit = audit.replace('Remove optional 2.5D reader, retain CoAtNet 384/224 + R6 Stage 3 unchanged', 'Bounded byte cache, balanced scheduling, independent 224 failure mask')
audit = audit.replace("'main_model_code_unchanged': True", "'model_math_unchanged': True, 'io_scheduling_and_fallback_changed': True")
audit = audit.replace('v3_ablation_receipt.json', 'v4_optimization_receipt.json').replace('EHSAN_V3_ABLATION_AUDIT:', 'EHSAN_V4_OPTIMIZATION_AUDIT:')
audit += '''
if _same_preview_ids:
    assert _preview_parity, 'V4 must reproduce the known three-study V3 submission before scoring'
_optimization = {
    'baseline_notebook_version': 360371736,
    'baseline_public_auc_displayed': 0.958,
    'candidate_public_auc': None,
    'baseline_preview_pipeline_seconds': 553.6,
    'candidate_preview_pipeline_seconds': round(time.time() - NB_T0, 1),
    'preview_identity_match': _same_preview_ids,
    'preview_submission_byte_parity': _preview_parity,
    'cache_limit_mib_per_loader': 128,
    'io_threads_per_loader': 8,
    'r224_failed': _blend_rcpt.get('raptor224_failed'),
    'accuracy_validation': 'Not independently measured; public score requires Kaggle submission',
}
(_audit_out / 'optimization_receipt.json').write_text(_audit_json.dumps(_optimization, indent=2))
print('EHSAN_OPTIMIZATION_RESULT', _audit_json.dumps(_optimization), flush=True)
'''
put(15, audit)

for i, cell in enumerate(nb['cells']):
    cell['metadata'] = {}
    if cell['cell_type'] == 'code':
        cell['execution_count'] = None
        cell['outputs'] = []
        source = ''.join(cell['source'])
        if source.startswith('%%writefile '):
            source = source.split('\n', 1)[1]
        ast.parse(source, filename=f'cell_{i}')
for name, code in sources.items():
    ast.parse(code, filename=name)
    (ROOT / 'embedded' / name).write_text(code, encoding='utf-8')
nb['metadata'].pop('papermill', None)
nb['metadata']['ehsan_experiment'] = {'baseline_version': 360371736, 'changes': 'bounded byte cache; cost-balanced scheduling; independent 224 failure tracking', 'status': 'requires Kaggle execution and scoring'}
nb['metadata']['kaggle']['isInternetEnabled'] = False
nb['metadata']['kaggle']['isGpuEnabled'] = True
nb['metadata']['kaggle']['accelerator'] = 'nvidiaTeslaT4'
dest = ROOT / 'Ehsan_RSNA_V4_Optimized.ipynb'
dest.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
(ROOT / 'blend_v4.py').write_text(get(13), encoding='utf-8')
assert get(4) == ''.join(original['cells'][4]['source'])
assert hashlib.sha256(sources['compact_infer.py'].encode()).hexdigest() == old_hashes['compact_infer.py']
print(json.dumps({'path': str(dest), 'bytes': dest.stat().st_size, 'code_cells_compiled': 9, 'embedded_sources_compiled': len(sources)}))
