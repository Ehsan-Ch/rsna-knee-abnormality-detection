# Stage 3: per-finding rank blend with weights fixed before any run: Raptor 0.5, ours 0.5.  Not fitted on any labels:
# [v3] Each model is ranked over its own non-failed studies; more than 50 % of the studies unscored by our stage stops the notebook;
# any fallback prints BLEND_FALLBACK.
# Per-finding override (user decision 10-10 17:15, fixed before any run): Medial OA 0.50, Lateral OA 0.50, PF OA 0.50 follow Raptor.
# Raptor trained on every training study and was selected on the 58 gold, so no clean labelled set exists to fit them.
# A study that one model failed on takes the other model's rank; more than 3 % Raptor failures stops the notebook.
import csv
import numpy as np
import pandas as pd
W_RAPTOR, W_OURS = 0.5, 0.5
W_RAPTOR_C = {'Medial OA': 0.5, 'Lateral OA': 0.2, 'PF OA': 0.6, 'Synovitis': 0.8, "Baker's": 0.35}   # [wc] per-finding Raptor weight (compact slot = 1 - it): heliosli's 0.958 notebook, 16 Oct   # per-finding Raptor weight (ours = 1 - it); every other finding uses W_RAPTOR
assert all(0.0 <= w_ <= 1.0 for w_ in (W_RAPTOR, W_OURS) + tuple(W_RAPTOR_C.values())), 'blend weights outside [0, 1]'   # [v3] a per-finding Raptor weight may be exactly 1.0 (ours 0)
test_ids = pd.read_csv(DATA / 'test.csv')['StudyInstanceUID'].astype(str).tolist()
ss = DATA / 'sample_submission.csv'
ours = pd.read_csv(WORK / 'ours_sub.csv', dtype={'StudyInstanceUID': str})
cols = list(pd.read_csv(ss, nrows=1).columns) if ss.is_file() else list(ours.columns)
labels = cols[1:]
raw = pd.read_csv(WORK / 'raptor_raw.csv', dtype={'StudyInstanceUID': str})
rsub = pd.read_csv(WORK / 'raptor_sub.csv', dtype={'StudyInstanceUID': str})
for df, nm in ((ours, 'ours'), (raw, 'raptor_raw'), (rsub, 'raptor_sub')):
    assert sorted(df['StudyInstanceUID']) == sorted(test_ids), nm + ': rows differ from test.csv'
    assert set(labels) <= set(df.columns), nm + ': columns differ'
ours = ours.set_index('StudyInstanceUID').loc[test_ids, labels]
raw = raw.set_index('StudyInstanceUID').loc[test_ids, labels]
rsub = rsub.set_index('StudyInstanceUID').loc[test_ids, labels]
A224 = 0.35   # [s5] share of the Raptor slot given to the 224-crop model; fixed before any run
r224 = None
r224_ok = np.zeros(len(test_ids), dtype=bool)
if R224_OK:
    try:
        _r = pd.read_csv(WORK / 'raptor224_sub.csv', dtype={'StudyInstanceUID': str})
        assert sorted(_r['StudyInstanceUID']) == sorted(test_ids) and set(labels) <= set(_r.columns)
        r224 = _r.set_index('StudyInstanceUID').loc[test_ids, labels]
        assert np.isfinite(r224.values).all()
        _status = pd.read_csv(WORK / 'raptor224_status.csv', dtype={'StudyInstanceUID': str})
        assert _status['StudyInstanceUID'].is_unique and sorted(_status['StudyInstanceUID']) == sorted(test_ids)
        _status = _status.set_index('StudyInstanceUID').loc[test_ids, 'scored']
        assert _status.isin([0, 1]).all()
        r224_ok = _status.to_numpy(dtype=bool)
    except Exception as _e:
        print('R224_UNUSABLE', repr(_e), flush=True); r224 = None
n = len(test_ids)
rap_fail = (raw.values == 0.5).all(axis=1)             # Raptor leaves a failed study at 0.5 in every column
scored = set()                                          # ours fills a failed study with the column mean: read what was scored
for f in pred.glob('worker*/predictions.csv'):
    with open(f, encoding='utf-8') as fh:
        for r in csv.DictReader(fh):
            try:
                v = [float(r[L]) for L in labels]
            except (KeyError, ValueError, TypeError):
                continue
            if all(0. <= x <= 1. for x in v):
                scored.add(r['StudyInstanceUID'])
our_fail = np.array([u not in scored for u in test_ids])
if rap_fail.sum() > 0.03 * n:
    raise RuntimeError('Raptor failed on %d of %d studies' % (rap_fail.sum(), n))
if our_fail.sum() > 0.5 * n:                            # [v3] no silent Raptor-only submission under the blend's name
    raise RuntimeError('our stage did not score %d of %d studies (more than 50 %%)' % (our_fail.sum(), n))


def pct(v):
    return pd.Series(np.asarray(v, dtype=np.float64)).rank(method='average', pct=True).values


def pct_on(v, ok):
    # [v3] percentile rank among the rows in ok only (one model's non-failed studies); the other rows are NaN and are
    # replaced below by the other model's rank (or 0.5), so a mean-filled failed row no longer shifts the scored rows' ranks
    r = np.full(len(v), np.nan)
    if ok.any():
        r[ok] = pct(np.asarray(v, dtype=np.float64)[ok])
    return r


out = np.zeros((n, len(labels)))
rho_rr, rho_ro = {}, {}
for j, c in enumerate(labels):
    rr, ro = pct_on(raw[c].values, ~rap_fail), pct_on(ours[c].values, ~our_fail)   # [v3] each over its own non-failed rows
    if r224 is not None:
        # Use the 224 branch only on its own successfully scored studies.
        # Complete-data arithmetic stays identical to V3.
        ok224 = (~rap_fail) & r224_ok
        rr224 = pct_on(r224[c].values, ok224)
        rr[ok224] = (1.0 - A224) * rr[ok224] + A224 * rr224[ok224]
    wr = W_RAPTOR_C.get(c, W_RAPTOR)
    b = wr * rr + (1.0 - wr) * ro
    b[rap_fail] = ro[rap_fail]
    b[our_fail] = rr[our_fail]
    b[rap_fail & our_fail] = 0.5
    out[:, j] = b
    if n >= 3:
        rho_rr[c] = round(float(pd.Series(raw[c].values).corr(pd.Series(rsub[c].values), method='spearman')), 4)
        rho_ro[c] = round(float(pd.Series(raw[c].values).corr(pd.Series(ours[c].values), method='spearman')), 4)
if rho_rr and min(rho_rr.values()) < 0.99:
    raise RuntimeError('Raptor raw dump does not match its own submission ranks: %s' % rho_rr)
sub = pd.DataFrame(out.astype(np.float32), columns=labels)
sub.insert(0, 'StudyInstanceUID', test_ids)
sub = sub[cols]
assert sub['StudyInstanceUID'].tolist() == test_ids and np.isfinite(sub[labels].values).all()
assert ((sub[labels].values >= 0) & (sub[labels].values <= 1)).all()
sub.to_csv(WORK / 'submission.csv', index=False)
blend = {'weights': {'raptor': W_RAPTOR, 'ours': W_OURS, 'raptor_per_class': W_RAPTOR_C}, 'rows': n, 'raptor_failed': int(rap_fail.sum()),
         'ours_failed': int(our_fail.sum()), 'raptor_seconds': round(RAPTOR_SECONDS, 1),
         'total_seconds': round(time.time() - NB_T0, 1), 'spearman_raptor_raw_vs_sub': rho_rr,
         'spearman_raptor_vs_ours': rho_ro,
         'mode': 'blend', 'both_failed': int((rap_fail & our_fail).sum()),   # [v3] (> 50 % ours unscored stops above)
         'rank_basis': 'each model over its own non-failed rows', 'raptor224_available': r224 is not None, 'raptor224_failed': int((~r224_ok).sum())}
(WORK / 'blend_receipt.json').write_text(json.dumps(blend, indent=1))
if rap_fail.any() or our_fail.any():                     # [v3] loud: a fallback changed what this submission is
    print('BLEND_FALLBACK ours_failed=%d raptor_failed=%d both_failed=%d of %d' % (
        int(our_fail.sum()), int(rap_fail.sum()), int((rap_fail & our_fail).sum()), n), flush=True)
print('NOTEBOOK_COMPLETE', json.dumps(blend), flush=True)
display(sub.head())