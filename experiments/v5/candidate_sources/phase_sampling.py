"""Fixed sampling offsets using the original trainer's coverage sampler."""
import numpy as np
PHASE_WEIGHTS = (0.5, 0.25, 0.25)

class FixedPhase:
    def __init__(self, offset, stride):
        self.offset, self.stride = int(offset), int(stride)
    def integers(self, low, high):
        if low != 0 or high < 1 or self.stride < 1:
            raise ValueError('Unexpected trainer sampling range')
        physical_step = max(1, int(high) // self.stride)
        return (1 + self.offset * physical_step) % int(high)

def phase_rows(window_rows, blocks, meta_of, slices, stride, phys_mm, offset):
    if offset == 0:
        return window_rows(blocks, meta_of, 'eval', slices, stride, phys_mm=phys_mm)
    return window_rows(blocks, meta_of, 'train', slices, stride,
                       cover_stride=stride, phys_mm=phys_mm,
                       rng=FixedPhase(offset, stride), slice_jitter=0.0,
                       train_sampling='cover')

def pool_probabilities(probabilities, weights):
    """Weighted logit pool AFTER each pass's original specialist heads."""
    p = np.asarray(probabilities, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    if p.ndim != 2 or w.shape != (p.shape[0],) or not len(w):
        raise ValueError('Invalid probability/weight shapes')
    if not np.isfinite(p).all() or not ((p >= 0) & (p <= 1)).all():
        raise ValueError('Invalid probabilities')
    if not np.isfinite(w).all() or (w < 0).any() or w.sum() <= 0:
        raise ValueError('Invalid weights')
    if len(p) == 1 or np.array_equal(p, np.broadcast_to(p[0], p.shape)):
        return p[0].copy()
    q = np.clip(p, 1e-7, 1 - 1e-7)
    z = (w[:, None] * (np.log(q) - np.log1p(-q))).sum(0) / w.sum()
    return 1.0 / (1.0 + np.exp(-z))

def ensemble_passes(predict, weights=PHASE_WEIGHTS):
    """Baseline failure propagates; extra failures are logged and renormalized."""
    base = np.asarray(predict(0), dtype=np.float64)
    pool_probabilities([base], [1.0])
    probabilities, used_weights = [base], [weights[0]]
    errors = {}
    for index in range(1, len(weights)):
        try:
            p = np.asarray(predict(index), dtype=np.float64)
            if p.shape != base.shape:
                raise ValueError('Phase output shape differs from baseline')
            pool_probabilities([p], [1.0])
        except Exception as exc:
            errors[str(index)] = repr(exc)[:200]
            continue
        probabilities.append(p)
        used_weights.append(weights[index])
    return pool_probabilities(probabilities, used_weights), base, errors
