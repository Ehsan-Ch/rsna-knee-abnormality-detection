from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace


class OneStudySource:
    """Bounded byte cache; DICOM parsing and pixel arithmetic stay unchanged.

    Only immutable competition inputs are cached. Prefetch uses at most eight
    file reads at once and stops retaining data at 128 MiB per loader process.
    A failed speculative read is retried by the original synchronous path.
    """
    current = None
    cache_limit = 128 * 1024 * 1024
    io_workers = 8

    def __init__(self, root, split):
        self.root = Path(root)
        self.split = split
        self._cache = OrderedDict()
        self._bytes = 0
        self._listed = None

    def _remember(self, name, blob):
        if len(blob) > self.cache_limit:
            return
        old = self._cache.pop(name, None)
        if old is not None:
            self._bytes -= len(old)
        while self._cache and self._bytes + len(blob) > self.cache_limit:
            self._bytes -= len(self._cache.popitem(last=False)[1])
        self._cache[name] = blob
        self._bytes += len(blob)

    def read(self, item):
        name = item.filename if hasattr(item, 'filename') else item
        name = str(name)
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        blob = (self.root / name).read_bytes()
        self._remember(name, blob)
        return blob

    def infolist(self):
        if self._listed is not None:
            return self._listed
        base = self.root / (self.split + '_series') / OneStudySource.current
        paths = sorted(base.glob('*/*.dcm'))
        self._listed = [SimpleNamespace(filename=p.relative_to(self.root).as_posix()) for p in paths]
        try:
            total = sum(p.stat().st_size for p in paths)
        except OSError:
            total = self.cache_limit + 1
        # Larger studies use the bounded LRU path, without whole-study prefetch.
        if paths and total <= self.cache_limit:
            def read_optional(p):
                try:
                    return p.read_bytes()
                except OSError:
                    return None
            with ThreadPoolExecutor(max_workers=min(self.io_workers, len(paths))) as pool:
                # Batches bound queued results, including unexpectedly changed inputs.
                for offset in range(0, len(paths), self.io_workers):
                    batch = paths[offset:offset + self.io_workers]
                    for p, blob in zip(batch, pool.map(read_optional, batch)):
                        if blob is not None:
                            self._remember(p.relative_to(self.root).as_posix(), blob)
        return self._listed

    def close(self):
        self._cache.clear()
        self._bytes = 0


def balanced_parts(ids, costs, n_gpu):
    """Choose LPT only when its estimated makespan beats existing round robin."""
    order = sorted(ids, key=lambda u: (-costs[u], u))
    original = [order[i::n_gpu] for i in range(n_gpu)]
    candidate = [[] for _ in range(n_gpu)]
    loads = [0] * n_gpu
    for uid in order:
        g = min(range(n_gpu), key=lambda j: (loads[j], len(candidate[j]), j))
        candidate[g].append(uid)
        loads[g] += max(1, costs[uid])
    old_max = max((sum(max(1, costs[u]) for u in part) for part in original), default=0)
    selected = candidate if max(loads, default=0) < old_max else original
    return [part[::2] + part[1::2] for part in selected]
