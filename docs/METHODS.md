# Methods and validation

## Model pipeline

The inherited main ensemble has three prediction branches:

1. CoAtNet at 384 pixels with anatomical-mirror inference.
2. CoAtNet at 224 pixels with the mirror view.
3. Compact R6 with four fold models, specialist heads, and anatomical flip TTA.

The CoAtNet slot combines percentile ranks as 0.65 × rank(384) + 0.35 × rank(224). It is blended with R6 using the inherited fixed per-finding CoAtNet weights:

| Finding | CoAtNet weight | R6 weight |
| --- | ---: | ---: |
| Lateral OA | 0.20 | 0.80 |
| PF OA | 0.60 | 0.40 |
| Synovitis | 0.80 | 0.20 |
| Baker's cyst | 0.35 | 0.65 |
| All other findings | 0.50 | 0.50 |

No new outer-blend weight search was performed.

## V4 engineering changes

- Immutable DICOM bytes are cached within a 128 MiB bound per loader.
- Up to eight concurrent reads reduce repeated I/O.
- Estimated file-count work is assigned across two GPUs when this improves the predicted maximum load.
- Loader processes are limited by the number of assigned studies.
- The 224 branch uses its own success mask so failures do not silently enter the blend.
- Final audits check row ordering, IDs, dimensions, finite values, prediction bounds, and the known preview checksum.

Seven existing tests cover the implemented I/O, scheduling, blend parity, and failure cases. The local Spearman diagnostic uses Pearson correlation of average ranks because SciPy was unavailable in the original local test environment. Production Kaggle code was unchanged.

## V5 sampling experiment

R6 normally uses one stride-three coverage offset. V5 uses its original training coverage sampler with three fixed offsets, while preserving physical slice spacing and series/view boundaries.

Each bag passes through the existing scoring path, anatomical mirror TTA, and specialist heads. Predictions are pooled after those heads with fixed logit-space weights **0.50, 0.25, 0.25**. Identical bags reuse the baseline prediction. Failed extra phases are omitted and surviving weights renormalized; an invalid baseline remains a hard error.

DICOM decoding and rendering are reused. R6 GPU work can approach three times the baseline, and extra bags increase CPU memory pressure. No measured full-run performance is available. When physical spacing is nonzero, three offsets need not cover every raw DICOM slice.

The notebook exports a baseline control and the experimental output. More than 3% missing studies or failed extra phases stops the run. Known preview IDs require the control submission's byte checksum to match V4.

## V5 test coverage

The eight existing tests cover:

- Exact original baseline sampling and coverage over synthetic series lengths 1–100.
- No crossing of series or view boundaries.
- Preserved physical neighbor spacing.
- Stable probability pooling, identity cases, and input validation.
- Extra-phase failure handling and weight renormalization.
- Baseline failure propagation.
- The actual modified loader fragment with synthetic tensors.
- The actual modified worker fragment with mocked Torch/scoring/specialist heads.

These checks establish local logic behavior. They do not establish GPU correctness, full-dataset runtime, or accuracy.

## Packaging checks

`scripts/check_project.py` runs the 15 existing tests, verifies original import hashes, parses code cells and embedded sources, checks clean entry-notebook metadata, and scans the packaged text for common credential signatures. `scripts/build_notebooks.py` rebuilds in a temporary directory, preserving archived snapshots. Clean entry notebooks strip inherited execution output and stale run metadata, enable the intended GPU import flag, and retain model code and input references.
