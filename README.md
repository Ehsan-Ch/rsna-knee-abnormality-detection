# RSNA Knee Abnormality Detection

Ehsan Cheraghi's Kaggle project: community-model ensembles, controlled ablations, inference optimization, and an experimental three-offset sampling approach for 12 knee MRI findings.

**Last recorded official public score: 0.958 (V4). V5 is experimental and unscored.** This repository preserves the recovered work and distinguishes executed results from proposals. It is a research/competition project.

## Start here

- [V4 optimized notebook](notebooks/Ehsan_RSNA_V4_Optimized.ipynb): the engineering revision with the recorded 0.958 result.
- [V5 three-offset notebook](notebooks/Ehsan_RSNA_V5_Three_Offset_Experimental.ipynb): new sampling experiment, not GPU-validated or scored.
- [Kaggle inputs and execution](docs/KAGGLE_SETUP.md)
- [Experiment history](docs/PROJECT_HISTORY.md)
- [Design and validation](docs/METHODS.md)
- [Sources, attribution, and archive scope](docs/PROVENANCE.md)
- [Original model acknowledgements](THIRD_PARTY_NOTICES.md)

## Results and limits

| Experiment | Recorded public ROC-AUC | Evidence and interpretation |
| --- | ---: | --- |
| DINOv2 baseline | 0.891 | Historical project record; original source not recovered here |
| DINOv3-only test | 0.882 | Historical project record; did not improve the baseline |
| R6 ensemble V2 / V3 | 0.958 | Historical submissions; V3 executed snapshot included |
| V4 engineering optimization | 0.958 | Last result verified during the prior Kaggle session |
| V5 three-offset sampling | — | 8 synthetic/mocked tests; no GPU run or official score |

V4's three-study preview took **514.6 seconds**, compared with **553.6 seconds** for V3 (about 7% faster in one comparison). The preview submission was byte-identical. This is a small smoke test, not an accuracy validation or a full-test runtime benchmark.

The competition's recorded final submission deadline was 22 October 2026 at 23:59 UTC. V5 was prepared on 23 October. No private leaderboard result is claimed.

## Run on Kaggle

Import a notebook from `notebooks/`, attach the four [required inputs](docs/KAGGLE_SETUP.md), select **GPU T4 ×2**, and set **Internet off**. The trained checkpoints and competition images remain external inputs; this repository contains code and project records.

## Local verification

Use Python 3.12, then:

```bash
python -m pip install -r requirements-local.txt
python scripts/check_project.py
```

This runs the existing 7 V4 and 8 V5 tests and validates the packaged notebooks and imported file hashes. It does not run MRI model inference.

Rebuild clean notebooks without modifying the original snapshots:

```bash
python scripts/build_notebooks.py
```

## Repository layout

| Path | Contents |
| --- | --- |
| `notebooks/` | Clean Kaggle entry notebooks with corrected execution metadata |
| `experiments/v4/` | Recovered V3 baseline, executed V4 snapshot, builder, source, and 7 tests |
| `experiments/v5/` | V4 control, delivered V5 snapshot, candidate source, builder, and 8 tests |
| `evidence/` | Extracted preview receipts, submission-time status, import hashes, and local check report |
| `docs/` | Inputs, experiment history, methods, and provenance |
| `scripts/` | Portable build and verification commands |

Archived notebook metadata reflects the time of export and can contain stale execution flags. Use `notebooks/` for fresh imports. The original bytes are retained under `experiments/`.

## Credits

The pretrained models and original architectures are community work. Ehsan's contributions here are experiment execution, ablation, pipeline engineering, failure handling, and the V5 sampling experiment. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). No new blanket license is applied to upstream code or weights.
