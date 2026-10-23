# Project history

Assembled on 23 October 2026 from recovered project files and prior conversation records. These are experiment records, not reconstructed Git commit dates.

| Stage | Work performed | Recorded result | Source availability |
| --- | --- | --- | --- |
| DINOv2 baseline, 13–14 October | Ran an ensemble of 20 pretrained DINOv2 models; created a submission | Public ROC-AUC 0.891; initially selected for final evaluation | Historical record and notebook link only |
| DINOv3-only, 15 October | Evaluated the DINOv3-only alternative and compared it with the baseline | Public ROC-AUC 0.882; baseline retained at that point | Historical record and notebook link only |
| OOF inspection, 15 October | Inspected an artifact with 4,407 studies, 12 targets, and 58 gold-mask studies | Derived-label macro AUC 0.8486; not an official score or independent clean-label validation | Historical analysis record; OOF arrays not included |
| CoAtNet anatomical mirror, 15 October | Copied and ran the 384-pixel community approach | The source advertised 0.949; this is not presented as a verified score for Ehsan's copied notebook | Historical record and notebook link only |
| CoAtNet + Compact R7 | Tried a 50/50 ensemble; later project history records 0.956 | Historical public score record, not re-verified during packaging | Original notebook export not recovered |
| R6 ensemble V1 / V2 | Used R6 flip TTA with CoAtNet 384/224, fixed per-finding weights, and investigated the optional 2.5D reader | Project history records the 0.958 baseline | V3 snapshot retains explanatory history; separate V1/V2 exports not recovered |
| V3 ablation | Removed the optional 2.5D reader while keeping the main ensemble and weights | Recorded public score 0.958; preview 553.6 s, matching the known Stage 3 checksum | Executed baseline in `experiments/v4/baseline.ipynb` |
| V4 optimization | Added bounded DICOM byte caching, workload-balanced GPU scheduling, loader limits, and independent 224-branch failure tracking | 7 local tests; preview 514.6 s and identical bytes; last verified public score 0.958 | Full recovered notebook, sources, builder, tests, and receipts |
| V5, 23 October | Added three fixed slice-sampling offsets to R6, retained per-pass specialist heads and mirror TTA, and pooled probabilities in logit space | 8 local synthetic/mocked tests; no GPU run, official submission, or verified accuracy gain | Full delivered notebook, source, builder, and tests |

## Known notebook references

- [DINOv2 baseline](https://www.kaggle.com/code/ehsan1993/rsna-knee-baseline-v1)
- [DINOv3-only test](https://www.kaggle.com/code/ehsan1993/rsna-knee-dinov3-test-v1)
- [Copied anatomical mirror notebook](https://www.kaggle.com/code/ehsan1993/rsna-knee-0949-anatomical-mirror)
- [V4 optimized, version 361061645](https://www.kaggle.com/code/ehsan1993/ehsan-rsna-v4-optimized?scriptVersionId=361061645)

V3's recorded version is 360371736. No exact V3 slug has been inferred.

## How to read the evidence

`evidence/v4_status_at_submission.json` was captured while hidden-test execution was still running, so its candidate score is null. Later conversation history records the verified 0.958 result. Those are different observation times.

The V3/V4 notebook outputs contain the actual preview receipts. Their `official_score: null` fields are expected because preview execution cannot fetch the later official Kaggle score. Do not interpret this as a V5 score.

The 58 gold-mask studies were already used for model selection. They do not provide an untouched holdout for further tuning. V5 weights are fixed design choices, not a claim of fitted optimal weights.

## Remaining gaps

This repository includes all code and experiment files recovered from the two available project workspaces. Earlier DINO and R7 exports, complete historical Kaggle output archives, and the newly referenced screenshots were not available as file bytes during packaging. Their absence is explicit; no replacement source, measurements, or screenshots have been fabricated.

The last known selected submissions and final private leaderboard outcome were not re-checked during this archive operation.
