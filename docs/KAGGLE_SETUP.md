# Kaggle inputs and execution

These four inputs are required by V4 and V5.

| Type | Input | Purpose |
| --- | --- | --- |
| Competition | [rsna-knee-abnormality-detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data) | Test metadata, sample submission, MRI studies |
| Dataset | [nartaa/rsna-knee-publication-swa-weights-20261007](https://www.kaggle.com/datasets/nartaa/rsna-knee-publication-swa-weights-20261007) | CoAtNet 384 and 224 checkpoints |
| Dataset | [rostigor/rsna-knee-r6-fe055406](https://www.kaggle.com/datasets/rostigor/rsna-knee-r6-fe055406) | Frozen Compact R6 checkpoints and manifest |
| Dataset | [heliosli/rsna-orthofoundation-fold0-epoch7-20261004](https://www.kaggle.com/datasets/heliosli/rsna-orthofoundation-fold0-epoch7-20261004) | Supporting source and offline dependencies |

## Required asset checks

The SWA dataset must contain both:

- `raptor_ft_alldata_t16_blendjev_oai_d96_r384_swa.pt`
- `raptor_ft_alldata_t16_blendjev_oai_d96_c08_r224_swa.pt`

R6 requires its pinned `manifest.json` and the referenced model files. The supporting bundle supplies `receipt.json`, `cloud_code.zip`, `dependencies.zip`, and any manifest-required anatomy assets. Preserve the notebook's integrity checks.

The actual R6 configuration does not require the conditional `rsna-knee-ofl-ssl-weights` input. The optional 2.5D reader is disabled. V5 requires no additional dataset for its three sampling offsets.

## Settings

1. Import the V4 or V5 notebook from `notebooks/`.
2. Attach the four inputs above and accept any required Kaggle dataset access terms.
3. Select **GPU T4 ×2**.
4. Set **Internet off**.
5. Run all cells. Resolve any preflight or audit failure before using the output.

The historical notebook metadata contains extra pinned dataset-version references. The preflight and table above identify the four inputs used by the active paths. The published slugs alone do not guarantee that later dataset revisions have identical bytes; keep the embedded checkpoint hashes and manifests.

## Outputs

V4 produces `submission.csv` and blend/optimization audit receipts.

V5 produces:

- `submission.csv`: experimental ensemble.
- `baseline_submission.csv`: the matching V4-style control.
- `ours_sub.csv` and `ours_baseline.csv`: R6 predictions before the outer blend.
- `phase_sampling_receipt.json` and `v5_experiment_receipt.json`: coverage, failure, delta, and timing records.

The preview has only three studies. Matching its control checksum establishes pipeline parity, not improved accuracy. V5 runtime and memory use must still be measured on GPU.
