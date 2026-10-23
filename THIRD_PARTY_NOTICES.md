# Third-party sources and acknowledgements

## Model provenance and attribution

This notebook is Ehsan's engineering revision of an existing community ensemble.
It does not claim that Ehsan trained the public models or developed the original reader.

- [nartaa / Danial Zakaria](https://www.kaggle.com/code/nartaa/rsna-knee-0949-anatomical-mirror): CoAtNet 384/224 checkpoints and anatomical-mirror inference, based on [Dread Development's Raptor](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense).
- [heliosli](https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters): Compact R6, flip TTA, source bundle, and fixed per-finding blend weights.
- [rostigor](https://www.kaggle.com/datasets/rostigor/rsna-knee-r6-fe055406): pinned R6 dataset copy.
- [varshithakolupuri](https://www.kaggle.com/code/varshithakolupuri/rsna-knee-one-unused-checkpoint-lifts-the-0-956): 224-pixel model contribution to the CoAtNet slot.
- [goodpjw2008](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-s6-2-5d-mil-lb-0-954): source combined notebook and historical 2.5D reader, omitted here.

The original authors' comments remain inside their source. Ehsan V4 changes only byte I/O, scheduling, 224-branch status tracking, and fallback handling. Checkpoints, DICOM transforms, image sizes, test-time augmentation, and fixed complete-data blend weights are retained.

*OAI acknowledgement (required for the Part A1 weights).* Data and/or research tools used in the preparation of this
manuscript were obtained and analyzed from the controlled access datasets distributed from the Osteoarthritis
Initiative (OAI), a data repository housed within the NIMH Data Archive (NDA). OAI is a collaborative informatics system
created by the National Institute of Mental Health and the National Institute of Arthritis, Musculoskeletal and Skin
Diseases (NIAMS) to provide a worldwide resource to quicken the pace of biomarker identification, scientific
investigation and OA drug development. Dataset identifier(s): 10.15154/0hcg-f676.
[Shared NDA Study](https://nda.nih.gov/study.html?id=3407) · [DOI: 10.15154/0hcg-f676](https://doi.org/10.15154/0hcg-f676).



## Repository packaging

Upstream code and weights remain subject to their original terms. This archive does not assign a new blanket license to community code, data, or model weights. Dataset binaries are linked, not redistributed. Original source comments are preserved.
