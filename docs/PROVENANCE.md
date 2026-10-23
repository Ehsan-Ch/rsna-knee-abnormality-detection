# Provenance and archive scope

## Preserved artifacts

The `experiments/v4/` and `experiments/v5/` directories preserve recovered project files byte for byte. `evidence/import_manifest.json` records SHA-256 hashes, byte sizes, and source groups. Duplicate baseline/source files are retained where they are part of the original builder or tests.

The recovered work includes the V3 executed baseline, V4 executed snapshot, V5 as delivered, both builders, all test scripts, the original and candidate embedded source bundles, and extracted CoAtNet scripts.

Notebook code includes community sources. Original comments, notebook attribution, checkpoint integrity pins, and OAI acknowledgement are retained. Ehsan is not credited with training those public models.

## Clean entry notebooks

Files in `notebooks/` are regenerated from the recovered builders. Code cells and embedded models are unchanged by the packaging normalizer. The normalizer:

- Clears execution counts and outputs.
- Removes inherited Papermill run timestamps and stale experiment metadata.
- Records explicit historical V4 / experimental V5 status.
- Sets GPU enabled and Internet disabled in Kaggle import metadata.

Manual selection of T4 ×2 is still required in Kaggle. This metadata cleanup does not constitute a new inference run.

## Excluded material

Competition DICOM data, pretrained checkpoint binaries, caches, bytecode, credentials, machine-specific transfer helpers, and Library identifiers are excluded. Required external datasets are linked in the setup guide.

Historical browser screenshots are represented by available text records; screenshot files and unavailable attachment bytes are not included. Earlier notebooks whose exports could not be recovered are documented in the history rather than recreated.

## Evidence hierarchy

1. Included source files and recorded notebook outputs.
2. Previously verified results preserved in the conversation record.
3. Upstream notebook advertised scores, explicitly labeled as such.

A Git import commit records the current assembly. It does not pretend to be a sequence of original development commits. The original submission-time status is retained as an historical snapshot, not a current status claim.
