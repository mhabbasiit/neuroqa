# Reproducibility

NeuroQA is built through a deterministic 5-stage pipeline. Given identical inputs (raw images, FreeSurfer outputs, metadata snapshots), the pipeline produces bit-identical question text, options, answers, and `neuroqa_id` enumeration.

## Software pin

| Component | Version |
|---|---|
| Python | 3.11+ |
| numpy | 1.24+ |
| pandas | 2.0+ |
| FreeSurfer (when run by us) | 7.4.1 |
| Random seed | `seed=42` |

## Per-dataset FreeSurfer source

Different datasets contribute their own FreeSurfer outputs; for the rest we run FreeSurfer ourselves. To reproduce, match each dataset's specific FS source.

| Dataset | FS source | How to obtain |
|---|---|---|
| ADNI | UCSF FreeSurfer 7 (UCSFFSX7.csv) | LONI → Study Data → MRI → UCSF |
| AIBL | LONI download (similar to ADNI) | LONI |
| HCP-YA | bundled with release (HCP minimal preprocessing) | ConnectomeDB |
| HCP-Aging | bundled with release | ConnectomeDB |
| HCP-Development | bundled with release | ConnectomeDB |
| ABCD | DAIRC release (ABCD-BIDS pipeline) | NIMH DAC |
| PPMI | bundled (when available) | LONI |
| BraTS-GLI | **not bundled** — we run FS 7.4.1 | the preprocessing pipeline (link withheld for anonymous review) |
| BraTS-MEN | **not bundled** — we run FS 7.4.1 | the preprocessing pipeline (link withheld for anonymous review) |
| CC359 | **not bundled** — we run FS 7.4.1 | the preprocessing pipeline (link withheld for anonymous review) |
| IXI | **not bundled** — we run FS 7.4.1 | the preprocessing pipeline (link withheld for anonymous review) |
| WMH | **not bundled** — we run FS 7.4.1 | the preprocessing pipeline (link withheld for anonymous review) |

## Image preprocessing pipeline

For datasets where we run FreeSurfer, all preprocessing follows our deterministic preprocessing pipeline (link withheld for anonymous review):

1. **Skullstripping** with FreeSurfer SynthStrip (`mri_synthstrip -g` for GPU)
2. **Rigid + affine registration** to MNI-152 space using SimpleITK
3. **Z-score normalization** within brain mask
4. **Cropping** to standardized cropped MNI bounding box

Output per subject: `T1w_mni_zscore_fixed_cropped.nii.gz`

## What is bit-identical, what is approximate (Tier 2 generators)

| Element | Tier 2 (DUA) |
|---|---|
| Question wording | ✅ exact |
| MCQ options content | ✅ exact |
| MCQ option order | ✅ exact (`seed=42`) |
| Answer for non-threshold items | ✅ exact |
| Answer for threshold items (e.g., "above-median") | ⚠️ may flip near threshold if FreeSurfer version differs |
| `neuroqa_id` enumeration | ⚠️ shifts if subject set differs (e.g., new ADNI download) |

Public-dataset (Tier 1) questions are released as fixed files in `part2_public_qa/` and need no generation.

### Why Tier 2 may not be bit-identical

1. **DUA dataset updates** — providers (LONI, NIMH DAC) release new visits/subjects over time. A user downloading 6 months later will get a slightly different subject pool, which shifts `neuroqa_id` enumeration and runtime cohort medians.
2. **FreeSurfer drift** — even small (<1mm³) volume differences near a threshold can flip a "above-median" answer. We pin volumes from the dataset-bundled FS outputs (UCSFFSX7, HCP pipeline, DAIRC) rather than re-running, but if you re-run FS yourself you may see drift.

In practice, expect >99% match on Tier 2 templates and answers, with a small set of threshold-near items that may flip.

## Snapshot dates (for our build)

| Dataset | Build snapshot |
|---|---|
| ADNI | (LONI download date — see `tier2_pipeline/ADNI/README.md`) |
| ABCD | (NIMH DAC release date) |
| HCP-* | (ConnectomeDB release version) |
| BraTS-GLI/MEN | BraTS 2024 release |
| CC359, IXI, WMH | upstream release versions |

Each per-dataset README documents the specific snapshot used.

## Stage 1–5 determinism

- **Stage 1 (Generation)** — per-dataset scripts use sorted iteration over subject IDs and `random.seed(42)` for any sampling.
- **Stage 5 (Cleanup)** — `cleanup_benchmark.py` calls `random.seed(42)` at module load and uses `det_shuffle()` (sort-then-shuffle) for all randomization.
- **Splits** — each Tier 2 generator assigns splits by subject (`assign_splits` in `generate_<db>.py`) so that every template has the published count per split and no subject is in two splits; test_private is assigned but not written.

Each Tier 2 folder runs its own copy of the cleanup rules (`tier2_pipeline/<DATASET>/code/cleanup_benchmark.py`).
