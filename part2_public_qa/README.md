# Part 2: public-dataset questions (NeuroQA v1.1)

Questions of the six public datasets. test_private is not included.

| File | Split | Questions |
|---|---|---:|
| `all_public/train.json` | train | 13,598 |
| `all_public/val.json` | val | 1,892 |
| `all_public/test.json` | test_public | 1,813 |

`per_dataset/<DATASET>/{train,val,test}.json` holds the same questions split by dataset
(BraTS-GLI, BraTS-MEN, CC359, HCP-YA, IXI, WMH).

Each file is a JSON list. Fields:

| Field | Meaning |
|---|---|
| `neuroqa_id` | question ID (use it in prediction files) |
| `dataset` | source dataset |
| `category` | one of 11 question categories |
| `image_dependency` | `image-grounded` or `image-informed` |
| `question_type` | `YN`, `MCQ` or `Open` |
| `question`, `options`, `answer` | question text, options (MCQ only) and reference answer |
| `answerable_modalities` | MRI sequences that carry the needed information |
| `split`, `tier` | split and release tier |

Subject IDs and image paths are not included. Images can be obtained from each provider and preprocessed as described in
`../part1_generation/docs/reproducibility.md`.
