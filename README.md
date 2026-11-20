# NeuroQA v1.1 (anonymous release for review)

NeuroQA is a visual question answering benchmark on 3D brain MRI volumes.

- 56,948 question–answer pairs over 12 neuroimaging datasets and 11 question categories
- Yes/No, multiple-choice and open-ended questions
- Reference answers from FreeSurfer volumetry, structured clinical/demographic metadata, image headers, or radiology-report parsers
- Splits: train, val, test_public, and a held-out test_private (not distributed)

The repository has three parts.

| Part | Folder | What it gives you |
|---|---|---|
| 1. Question generation | `part1_generation/` | Code that builds the questions of the six data-use-agreement (DUA) datasets (ABCD, ADNI, AIBL, HCP-Aging, HCP-Development, PPMI), whose questions cannot be redistributed: obtain the data from each provider and run the generator in `part1_generation/tier2_pipeline/<DATASET>/` (see its README). |
| 2. Public questions | `part2_public_qa/` | Ready-to-use questions of the six public datasets (BraTS-GLI, BraTS-MEN, CC359, HCP-YA, IXI, WMH): train, val and test (= test_public). |
| 3. Scoring | `part3_evaluation/` | The scorer used for all results in the paper (`evaluate.py`). |

Images are not included. Download them from the original providers and preprocess them as described in `part1_generation/docs/reproducibility.md`.

## Quick start

```bash
pip install -r requirements.txt

# score your predictions on the public test questions
python part3_evaluation/evaluate.py \
    --gold  part2_public_qa/all_public/test.json \
    --train part2_public_qa/all_public/train.json \
    --pred_dir my_predictions/ --out my_results/
```

See `part3_evaluation/README.md` for the prediction file format.

## Notes

- Subjects in the official test splits may appear in a train split you generate yourself with the Tier 2 scripts. Do not train on a self-generated train split and test on the official test split.
- test_private is held out and is not part of this release.

## License

MIT (code). The questions of each dataset are subject to the terms of that dataset's provider.
