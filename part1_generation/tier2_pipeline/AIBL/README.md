# NeuroQA generator for AIBL

Builds the AIBL questions of NeuroQA from your own licensed data. No subject list, question or answer is stored here.

## What you need

From LONI IDA (https://ida.loni.usc.edu/), in one folder (`--metadata`); date suffixes in the names are fine (checked version `01-Jun-2018`): `aibl_pdxconv*.csv`, `aibl_mmse*.csv`, `aibl_cdr*.csv`, `aibl_neurobat*.csv`, `aibl_ptdemog*.csv`, `aibl_apoeres*.csv`, `aibl_mri3meta*.csv`.

The T1 and T2 metadata files are outputs of the preprocessing pipeline (link withheld for anonymous review). Run it on your own scans first. The script reads these columns:

| File | Argument | Columns used |
|---|---|---|
| `AIBL_T1_metadata.csv` | `--t1meta` | `subjectId`, `sessionName`, `MNI_Z_Cropped`, `age`, `sex`, `smoking_history`, `current_diagnosis`, `scanner_manufacturer`, `overall_qc_score` |
| `AIBL_T2_metadata.csv` | `--t2meta` | the same columns |

## Run

```bash
pip install -r requirements.txt
python generate_aibl.py --out aibl.json \
  --t1meta AIBL_T1_metadata.csv --t2meta AIBL_T2_metadata.csv --metadata <loni_tables>
```

## Output

One JSON list of questions. Each question has:

| Field | Meaning |
|---|---|
| `neuroqa_id` | `AIBL-GEN-#####` |
| `split` | `train`, `val` or `test_public`. The test_private split is held out: the script assigns it, so the other splits keep the published counts, but does not write it. |
| `subject_id`, `session_date` | the subject and session the question is about |
| `prior_session_date` | Longitudinal only: the earlier session |
| `image_path` | the T1 image of that session, taken from `MNI_Z_Cropped` in your T1 metadata |
| `image_path_t2` | the T2 image of that session, when your T2 metadata has it |
| `image_path_prior` | Longitudinal only: the T1 image of the earlier session |
| `question`, `question_type`, `options`, `answer` | the question; `question_type` is `YN`, `MCQ` or `Open` |
| `category`, `image_required`, `image_dependency`, `answerable_modalities`, `human_assessable` | labels from the templates |

The image paths are the paths on your computer, as written in your sMRI metadata. If a session has no image there, the field is `#`.

## Do not share the output

The output has subject IDs, sessions and answers derived from restricted data. Do not publish or share it unless your data use agreement allows it.

## What the output is

Questions from the same templates and rules as the release, with the same count of each template in each split as the evaluation version. No subject is in two splits. Yes and No are equal within each split, and MCQ answer positions are balanced. The script checks these counts at the end and prints `ALL COUNT CHECKS PASSED`, or `SHORTAGE` with each template that has too few valid questions (the usual cause is another version of the source files).

The paper's results and the official private test belong to the fixed evaluation version. This script implements the generation method; its subjects and questions can differ from that version. Question IDs contain `-GEN-` so the two are not confused.

The same inputs give the same output file. Intermediate files are written beside the output and removed at the end.

## Session rules

- An MMSE below zero is a LONI missing code, not a score. A question whose answer used it is replaced with another question of the same template.
- A question keeps the session it was generated from, for its image, age and answer. A Longitudinal question compares its session with the one before it; a volume change question keeps the pair of sessions rule R37 checked against its answer.

`code/` holds the release generator, cleanup rules, templates and wording for this database. `code/targets*.json` hold counts only.
