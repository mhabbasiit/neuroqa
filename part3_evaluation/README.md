# Part 3: scoring

`evaluate.py` is the scorer used for every result in the paper.

```bash
python evaluate.py --gold <test.json> --train <train.json> --pred_dir <predictions/> --out <new_output_dir/>
```

- `--gold`: the questions to score, e.g. `../part2_public_qa/all_public/test.json`, or a file built with the Part 1 generators.
- `--train`: the train questions of the same release; used only for the text-only reference.
- `--pred_dir`: one file per run, named `<Model>__<condition>.jsonl` (for several seeds `<Model>__<condition>__seed<k>.jsonl`; seeds are averaged).
  Each line is `{"neuroqa_id": "...", "response": "..."}`.
- `--out`: must not exist yet. The scorer writes `results.json` and `results.md`.

Rules:
- Every question in the gold file is scored. A question without a response counts as wrong.
- Yes/No and multiple-choice: the answer is read from the response (a leading Yes/No or option letter, an explicit "Answer: ..." line, or
  the option text; see `extract_answer`) and compared with the reference after normalization.
- Open-ended: exact match after normalization (lower case, whitespace collapsed, punctuation other than `% . , -` removed) and removal of a
  single trailing period on both sides.
- Reported groups: YN, MCQ, closed (YN + MCQ), closed image-grounded, closed image-informed, open EM, and closed accuracy per category.
- Text-only reference: for each closed question, the question's own option (Yes/No, or its MCQ options) that is most frequent among
  train answers of the same template (numbers replaced by N); ties go to the alphabetically first option. An in-sample version
  (counts from the gold file itself) is reported as a descriptive measure only.
