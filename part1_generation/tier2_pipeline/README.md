# NeuroQA Tier 2 generators

Tier 2 questions come from data under a data use agreement, so they are not published. Each folder builds the questions of one database from your own licensed data, and works on its own: you need only the folders of the databases you have access to.

| Folder | Database | Script |
|---|---|---|
| `ABCD/` | ABCD | `generate_abcd.py` |
| `ADNI/` | ADNI | `generate_adni.py` |
| `AIBL/` | AIBL | `generate_aibl.py` |
| `HCP-Aging/` | HCP-Aging | `generate_hcpaging.py` |
| `HCP-Development/` | HCP-Development | `generate_hcpdev.py` |
| `PPMI/` | PPMI | `generate_ppmi.py` |

Every script needs the T1 (and T2) metadata from the preprocessing pipeline (link withheld for anonymous review) and the database's own tables. See the README in each folder.

To build several databases, run each script and join the JSON lists:

```bash
python -c "import json,sys; json.dump(sum((json.load(open(f)) for f in sys.argv[2:]), []), open(sys.argv[1], 'x'), indent=1)" all.json abcd.json adni.json aibl.json
```

Subjects of the official test can be in a generated train split, so do not train on a generated train split and test on the official test.
