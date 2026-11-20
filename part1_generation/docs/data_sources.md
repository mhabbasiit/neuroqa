# Data Sources

NeuroQA is built from 12 neuroimaging datasets. Six are publicly licensed (Tier 1) and six are DUA-restricted (Tier 2).

## Tier 1 — Public datasets (6)

The QA pairs are released ready to use in `part2_public_qa/` of this repository (train, val, test_public). Users obtain images from the original sources.

| Dataset | Items | Subjects | Domain | Source |
|---|---|---|---|---|
| BraTS-GLI | 3,076 | ~580 | Glioma | [synapse.org/brats](https://www.synapse.org/brats) |
| BraTS-MEN | 2,670 | ~500 | Meningioma | [synapse.org/brats](https://www.synapse.org/brats) |
| HCP-YA | 6,684 | ~1,100 | Healthy young adults (22–35) | [humanconnectome.org](https://www.humanconnectome.org/) |
| CC359 | 3,402 | ~360 | Healthy adults (Calgary-Campinas) | [sites.google.com/view/calgary-campinas-dataset](https://sites.google.com/view/calgary-campinas-dataset) |
| IXI | 3,247 | ~580 | Healthy adults (London) | [brain-development.org/ixi-dataset](https://brain-development.org/ixi-dataset/) |
| WMH | 183 | ~60 | White matter hyperintensities | [wmh.isi.uu.nl](https://wmh.isi.uu.nl/) |

## Tier 2 — DUA-restricted datasets (6)

QA pairs are generated from the user's own licensed data. **Subject IDs are not redistributed.** Users obtain DUA access from each provider and run `part1_generation/tier2_pipeline/{DATASET}/generate_<db>.py` (see its README).

| Dataset | Items | Subjects | Domain | Access |
|---|---|---|---|---|
| ABCD | 11,331 | ~3,500 | Adolescent neurodevelopment | [NIMH DAC](https://nda.nih.gov/) |
| ADNI | 5,319 | ~1,500 | Alzheimer's disease + MCI | [LONI](https://adni.loni.usc.edu/) |
| AIBL | 2,545 | ~750 | Alzheimer's disease (Australian) | [LONI](https://aibl.org.au/) |
| HCP-Aging | 8,178 | ~1,400 | Adult aging (36–100) | [ConnectomeDB](https://www.humanconnectome.org/study/hcp-lifespan-aging) |
| HCP-Development | 6,580 | ~1,300 | Pediatric development (5–21) | [ConnectomeDB](https://www.humanconnectome.org/study/hcp-lifespan-development) |
| PPMI | 3,738 | ~1,000 | Parkinson's disease | [LONI](https://www.ppmi-info.org/) |

## Five clinical domains covered

1. **Alzheimer's disease and MCI** — ADNI, AIBL
2. **Parkinson's disease** — PPMI
3. **Brain tumors** — BraTS-GLI, BraTS-MEN
4. **White matter disease** — WMH, CC359
5. **Healthy aging and neurodevelopment** — HCP-YA, HCP-Aging, HCP-Development, ABCD, IXI

## Standardized clinical instruments

Ground-truth labels for clinical questions are anchored to standardized instruments:

- **CDR** (Clinical Dementia Rating)
- **MoCA** (Montreal Cognitive Assessment)
- **MMSE** (Mini-Mental State Examination)
- **NIH Toolbox** cognition battery
- **CBCL** (Child Behavior Checklist)
- **UPDRS** (Unified Parkinson's Disease Rating Scale)
- **Hoehn-Yahr** staging
