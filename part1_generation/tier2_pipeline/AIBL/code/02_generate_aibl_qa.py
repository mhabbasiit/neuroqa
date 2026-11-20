#!/usr/bin/env python3
"""
NeuroQA — Step 2: AIBL Template-Based QA Generator

Generates question-answer pairs from AIBL structured metadata CSV.
All questions are template-based (no LLM calls). Anti-shortcut by design:
  - 50/50 YN balance via template negation
  - Zero metadata in question text (no age/DX/score leakage)
  - MCQ options shuffled per instance
  - DX-stratified subsampling (Normal/MCI/AD/Unknown)
  - Two image-dependency tiers (grounded / informed)
  - Median-based thresholds computed at runtime (no fixed cutoffs)

KEY DIFFERENCE FROM ADNI/PPMI/ABCD:
  AIBL does NOT have FreeSurfer volumetric data. Categories that rely on
  brain volumes/thickness in other datasets (Location, Anatomy, Measurement,
  Presence) use clinical cognitive scores (MMSE, CDR, memory) instead.
  This makes AIBL a complementary dataset — testing clinical-imaging reasoning
  without volumetric shortcuts.

ANTI-SHORTCUT NOTES (v2):
  - Removed Modality category: all scans are T1, answers never vary
  - Removed Location category: all scans are sagittal MPRAGE, answers never vary
  - Replaced fixed clinical thresholds with median-based computation
  - All YN threshold templates use data-derived medians for ~50/50 split
  - MCQ age bug fixed: correct answer always guaranteed in options

PREREQUISITES:
  Run 01_build_aibl_metadata.py first to produce aibl_neuroqa_merged.csv.

USAGE:
  python 02_generate_aibl_qa.py \\
    --input  ../outputs/aibl_neuroqa_merged.csv \\
    --output ../outputs/
"""

import pandas as pd
import numpy as np
import json
import random
import argparse
import os
from collections import defaultdict
import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from skip_templates import filter_qa_list
from wording_fixes import apply_wording_fixes

random.seed(42)
np.random.seed(42)

AGE_BINS = [(55, 65), (65, 75), (75, 85), (85, 100)]
AGE_LABELS = ["55-65", "65-75", "75-85", "85+"]

# ============================================================
# HELPERS
# ============================================================
def sv(row, col):
    """Safe value — returns None if NaN."""
    v = row.get(col)
    return None if pd.isna(v) else v

def safe_float(v):
    if v is None:
        return None
    try:
        return float(str(v).replace('>', '').replace('<', ''))
    except (ValueError, TypeError):
        return None


# ============================================================
# QA GENERATOR — with balanced YN via template negation
# ============================================================
def generate_qa(df):
    all_qa = []

    # ----------------------------------------------------------
    # Compute median thresholds from data (anti-shortcut)
    # ----------------------------------------------------------
    median_mmse = df['MMSE'].dropna().median() if 'MMSE' in df.columns else None
    median_cdr = df['CDR_Global'].dropna().median() if 'CDR_Global' in df.columns else None
    median_limm = df['LIMMTOTAL'].dropna().median() if 'LIMMTOTAL' in df.columns else None
    median_ldel = df['LDELTOTAL'].dropna().median() if 'LDELTOTAL' in df.columns else None
    median_qc = df['qc_score'].dropna().median() if 'qc_score' in df.columns else None
    median_age = df['age'].dropna().median() if 'age' in df.columns else None

    print(f"  Median thresholds: MMSE={median_mmse}, CDR={median_cdr}, "
          f"LIMM={median_limm}, LDEL={median_ldel}, QC={median_qc}, Age={median_age}")

    # Pre-compute longitudinal
    subj_sessions = {}
    for subj, group in df.groupby('aibl_subject_id'):
        subj_sessions[subj] = group.sort_values('session_date').index.tolist()

    for idx, row in df.iterrows():
        scan_id = row['neuroqa_id']
        subj_id = row['aibl_subject_id']
        qa = []

        # ========== MODALITY — REMOVED ==========
        # All AIBL scans are T1-weighted; every Modality template produces
        # the same answer for every subject → pure shortcut. Removed entirely.

        # ========== LOCATION — REMOVED ==========
        # All AIBL scans are sagittal MPRAGE acquisitions; every Location
        # template produces the same answer for every subject → pure shortcut.

        # ========== ANATOMY / MORPHOLOGY (clinical-score-based) ==========
        mmse = safe_float(sv(row, 'MMSE'))
        cdr = safe_float(sv(row, 'CDR_Global'))
        dx = sv(row, 'DX')

        # MMSE-based brain health inference (median threshold)
        if mmse is not None and median_mmse is not None:
            impaired = mmse < median_mmse
            # YN pair
            qa.append({'cat': 'Anatomy', 'type': 'YN', 'img': True,
                       'q': 'Does this scan suggest cognitive impairment based on clinical assessment?',
                       'a': 'Yes' if impaired else 'No'})
            qa.append({'cat': 'Anatomy', 'type': 'YN', 'img': True,
                       'q': 'Is this scan consistent with normal cognitive function?',
                       'a': 'No' if impaired else 'Yes'})
            # MCQ — cognitive status (uses standard clinical cutoffs for category labels,
            # but the YN threshold above is median-based for balance)
            if mmse >= 27: cog_status = 'Normal cognition'
            elif mmse >= 21: cog_status = 'Mild cognitive impairment'
            elif mmse >= 10: cog_status = 'Moderate cognitive impairment'
            else: cog_status = 'Severe cognitive impairment'
            opts = ['Normal cognition', 'Mild cognitive impairment',
                    'Moderate cognitive impairment', 'Severe cognitive impairment']
            random.shuffle(opts)
            qa.append({'cat': 'Anatomy', 'type': 'MCQ', 'img': True,
                       'q': 'What cognitive status is most consistent with this brain scan?',
                       'a': cog_status, 'options': opts})
            # Open
            qa.append({'cat': 'Anatomy', 'type': 'Open', 'img': True,
                       'q': 'Does this scan suggest cognitive decline?',
                       'a': 'Yes' if impaired else 'No'})

        # CDR-based morphology inference (median threshold)
        if cdr is not None and median_cdr is not None:
            has_dementia_signs = cdr > median_cdr
            # YN pair
            qa.append({'cat': 'Anatomy', 'type': 'YN', 'img': True,
                       'q': 'Does this brain scan show features consistent with dementia?',
                       'a': 'Yes' if has_dementia_signs else 'No'})
            qa.append({'cat': 'Anatomy', 'type': 'YN', 'img': True,
                       'q': 'Is the overall brain morphology within normal limits in this scan?',
                       'a': 'No' if has_dementia_signs else 'Yes'})

        # ========== SIGNAL & IMAGING FEATURE ==========
        has_t2 = sv(row, 'has_t2')
        # T2 availability varies by subject — keep
        if has_t2 is not None:
            has_t2_bool = has_t2 == 'Yes'
            qa.append({'cat': 'Signal', 'type': 'YN', 'img': True,
                       'q': 'Is a T2-weighted scan available for this subject at this visit?',
                       'a': 'Yes' if has_t2_bool else 'No'})
            qa.append({'cat': 'Signal', 'type': 'YN', 'img': True,
                       'q': 'Is this the only structural MRI modality available for this visit?',
                       'a': 'No' if has_t2_bool else 'Yes'})
            # MCQ — available modalities
            if has_t2_bool:
                ans_mod = 'T1 and T2'
            else:
                ans_mod = 'T1 only'
            opts = ['T1 only', 'T1 and T2', 'T1 and FLAIR', 'T1, T2, and FLAIR']
            random.shuffle(opts)
            qa.append({'cat': 'Signal', 'type': 'MCQ', 'img': True,
                       'q': 'Which structural MRI modalities are available for this visit?',
                       'a': ans_mod, 'options': opts})
            # Open — answer varies by subject
            qa.append({'cat': 'Signal', 'type': 'Open', 'img': True,
                       'q': 'Is a T2-weighted scan available for this visit?',
                       'a': 'Yes' if has_t2_bool else 'No'})

        # QC-based signal quality (median threshold)
        qc = safe_float(sv(row, 'qc_score'))
        if qc is not None and qc > 0 and median_qc is not None:
            good_qc = qc > median_qc
            qa.append({'cat': 'Signal', 'type': 'YN', 'img': True,
                       'q': 'Is this scan of above-average quality?',
                       'a': 'Yes' if good_qc else 'No'})
            qa.append({'cat': 'Signal', 'type': 'YN', 'img': True,
                       'q': 'Is this scan of below-average quality?',
                       'a': 'No' if good_qc else 'Yes'})

        # ========== PRESENCE / ABSENCE ==========
        if dx is not None and dx in ('Normal', 'MCI', 'AD'):
            # Cognitive impairment presence — varies by DX
            has_ci = dx in ('MCI', 'AD')
            qa.append({'cat': 'Presence', 'type': 'YN', 'img': True,
                       'q': 'Is cognitive impairment present in this subject?',
                       'a': 'Yes' if has_ci else 'No'})
            qa.append({'cat': 'Presence', 'type': 'YN', 'img': True,
                       'q': 'Is this subject cognitively normal?',
                       'a': 'No' if has_ci else 'Yes'})

            # AD presence — varies by DX
            has_ad = dx == 'AD'
            qa.append({'cat': 'Presence', 'type': 'YN', 'img': True,
                       'q': "Is Alzheimer's disease present in this subject?",
                       'a': 'Yes' if has_ad else 'No'})
            qa.append({'cat': 'Presence', 'type': 'YN', 'img': True,
                       'q': "Is this subject free of Alzheimer's disease?",
                       'a': 'No' if has_ad else 'Yes'})

            # MCQ — which finding present
            findings = []
            absent = []
            if has_ci:
                findings.append('Cognitive impairment')
            else:
                absent.append('Cognitive impairment')
            if has_ad:
                findings.append("Alzheimer's disease")
            else:
                absent.append("Alzheimer's disease")
            if mmse is not None and median_mmse is not None and mmse < median_mmse:
                findings.append('Below-median MMSE')
            else:
                absent.append('Below-median MMSE')

            if findings:
                ans = random.choice(findings)  # Randomize to avoid always picking first
                distractors = [f for f in absent if f != ans] + ['Tumor mass', 'Acute infarction', 'Hemorrhage', 'Cortical calcification']
                distractors = [x for x in distractors if x != ans]
                opts = [ans] + random.sample(distractors, min(3, len(distractors)))
                opts = opts[:4]
                random.shuffle(opts)
                qa.append({'cat': 'Presence', 'type': 'MCQ', 'img': True,
                           'q': 'Which of the following findings is present in this subject?',
                           'a': ans, 'options': opts})
            if absent:
                ans_absent = absent[0]
                present_list = findings + ['Tumor mass', 'Acute infarction', 'Hemorrhage', 'Cortical calcification']
                present_list = [x for x in present_list if x != ans_absent]
                opts_a = [ans_absent] + random.sample(present_list, min(3, len(present_list)))
                opts_a = opts_a[:4]
                random.shuffle(opts_a)
                qa.append({'cat': 'Presence', 'type': 'MCQ', 'img': True,
                           'q': 'Which of the following findings is NOT present in this subject?',
                           'a': ans_absent, 'options': opts_a})

            # Open — answer varies by DX
            qa.append({'cat': 'Presence', 'type': 'Open', 'img': True,
                       'q': 'Is this scan from a subject with normal cognition, MCI, or AD?',
                       'a': dx})

        # ========== MEASUREMENT / COUNT (clinical scores) ==========
        # MMSE measurements (median threshold)
        if mmse is not None and median_mmse is not None:
            # Open — MMSE score (varies per subject)
            qa.append({'cat': 'Measurement', 'type': 'Open', 'img': True,
                       'q': 'What is the MMSE score for this subject?',
                       'a': str(int(mmse))})
            # YN — MMSE threshold (median-based)
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the MMSE score below the cohort median ({median_mmse:.0f}) for this subject?',
                       'a': 'Yes' if mmse < median_mmse else 'No'})
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the MMSE score at or above the cohort median ({median_mmse:.0f}) for this subject?',
                       'a': 'No' if mmse < median_mmse else 'Yes'})
            # MCQ — MMSE range (clinical categories — answers vary)
            if mmse >= 27: mmse_range = '27-30 (Normal)'
            elif mmse >= 21: mmse_range = '21-26 (Mild)'
            elif mmse >= 10: mmse_range = '10-20 (Moderate)'
            else: mmse_range = '< 10 (Severe)'
            opts = ['27-30 (Normal)', '21-26 (Mild)', '10-20 (Moderate)', '< 10 (Severe)']
            random.shuffle(opts)
            qa.append({'cat': 'Measurement', 'type': 'MCQ', 'img': True,
                       'q': 'What MMSE score range does this subject fall into?',
                       'a': mmse_range, 'options': opts})

        # CDR measurements (median threshold)
        if cdr is not None and median_cdr is not None:
            qa.append({'cat': 'Measurement', 'type': 'Open', 'img': True,
                       'q': 'What is the CDR global score for this subject?',
                       'a': f'{cdr:.1f}' if cdr != int(cdr) else str(int(cdr))})
            # YN (median-based)
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the CDR global score above the cohort median ({median_cdr:.1f}) for this subject?',
                       'a': 'Yes' if cdr > median_cdr else 'No'})
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the CDR global score at or below the cohort median ({median_cdr:.1f}) for this subject?',
                       'a': 'No' if cdr > median_cdr else 'Yes'})
            # MCQ — CDR staging (answers vary)
            if cdr == 0: cdr_stage = 'CDR 0 (Normal)'
            elif cdr == 0.5: cdr_stage = 'CDR 0.5 (Questionable)'
            elif cdr == 1: cdr_stage = 'CDR 1 (Mild)'
            else: cdr_stage = 'CDR 2+ (Moderate-Severe)'
            opts = ['CDR 0 (Normal)', 'CDR 0.5 (Questionable)', 'CDR 1 (Mild)', 'CDR 2+ (Moderate-Severe)']
            random.shuffle(opts)
            qa.append({'cat': 'Measurement', 'type': 'MCQ', 'img': True,
                       'q': 'What CDR stage does this subject belong to?',
                       'a': cdr_stage, 'options': opts})

        # Memory score measurements (median threshold)
        limm = safe_float(sv(row, 'LIMMTOTAL'))
        ldel = safe_float(sv(row, 'LDELTOTAL'))
        if limm is not None and median_limm is not None:
            qa.append({'cat': 'Measurement', 'type': 'Open', 'img': True,
                       'q': 'What is the immediate recall score (LIMMTOTAL) for this subject?',
                       'a': str(int(limm))})
            limm_below = limm <= median_limm  # Use <= for closer to 50/50 split
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the immediate recall score at or below the cohort median ({median_limm:.0f}) for this subject?',
                       'a': 'Yes' if limm_below else 'No'})
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the immediate recall score above the cohort median ({median_limm:.0f}) for this subject?',
                       'a': 'No' if limm_below else 'Yes'})
        if ldel is not None and median_ldel is not None:
            qa.append({'cat': 'Measurement', 'type': 'Open', 'img': True,
                       'q': 'What is the delayed recall score (LDELTOTAL) for this subject?',
                       'a': str(int(ldel))})
            ldel_below = ldel <= median_ldel  # Use <= for closer to 50/50 split
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the delayed recall score at or below the cohort median ({median_ldel:.0f}) for this subject?',
                       'a': 'Yes' if ldel_below else 'No'})
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the delayed recall score above the cohort median ({median_ldel:.0f}) for this subject?',
                       'a': 'No' if ldel_below else 'Yes'})

        # ========== DIAGNOSIS / CLASSIFICATION ==========
        # AIBL has mixed DX (Normal/MCI/AD) — answers vary by subject. KEEP.
        if dx is not None and dx in ('Normal', 'MCI', 'AD'):
            dx_full_map = {'Normal': 'Cognitively Normal', 'MCI': 'Mild Cognitive Impairment', 'AD': "Alzheimer's Disease"}
            dx_full = dx_full_map.get(dx, dx)

            # YN pairs — balanced by template negation
            qa.append({'cat': 'Diagnosis', 'type': 'YN', 'img': True,
                       'q': 'Is this brain scan from a cognitively normal subject?',
                       'a': 'Yes' if dx == 'Normal' else 'No'})
            qa.append({'cat': 'Diagnosis', 'type': 'YN', 'img': True,
                       'q': 'Does this brain scan show signs of cognitive impairment?',
                       'a': 'No' if dx == 'Normal' else 'Yes'})
            qa.append({'cat': 'Diagnosis', 'type': 'YN', 'img': True,
                       'q': "Does this brain scan show features consistent with Alzheimer's disease?",
                       'a': 'Yes' if dx == 'AD' else 'No'})
            qa.append({'cat': 'Diagnosis', 'type': 'YN', 'img': True,
                       'q': "Is this brain scan inconsistent with an Alzheimer's disease diagnosis?",
                       'a': 'No' if dx == 'AD' else 'Yes'})

            # MCQ 1 — classification
            opts = ['Cognitively Normal', 'Mild Cognitive Impairment',
                    "Alzheimer's Disease", 'Frontotemporal Dementia']
            random.shuffle(opts)
            qa.append({'cat': 'Diagnosis', 'type': 'MCQ', 'img': True,
                       'q': 'What is the most likely clinical classification based on this brain scan?',
                       'a': dx_full, 'options': opts})
            # MCQ 2 — disease type (varies by DX)
            if dx == 'AD':
                opts2 = ['Neurodegenerative disease', 'Vascular disease', 'Traumatic injury', 'Normal aging']
                random.shuffle(opts2)
                qa.append({'cat': 'Diagnosis', 'type': 'MCQ', 'img': True,
                           'q': 'What type of condition is most consistent with the findings in this scan?',
                           'a': 'Neurodegenerative disease', 'options': opts2})
            elif dx == 'Normal':
                opts2 = ['Normal aging', 'Neurodegenerative disease', 'Vascular disease', 'Traumatic injury']
                random.shuffle(opts2)
                qa.append({'cat': 'Diagnosis', 'type': 'MCQ', 'img': True,
                           'q': 'What type of condition is most consistent with the findings in this scan?',
                           'a': 'Normal aging', 'options': opts2})
            # Open
            qa.append({'cat': 'Diagnosis', 'type': 'Open', 'img': True,
                       'q': 'What is the clinical diagnosis group for this subject?',
                       'a': dx})

        # ========== DIAGNOSTIC REASONING ==========
        if dx is not None and mmse is not None and median_mmse is not None:
            low_mmse = mmse < median_mmse

            # MCQ — reasoning about clinical concordance (answer varies by DX x MMSE)
            # Uses actual score-diagnosis concordance rather than tautological DX-implies-answer
            if dx == 'AD' and low_mmse:
                reasoning_ans = 'Low cognitive scores with dementia diagnosis'
            elif dx == 'AD' and not low_mmse:
                reasoning_ans = 'Preserved scores despite dementia diagnosis'
            elif dx == 'MCI' and low_mmse:
                reasoning_ans = 'Low cognitive scores with MCI diagnosis'
            elif dx == 'MCI' and not low_mmse:
                reasoning_ans = 'Preserved scores with MCI diagnosis'
            elif dx == 'Normal' and not low_mmse:
                reasoning_ans = 'Normal scores with normal diagnosis'
            elif dx == 'Normal' and low_mmse:
                reasoning_ans = 'Low cognitive scores despite normal diagnosis'
            else:
                reasoning_ans = None
            if reasoning_ans is not None:
                reasoning_opts = [
                    'Low cognitive scores with dementia diagnosis',
                    'Preserved scores despite dementia diagnosis',
                    'Low cognitive scores with MCI diagnosis',
                    'Preserved scores with MCI diagnosis',
                    'Normal scores with normal diagnosis',
                    'Low cognitive scores despite normal diagnosis',
                ]
                # Keep 4 options: correct + 3 distractors
                distractors = [o for o in reasoning_opts if o != reasoning_ans]
                mcq_opts = [reasoning_ans] + random.sample(distractors, 3)
                random.shuffle(mcq_opts)
                qa.append({'cat': 'Reasoning', 'type': 'MCQ', 'img': True,
                           'q': 'Which combination of clinical findings and diagnosis best describes this subject?',
                           'a': reasoning_ans, 'options': mcq_opts})

            # YN pairs — BALANCED (median-based threshold)
            # Template 1: Does scan suggest neurodegeneration?
            if low_mmse and dx in ('AD', 'MCI'):
                qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                           'q': 'Do the clinical findings suggest neurodegenerative changes in this subject?',
                           'a': 'Yes'})
            elif not low_mmse and dx == 'Normal':
                qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                           'q': 'Do the clinical findings suggest neurodegenerative changes in this subject?',
                           'a': 'No'})

            # Template 2: Is cognitive function within normal limits?
            if not low_mmse and dx == 'Normal':
                qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                           'q': 'Is the overall cognitive function within normal limits for this subject?',
                           'a': 'Yes'})
            elif low_mmse and dx in ('AD', 'MCI'):
                qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                           'q': 'Is the overall cognitive function within normal limits for this subject?',
                           'a': 'No'})

            # Template 3: Concordance
            qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                       'q': 'Does the clinical pattern suggest cognitive impairment in this subject?',
                       'a': 'Yes' if (low_mmse and dx != 'Normal') else 'No'})

            # Template 4: Mismatch (answer varies by DX-MMSE concordance)
            if dx == 'Normal' and low_mmse:
                qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                           'q': 'Is there a mismatch between the cognitive scores and the clinical diagnosis?',
                           'a': 'Yes'})
            elif dx == 'AD' and not low_mmse:
                qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                           'q': 'Is there a mismatch between the cognitive scores and the clinical diagnosis?',
                           'a': 'Yes'})
            elif dx == 'Normal' and not low_mmse:
                qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                           'q': 'Is there a mismatch between the cognitive scores and the clinical diagnosis?',
                           'a': 'No'})
            elif dx == 'AD' and low_mmse:
                qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                           'q': 'Is there a mismatch between the cognitive scores and the clinical diagnosis?',
                           'a': 'No'})

            # Open — answer varies
            qa.append({'cat': 'Reasoning', 'type': 'Open', 'img': True,
                       'q': 'What is the most prominent clinical finding for this subject?',
                       'a': 'Impaired' if low_mmse else 'Normal'})

        # ========== SEVERITY / STAGING ==========
        if cdr is not None and median_cdr is not None:
            # YN pairs (median-based)
            severe = cdr > median_cdr
            qa.append({'cat': 'Severity', 'type': 'YN', 'img': True,
                       'q': 'Is this subject above the cohort median CDR score?',
                       'a': 'Yes' if severe else 'No'})
            qa.append({'cat': 'Severity', 'type': 'YN', 'img': True,
                       'q': 'Is this subject at or below the cohort median CDR score?',
                       'a': 'No' if severe else 'Yes'})

            # MCQ — severity level (answers vary)
            if cdr == 0: sev = 'No impairment'
            elif cdr == 0.5: sev = 'Questionable'
            elif cdr == 1: sev = 'Mild dementia'
            else: sev = 'Moderate to severe dementia'
            opts = ['No impairment', 'Questionable', 'Mild dementia', 'Moderate to severe dementia']
            random.shuffle(opts)
            qa.append({'cat': 'Severity', 'type': 'MCQ', 'img': True,
                       'q': 'What severity level is most consistent with this subject?',
                       'a': sev, 'options': opts})

        if mmse is not None:
            if mmse >= 27: mc = 'Normal'
            elif mmse >= 21: mc = 'Mild'
            elif mmse >= 10: mc = 'Moderate'
            else: mc = 'Severe'
            qa.append({'cat': 'Severity', 'type': 'Open', 'img': True,
                       'q': 'What cognitive severity level is most consistent with this subject?',
                       'a': mc})

        # APOE risk — varies by subject
        apoe4 = sv(row, 'APOE4_carrier')
        if apoe4 is not None and apoe4 in ('Yes', 'No'):
            qa.append({'cat': 'Severity', 'type': 'YN', 'img': True,
                       'q': 'Is this subject an APOE \u03b54 carrier?',
                       'a': apoe4})
            qa.append({'cat': 'Severity', 'type': 'YN', 'img': True,
                       'q': 'Is this subject a non-carrier of APOE \u03b54?',
                       'a': 'No' if apoe4 == 'Yes' else 'Yes'})

        # ========== LONGITUDINAL / PROGRESSION ==========
        sessions = subj_sessions.get(subj_id, [])
        if len(sessions) > 1:
            curr_pos = sessions.index(idx) if idx in sessions else -1
            if curr_pos > 0:
                prev = df.loc[sessions[curr_pos - 1]]

                # MMSE change
                prev_mmse = safe_float(sv(prev, 'MMSE'))
                if mmse is not None and prev_mmse is not None:
                    dec = mmse < prev_mmse
                    change = mmse - prev_mmse
                    # YN pair
                    qa.append({'cat': 'Longitudinal', 'type': 'YN', 'img': True,
                               'q': 'Has the MMSE score decreased compared to the previous visit?',
                               'a': 'Yes' if dec else 'No'})
                    qa.append({'cat': 'Longitudinal', 'type': 'YN', 'img': True,
                               'q': 'Has the MMSE score remained stable or improved compared to the previous visit?',
                               'a': 'No' if dec else 'Yes'})
                    if abs(change) >= 2:
                        qa.append({'cat': 'Longitudinal', 'type': 'Open', 'img': True,
                                   'q': 'What is the change in MMSE score from the previous visit?',
                                   'a': f'{change:+.0f}'})

                # CDR change
                prev_cdr = safe_float(sv(prev, 'CDR_Global'))
                if cdr is not None and prev_cdr is not None:
                    cdr_inc = cdr > prev_cdr
                    qa.append({'cat': 'Longitudinal', 'type': 'YN', 'img': True,
                               'q': 'Has the CDR score increased (worsened) compared to the previous visit?',
                               'a': 'Yes' if cdr_inc else 'No'})
                    qa.append({'cat': 'Longitudinal', 'type': 'YN', 'img': True,
                               'q': 'Has the CDR score remained stable or improved compared to the previous visit?',
                               'a': 'No' if cdr_inc else 'Yes'})

                # DX progression
                pdx = sv(prev, 'DX')
                cdx = sv(row, 'DX')
                if pdx is not None and cdx is not None:
                    dx_ord = {'Normal': 0, 'MCI': 1, 'AD': 2}
                    if pdx in dx_ord and cdx in dx_ord:
                        if dx_ord[cdx] > dx_ord[pdx]: prog = 'Progressed'
                        elif dx_ord[cdx] < dx_ord[pdx]: prog = 'Improved'
                        else: prog = 'Stable'
                        opts = ['Progressed', 'Stable', 'Improved', 'Undetermined']
                        random.shuffle(opts)
                        qa.append({'cat': 'Longitudinal', 'type': 'MCQ', 'img': True,
                                   'q': 'Has the clinical status progressed, remained stable, or improved compared to the previous visit?',
                                   'a': prog, 'options': opts})
                        qa.append({'cat': 'Longitudinal', 'type': 'Open', 'img': True,
                                   'q': 'What is the clinical progression status compared to the previous visit?',
                                   'a': prog})

                # Memory change
                prev_ldel = safe_float(sv(prev, 'LDELTOTAL'))
                if ldel is not None and prev_ldel is not None:
                    mem_dec = ldel < prev_ldel
                    qa.append({'cat': 'Longitudinal', 'type': 'YN', 'img': True,
                               'q': 'Has the delayed recall score decreased compared to the previous visit?',
                               'a': 'Yes' if mem_dec else 'No'})
                    qa.append({'cat': 'Longitudinal', 'type': 'YN', 'img': True,
                               'q': 'Has the delayed recall score remained stable or improved compared to the previous visit?',
                               'a': 'No' if mem_dec else 'Yes'})

        # ========== DEMOGRAPHIC (age + sex only — image-informed) ==========
        age = safe_float(sv(row, 'age'))
        sex = sv(row, 'sex')

        if sex is not None:
            qa.append({'cat': 'Demographic', 'type': 'YN', 'img': False,
                       'q': 'Is this subject male?',
                       'a': 'Yes' if sex == 'Male' else 'No'})
            qa.append({'cat': 'Demographic', 'type': 'YN', 'img': False,
                       'q': 'Is this subject female?',
                       'a': 'No' if sex == 'Male' else 'Yes'})
            qa.append({'cat': 'Demographic', 'type': 'Open', 'img': False,
                       'q': 'What is the sex of this subject?', 'a': sex})

        if age is not None and median_age is not None:
            # YN — median-based threshold
            qa.append({'cat': 'Demographic', 'type': 'YN', 'img': False,
                       'q': f'Is this subject older than {median_age:.0f} years?',
                       'a': 'Yes' if age > median_age else 'No'})
            qa.append({'cat': 'Demographic', 'type': 'YN', 'img': False,
                       'q': f'Is this subject {median_age:.0f} years old or younger?',
                       'a': 'No' if age > median_age else 'Yes'})
            # MCQ — age range (fix: ensure correct answer is always in options)
            age_ans = None
            for i, (lo, hi) in enumerate(AGE_BINS):
                if i == len(AGE_BINS) - 1:
                    if age >= lo:
                        age_ans = AGE_LABELS[i]
                        break
                else:
                    if lo <= age < hi:
                        age_ans = AGE_LABELS[i]
                        break
            if age_ans is None:
                # Handle edge case: age below lowest bin
                if age < AGE_BINS[0][0]:
                    age_ans = f"<{AGE_BINS[0][0]}"
                else:
                    age_ans = AGE_LABELS[-1]
            # Build options: always include the correct answer
            opts = AGE_LABELS.copy()
            if age_ans not in opts:
                opts.append(age_ans)
            random.shuffle(opts)
            qa.append({'cat': 'Demographic', 'type': 'MCQ', 'img': False,
                       'q': 'What age range does this subject belong to?',
                       'a': age_ans, 'options': opts})

        # === Add metadata ===
        IMG_DEP_MAP = {
            'Anatomy': 'image-informed',
            'Presence': 'image-informed',
            'Measurement': 'image-informed',
            'Longitudinal': 'image-informed',
            'Diagnosis': 'image-informed',
            'Reasoning': 'image-informed',
            'Signal': 'image-informed',
            'Severity': 'image-informed',
            'Demographic': 'image-informed',
        }
        for q in qa:
            q['neuroqa_id'] = scan_id
            q['subject_id'] = subj_id
            q['dataset'] = 'AIBL'
            q['image_required'] = q.pop('img')
            cat = q.pop('cat')
            q['category'] = cat
            q['image_dependency'] = IMG_DEP_MAP.get(cat, 'unknown')
            q['question_type'] = q.pop('type')
            q['question'] = q.pop('q')
            q['answer'] = q.pop('a')

        all_qa.extend(qa)

    return all_qa


# ============================================================
# MCQ POSITION BALANCE
# ============================================================
def balance_mcq_positions(df):
    """Redistribute MCQ correct answer positions to be uniform across A/B/C/D."""
    rng = np.random.RandomState(42)
    mcq_mask = df['question_type'] == 'MCQ'
    mcq_idx = df.index[mcq_mask].tolist()
    n_mcq = len(mcq_idx)
    if n_mcq == 0:
        return df

    for n_opts in df.loc[mcq_mask, 'options'].apply(len).unique():
        subset_idx = [i for i in mcq_idx if len(df.at[i, 'options']) == n_opts]
        n = len(subset_idx)
        targets = list(range(n_opts)) * (n // n_opts + 1)
        targets = targets[:n]
        rng.shuffle(targets)

        for idx, target_pos in zip(subset_idx, targets):
            opts = list(df.at[idx, 'options'])
            ans = df.at[idx, 'answer']
            try:
                cur_pos = opts.index(ans)
            except ValueError:
                continue
            if cur_pos != target_pos:
                opts[cur_pos], opts[target_pos] = opts[target_pos], opts[cur_pos]
            df.at[idx, 'options'] = opts

    pos_counts = {}
    for idx in mcq_idx:
        opts = df.at[idx, 'options']
        ans = df.at[idx, 'answer']
        try:
            pos = opts.index(ans)
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        except ValueError:
            pass
    labels = 'ABCDEFG'
    dist = ', '.join(f'{labels[p]}:{c} ({100*c/n_mcq:.1f}%)' for p, c in sorted(pos_counts.items()))
    print(f"MCQ position balance: {dist}")
    return df


# ============================================================
# BALANCING & SUBSAMPLING
# ============================================================
def balance_and_subsample(qa_df, df_meta, target_total=5000, max_per_subj_per_cat=3):
    """
    1. Keep ALL subjects (including Unknown DX — important for anti-shortcut)
    2. Balance Yes/No 50/50 per category via subsampling min(Yes,No)
    3. Max N questions per subject per category (YN-balance-aware)
    4. Re-balance YN after limiting
    5. Subsample to target, stratified by category and DX (Normal/MCI/AD/Unknown)
    """
    id_to_dx = df_meta.set_index('neuroqa_id')['DX'].to_dict()
    qa_df = qa_df.copy()
    qa_df['_dx'] = qa_df['neuroqa_id'].map(id_to_dx)
    qa_df['_dx'] = qa_df['_dx'].fillna('Unknown')
    print(f"DX distribution in raw QA: {qa_df['_dx'].value_counts().to_dict()}")

    cat_order = ['Anatomy', 'Signal', 'Presence',
                 'Measurement', 'Diagnosis', 'Reasoning', 'Severity',
                 'Longitudinal', 'Demographic']

    balanced = []
    for cat in cat_order:
        sub = qa_df[qa_df['category'] == cat].copy()

        # Step 1: Balance YN
        yn = sub[sub['question_type'] == 'YN']
        yes_q = yn[yn['answer'] == 'Yes']
        no_q = yn[yn['answer'] == 'No']
        n_min = min(len(yes_q), len(no_q))
        if n_min > 0:
            yes_sample = yes_q.sample(n_min, random_state=42)
            no_sample = no_q.sample(n_min, random_state=42)
            yn_balanced = pd.concat([yes_sample, no_sample])
        else:
            yn_balanced = yn

        non_yn = sub[sub['question_type'] != 'YN']
        cat_balanced = pd.concat([yn_balanced, non_yn])
        balanced.append(cat_balanced)

    all_balanced = pd.concat(balanced, ignore_index=True)
    print(f"After YN balancing: {len(all_balanced):,} QA pairs")

    # Step 2: Limit per subject per category (YN-balance-aware)
    limited = []
    yn_toggle = {}
    for (subj, cat), group in all_balanced.groupby(['subject_id', 'category']):
        if len(group) > max_per_subj_per_cat:
            sampled = []
            for qtype in ['YN', 'MCQ', 'Open']:
                type_q = group[group['question_type'] == qtype]
                if len(type_q) == 0:
                    continue
                if qtype == 'YN':
                    yes_q = type_q[type_q['answer'] == 'Yes']
                    no_q = type_q[type_q['answer'] == 'No']
                    toggle_key = cat
                    cnt = yn_toggle.get(toggle_key, 0)
                    if cnt % 2 == 0 and len(yes_q) > 0:
                        sampled.append(yes_q.sample(1, random_state=42 + cnt))
                    elif len(no_q) > 0:
                        sampled.append(no_q.sample(1, random_state=42 + cnt))
                    elif len(yes_q) > 0:
                        sampled.append(yes_q.sample(1, random_state=42 + cnt))
                    else:
                        sampled.append(type_q.sample(1, random_state=42 + cnt))
                    yn_toggle[toggle_key] = cnt + 1
                else:
                    sampled.append(type_q.sample(1, random_state=42))
            remaining = max_per_subj_per_cat - len(sampled)
            if remaining > 0:
                used_idx = set()
                for s in sampled:
                    used_idx.update(s.index)
                leftover = group[~group.index.isin(used_idx)]
                if len(leftover) > remaining:
                    sampled.append(leftover.sample(remaining, random_state=42))
                else:
                    sampled.append(leftover)
            limited.append(pd.concat(sampled))
        else:
            limited.append(group)

    all_limited = pd.concat(limited, ignore_index=True)
    print(f"After per-subject limit ({max_per_subj_per_cat}/cat): {len(all_limited):,} QA pairs")

    # Step 3: Enforce YN 50/50 balance AGAIN after per-subject limiting
    rebalanced = []
    for cat in cat_order:
        sub = all_limited[all_limited['category'] == cat]
        yn = sub[sub['question_type'] == 'YN']
        non_yn = sub[sub['question_type'] != 'YN']

        yes_q = yn[yn['answer'] == 'Yes']
        no_q = yn[yn['answer'] == 'No']
        n_min = min(len(yes_q), len(no_q))
        if n_min > 0:
            yn_bal = pd.concat([
                yes_q.sample(n_min, random_state=42),
                no_q.sample(n_min, random_state=42)
            ])
        else:
            yn_bal = yn
        rebalanced.append(pd.concat([yn_bal, non_yn]))

    all_rebalanced = pd.concat(rebalanced, ignore_index=True)
    print(f"After YN re-balancing: {len(all_rebalanced):,} QA pairs")

    # Step 4: Subsample to target, stratified by category AND DX
    if '_dx' not in all_rebalanced.columns:
        all_rebalanced['_dx'] = all_rebalanced['neuroqa_id'].map(id_to_dx)

    if len(all_rebalanced) > target_total:
        per_cat = target_total // len(cat_order)
        final = []
        for cat in cat_order:
            sub = all_rebalanced[all_rebalanced['category'] == cat]
            if len(sub) <= per_cat:
                final.append(sub)
                continue

            dx_groups = sub.groupby('_dx', dropna=False)
            dx_counts = dx_groups.size()
            n_dx = len(dx_counts)
            per_dx = per_cat // max(n_dx, 1)

            dx_sampled = []
            for dx_val, dx_sub in dx_groups:
                n_take = min(per_dx, len(dx_sub))
                if n_take == 0:
                    continue
                type_sampled = []
                for qtype in ['YN', 'MCQ', 'Open']:
                    type_sub = dx_sub[dx_sub['question_type'] == qtype]
                    type_share = len(type_sub) / len(dx_sub) if len(dx_sub) > 0 else 0.33
                    type_n = max(1, int(n_take * type_share))
                    if len(type_sub) <= type_n:
                        type_sampled.append(type_sub)
                    elif qtype == 'YN':
                        yes_q = type_sub[type_sub['answer'] == 'Yes']
                        no_q = type_sub[type_sub['answer'] == 'No']
                        half = type_n // 2
                        yn_s = pd.concat([
                            yes_q.sample(min(half, len(yes_q)), random_state=42),
                            no_q.sample(min(half, len(no_q)), random_state=42)
                        ])
                        type_sampled.append(yn_s)
                    else:
                        type_sampled.append(type_sub.sample(type_n, random_state=42))
                dx_sampled.append(pd.concat(type_sampled))
            final.append(pd.concat(dx_sampled))

        all_final = pd.concat(final, ignore_index=True)
    else:
        all_final = all_rebalanced

    # Drop internal column
    all_final = all_final.drop(columns=['_dx'], errors='ignore')

    # MCQ position balance
    all_final = balance_mcq_positions(all_final)

    print(f"After subsampling to ~{target_total}: {len(all_final):,} QA pairs")
    return all_final


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate AIBL QA pairs for NeuroQA')
    parser.add_argument('--input', default='../outputs/aibl_neuroqa_merged.csv',
                        help='Path to AIBL merged metadata CSV (from step 1)')
    parser.add_argument('--output', default='../outputs/',
                        help='Output directory for QA files')
    args = parser.parse_args()

    print("Loading AIBL data...")
    df = pd.read_csv(args.input, low_memory=False)
    df['session_date'] = pd.to_datetime(df['session_date'].str[:10], errors='coerce')
    df = df.sort_values(['aibl_subject_id', 'session_date']).reset_index(drop=True)
    print(f"Loaded {len(df):,} scans, {df['aibl_subject_id'].nunique():,} subjects\n")

    # Generate all
    print("Generating all QA pairs...")
    all_qa = generate_qa(df)
    all_qa, _n_skipped = filter_qa_list(all_qa)
    qa_df = pd.DataFrame(all_qa)
    print(f"Total raw QA: {len(qa_df):,}")

    # === RAW STATS ===
    print("\n" + "="*80)
    print("RAW QA STATS (before balancing)")
    print("="*80)
    cat_order = ['Anatomy', 'Signal', 'Presence',
                 'Measurement', 'Diagnosis', 'Reasoning', 'Severity',
                 'Longitudinal', 'Demographic']

    print(f"\n{'Category':<25} {'YN':>8} {'MCQ':>8} {'Open':>8} {'Total':>8}  {'Y%':>5} {'N%':>5}")
    print("-"*80)
    for cat in cat_order:
        s = qa_df[qa_df['category'] == cat]
        yn = s[s['question_type'] == 'YN']
        mcq = len(s[s['question_type'] == 'MCQ'])
        opn = len(s[s['question_type'] == 'Open'])
        y_pct = (yn['answer'] == 'Yes').mean() * 100 if len(yn) > 0 else 0
        n_pct = 100 - y_pct
        print(f"{cat:<25} {len(yn):>8,} {mcq:>8,} {opn:>8,} {len(s):>8,}  {y_pct:>4.0f}% {n_pct:>4.0f}%")

    # Balance & subsample
    print("\n" + "="*80)
    print("BALANCING & SUBSAMPLING")
    print("="*80)
    final_df = balance_and_subsample(qa_df, df, target_total=5000, max_per_subj_per_cat=3)

    # === FINAL STATS ===
    print("\n" + "="*80)
    print("FINAL QA STATS (after balancing)")
    print("="*80)
    print(f"\n{'Category':<25} {'YN':>8} {'MCQ':>8} {'Open':>8} {'Total':>8}  {'Y%':>5} {'N%':>5}")
    print("-"*80)
    gt_yn = gt_mcq = gt_op = 0
    for cat in cat_order:
        s = final_df[final_df['category'] == cat]
        yn = s[s['question_type'] == 'YN']
        mcq = len(s[s['question_type'] == 'MCQ'])
        opn = len(s[s['question_type'] == 'Open'])
        gt_yn += len(yn)
        gt_mcq += mcq
        gt_op += opn
        y_pct = (yn['answer'] == 'Yes').mean() * 100 if len(yn) > 0 else 0
        n_pct = 100 - y_pct
        print(f"{cat:<25} {len(yn):>8,} {mcq:>8,} {opn:>8,} {len(s):>8,}  {y_pct:>4.0f}% {n_pct:>4.0f}%")
    print("-"*80)
    total = gt_yn + gt_mcq + gt_op
    print(f"{'TOTAL':<25} {gt_yn:>8,} {gt_mcq:>8,} {gt_op:>8,} {total:>8,}")

    # Image-required
    img_req = final_df['image_required'].sum()
    print(f"\nImage-required: {img_req:,} / {total:,} ({100*img_req/total:.1f}%)")
    print(f"Metadata-only:  {total - img_req:,} / {total:,} ({100*(total-img_req)/total:.1f}%)")

    # DX distribution
    print(f"\nDiagnosis distribution in final QA:")
    id_to_dx = df.set_index('neuroqa_id')['DX'].to_dict()
    final_df['_dx'] = final_df['neuroqa_id'].map(id_to_dx)
    dx_dist = final_df['_dx'].value_counts(dropna=False)
    for k, v in dx_dist.items():
        label = k if pd.notna(k) and k else 'Unknown'
        print(f"  {label}: {v:,} ({100*v/total:.1f}%)")

    # Sample questions
    print("\n" + "="*80)
    print("SAMPLE QUESTIONS (3 per category)")
    print("="*80)
    for cat in cat_order:
        sub = final_df[final_df['category'] == cat]
        if len(sub) == 0:
            continue
        print(f"\n--- {cat} ---")
        for qtype in ['YN', 'MCQ', 'Open']:
            type_q = sub[sub['question_type'] == qtype]
            if len(type_q) == 0:
                continue
            s = type_q.sample(1, random_state=42).iloc[0]
            if qtype == 'MCQ' and 'options' in s and isinstance(s['options'], list):
                print(f"  [{qtype}] Q: {s['question']}")
                print(f"       Opts: {' | '.join(s['options'])}")
                print(f"       A: {s['answer']}")
            else:
                print(f"  [{qtype}] Q: {s['question']}  ->  A: {s['answer']}")

    # === TEMPLATE COUNT ===
    print("\n" + "="*80)
    print("UNIQUE QUESTION TEMPLATES")
    print("="*80)
    import re
    def extract_template(q):
        t = re.sub(r'[\d,]+(?:\.\d+)?%?', '{N}', q)
        return t

    for cat in cat_order:
        sub = final_df[final_df['category'] == cat]
        if len(sub) == 0:
            continue
        templates = {}
        for qtype in ['YN', 'MCQ', 'Open']:
            type_q = sub[sub['question_type'] == qtype]
            unique_templates = type_q['question'].apply(extract_template).unique()
            templates[qtype] = list(unique_templates)

        total_templates = sum(len(v) for v in templates.values())
        print(f"\n{cat} ({total_templates} templates):")
        for qtype in ['YN', 'MCQ', 'Open']:
            tlist = templates.get(qtype, [])
            print(f"  [{qtype}] {len(tlist)} templates:")
            for t in tlist[:6]:
                print(f"    - {t}")
            if len(tlist) > 6:
                print(f"    ... +{len(tlist)-6} more")

    # Save
    final_df = final_df.drop(columns=['_dx'], errors='ignore')
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    # Apply expert-reviewed wording fixes (in-place on records) before saving
    _recs = final_df.to_dict(orient='records')
    apply_wording_fixes(_recs)
    final_df = pd.DataFrame(_recs)
    final_df.to_json(os.path.join(out_dir, 'aibl_qa_final.json'), orient='records', indent=2)
    final_df.to_csv(os.path.join(out_dir, 'aibl_qa_final.csv'), index=False)
    qa_df.to_json(os.path.join(out_dir, 'aibl_qa_raw_all.json'), orient='records', indent=2)
    print(f"\nSaved to {out_dir}/: aibl_qa_final.json ({len(final_df):,} QA) + aibl_qa_raw_all.json ({len(qa_df):,} QA)")


# Apply expert-reviewed wording fixes
# apply_wording_fixes(final_qa)  # TODO: add correct variable name