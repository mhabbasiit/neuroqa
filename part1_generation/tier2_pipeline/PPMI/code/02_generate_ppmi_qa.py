#!/usr/bin/env python3
"""
NeuroQA — Step 2: PPMI Template-Based QA Generator

Generates question-answer pairs from PPMI structured metadata CSV.
All questions are template-based (no LLM calls). Anti-shortcut by design:
  - 50/50 YN balance via template negation
  - Zero metadata in question text
  - MCQ options shuffled per instance
  - Median-based thresholds computed from data (no fixed constants)
  - No Modality category (all T1 = pure shortcut)

Shared structural categories (Location, Anatomy, Measurement, Presence,
Reasoning-structural, Demographic) mirror the ADNI generator verbatim.
PPMI-specific categories (Diagnosis, Severity, Signal, Reasoning-motor,
Longitudinal-motor) use Parkinson's instruments: research_group, hoehn_yahr,
UPDRS-III, MoCA, and FreeSurfer WM-hypointensity volume.

PREREQUISITES:
  Run 01_build_ppmi_metadata.py first to produce ppmi_neuroqa_merged.csv.

USAGE:
  python 02_generate_ppmi_qa.py \\
    --input  ../outputs/ppmi_neuroqa_merged.csv \\
    --output ../outputs/
"""

import pandas as pd
import numpy as np
import json
import random
import argparse
import os
import re
from collections import defaultdict

random.seed(42)
np.random.seed(42)

# ============================================================
# HELPERS
# ============================================================
def sv(row, col):
    v = row.get(col)
    return None if pd.isna(v) else v

def safe_float(v):
    if v is None:
        return None
    try:
        return float(str(v).replace('>', '').replace('<', ''))
    except (ValueError, TypeError):
        return None

# ------------------------------------------------------------
# BASELINE MOTOR SCORE (motor-severity ground truth)
# The benchmark's motor-severity label is anchored to the subject's
# BASELINE-visit MDS-UPDRS Part III total (NP3TOT at EVENT_ID='BL',
# fallback 'SC', then earliest visit) — NOT the visit nearest the scan.
# Reproduces frozen motor severity at ~96% (vs ~88.7% for the
# nearest-visit value the merged CSV carries). Set PPMI_UPDRS3_RAW to the
# LONI 'MDS-UPDRS_Part_III' CSV; if unavailable the code falls back to the
# merged-CSV UPDRS_III_total.
# ------------------------------------------------------------
def load_baseline_updrs3(path=None):
    """Return {PATNO(str): baseline NP3TOT(float)}. Empty dict if unavailable."""
    import csv as _csv
    path = path or os.environ.get('PPMI_UPDRS3_RAW', '')
    out = {}
    if not path or not os.path.exists(path):
        return out
    try:
        rows = list(_csv.DictReader(open(path, encoding='latin-1')))
    except Exception:
        return out
    # priority: BL > SC > earliest-by-INFODT
    pref = {}
    for r in rows:
        p = str(r.get('PATNO')); ev = r.get('EVENT_ID'); d = r.get('INFODT', '')
        try:
            n = float(r.get('NP3TOT'))
        except (ValueError, TypeError):
            continue
        rank = 0 if ev == 'BL' else (1 if ev == 'SC' else 2)
        cur = pref.get(p)
        if cur is None or (rank, d) < (cur[0], cur[1]):
            pref[p] = (rank, d, n)
    return {p: v[2] for p, v in pref.items()}

# ============================================================
# REGION MAPPINGS (shared with ADNI schema)
# ============================================================
SUBCORT_PAIRS = {
    'hippocampus': ('FS_Vol_L_Hippocampus', 'FS_Vol_R_Hippocampus'),
    'amygdala': ('FS_Vol_L_Amygdala', 'FS_Vol_R_Amygdala'),
    'caudate': ('FS_Vol_L_Caudate', 'FS_Vol_R_Caudate'),
    'putamen': ('FS_Vol_L_Putamen', 'FS_Vol_R_Putamen'),
    'pallidum': ('FS_Vol_L_Pallidum', 'FS_Vol_R_Pallidum'),
    'thalamus': ('FS_Vol_L_Thalamus', 'FS_Vol_R_Thalamus'),
    'lateral ventricle': ('FS_Vol_L_LatVentricle', 'FS_Vol_R_LatVentricle'),
    'accumbens': ('FS_Vol_L_Accumbens', 'FS_Vol_R_Accumbens'),
}

# PPMI diagnosis (research_group -> clinical label)
DX_LABEL = {
    'PD': "Parkinson's Disease",
    'Control': 'Healthy Control',
    'Prodromal': 'Prodromal Parkinsonism',
    'SWEDD': 'SWEDD',
}

AGE_BINS = [(0, 50), (50, 65), (65, 75), (75, 200)]
AGE_LABELS = ['30-49', '50-64', '65-74', '75+']

IMG_DEP_MAP = {
    'Location': 'image-grounded', 'Anatomy': 'image-grounded',
    'Presence': 'image-grounded', 'Measurement': 'image-grounded',
    'Longitudinal': 'image-grounded', 'Signal': 'image-grounded',
    'Diagnosis': 'image-informed', 'Reasoning': 'image-informed',
    'Severity': 'image-informed', 'Demographic': 'image-informed',
    'Modality': 'image-grounded',
}

# ============================================================
# MEDIANS — thresholds from data (anti-shortcut)
# ============================================================
def compute_medians(df):
    med = {}
    hl = pd.to_numeric(df.get('FS_Vol_L_Hippocampus'), errors='coerce')
    hr = pd.to_numeric(df.get('FS_Vol_R_Hippocampus'), errors='coerce')
    avg_h = ((hl + hr) / 2).dropna()
    med['hippo'] = int(round(avg_h.median() / 100) * 100) if len(avg_h) else 3200

    vent = pd.to_numeric(df.get('FS_Ventricles'), errors='coerce')
    icv = pd.to_numeric(df.get('FS_ICV'), errors='coerce')
    valid = vent.notna() & icv.notna() & (icv > 0)
    if valid.sum():
        ratio = (vent[valid] / icv[valid])
        med['vent_icv'] = ratio.median()
        med['vent_pct'] = round(ratio.median() * 100, 2)
    else:
        med['vent_icv'], med['vent_pct'] = 0.0177, 1.77

    el = pd.to_numeric(df.get('FS_Thick_L_Entorhinal'), errors='coerce')
    er = pd.to_numeric(df.get('FS_Thick_R_Entorhinal'), errors='coerce')
    avg_e = ((el + er) / 2).dropna()
    med['entorhinal'] = round(avg_e.median(), 1) if len(avg_e) else 3.5

    wb = pd.to_numeric(df.get('FS_WholeBrain'), errors='coerce').dropna()
    med['brain'] = int(round(wb.median() / 10000) * 10000) if len(wb) else 1000000

    # Lateral-ventricle ratio median (used by cleanup R11 compound-Presence logic)
    lvl = pd.to_numeric(df.get('FS_Vol_L_LatVentricle'), errors='coerce')
    lvr = pd.to_numeric(df.get('FS_Vol_R_LatVentricle'), errors='coerce')
    lv_ratio = ((lvl + lvr) / icv * 100).replace([np.inf, -np.inf], np.nan).dropna()
    med['latvent_pct'] = lv_ratio.median() if len(lv_ratio) else 2.0
    # raw hippo median (unrounded) — R11 uses the unrounded median
    med['hippo_raw'] = float(avg_h.median()) if len(avg_h) else 3200.0
    return med

# Signal WM-hypointensity severity thresholds (from data separation)
WM_PRESENT = 1000
WM_MILD = 1000
WM_MODERATE = 3000
WM_SEVERE = 8000

# ============================================================
# QA GENERATOR
# ============================================================
def generate_qa(df):
    MED = compute_medians(df)
    print(f"  Median avg hippocampal volume: {MED['hippo']:,} mm^3")
    print(f"  Median vent/ICV ratio: {MED['vent_pct']:.2f}%")
    print(f"  Median entorhinal thickness: {MED['entorhinal']:.1f} mm")
    print(f"  Median whole-brain volume: {MED['brain']:,} mm^3")

    MED_HIPPO, MED_VENT, MED_VPCT, MED_BRAIN = (
        MED['hippo'], MED['vent_icv'], MED['vent_pct'], MED['brain'])

    # brain YN template constants (no longer used in the text; unit is mm^3)
    BRAIN_HI = MED_BRAIN + 50000    # 1,170,000
    BRAIN_LO = MED_BRAIN - 100000   # 1,020,000

    # longitudinal: previous UPDRS-III per subject session order
    df = df.sort_values(['ppmi_subject_id', 'session_date']).reset_index(drop=True)
    subj_sessions = {s: g.index.tolist() for s, g in df.groupby('ppmi_subject_id')}
    baseline_u3 = load_baseline_updrs3()  # {PATNO: baseline NP3TOT} for motor severity

    all_qa = []
    for idx, row in df.iterrows():
        subj = row['ppmi_subject_id']
        qa = []

        hl, hr = sv(row, 'FS_Vol_L_Hippocampus'), sv(row, 'FS_Vol_R_Hippocampus')
        vent, icv = sv(row, 'FS_Ventricles'), sv(row, 'FS_ICV')
        el, er = sv(row, 'FS_Thick_L_Entorhinal'), sv(row, 'FS_Thick_R_Entorhinal')
        wmh = safe_float(sv(row, 'FS_Vol_WMHypo'))
        rg = sv(row, 'research_group')
        age = safe_float(sv(row, 'age'))
        sex = sv(row, 'sex')

        avg_h = (hl + hr) / 2 if (hl is not None and hr is not None) else None
        atrophy = (avg_h < MED_HIPPO) if avg_h is not None else None
        enl = (vent / icv > MED_VENT) if (vent is not None and icv not in (None, 0)) else None
        wm_present = (wmh is not None and wmh >= WM_PRESENT)

        def add(cat, typ, q, a, opts=None, img=True, ha=True):
            d = {'cat': cat, 'type': typ, 'q': q, 'a': a, 'img': img, 'ha': ha}
            if opts is not None:
                d['options'] = opts
            qa.append(d)

        # ---------- LOCATION (shared) ----------
        for rname, (lcol, rcol) in SUBCORT_PAIRS.items():
            lv, rv = sv(row, lcol), sv(row, rcol)
            if lv is None or rv is None:
                continue
            if abs(lv - rv) / max(lv, rv) * 100 < 5:
                continue
            smaller = 'left' if lv < rv else 'right'
            larger = 'right' if lv < rv else 'left'
            add('Location', 'YN',
                f'Is the {smaller} {rname} smaller than the {larger} {rname} in this scan?', 'Yes')
            add('Location', 'YN',
                f'Is the {larger} {rname} smaller than the {smaller} {rname} in this scan?', 'No')
            add('Location', 'Open',
                f'Which side has a smaller {rname} volume in this brain scan?', smaller.capitalize())

        # ---------- ANATOMY (shared) ----------
        if avg_h is not None:
            add('Anatomy', 'YN', 'Is the hippocampal volume below normal in this scan?',
                'Yes' if atrophy else 'No')
            add('Anatomy', 'YN', 'Is the hippocampal volume within normal range in this scan?',
                'No' if atrophy else 'Yes')
            ans = 'Hippocampal atrophy' if atrophy else 'Normal hippocampal volume'
            add('Anatomy', 'MCQ', 'What best describes the hippocampal morphology in this scan?',
                ans, ['Hippocampal atrophy', 'Normal hippocampal volume',
                      'Hippocampal enlargement', 'Hippocampal calcification'])
            most = 'Hippocampus' if atrophy else 'None (all normal)'
            add('Anatomy', 'MCQ', 'Which subcortical structure shows the most volume reduction in this scan?',
                most, ['Hippocampus', 'Caudate nucleus', 'Thalamus', 'None (all normal)'])
            add('Anatomy', 'YN', 'Is hippocampal atrophy present in this scan?',
                'Yes' if atrophy else 'No')
        if enl is not None:
            add('Anatomy', 'YN', 'Are the ventricles enlarged relative to brain size in this scan?',
                'Yes' if enl else 'No')
            add('Anatomy', 'YN', 'Are the ventricles normal in size relative to brain size in this scan?',
                'No' if enl else 'Yes')
            add('Anatomy', 'Open', 'Are the ventricles enlarged or normal in this scan?',
                'Enlarged' if enl else 'Normal')
        if el is not None and er is not None:
            thin = (el + er) / 2 < MED['entorhinal']
            add('Anatomy', 'YN', 'Is the entorhinal cortex thinner than normal in this scan?',
                'Yes' if thin else 'No')
            add('Anatomy', 'YN', 'Is the entorhinal cortical thickness within normal range in this scan?',
                'No' if thin else 'Yes')

        # ---------- MEASUREMENT (shared, PPMI ranges) ----------
        if avg_h is not None:
            add('Measurement', 'YN',
                f'Is the average hippocampal volume above {MED_HIPPO:,} mm³ in this scan?',
                'No' if atrophy else 'Yes')
            add('Measurement', 'YN',
                f'Is the average hippocampal volume below {MED_HIPPO:,} mm³ in this scan?',
                'Yes' if atrophy else 'No')
            if avg_h < 3000: h_ans = '< 3,000'
            elif avg_h < 3500: h_ans = '3,000 – 3,500'
            elif avg_h < 4000: h_ans = '3,500 – 4,000'
            else: h_ans = '> 4,000'
            add('Measurement', 'MCQ', 'What range best describes the average hippocampal volume (mm³)?',
                h_ans, ['< 3,000', '3,000 – 3,500', '3,500 – 4,000', '> 4,000'])
        if vent is not None and icv not in (None, 0):
            ratio = vent / icv * 100
            add('Measurement', 'YN',
                f'Is the ventricular-to-brain volume ratio greater than {MED_VPCT}% in this scan?',
                'Yes' if ratio > MED_VPCT else 'No')
            add('Measurement', 'YN',
                f'Is the ventricular-to-brain volume ratio less than {MED_VPCT}% in this scan?',
                'No' if ratio > MED_VPCT else 'Yes')
        wb = safe_float(sv(row, 'FS_WholeBrain'))
        if wb is not None:
            add('Measurement', 'YN',
                f'Is the total brain volume greater than {MED_BRAIN:,} mm^3 in this scan?',
                'Yes' if wb > MED_BRAIN else 'No')
            add('Measurement', 'YN',
                f'Is the total brain volume less than {MED_BRAIN:,} mm^3 in this scan?',
                'No' if wb > MED_BRAIN else 'Yes')
        if icv is not None:
            if icv < 1200000: i_ans = '< 1,200,000'
            elif icv < 1400000: i_ans = '1,200,000 – 1,400,000'
            elif icv < 1600000: i_ans = '1,400,000 – 1,600,000'
            else: i_ans = '> 1,600,000'
            add('Measurement', 'MCQ', 'What range best describes the intracranial volume of this scan?',
                i_ans, ['< 1,200,000', '1,200,000 – 1,400,000',
                        '1,400,000 – 1,600,000', '> 1,600,000'])

        # ---------- PRESENCE (shared) ----------
        if avg_h is not None:
            loss = min(hl, hr) < MED_HIPPO
            add('Presence', 'YN', 'Is hippocampal volume loss present in this scan?',
                'Yes' if loss else 'No')
            add('Presence', 'YN', 'Are both hippocampi within normal volume range in this scan?',
                'No' if loss else 'Yes')
        if enl is not None:
            add('Presence', 'YN', 'Is ventricular enlargement present in this scan?',
                'Yes' if enl else 'No')
            add('Presence', 'YN', 'Are the ventricles normal-sized in this scan?',
                'No' if enl else 'Yes')
        # compound / present MCQ — replicates cleanup R11 exactly
        # (findings use LATERAL-ventricle ratio and WMHypo>3000; question is
        #  rewritten to "Which best describes..." when >=2 findings).
        lvl_v, lvr_v = sv(row, 'FS_Vol_L_LatVentricle'), sv(row, 'FS_Vol_R_LatVentricle')
        if avg_h is not None:
            findings = []
            if avg_h < MED['hippo_raw']:
                findings.append('Hippocampal atrophy')
            if lvl_v is not None and lvr_v is not None and icv not in (None, 0):
                if (lvl_v + lvr_v) / icv * 100 > MED['latvent_pct']:
                    findings.append('Ventricular enlargement')
            if wmh is not None and wmh > 3000:
                findings.append('White matter changes')

            if len(findings) == 0:
                ans, opts, q = ('No significant findings',
                    ['Hippocampal atrophy', 'Ventricular enlargement',
                     'White matter changes', 'No significant findings'],
                    'Which of the following findings is present in this scan?')
            elif len(findings) == 1:
                others = [f for f in ['Hippocampal atrophy', 'Ventricular enlargement',
                                      'White matter changes'] if f != findings[0]]
                ans, opts, q = (findings[0], [findings[0]] + others[:2] + ['No significant findings'],
                                'Which of the following findings is present in this scan?')
            elif len(findings) == 2:
                comp = f'Both {findings[0].lower()} and {findings[1].lower()}'
                ans, opts, q = (comp, [findings[0] + ' only', findings[1] + ' only', comp,
                                       'No significant findings'],
                                'Which best describes the structural findings in this scan?')
            else:
                comp = f'{findings[0]}, {findings[1].lower()}, and {findings[2].lower()}'
                ans, opts, q = (comp, [findings[0] + ' only',
                                       f'Both {findings[0].lower()} and {findings[1].lower()}',
                                       comp, 'No significant findings'],
                                'Which best describes the structural findings in this scan?')
            add('Presence', 'MCQ', q, ans, opts)
            # NOT-present MCQ (independent template, retained)
            not_present = [f for f in ['Hippocampal atrophy', 'Ventricular enlargement',
                                       'White matter changes'] if f not in findings]
            if not_present:
                a2 = not_present[0]
                add('Presence', 'MCQ', 'Which of the following findings is NOT present in this scan?',
                    a2, ([a2] + findings + ['Tumor mass', 'Acute infarction'])[:4])

        # ---------- SIGNAL (PPMI: WM changes) ----------
        if wmh is not None:
            add('Signal', 'YN', 'Are white matter signal abnormalities present in this scan?',
                'Yes' if wm_present else 'No')
            add('Signal', 'YN', 'Does this scan show white matter signal abnormalities?',
                'Yes' if wm_present else 'No')
            add('Signal', 'YN', 'Is the white matter signal normal in this scan?',
                'No' if wm_present else 'Yes')
            if wmh < WM_MODERATE and wm_present: sev = 'Mild'
            elif not wm_present: sev = 'No changes'
            elif wmh < WM_SEVERE: sev = 'Moderate'
            else: sev = 'Severe'
            add('Signal', 'MCQ', 'What is the severity of white matter changes in this scan?',
                sev, ['No changes', 'Mild', 'Moderate', 'Severe'])

        # ---------- DIAGNOSIS (PPMI: research_group) ----------
        if rg in DX_LABEL:
            is_pd, is_ctrl = rg == 'PD', rg == 'Control'
            add('Diagnosis', 'MCQ', 'What is the most likely clinical classification based on this brain scan?',
                DX_LABEL[rg], ["Parkinson's Disease", 'Healthy Control',
                               'Prodromal Parkinsonism', 'SWEDD'])
            if is_pd or is_ctrl:
                add('Diagnosis', 'MCQ', 'What type of condition is most consistent with the findings in this scan?',
                    'Neurodegenerative movement disorder' if is_pd else 'Normal aging',
                    ['Neurodegenerative movement disorder', 'Vascular parkinsonism', 'Normal aging', 'Essential tremor'])
            add('Diagnosis', 'YN', 'Does this brain scan show structural findings consistent with Parkinson\'s disease?',
                'Yes' if is_pd else 'No')
            add('Diagnosis', 'YN',
                'Are there any structural findings that suggest an alternative diagnosis to idiopathic Parkinson\'s disease?',
                'No' if is_pd else 'Yes')
            add('Diagnosis', 'YN', 'Does this brain scan show any structural abnormalities?',
                'No' if is_ctrl else 'Yes')
            add('Diagnosis', 'YN', 'Is this a brain scan from a neurologically healthy subject?',
                'Yes' if is_ctrl else 'No')

        # ---------- SEVERITY (PPMI: hoehn_yahr / UPDRS-III / MoCA) ----------
        hy = safe_float(sv(row, 'hoehn_yahr'))
        u3 = safe_float(sv(row, 'UPDRS_III_total'))
        moca = safe_float(sv(row, 'MoCA'))
        if hy is not None:
            if hy <= 0: stage = 'No signs'
            elif hy <= 1: stage = 'Unilateral'
            elif hy <= 2: stage = 'Bilateral without balance impairment'
            else: stage = 'Bilateral with postural instability'
            add('Severity', 'MCQ', 'What stage of parkinsonism is most consistent with this subject\'s brain scan?',
                stage, ['No signs', 'Unilateral', 'Bilateral without balance impairment',
                        'Bilateral with postural instability'], ha=False)
        # Motor severity uses the BASELINE-visit UPDRS-III (see load_baseline_updrs3);
        # falls back to the merged-CSV nearest-visit value if raw is unavailable.
        u3_motor = baseline_u3.get(str(subj))
        if u3_motor is None:
            u3_motor = u3
        if u3_motor is not None:
            # Baseline UPDRS-III thresholds reproduce frozen motor severity ~96%.
            if u3_motor < 10: motor = 'Minimal'
            elif u3_motor < 31: motor = 'Mild-to-moderate'
            else: motor = 'Severe'
            add('Severity', 'MCQ', 'What is the motor severity level most consistent with this brain scan?',
                motor, ['Minimal', 'Mild-to-moderate', 'Severe'], ha=False)
        if moca is not None:
            # Reverse-engineered vs frozen (best single-field fit ~88.6%)
            if moca >= 26: cog = 'Normal'
            elif moca >= 18: cog = 'Mild cognitive impairment'
            else: cog = 'Moderate-to-severe cognitive impairment'
            add('Severity', 'Open', 'What cognitive severity level is most consistent with this scan?',
                cog, ha=False)

        # ---------- REASONING (shared-structural + PPMI motor) ----------
        if avg_h is not None and enl is not None:
            if atrophy and enl: conc = 'Hippocampal atrophy with ventricular enlargement'
            elif atrophy and not enl: conc = 'Hippocampal atrophy without ventricular enlargement'
            elif not atrophy and enl: conc = 'Ventricular enlargement without hippocampal atrophy'
            else: conc = 'No significant structural abnormality'
            add('Reasoning', 'MCQ', 'Which structural pattern best describes the findings in this scan?',
                conc, ['Hippocampal atrophy with ventricular enlargement',
                       'Hippocampal atrophy without ventricular enlargement',
                       'Ventricular enlargement without hippocampal atrophy',
                       'No significant structural abnormality'])
            add('Reasoning', 'Open', 'What is the most prominent structural finding in this brain scan?',
                'Atrophy' if atrophy else 'Normal')
            add('Reasoning', 'YN', 'Does the brain structure in this scan suggest neurodegenerative changes?',
                'Yes' if atrophy else 'No')
            add('Reasoning', 'YN', 'Is the overall brain morphology within normal limits in this scan?',
                'No' if atrophy else 'Yes')
        if rg in DX_LABEL:
            motor_dz = rg in ('PD', 'SWEDD')
            add('Reasoning', 'YN',
                'Do the structural patterns in this scan suggest motor-related changes in the subject\'s brain?',
                'Yes' if motor_dz else 'No')
            add('Reasoning', 'YN',
                'Do the structural patterns in this scan suggest preserved motor function in the subject\'s brain?',
                'No' if motor_dz else 'Yes')

        # ---------- DEMOGRAPHIC (shared) ----------
        if sex is not None:
            add('Demographic', 'YN', 'Is this subject female?',
                'Yes' if sex == 'Female' else 'No', img=False, ha=False)
            add('Demographic', 'Open', 'What is the sex of this subject?', sex, img=False, ha=False)
        if age is not None:
            add('Demographic', 'YN', 'Is this subject older than 65 years old?',
                'Yes' if age > 65 else 'No', img=False, ha=False)
            add('Demographic', 'YN', 'Is this subject younger than 65 years old?',
                'No' if age > 65 else 'Yes', img=False, ha=False)
            age_ans = None
            for (lo, hi), lab in zip(AGE_BINS, AGE_LABELS):
                if lo <= age < hi:
                    age_ans = lab
                    break
            if age_ans:
                add('Demographic', 'MCQ', 'What age range does this subject likely belong to?',
                    age_ans, list(AGE_LABELS), img=False, ha=False)

        # ---------- LONGITUDINAL (PPMI motor) ----------
        sess = subj_sessions.get(subj, [])
        pos = sess.index(idx) if idx in sess else -1
        if pos > 0:
            prev = df.loc[sess[pos - 1]]
            pu3 = safe_float(sv(prev, 'UPDRS_III_total'))
            if u3 is not None and pu3 is not None and u3 != pu3:
                worse = u3 > pu3
                add('Longitudinal', 'YN',
                    'Has motor function worsened compared to the previous visit?',
                    'Yes' if worse else 'No')
                add('Longitudinal', 'YN',
                    'Has motor function improved compared to the previous visit?',
                    'No' if worse else 'Yes')

        # attach metadata
        # Per-scan neuroqa_id (mirrors ADNI: cleanup_benchmark.py reassigns the
        # final global PPMI-##### id after det_shuffle; see VALIDATION.md).
        scan_id = row.get('neuroqa_id')
        for q in qa:
            cat = q.pop('cat')
            q['neuroqa_id'] = scan_id
            q['subject_id'] = str(subj)
            q['dataset'] = 'PPMI'
            q['image_required'] = q.pop('img')
            q['category'] = cat
            q['image_dependency'] = IMG_DEP_MAP.get(cat, 'unknown')
            q['question_type'] = q.pop('type')
            q['question'] = q.pop('q')
            q['answer'] = q.pop('a')
            q['session_date'] = str(row['session_date'])
            q['image_path'] = row.get('T1w_image_path')
            q['answerable_modalities'] = ['T1w', 'T2w', 'T1w+T2w']
            q['human_assessable'] = q.pop('ha')
        all_qa.extend(qa)

    return all_qa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', default='../outputs/ppmi_neuroqa_merged.csv')
    ap.add_argument('--output', default='../outputs/')
    args = ap.parse_args()

    df = pd.read_csv(args.input, low_memory=False)
    df['session_date'] = df['session_date'].astype(str)
    print(f"Loaded {len(df):,} scans, {df['ppmi_subject_id'].nunique():,} subjects")

    qa = generate_qa(df)
    print(f"Generated {len(qa):,} raw QA pairs")
    os.makedirs(args.output, exist_ok=True)
    with open(os.path.join(args.output, 'ppmi_qa_raw_all.json'), 'w') as f:
        json.dump(qa, f, indent=2, ensure_ascii=False)
    from collections import Counter
    print("By category:", dict(Counter(q['category'] for q in qa)))


if __name__ == '__main__':
    main()
