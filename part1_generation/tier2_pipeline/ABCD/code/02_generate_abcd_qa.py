#!/usr/bin/env python3
"""
NeuroQA -- Step 2: ABCD Template-Based QA Generator

Generates question-answer pairs from ABCD structured metadata CSV.
Reuses FreeSurfer-based templates from ADNI/PPMI (Location, Anatomy, Measurement,
Presence) and adapts for PEDIATRIC healthy population:
  - No disease diagnosis -- uses NIH Toolbox cognitive groups for stratification
  - "Below expected range" language instead of "atrophy"
  - Brain GROWTH in longitudinal (volumes increase in developing brains)
  - CBCL T-scores for behavioral severity
  - Data-driven median thresholds (no fixed constants)

Anti-shortcut by design:
  - 50/50 YN balance via template negation
  - Zero metadata in question text
  - MCQ options shuffled per instance
  - Cognitive-group-stratified subsampling
  - Two image-dependency tiers (grounded / informed)
  - Median-based thresholds computed from data at runtime
  - No fixed-answer templates (Modality, fixed health, fixed disease removed)

USAGE:
  python 02_generate_abcd_qa.py \
    --input  ../outputs/abcd_neuroqa_merged.csv \
    --output ../outputs/
"""

import pandas as pd
import numpy as np
import json
import random
import argparse
import os
import re
from collections import defaultdict, Counter
import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from skip_templates import filter_qa_list
from wording_fixes import apply_wording_fixes

random.seed(42)
np.random.seed(42)

# ============================================================
# THRESHOLDS -- only non-structural ones kept as constants
# ============================================================
WMH_PRESENT = 500            # mm3 (WM hypointensities -- rare in pediatric)
WMH_MODERATE = 1500
WMH_SEVERE = 5000

# NIH Toolbox cognitive thresholds
NIH_ABOVE_AVG = 115
NIH_AVG = 85
NIH_BELOW_AVG = 70

# CBCL T-score thresholds (Achenbach)
CBCL_NORMAL = 60             # T < 60 = Normal
CBCL_BORDERLINE = 65         # 60 <= T < 65 = Borderline
                              # T >= 65 = Clinical

AGE_BINS = [(8, 10), (10, 11), (11, 13), (13, 15)]
AGE_LABELS = ["8-10", "10-11", "11-13", "13-15"]

# ============================================================
# MEDIAN-BASED THRESHOLDS (computed at runtime)
# ============================================================
MED_HIPPO = None       # will be set from data
MED_VENT_PCT = None    # will be set from data
MED_ENTORH = None      # will be set from data
MED_BRAIN = None       # will be set from data
MED_CBCL_TOTAL = None  # median CBCL total T-score (replaces fixed 60/64)
MED_CBCL_INT = None    # median CBCL internalizing T-score
MED_CBCL_EXT = None    # median CBCL externalizing T-score
MED_NIH_TOTAL = None   # median NIH total composite score (replaces fixed 115)

# ============================================================
# REGION MAPPINGS (same as ADNI/PPMI)
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

THICK_PAIRS = {
    'entorhinal cortex': ('FS_Thick_L_Entorhinal', 'FS_Thick_R_Entorhinal'),
    'fusiform gyrus': ('FS_Thick_L_Fusiform', 'FS_Thick_R_Fusiform'),
    'precuneus': ('FS_Thick_L_Precuneus', 'FS_Thick_R_Precuneus'),
    'superior temporal gyrus': ('FS_Thick_L_SupTemporal', 'FS_Thick_R_SupTemporal'),
    'inferior temporal gyrus': ('FS_Thick_L_InfTemporal', 'FS_Thick_R_InfTemporal'),
    'posterior cingulate': ('FS_Thick_L_PostCing', 'FS_Thick_R_PostCing'),
    'superior frontal gyrus': ('FS_Thick_L_SupFrontal', 'FS_Thick_R_SupFrontal'),
    'superior parietal lobule': ('FS_Thick_L_SupParietal', 'FS_Thick_R_SupParietal'),
    'isthmus cingulate': ('FS_Thick_L_IsthmusCing', 'FS_Thick_R_IsthmusCing'),
    'medial orbitofrontal cortex': ('FS_Thick_L_MedOrbFrontal', 'FS_Thick_R_MedOrbFrontal'),
}

LOBE_MAP = {
    'temporal': ['hippocampus', 'amygdala', 'entorhinal cortex', 'fusiform gyrus',
                 'superior temporal gyrus', 'inferior temporal gyrus'],
    'frontal': ['superior frontal gyrus', 'medial orbitofrontal cortex'],
    'parietal': ['precuneus', 'superior parietal lobule'],
    'limbic': ['posterior cingulate', 'isthmus cingulate'],
    'subcortical': ['caudate', 'putamen', 'pallidum', 'thalamus', 'accumbens'],
}

# ============================================================
# HELPERS
# ============================================================
def sv(row, col):
    v = row.get(col)
    return None if pd.isna(v) else v

def round_vol(v):
    return int(round(v / 100) * 100)

def safe_float(v):
    if v is None:
        return None
    try:
        return float(str(v).replace('>', '').replace('<', ''))
    except (ValueError, TypeError):
        return None


def compute_medians(df):
    """Compute median thresholds from data at runtime."""
    global MED_HIPPO, MED_VENT_PCT, MED_ENTORH, MED_BRAIN
    global MED_CBCL_TOTAL, MED_CBCL_INT, MED_CBCL_EXT, MED_NIH_TOTAL

    # Hippocampal volume: average of L+R, round to nearest 100
    hl = pd.to_numeric(df.get('FS_Vol_L_Hippocampus'), errors='coerce')
    hr = pd.to_numeric(df.get('FS_Vol_R_Hippocampus'), errors='coerce')
    avg_h = ((hl + hr) / 2).dropna()
    if len(avg_h) > 0:
        MED_HIPPO = int(round(avg_h.median() / 100) * 100)
    else:
        MED_HIPPO = 3800

    # Ventricle/ICV ratio as percentage, round to 2 decimals
    vent = pd.to_numeric(df.get('FS_Ventricles'), errors='coerce')
    icv = pd.to_numeric(df.get('FS_ICV'), errors='coerce')
    mask = (vent.notna()) & (icv.notna()) & (icv > 0)
    if mask.sum() > 0:
        ratios = (vent[mask] / icv[mask]) * 100
        MED_VENT_PCT = round(ratios.median(), 2)
    else:
        MED_VENT_PCT = 1.5

    # Entorhinal thickness: average of L+R, round to 0.1
    el = pd.to_numeric(df.get('FS_Thick_L_Entorhinal'), errors='coerce')
    er = pd.to_numeric(df.get('FS_Thick_R_Entorhinal'), errors='coerce')
    avg_e = ((el + er) / 2).dropna()
    if len(avg_e) > 0:
        MED_ENTORH = round(avg_e.median(), 1)
    else:
        MED_ENTORH = 3.4

    # Brain volume: use FS_WholeBrain if values > 100000, else FS_Vol_TotalGM + FS_Vol_CerebralWM
    wb = pd.to_numeric(df.get('FS_WholeBrain'), errors='coerce')
    if wb.notna().sum() > 0 and wb.dropna().median() > 100000:
        brain_vals = wb.dropna()
    else:
        tgm = pd.to_numeric(df.get('FS_Vol_TotalGM'), errors='coerce')
        cwm = pd.to_numeric(df.get('FS_Vol_CerebralWM'), errors='coerce')
        brain_vals = (tgm + cwm).dropna()
    if len(brain_vals) > 0:
        MED_BRAIN = int(round(brain_vals.median() / 10000) * 10000)
    else:
        MED_BRAIN = 1100000

    # CBCL T-score medians (replace fixed 60/64 thresholds)
    cbcl_t = pd.to_numeric(df.get('CBCL_TotalProblems'), errors='coerce').dropna()
    MED_CBCL_TOTAL = round(cbcl_t.median(), 1) if len(cbcl_t) > 0 else 50.0

    cbcl_i = pd.to_numeric(df.get('CBCL_Internalizing'), errors='coerce').dropna()
    MED_CBCL_INT = round(cbcl_i.median(), 1) if len(cbcl_i) > 0 else 50.0

    cbcl_e = pd.to_numeric(df.get('CBCL_Externalizing'), errors='coerce').dropna()
    MED_CBCL_EXT = round(cbcl_e.median(), 1) if len(cbcl_e) > 0 else 50.0

    # NIH total composite median (replace fixed 115 threshold)
    nih_t = pd.to_numeric(df.get('NIH_TotalComp'), errors='coerce').dropna()
    MED_NIH_TOTAL = round(nih_t.median(), 1) if len(nih_t) > 0 else 100.0

    print(f"Computed medians from data:")
    print(f"  MED_HIPPO     = {MED_HIPPO:,} mm3")
    print(f"  MED_VENT_PCT  = {MED_VENT_PCT}%")
    print(f"  MED_ENTORH    = {MED_ENTORH} mm")
    print(f"  MED_BRAIN     = {MED_BRAIN:,} mm3")
    print(f"  MED_CBCL_TOTAL= {MED_CBCL_TOTAL}")
    print(f"  MED_CBCL_INT  = {MED_CBCL_INT}")
    print(f"  MED_CBCL_EXT  = {MED_CBCL_EXT}")
    print(f"  MED_NIH_TOTAL = {MED_NIH_TOTAL}")


# ============================================================
# QA GENERATOR
# ============================================================
def generate_qa(df):
    # Compute median-based thresholds from data
    compute_medians(df)

    all_qa = []

    subj_sessions = {}
    for subj, group in df.groupby('abcd_subject_id'):
        # visits in time order (sorting the session name as text puts the 2-year visit before baseline)
        subj_sessions[subj] = group.sort_values('session_nda', key=lambda s: s.map(lambda v: 0 if str(v).startswith('baseline') else int(str(v).split('_')[0]) if str(v).split('_')[0].isdigit() else 99)).index.tolist()

    for idx, row in df.iterrows():
        scan_id = row['neuroqa_id']
        subj_id = row['abcd_subject_id']
        qa = []

        # ========== MODALITY REMOVED -- all scans are T1, pure shortcut ==========

        # ========== 2. LOCATION (FS subcortical asymmetry -- same as ADNI/PPMI) ==========
        for rname, (lcol, rcol) in SUBCORT_PAIRS.items():
            lv, rv = sv(row, lcol), sv(row, rcol)
            if lv is None or rv is None:
                continue
            lv, rv = float(lv), float(rv)
            asym = abs(lv - rv) / max(lv, rv) * 100
            if asym < 5:
                continue
            smaller = 'left' if lv < rv else 'right'
            larger = 'right' if lv < rv else 'left'

            qa.append({'cat': 'Location', 'type': 'YN', 'img': True,
                       'q': f'Is the {smaller} {rname} smaller than the {larger} {rname} in this scan?',
                       'a': 'Yes'})
            qa.append({'cat': 'Location', 'type': 'YN', 'img': True,
                       'q': f'Is the {larger} {rname} smaller than the {smaller} {rname} in this scan?',
                       'a': 'No'})
            qa.append({'cat': 'Location', 'type': 'Open', 'img': True,
                       'q': f'Which side has a smaller {rname} volume in this scan?',
                       'a': smaller.capitalize()})
            # REMOVED: "In which brain region is X located?" MCQ -- 100% same-answer
            # anatomical knowledge shortcut (answer depends only on region name, not scan)
            # for lobe, regions in LOBE_MAP.items():
            #     if rname in regions:
            #         others = [l for l in LOBE_MAP if l != lobe]
            #         opts = [lobe.capitalize()] + [l.capitalize() for l in random.sample(others, 3)]
            #         random.shuffle(opts)
            #         qa.append({'cat': 'Location', 'type': 'MCQ', 'img': True,
            #                    'q': f'In which brain region is the {rname} located?',
            #                    'a': lobe.capitalize(), 'options': opts})
            #         break

        # ========== 3. ANATOMY (PEDIATRIC -- median-based thresholds) ==========
        hl, hr = sv(row, 'FS_Vol_L_Hippocampus'), sv(row, 'FS_Vol_R_Hippocampus')
        if hl is not None and hr is not None:
            hl, hr = float(hl), float(hr)
            avg_h = (hl + hr) / 2
            below = avg_h < MED_HIPPO
            qa.append({'cat': 'Anatomy', 'type': 'YN', 'img': True,
                       'q': 'Is the hippocampal volume below the expected range for this age group?',
                       'a': 'Yes' if below else 'No'})
            qa.append({'cat': 'Anatomy', 'type': 'YN', 'img': True,
                       'q': 'Is the hippocampal volume within the expected range for this age group?',
                       'a': 'No' if below else 'Yes'})
            ans = 'Below expected hippocampal volume' if below else 'Normal hippocampal volume'
            opts = ['Below expected hippocampal volume', 'Normal hippocampal volume',
                    'Enlarged hippocampal volume', 'Hippocampal asymmetry']
            random.shuffle(opts)
            qa.append({'cat': 'Anatomy', 'type': 'MCQ', 'img': True,
                       'q': 'What best describes the hippocampal morphology in this pediatric scan?',
                       'a': ans, 'options': opts})
            most_anom = 'Hippocampus' if below else 'None (all within expected range)'
            opts2 = ['Hippocampus', 'Caudate nucleus', 'Thalamus', 'None (all within expected range)']
            random.shuffle(opts2)
            qa.append({'cat': 'Anatomy', 'type': 'MCQ', 'img': True,
                       'q': 'Which subcortical structure shows the most volume deviation from expected range?',
                       'a': most_anom, 'options': opts2})
            qa.append({'cat': 'Anatomy', 'type': 'Open', 'img': True,
                       'q': 'Is hippocampal volume below expected for this age group?',
                       'a': 'Yes' if below else 'No'})

        vent_val = sv(row, 'FS_Ventricles')
        icv_val = sv(row, 'FS_ICV')
        if vent_val is not None and icv_val is not None:
            vent, icv = float(vent_val), float(icv_val)
            if icv > 0:
                ratio = vent / icv
                enlarged = (ratio * 100) > MED_VENT_PCT
                qa.append({'cat': 'Anatomy', 'type': 'YN', 'img': True,
                           'q': 'Are the ventricles relatively enlarged for a pediatric brain?',
                           'a': 'Yes' if enlarged else 'No'})
                qa.append({'cat': 'Anatomy', 'type': 'YN', 'img': True,
                           'q': 'Are the ventricles within normal size for a developing brain?',
                           'a': 'No' if enlarged else 'Yes'})
                qa.append({'cat': 'Anatomy', 'type': 'Open', 'img': True,
                           'q': 'Are the ventricles relatively enlarged or normal in this pediatric scan?',
                           'a': 'Enlarged' if enlarged else 'Normal'})

        el, er = sv(row, 'FS_Thick_L_Entorhinal'), sv(row, 'FS_Thick_R_Entorhinal')
        if el is not None and er is not None:
            el, er = float(el), float(er)
            avg_e = (el + er) / 2
            thin = avg_e < MED_ENTORH
            qa.append({'cat': 'Anatomy', 'type': 'YN', 'img': True,
                       'q': 'Is the entorhinal cortex thinner than expected for this age group?',
                       'a': 'Yes' if thin else 'No'})
            qa.append({'cat': 'Anatomy', 'type': 'YN', 'img': True,
                       'q': 'Is the entorhinal cortical thickness within expected range for this age group?',
                       'a': 'No' if thin else 'Yes'})

        # ========== 4. SIGNAL (WMH only -- T1 contrast templates removed) ==========
        wmh_fs = sv(row, 'FS_Vol_WMHypo')
        if wmh_fs is not None:
            wmh_fs = float(wmh_fs)
            has_wmh = wmh_fs > WMH_PRESENT
            qa.append({'cat': 'Signal', 'type': 'YN', 'img': True,
                       'q': 'Does this scan show white matter signal abnormalities?',
                       'a': 'Yes' if has_wmh else 'No'})
            qa.append({'cat': 'Signal', 'type': 'YN', 'img': True,
                       'q': 'Is the white matter signal normal in this scan?',
                       'a': 'No' if has_wmh else 'Yes'})
            if has_wmh:
                sev = 'Severe' if wmh_fs > WMH_SEVERE else ('Moderate' if wmh_fs > WMH_MODERATE else 'Mild')
            else:
                sev = 'None'
            opts = ['None', 'Mild', 'Moderate', 'Severe']
            random.shuffle(opts)
            qa.append({'cat': 'Signal', 'type': 'MCQ', 'img': True,
                       'q': 'What is the severity of white matter changes in this scan?',
                       'a': sev, 'options': opts})
            qa.append({'cat': 'Signal', 'type': 'Open', 'img': True,
                       'q': 'Are white matter signal abnormalities present in this scan?',
                       'a': 'Yes' if has_wmh else 'No'})

        # ========== 5. PRESENCE (per-subject findings only -- fixed health templates removed) ==========
        if hl is not None and hr is not None:
            loss = min(hl, hr) < MED_HIPPO
            qa.append({'cat': 'Presence', 'type': 'YN', 'img': True,
                       'q': 'Is hippocampal volume below expected range in this scan?',
                       'a': 'Yes' if loss else 'No'})
            qa.append({'cat': 'Presence', 'type': 'YN', 'img': True,
                       'q': 'Are both hippocampi within expected volume range for this age group?',
                       'a': 'No' if loss else 'Yes'})

        if vent_val is not None and icv_val is not None and float(icv_val) > 0:
            enl = (float(vent_val) / float(icv_val) * 100) > MED_VENT_PCT
            qa.append({'cat': 'Presence', 'type': 'YN', 'img': True,
                       'q': 'Is ventricular enlargement present in this scan?',
                       'a': 'Yes' if enl else 'No'})
            qa.append({'cat': 'Presence', 'type': 'YN', 'img': True,
                       'q': 'Are the ventricles normal-sized in this scan?',
                       'a': 'No' if enl else 'Yes'})

            # MCQ -- which finding present
            findings = []
            absent = []
            if hl is not None and avg_h < MED_HIPPO:
                findings.append('Below expected hippocampal volume')
            else:
                absent.append('Below expected hippocampal volume')
            if enl:
                findings.append('Ventricular enlargement')
            else:
                absent.append('Ventricular enlargement')
            # ABCD is pediatric -- WM hypointensities are rare and NOT treated as a Presence
            # "finding" present (frozen ABCD Presence "present" uses only hippocampal +
            # ventricular). But when WMH is absent it stays a valid "NOT present" answer.
            if not (wmh_fs is not None and wmh_fs > WMH_PRESENT):
                absent.append('White matter changes')

            # Present question: first finding, else "No significant findings"
            ans = findings[0] if findings else 'No significant findings'
            distractors = (absent + ['Tumor mass', 'Cortical malformation', 'Cyst']) if findings \
                          else ['Below expected hippocampal volume', 'Ventricular enlargement',
                                'White matter changes', 'Tumor mass']
            opts = [ans] + random.sample(distractors, min(3, len(distractors)))
            opts = opts[:4]
            random.shuffle(opts)
            qa.append({'cat': 'Presence', 'type': 'MCQ', 'img': True,
                       'q': 'Which of the following findings is present in this scan?',
                       'a': ans, 'options': opts})
            if absent:
                ans_a = absent[0]
                present_list = findings + ['Tumor mass', 'Cortical malformation', 'Cyst', 'Hemorrhage']
                present_list = [x for x in present_list if x != ans_a]
                opts_a = [ans_a] + random.sample(present_list, min(3, len(present_list)))
                opts_a = opts_a[:4]
                random.shuffle(opts_a)
                qa.append({'cat': 'Presence', 'type': 'MCQ', 'img': True,
                           'q': 'Which of the following findings is NOT present in this scan?',
                           'a': ans_a, 'options': opts_a})

        # REMOVED: Fixed health Presence templates (always Yes/No) -- pure shortcut
        # REMOVED: "Is this brain scan from a healthy pediatric subject?" -- always Yes

        # ========== 6. MEASUREMENT (median-based thresholds in question text) ==========
        if hl is not None:
            qa.append({'cat': 'Measurement', 'type': 'Open', 'img': True,
                       'q': 'What is the approximate left hippocampal volume (mm3)?',
                       'a': str(round_vol(hl))})
        if hr is not None:
            qa.append({'cat': 'Measurement', 'type': 'Open', 'img': True,
                       'q': 'What is the approximate right hippocampal volume (mm3)?',
                       'a': str(round_vol(hr))})
        if vent_val is not None:
            qa.append({'cat': 'Measurement', 'type': 'Open', 'img': True,
                       'q': 'What is the approximate total ventricular volume (mm3)?',
                       'a': str(int(round(float(vent_val) / 1000) * 1000))})
        tgm = sv(row, 'FS_Vol_TotalGM')
        if tgm is not None:
            qa.append({'cat': 'Measurement', 'type': 'Open', 'img': True,
                       'q': 'What is the approximate total gray matter volume (mm3)?',
                       'a': str(int(round(float(tgm) / 1000) * 1000))})

        # YN -- median-based threshold templates
        if hl is not None and hr is not None:
            avg_h = (hl + hr) / 2
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the average hippocampal volume below {MED_HIPPO:,} mm3 in this scan?',
                       'a': 'Yes' if avg_h < MED_HIPPO else 'No'})
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the average hippocampal volume above {MED_HIPPO:,} mm3 in this scan?',
                       'a': 'No' if avg_h < MED_HIPPO else 'Yes'})

        if vent_val is not None and icv_val is not None and float(icv_val) > 0:
            ratio_pct = float(vent_val) / float(icv_val) * 100
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the ventricle-to-brain ratio greater than {MED_VENT_PCT}% in this scan?',
                       'a': 'Yes' if ratio_pct > MED_VENT_PCT else 'No'})
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the ventricle-to-brain ratio less than {MED_VENT_PCT}% in this scan?',
                       'a': 'No' if ratio_pct > MED_VENT_PCT else 'Yes'})

        wb = sv(row, 'FS_WholeBrain')
        if wb is not None:
            wb = float(wb)
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the whole brain volume greater than {MED_BRAIN:,} mm3 in this scan?',
                       'a': 'Yes' if wb > MED_BRAIN else 'No'})
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the whole brain volume less than {MED_BRAIN:,} mm3 in this scan?',
                       'a': 'No' if wb > MED_BRAIN else 'Yes'})

        # MCQ -- ICV range
        if icv_val is not None:
            icv = float(icv_val)
            ranges = ['< 1,200,000', '1,200,000 - 1,400,000', '1,400,000 - 1,600,000', '> 1,600,000']
            if icv < 1200000: ans = ranges[0]
            elif icv < 1400000: ans = ranges[1]
            elif icv < 1600000: ans = ranges[2]
            else: ans = ranges[3]
            random.shuffle(ranges)
            qa.append({'cat': 'Measurement', 'type': 'MCQ', 'img': True,
                       'q': 'What range best describes the intracranial volume of this scan?',
                       'a': ans, 'options': ranges})

        # MCQ -- hippocampal volume range (pediatric)
        if hl is not None and hr is not None:
            avg_h = (hl + hr) / 2
            h_ranges = ['< 3,500', '3,500 - 3,800', '3,800 - 4,200', '> 4,200']
            if avg_h < 3500: h_ans = h_ranges[0]
            elif avg_h < 3800: h_ans = h_ranges[1]
            elif avg_h < 4200: h_ans = h_ranges[2]
            else: h_ans = h_ranges[3]
            random.shuffle(h_ranges)
            qa.append({'cat': 'Measurement', 'type': 'MCQ', 'img': True,
                       'q': 'What range best describes the average hippocampal volume (mm3)?',
                       'a': h_ans, 'options': h_ranges})

        # MCQ -- ventricular volume range
        if vent_val is not None:
            vent = float(vent_val)
            v_ranges = ['< 10,000', '10,000 - 20,000', '20,000 - 30,000', '> 30,000']
            if vent < 10000: v_ans = v_ranges[0]
            elif vent < 20000: v_ans = v_ranges[1]
            elif vent < 30000: v_ans = v_ranges[2]
            else: v_ans = v_ranges[3]
            random.shuffle(v_ranges)
            qa.append({'cat': 'Measurement', 'type': 'MCQ', 'img': True,
                       'q': 'What range best describes the total ventricular volume (mm3)?',
                       'a': v_ans, 'options': v_ranges})

        # Open -- thickness (pick 1 random region)
        thick_regions = list(THICK_PAIRS.items())
        random.shuffle(thick_regions)
        for rname, (lcol, rcol) in thick_regions[:1]:
            lt = sv(row, lcol)
            if lt is not None:
                qa.append({'cat': 'Measurement', 'type': 'Open', 'img': True,
                           'q': f'What is the approximate left {rname} cortical thickness (mm)?',
                           'a': f'{float(lt):.1f}'})

        # ========== 7. DIAGNOSIS (cognitive group only -- fixed health templates removed) ==========
        cg = sv(row, 'cognitive_group')
        if cg is not None and cg in ('Above', 'Average', 'Below'):
            cg_full = {'Above': 'Above average cognitive performance',
                       'Average': 'Average cognitive performance',
                       'Below': 'Below average cognitive performance'}

            # Balanced YN pair: pick one true and one false statement per subject
            if cg == 'Above':
                qa.append({'cat': 'Diagnosis', 'type': 'YN', 'img': True,
                           'q': 'Is this brain scan from a child with above average cognitive performance?',
                           'a': 'Yes'})
                qa.append({'cat': 'Diagnosis', 'type': 'YN', 'img': True,
                           'q': 'Is this brain scan from a child with below average cognitive performance?',
                           'a': 'No'})
            elif cg == 'Average':
                qa.append({'cat': 'Diagnosis', 'type': 'YN', 'img': True,
                           'q': 'Is this brain scan from a child with average cognitive performance?',
                           'a': 'Yes'})
                qa.append({'cat': 'Diagnosis', 'type': 'YN', 'img': True,
                           'q': 'Is this brain scan from a child with above average cognitive performance?',
                           'a': 'No'})
            else:  # Below
                qa.append({'cat': 'Diagnosis', 'type': 'YN', 'img': True,
                           'q': 'Is this brain scan from a child with below average cognitive performance?',
                           'a': 'Yes'})
                qa.append({'cat': 'Diagnosis', 'type': 'YN', 'img': True,
                           'q': 'Is this brain scan from a child with average cognitive performance?',
                           'a': 'No'})

            # REMOVED: Fixed healthy status pairs (always Yes/No) -- pure shortcut

            opts_cg = ['Above average cognitive performance', 'Average cognitive performance',
                       'Below average cognitive performance']
            random.shuffle(opts_cg)
            qa.append({'cat': 'Diagnosis', 'type': 'MCQ', 'img': True,
                       'q': 'What cognitive performance group does this subject belong to?',
                       'a': cg_full[cg], 'options': opts_cg + ['Unknown']})

            qa.append({'cat': 'Diagnosis', 'type': 'Open', 'img': True,
                       'q': 'What is the cognitive performance group for this subject?',
                       'a': cg})

        # ========== 8. REASONING (structure-cognition concordance only) ==========
        nih_total = safe_float(sv(row, 'NIH_TotalComp'))
        nih_fluid = safe_float(sv(row, 'NIH_FluidComp'))

        if cg is not None and hl is not None and hr is not None and vent_val is not None and icv_val is not None:
            avg_h = (hl + hr) / 2
            below = avg_h < MED_HIPPO
            enl = (float(vent_val) / float(icv_val) * 100) > MED_VENT_PCT if float(icv_val) > 0 else False

            # Data-driven MCQ: options reflect actual subject state
            if below and enl:
                reasoning_ans = 'Reduced hippocampal volume with enlarged ventricles'
            elif below and not enl:
                reasoning_ans = 'Reduced hippocampal volume with normal ventricles'
            elif not below and enl:
                reasoning_ans = 'Normal hippocampal volume with enlarged ventricles'
            else:
                reasoning_ans = 'Normal hippocampal volume with normal ventricles'
            reasoning_opts = [
                'Reduced hippocampal volume with enlarged ventricles',
                'Reduced hippocampal volume with normal ventricles',
                'Normal hippocampal volume with enlarged ventricles',
                'Normal hippocampal volume with normal ventricles',
            ]
            random.shuffle(reasoning_opts)
            qa.append({'cat': 'Reasoning', 'type': 'MCQ', 'img': True,
                       'q': 'Which description best matches the structural findings in this scan?',
                       'a': reasoning_ans, 'options': reasoning_opts})

            # Structure-cognition concordance YN
            concordant = (cg in ('Above', 'Average') and not below) or \
                         (cg == 'Below' and below)
            qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                       'q': 'Is there concordance between brain structure and cognitive performance in this subject?',
                       'a': 'Yes' if concordant else 'No'})
            qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                       'q': 'Is there a mismatch between brain structure and cognitive performance in this subject?',
                       'a': 'No' if concordant else 'Yes'})

            # Data-driven: brain morphology vs median
            qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                       'q': 'Is the overall brain morphology within expected limits for a developing brain?',
                       'a': 'Yes' if not below and not enl else 'No'})
            qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                       'q': 'Does the brain show structural deviations from expected developmental norms?',
                       'a': 'No' if not below and not enl else 'Yes'})

            # Data-driven: developmental concern depends on subject measurements
            qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                       'q': 'Do the brain measurements suggest any developmental concern?',
                       'a': 'Yes' if below or enl else 'No'})
            qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                       'q': 'Do the brain measurements suggest normal development?',
                       'a': 'No' if below or enl else 'Yes'})

            # REMOVED: Fixed "neurodegenerative changes" (always No) -- pure shortcut
            # REMOVED: Fixed "consistent with healthy developing brain" (always Yes) -- pure shortcut

            qa.append({'cat': 'Reasoning', 'type': 'Open', 'img': True,
                       'q': 'What is the most prominent structural finding in this brain scan?',
                       'a': 'Below expected volume' if below else 'Normal'})

        # ========== 9. SEVERITY (CBCL T-scores + NIH Fluid) ==========
        cbcl_int = safe_float(sv(row, 'CBCL_Internalizing'))
        cbcl_ext = safe_float(sv(row, 'CBCL_Externalizing'))
        cbcl_total = safe_float(sv(row, 'CBCL_TotalProblems'))

        if cbcl_total is not None:
            # Use median-based thresholds to avoid fixed-answer shortcuts
            # MED_CBCL_TOTAL splits ~50/50; borderline = median + 5
            med_border = MED_CBCL_TOTAL + 5
            if cbcl_total < MED_CBCL_TOTAL:
                cbcl_sev = 'Normal'
            elif cbcl_total < med_border:
                cbcl_sev = 'Borderline'
            else:
                cbcl_sev = 'Clinical'

            qa.append({'cat': 'Severity', 'type': 'YN', 'img': True,
                       'q': 'Does this subject have behavioral scores in the clinical range?',
                       'a': 'Yes' if cbcl_sev == 'Clinical' else 'No'})
            qa.append({'cat': 'Severity', 'type': 'YN', 'img': True,
                       'q': 'Are the behavioral scores in the normal range for this subject?',
                       'a': 'Yes' if cbcl_sev == 'Normal' else 'No'})

            opts_sev = ['Normal', 'Borderline', 'Clinical']
            random.shuffle(opts_sev)
            qa.append({'cat': 'Severity', 'type': 'MCQ', 'img': True,
                       'q': 'What is the behavioral severity level for this subject?',
                       'a': cbcl_sev, 'options': opts_sev + ['Severe']})

            qa.append({'cat': 'Severity', 'type': 'Open', 'img': True,
                       'q': 'What is the overall behavioral severity classification?',
                       'a': cbcl_sev})

        if cbcl_int is not None and cbcl_ext is not None:
            # Use median-based thresholds to avoid "Neither elevated" dominance
            if cbcl_int >= MED_CBCL_INT and cbcl_ext < MED_CBCL_EXT:
                pattern = 'Internalizing predominant'
            elif cbcl_ext >= MED_CBCL_EXT and cbcl_int < MED_CBCL_INT:
                pattern = 'Externalizing predominant'
            elif cbcl_int >= MED_CBCL_INT and cbcl_ext >= MED_CBCL_EXT:
                pattern = 'Both elevated'
            else:
                pattern = 'Neither elevated'
            opts_pat = ['Internalizing predominant', 'Externalizing predominant',
                        'Both elevated', 'Neither elevated']
            random.shuffle(opts_pat)
            qa.append({'cat': 'Severity', 'type': 'MCQ', 'img': True,
                       'q': 'What behavioral pattern best characterizes this subject?',
                       'a': pattern, 'options': opts_pat})

        if nih_fluid is not None:
            if nih_fluid >= 115: fluid_label = 'Above average'
            elif nih_fluid >= 85: fluid_label = 'Average'
            elif nih_fluid >= 70: fluid_label = 'Below average'
            else: fluid_label = 'Low'
            qa.append({'cat': 'Severity', 'type': 'Open', 'img': True,
                       'q': 'What is the fluid cognition level for this subject?',
                       'a': fluid_label})

        if nih_total is not None:
            # Use median-based threshold to avoid 100% No shortcut
            nih_thresh = MED_NIH_TOTAL
            qa.append({'cat': 'Severity', 'type': 'YN', 'img': True,
                       'q': f'Is the total cognitive composite score above average (> {nih_thresh})?',
                       'a': 'Yes' if nih_total > nih_thresh else 'No'})
            qa.append({'cat': 'Severity', 'type': 'YN', 'img': True,
                       'q': f'Is the total cognitive composite score below average (< {nih_thresh})?',
                       'a': 'Yes' if nih_total < nih_thresh else 'No'})

        # ========== 10. LONGITUDINAL (brain GROWTH -- volumes increase in developing brains) ==========
        sessions = subj_sessions.get(subj_id, [])
        if len(sessions) > 1:
            curr_pos = sessions.index(idx) if idx in sessions else -1
            if curr_pos > 0:
                prev_row = df.loc[sessions[curr_pos - 1]]

                # Hippocampal growth
                ph = sv(prev_row, 'FS_Vol_L_Hippocampus')
                if hl is not None and ph is not None:
                    ph = float(ph)
                    if ph > 0:
                        grew = hl > ph
                        pct = (hl - ph) / ph * 100
                        qa.append({'cat': 'Longitudinal', 'type': 'YN', 'img': True,
                                   'q': 'Has the left hippocampal volume increased compared to the previous scan?',
                                   'a': 'Yes' if grew else 'No'})
                        qa.append({'cat': 'Longitudinal', 'type': 'YN', 'img': True,
                                   'q': 'Has the left hippocampal volume remained stable or decreased compared to the previous scan?',
                                   'a': 'No' if grew else 'Yes'})
                        if abs(pct) > 1:
                            qa.append({'cat': 'Longitudinal', 'type': 'Open', 'img': True,
                                       'q': 'What is the approximate percentage change in left hippocampal volume?',
                                       'a': f'{pct:+.0f}%'})

                # Whole brain growth
                pwb = sv(prev_row, 'FS_WholeBrain')
                cwb = sv(row, 'FS_WholeBrain')
                if cwb is not None and pwb is not None:
                    cwb_f, pwb_f = float(cwb), float(pwb)
                    grew_wb = cwb_f > pwb_f
                    qa.append({'cat': 'Longitudinal', 'type': 'YN', 'img': True,
                               'q': 'Has the whole brain volume increased compared to the previous scan?',
                               'a': 'Yes' if grew_wb else 'No'})
                    qa.append({'cat': 'Longitudinal', 'type': 'YN', 'img': True,
                               'q': 'Has the whole brain volume remained stable or decreased compared to the previous scan?',
                               'a': 'No' if grew_wb else 'Yes'})

                # Cognitive change
                p_nih = safe_float(sv(prev_row, 'NIH_TotalComp'))
                c_nih = safe_float(sv(row, 'NIH_TotalComp'))
                if p_nih is not None and c_nih is not None:
                    improved = c_nih > p_nih
                    qa.append({'cat': 'Longitudinal', 'type': 'YN', 'img': True,
                               'q': 'Has the cognitive composite score improved compared to the previous visit?',
                               'a': 'Yes' if improved else 'No'})
                    qa.append({'cat': 'Longitudinal', 'type': 'YN', 'img': True,
                               'q': 'Has the cognitive composite score remained stable or declined compared to the previous visit?',
                               'a': 'No' if improved else 'Yes'})

                # Cognitive group progression
                pcg = sv(prev_row, 'cognitive_group')
                ccg = sv(row, 'cognitive_group')
                if pcg is not None and ccg is not None:
                    if pcg == ccg:
                        prog = 'Stable'
                    else:
                        cg_order = {'Below': 0, 'Average': 1, 'Above': 2}
                        if cg_order.get(ccg, 0) > cg_order.get(pcg, 0):
                            prog = 'Improved'
                        else:
                            prog = 'Declined'
                    opts_prog = ['Improved', 'Stable', 'Declined', 'Changed']
                    random.shuffle(opts_prog)
                    qa.append({'cat': 'Longitudinal', 'type': 'MCQ', 'img': True,
                               'q': 'Has the cognitive performance improved, remained stable, or declined compared to the previous visit?',
                               'a': prog, 'options': opts_prog})
                    qa.append({'cat': 'Longitudinal', 'type': 'Open', 'img': True,
                               'q': 'What is the cognitive trajectory compared to the previous visit?',
                               'a': prog})

        # ========== 11. DEMOGRAPHIC ==========
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

        if age is not None:
            qa.append({'cat': 'Demographic', 'type': 'YN', 'img': False,
                       'q': 'Is this subject older than 12 years?',
                       'a': 'Yes' if age > 12 else 'No'})
            qa.append({'cat': 'Demographic', 'type': 'YN', 'img': False,
                       'q': 'Is this subject younger than 12 years?',
                       'a': 'No' if age > 12 else 'Yes'})
            age_ans = None
            for j, (lo, hi) in enumerate(AGE_BINS):
                if lo <= age < hi or (j == len(AGE_BINS) - 1 and age >= lo):
                    age_ans = AGE_LABELS[j]
                    break
            if age_ans:
                opts = AGE_LABELS.copy()
                random.shuffle(opts)
                qa.append({'cat': 'Demographic', 'type': 'MCQ', 'img': False,
                           'q': 'What age range does this subject belong to?',
                           'a': age_ans, 'options': opts})

        # === Add metadata ===
        IMG_DEP_MAP = {
            'Location': 'image-grounded',
            'Anatomy': 'image-grounded',
            'Presence': 'image-grounded',
            'Measurement': 'image-grounded',
            'Longitudinal': 'image-grounded',
            'Diagnosis': 'image-informed',
            'Reasoning': 'image-informed',
            'Signal': 'image-informed',
            'Severity': 'image-informed',
            'Demographic': 'image-informed',
        }
        for q in qa:
            q['neuroqa_id'] = scan_id
            q['subject_id'] = str(subj_id)
            q['dataset'] = 'ABCD'
            q['image_required'] = q.pop('img')
            cat = q.pop('cat')
            q['category'] = cat
            q['image_dependency'] = IMG_DEP_MAP.get(cat, 'unknown')
            q['question_type'] = q.pop('type')
            q['question'] = q.pop('q')
            q['answer'] = q.pop('a')

        all_qa.extend(qa)

    # ========== LATERALITY POST-HOC BALANCING ==========
    # Balance laterality questions per template to avoid Left/Right bias
    lat_by_q = defaultdict(lambda: defaultdict(list))
    non_lat = []
    for item in all_qa:
        if item['category'] == 'Location' and ('smaller than' in item['question'] or
                                                 item.get('answer', '') in ('Left', 'Right')):
            lat_by_q[item['question']][item['answer']].append(item)
        else:
            non_lat.append(item)
    balanced_lat = []
    removed_lat = 0
    for q_text, by_ans in lat_by_q.items():
        counts = {a: len(items) for a, items in by_ans.items()}
        min_count = min(counts.values())
        if min_count < 3:
            removed_lat += sum(counts.values())
            continue
        for a, items in by_ans.items():
            random.shuffle(items)
            balanced_lat.extend(items[:min_count])
    if removed_lat > 0:
        print(f"Laterality balancing: removed {removed_lat} QA from heavily skewed templates")
    print(f"Laterality balancing: kept {len(balanced_lat)} QA")
    all_qa = non_lat + balanced_lat

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

    # Group by number of options
    for n_opts in df.loc[mcq_mask, 'options'].apply(len).unique():
        subset_idx = [i for i in mcq_idx if len(df.at[i, 'options']) == n_opts]
        n = len(subset_idx)
        # Create target positions: cycle through 0..n_opts-1 uniformly
        targets = list(range(n_opts)) * (n // n_opts + 1)
        targets = targets[:n]
        rng.shuffle(targets)

        for idx, target_pos in zip(subset_idx, targets):
            opts = list(df.at[idx, 'options'])
            ans = df.at[idx, 'answer']
            # Find current position of correct answer
            try:
                cur_pos = opts.index(ans)
            except ValueError:
                continue
            # Swap correct answer to target position
            if cur_pos != target_pos:
                opts[cur_pos], opts[target_pos] = opts[target_pos], opts[cur_pos]
            df.at[idx, 'options'] = opts

    # Report
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
# BALANCING & SUBSAMPLING (adapted from PPMI -- uses cognitive groups)
# ============================================================
def balance_and_subsample(qa_df, df_meta, target_total=15000, max_per_subj_per_cat=3):
    id_to_cg = df_meta.set_index('neuroqa_id')['cognitive_group'].to_dict()
    qa_df = qa_df.copy()
    qa_df['_cg'] = qa_df['neuroqa_id'].map(id_to_cg).fillna('Unknown')
    print(f"Cognitive group distribution in raw QA: {qa_df['_cg'].value_counts().to_dict()}")

    cat_order = ['Location', 'Anatomy', 'Signal', 'Presence',
                 'Measurement', 'Diagnosis', 'Reasoning', 'Severity',
                 'Longitudinal', 'Demographic']

    # Step 1: Balance YN per category
    balanced = []
    for cat in cat_order:
        sub = qa_df[qa_df['category'] == cat].copy()
        yn = sub[sub['question_type'] == 'YN']
        yes_q = yn[yn['answer'] == 'Yes']
        no_q = yn[yn['answer'] == 'No']
        n_min = min(len(yes_q), len(no_q))
        if n_min > 0:
            yn_balanced = pd.concat([yes_q.sample(n_min, random_state=42),
                                     no_q.sample(n_min, random_state=42)])
        else:
            yn_balanced = yn
        balanced.append(pd.concat([yn_balanced, sub[sub['question_type'] != 'YN']]))
    all_balanced = pd.concat(balanced, ignore_index=True)
    print(f"After YN balancing: {len(all_balanced):,}")

    # Step 2: Limit per subject per category (YN-balance-aware)
    limited = []
    yn_toggle = {}  # (cat,) -> counter to alternate Yes/No picks
    for (subj, cat), group in all_balanced.groupby(['subject_id', 'category']):
        if len(group) > max_per_subj_per_cat:
            sampled = []
            for qtype in ['YN', 'MCQ', 'Open']:
                type_q = group[group['question_type'] == qtype]
                if len(type_q) == 0:
                    continue
                if qtype == 'YN':
                    # Alternate between Yes/No to preserve balance
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
    print(f"After per-subject limit ({max_per_subj_per_cat}/cat): {len(all_limited):,}")

    # Step 3: Re-balance YN
    rebalanced = []
    for cat in cat_order:
        sub = all_limited[all_limited['category'] == cat]
        yn = sub[sub['question_type'] == 'YN']
        non_yn = sub[sub['question_type'] != 'YN']
        yes_q = yn[yn['answer'] == 'Yes']
        no_q = yn[yn['answer'] == 'No']
        n_min = min(len(yes_q), len(no_q))
        if n_min > 0:
            yn_bal = pd.concat([yes_q.sample(n_min, random_state=42),
                                no_q.sample(n_min, random_state=42)])
        else:
            yn_bal = yn
        rebalanced.append(pd.concat([yn_bal, non_yn]))
    all_rebalanced = pd.concat(rebalanced, ignore_index=True)
    print(f"After YN re-balancing: {len(all_rebalanced):,}")

    # Step 4: Subsample to target, stratified by category AND cognitive group
    if '_cg' not in all_rebalanced.columns:
        all_rebalanced['_cg'] = all_rebalanced['neuroqa_id'].map(id_to_cg)

    if len(all_rebalanced) > target_total:
        per_cat = target_total // len(cat_order)
        final = []
        for cat in cat_order:
            sub = all_rebalanced[all_rebalanced['category'] == cat]
            if len(sub) <= per_cat:
                final.append(sub)
                continue
            cg_groups = sub.groupby('_cg', dropna=False)
            n_cg = len(cg_groups)
            per_cg = per_cat // max(n_cg, 1)
            cg_sampled = []
            for cg_val, cg_sub in cg_groups:
                n_take = min(per_cg, len(cg_sub))
                if n_take == 0:
                    continue
                type_sampled = []
                for qtype in ['YN', 'MCQ', 'Open']:
                    type_sub = cg_sub[cg_sub['question_type'] == qtype]
                    type_share = len(type_sub) / len(cg_sub) if len(cg_sub) > 0 else 0.33
                    type_n = max(1, int(n_take * type_share))
                    if len(type_sub) <= type_n:
                        type_sampled.append(type_sub)
                    elif qtype == 'YN':
                        yes_q = type_sub[type_sub['answer'] == 'Yes']
                        no_q = type_sub[type_sub['answer'] == 'No']
                        half = type_n // 2
                        type_sampled.append(pd.concat([
                            yes_q.sample(min(half, len(yes_q)), random_state=42),
                            no_q.sample(min(half, len(no_q)), random_state=42)]))
                    else:
                        type_sampled.append(type_sub.sample(type_n, random_state=42))
                cg_sampled.append(pd.concat(type_sampled))
            final.append(pd.concat(cg_sampled))
        all_final = pd.concat(final, ignore_index=True)
    else:
        all_final = all_rebalanced

    # Final YN balance enforcement
    final_parts = []
    for cat in cat_order:
        sub = all_final[all_final['category'] == cat]
        yn = sub[sub['question_type'] == 'YN']
        non_yn = sub[sub['question_type'] != 'YN']
        yes_q = yn[yn['answer'] == 'Yes']
        no_q = yn[yn['answer'] == 'No']
        n_min = min(len(yes_q), len(no_q))
        if n_min > 0:
            yn_bal = pd.concat([yes_q.sample(n_min, random_state=42),
                                no_q.sample(n_min, random_state=42)])
        else:
            yn_bal = yn
        final_parts.append(pd.concat([yn_bal, non_yn]))

    all_final = pd.concat(final_parts, ignore_index=True)
    all_final = all_final.drop(columns=['_cg'], errors='ignore')

    # MCQ position balance: ensure correct answer is uniformly distributed across positions
    all_final = balance_mcq_positions(all_final)

    print(f"Final: {len(all_final):,} QA pairs")
    return all_final


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate ABCD QA pairs for NeuroQA')
    parser.add_argument('--input', default='../outputs/abcd_neuroqa_merged.csv')
    parser.add_argument('--output', default='../outputs/')
    args = parser.parse_args()

    print("Loading ABCD data...")
    df = pd.read_csv(args.input, low_memory=False)
    df = df.sort_values(['abcd_subject_id', 'session_nda']).reset_index(drop=True)
    print(f"Loaded {len(df):,} scans, {df['abcd_subject_id'].nunique():,} subjects\n")

    print("Generating all QA pairs...")
    all_qa = generate_qa(df)
    all_qa, _n_skipped = filter_qa_list(all_qa)
    qa_df = pd.DataFrame(all_qa)
    print(f"Total raw QA: {len(qa_df):,}")

    cat_order = ['Location', 'Anatomy', 'Signal', 'Presence',
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
        print(f"{cat:<25} {len(yn):>8,} {mcq:>8,} {opn:>8,} {len(s):>8,}  {y_pct:>4.0f}% {100-y_pct:>4.0f}%")

    print("\n" + "="*80)
    print("BALANCING & SUBSAMPLING")
    print("="*80)
    final_df = balance_and_subsample(qa_df, df, target_total=15000, max_per_subj_per_cat=3)

    print(f"\n{'Category':<25} {'YN':>8} {'MCQ':>8} {'Open':>8} {'Total':>8}  {'Y%':>5} {'N%':>5}")
    print("-"*80)
    gt_yn = gt_mcq = gt_op = 0
    for cat in cat_order:
        s = final_df[final_df['category'] == cat]
        yn = s[s['question_type'] == 'YN']
        mcq = len(s[s['question_type'] == 'MCQ'])
        opn = len(s[s['question_type'] == 'Open'])
        gt_yn += len(yn); gt_mcq += mcq; gt_op += opn
        y_pct = (yn['answer'] == 'Yes').mean() * 100 if len(yn) > 0 else 0
        print(f"{cat:<25} {len(yn):>8,} {mcq:>8,} {opn:>8,} {len(s):>8,}  {y_pct:>4.0f}% {100-y_pct:>4.0f}%")
    print("-"*80)
    total = gt_yn + gt_mcq + gt_op
    print(f"{'TOTAL':<25} {gt_yn:>8,} {gt_mcq:>8,} {gt_op:>8,} {total:>8,}")

    # Cognitive group distribution
    id_to_cg = df.set_index('neuroqa_id')['cognitive_group'].to_dict()
    final_df['_cg'] = final_df['neuroqa_id'].map(id_to_cg)
    print(f"\nCognitive group distribution:")
    for k, v in final_df['_cg'].value_counts(dropna=False).items():
        print(f"  {k}: {v:,} ({100*v/total:.1f}%)")

    # Template count
    def extract_template(q):
        return re.sub(r'[\d,]+(?:\.\d+)?%?', '{N}', q)
    total_templates = 0
    for cat in cat_order:
        sub = final_df[final_df['category'] == cat]
        tpl = sub['question'].apply(extract_template).nunique()
        total_templates += tpl
    print(f"\nTotal unique templates: {total_templates}")

    # Save
    final_df = final_df.drop(columns=['_cg'], errors='ignore')
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)
    _recs = final_df.to_dict(orient='records')
    apply_wording_fixes(_recs)  # apply expert-reviewed wording fixes (in-place)
    final_df = pd.DataFrame(_recs)
    final_df.to_json(os.path.join(out_dir, 'abcd_qa_final.json'), orient='records', indent=2)
    final_df.to_csv(os.path.join(out_dir, 'abcd_qa_final.csv'), index=False)
    qa_df.to_json(os.path.join(out_dir, 'abcd_qa_raw_all.json'), orient='records', indent=2)
    print(f"\nSaved: abcd_qa_final.json ({len(final_df):,}) + abcd_qa_raw_all.json ({len(qa_df):,})")


# Apply expert-reviewed wording fixes
# apply_wording_fixes(final_qa)  # TODO: add correct variable name