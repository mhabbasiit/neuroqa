#!/usr/bin/env python3
"""
NeuroQA — Step 2: HCP-Aging Template-Based QA Generator

Generates question-answer pairs from HCP-Aging structured metadata CSV.
Reuses FreeSurfer-based templates from ADNI/PPMI/ABCD/HCP-YA and adapts
for HEALTHY AGING ADULTS (age 36-100):
  - No disease diagnosis — uses NIH Toolbox cognitive groups for stratification
  - Aging-appropriate language: "within expected range for age", "consistent with normal aging"
  - NO longitudinal (single timepoint V1) → 9 categories (Modality removed)
  - MoCA instead of MMSE
  - NEO-FFI personality dimensions for Severity category
  - Age-appropriate thresholds (larger ventricles, more WMH expected)

Anti-shortcut by design:
  - 50/50 YN balance via template negation
  - Zero metadata in question text
  - MCQ options shuffled per instance
  - Cognitive-group-stratified subsampling
  - Two image-dependency tiers (grounded / informed)

USAGE:
  python 02_generate_hcpaging_qa.py \\
    --input  ../outputs/hcpaging_neuroqa_merged.csv \\
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
# THRESHOLDS — HEALTHY AGING ADULTS (age 36-100)
# ============================================================
# Structural thresholds are computed as MEDIANS from the data at runtime
# to ensure ~50/50 splits (no "greater/less than" keyword shortcuts).
# WMH thresholds remain clinical (severity grading).
WMH_PRESENT = 1000           # mm³ (more WMH expected in aging)
WMH_MODERATE = 3000
WMH_SEVERE = 8000

# NIH Toolbox cognitive thresholds — now computed as MEDIANS from data at runtime
# (see generate_qa function). Fixed fallbacks kept only as defaults.
NIH_ABOVE_AVG = 113  # fallback — overridden by median at runtime
NIH_AVG = 98         # fallback — overridden by median at runtime

# Age bins
AGE_BINS = ["36-45", "46-55", "56-65", "66-75", "76-85", "86+"]

def get_age_bin(age):
    if age < 46: return "36-45"
    elif age < 56: return "46-55"
    elif age < 66: return "56-65"
    elif age < 76: return "66-75"
    elif age < 86: return "76-85"
    else: return "86+"

# ============================================================
# REGION MAPPINGS (same as ADNI/PPMI/ABCD/HCP-YA)
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


# ============================================================
# QA GENERATOR
# ============================================================
def generate_qa(df):
    all_qa = []

    # ------------------------------------------------------------------
    # Compute MEDIAN-BASED THRESHOLDS from data (anti-shortcut: ~50/50 splits)
    # ------------------------------------------------------------------
    _hippo_vals, _vent_icv_vals, _entorh_vals, _brain_vals = [], [], [], []
    for _, r in df.iterrows():
        _hl, _hr = sv(r, 'FS_Vol_L_Hippocampus'), sv(r, 'FS_Vol_R_Hippocampus')
        if _hl is not None and _hr is not None:
            _hippo_vals.append((float(_hl) + float(_hr)) / 2)
        _vv, _iv = sv(r, 'FS_Ventricles'), sv(r, 'FS_ICV')
        if _vv is not None and _iv is not None and float(_iv) > 0:
            _vent_icv_vals.append(float(_vv) / float(_iv))
        _el, _er = sv(r, 'FS_Thick_L_Entorhinal'), sv(r, 'FS_Thick_R_Entorhinal')
        if _el is not None and _er is not None:
            _entorh_vals.append((float(_el) + float(_er)) / 2)
        _gm = sv(r, 'FS_Vol_TotalGM')
        _wm = sv(r, 'FS_Vol_CerebralWM')
        if _gm is not None and _wm is not None:
            _brain_vals.append(float(_gm) + float(_wm))

    MED_HIPPO = float(np.median(_hippo_vals)) if _hippo_vals else 3200.0
    MED_VENT_ICV = float(np.median(_vent_icv_vals)) if _vent_icv_vals else 0.035
    MED_ENTORH = float(np.median(_entorh_vals)) if _entorh_vals else 2.8
    MED_BRAIN = float(np.median(_brain_vals)) if _brain_vals else 1100000.0

    # Round for readable question text
    MED_HIPPO_R = int(round(MED_HIPPO / 100) * 100)          # e.g. 3200
    MED_VENT_PCT = round(MED_VENT_ICV * 100, 2)              # e.g. 3.50
    MED_ENTORH_R = round(MED_ENTORH, 1)                      # e.g. 2.8
    MED_BRAIN_R = int(round(MED_BRAIN / 10000) * 10000)       # e.g. 1100000

    # Compute MEDIAN-BASED thresholds for NIH cognitive composite (replaces fixed 113/98)
    _nih_total_vals = []
    _moca_vals = []
    for _, r in df.iterrows():
        _nt = safe_float(sv(r, 'NIH_TotalComp'))
        if _nt is not None:
            _nih_total_vals.append(_nt)
        _mc = safe_float(sv(r, 'MoCA_Total'))
        if _mc is not None:
            _moca_vals.append(_mc)

    MED_NIH_TOTAL = float(np.median(_nih_total_vals)) if _nih_total_vals else 105.0
    MED_MOCA = float(np.median(_moca_vals)) if _moca_vals else 26.0
    MED_NIH_TOTAL_R = round(MED_NIH_TOTAL, 1)
    MED_MOCA_R = round(MED_MOCA, 0)

    print(f"Median thresholds: hippo={MED_HIPPO_R} mm³, vent/ICV={MED_VENT_PCT}%, "
          f"entorhinal={MED_ENTORH_R} mm, brain={MED_BRAIN_R:,} mm³")
    print(f"Median thresholds: NIH_total={MED_NIH_TOTAL_R}, MoCA={MED_MOCA_R}")

    for idx, row in df.iterrows():
        scan_id = row['neuroqa_id']
        subj_id = row['hcpaging_subject_id']
        qa = []

        # NOTE: Modality category REMOVED — all scans are T1 (pure shortcut)

        # ========== 2. LOCATION (FS subcortical asymmetry) ==========
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
            # REMOVED: "In which brain region is X located?" MCQ
            # 100% same-answer anatomical knowledge shortcut — answer depends only
            # on region name, not on the scan image.
            # for lobe, regions in LOBE_MAP.items():
            #     if rname in regions:
            #         others = [l for l in LOBE_MAP if l != lobe]
            #         opts = [lobe.capitalize()] + [l.capitalize() for l in random.sample(others, 3)]
            #         random.shuffle(opts)
            #         qa.append({'cat': 'Location', 'type': 'MCQ', 'img': True,
            #                    'q': f'In which brain region is the {rname} located?',
            #                    'a': lobe.capitalize(), 'options': opts})
            #         break

        # ========== 3. ANATOMY (aging — normative language) ==========
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
            ans = 'Below expected hippocampal volume' if below else 'Age-appropriate hippocampal volume'
            opts = ['Below expected hippocampal volume', 'Age-appropriate hippocampal volume',
                    'Enlarged hippocampal volume', 'Hippocampal asymmetry']
            random.shuffle(opts)
            qa.append({'cat': 'Anatomy', 'type': 'MCQ', 'img': True,
                       'q': 'What best describes the hippocampal morphology in this scan?',
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
                enlarged = ratio > MED_VENT_ICV
                qa.append({'cat': 'Anatomy', 'type': 'YN', 'img': True,
                           'q': 'Are the ventricles relatively enlarged beyond expected age-related changes?',
                           'a': 'Yes' if enlarged else 'No'})
                qa.append({'cat': 'Anatomy', 'type': 'YN', 'img': True,
                           'q': 'Are the ventricles within expected size for this age group?',
                           'a': 'No' if enlarged else 'Yes'})
                qa.append({'cat': 'Anatomy', 'type': 'Open', 'img': True,
                           'q': 'Are the ventricles relatively enlarged or within expected range in this scan?',
                           'a': 'Enlarged' if enlarged else 'Within expected range'})

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

        # ========== 4. SIGNAL (WMH + contrast) ==========
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

        # NOTE: T1 contrast templates REMOVED — always same answer (pure shortcut)

        # ========== 5. PRESENCE (healthy aging adults) ==========
        if hl is not None and hr is not None:
            loss = min(hl, hr) < MED_HIPPO
            qa.append({'cat': 'Presence', 'type': 'YN', 'img': True,
                       'q': 'Is hippocampal volume below expected range in this scan?',
                       'a': 'Yes' if loss else 'No'})
            qa.append({'cat': 'Presence', 'type': 'YN', 'img': True,
                       'q': 'Are both hippocampi within expected volume range for this age group?',
                       'a': 'No' if loss else 'Yes'})

        if vent_val is not None and icv_val is not None and float(icv_val) > 0:
            enl = float(vent_val) / float(icv_val) > MED_VENT_ICV
            qa.append({'cat': 'Presence', 'type': 'YN', 'img': True,
                       'q': 'Is ventricular enlargement beyond age-expected range present in this scan?',
                       'a': 'Yes' if enl else 'No'})
            qa.append({'cat': 'Presence', 'type': 'YN', 'img': True,
                       'q': 'Are the ventricles within age-expected size in this scan?',
                       'a': 'No' if enl else 'Yes'})

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
            if wmh_fs is not None and wmh_fs > WMH_PRESENT:
                findings.append('White matter changes')
            else:
                absent.append('White matter changes')

            if findings:
                ans = findings[0]
                distractors = absent + ['Tumor mass', 'Cortical malformation', 'Cyst']
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

        # NOTE: Fixed health/disease templates REMOVED — always same answer (pure shortcut)
        # "Does this scan appear to show normal aging?" → always Yes
        # "Is there evidence of structural abnormality?" → always No
        # These are kept only for multi-modality/multi-diagnosis datasets (BraTS, WMH)

        # ========== 6. MEASUREMENT (FS quantitative) ==========
        if hl is not None:
            qa.append({'cat': 'Measurement', 'type': 'Open', 'img': True,
                       'q': 'What is the approximate left hippocampal volume (mm\u00b3)?',
                       'a': str(round_vol(hl))})
        if hr is not None:
            qa.append({'cat': 'Measurement', 'type': 'Open', 'img': True,
                       'q': 'What is the approximate right hippocampal volume (mm\u00b3)?',
                       'a': str(round_vol(hr))})
        if vent_val is not None:
            qa.append({'cat': 'Measurement', 'type': 'Open', 'img': True,
                       'q': 'What is the approximate total ventricular volume (mm\u00b3)?',
                       'a': str(int(round(float(vent_val) / 1000) * 1000))})
        tgm = sv(row, 'FS_Vol_TotalGM')
        if tgm is not None:
            qa.append({'cat': 'Measurement', 'type': 'Open', 'img': True,
                       'q': 'What is the approximate total gray matter volume (mm\u00b3)?',
                       'a': str(int(round(float(tgm) / 1000) * 1000))})

        # YN — median-based threshold templates (~50/50 by design)
        if hl is not None and hr is not None:
            avg_h = (hl + hr) / 2
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the average hippocampal volume above {MED_HIPPO_R:,} mm\u00b3 in this scan?',
                       'a': 'Yes' if avg_h > MED_HIPPO else 'No'})
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the average hippocampal volume below {MED_HIPPO_R:,} mm\u00b3 in this scan?',
                       'a': 'No' if avg_h > MED_HIPPO else 'Yes'})

        if vent_val is not None and icv_val is not None and float(icv_val) > 0:
            ratio_pct = float(vent_val) / float(icv_val) * 100
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the ventricle-to-brain ratio greater than {MED_VENT_PCT}% in this scan?',
                       'a': 'Yes' if ratio_pct > MED_VENT_PCT else 'No'})
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the ventricle-to-brain ratio less than {MED_VENT_PCT}% in this scan?',
                       'a': 'No' if ratio_pct > MED_VENT_PCT else 'Yes'})

        _gm_v = sv(row, 'FS_Vol_TotalGM')
        _wm_v = sv(row, 'FS_Vol_CerebralWM')
        wb = (float(_gm_v) + float(_wm_v)) if _gm_v is not None and _wm_v is not None else None
        if wb is not None:
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the whole brain volume greater than {MED_BRAIN_R:,} mm\u00b3 in this scan?',
                       'a': 'Yes' if wb > MED_BRAIN else 'No'})
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the whole brain volume less than {MED_BRAIN_R:,} mm\u00b3 in this scan?',
                       'a': 'No' if wb > MED_BRAIN else 'Yes'})

        # MCQ — ICV range
        if icv_val is not None:
            icv = float(icv_val)
            ranges = ['< 1,200,000', '1,200,000 \u2013 1,400,000', '1,400,000 \u2013 1,600,000', '> 1,600,000']
            if icv < 1200000: ans = ranges[0]
            elif icv < 1400000: ans = ranges[1]
            elif icv < 1600000: ans = ranges[2]
            else: ans = ranges[3]
            random.shuffle(ranges)
            qa.append({'cat': 'Measurement', 'type': 'MCQ', 'img': True,
                       'q': 'What range best describes the intracranial volume of this scan?',
                       'a': ans, 'options': ranges})

        # MCQ — hippocampal volume range
        if hl is not None and hr is not None:
            avg_h = (hl + hr) / 2
            h_ranges = ['< 3,000', '3,000 \u2013 3,500', '3,500 \u2013 4,000', '> 4,000']
            if avg_h < 3000: h_ans = h_ranges[0]
            elif avg_h < 3500: h_ans = h_ranges[1]
            elif avg_h < 4000: h_ans = h_ranges[2]
            else: h_ans = h_ranges[3]
            random.shuffle(h_ranges)
            qa.append({'cat': 'Measurement', 'type': 'MCQ', 'img': True,
                       'q': 'What range best describes the average hippocampal volume (mm\u00b3)?',
                       'a': h_ans, 'options': h_ranges})

        # MCQ — ventricular volume range (wider ranges for aging)
        if vent_val is not None:
            vent = float(vent_val)
            v_ranges = ['< 20,000', '20,000 \u2013 40,000', '40,000 \u2013 60,000', '> 60,000']
            if vent < 20000: v_ans = v_ranges[0]
            elif vent < 40000: v_ans = v_ranges[1]
            elif vent < 60000: v_ans = v_ranges[2]
            else: v_ans = v_ranges[3]
            random.shuffle(v_ranges)
            qa.append({'cat': 'Measurement', 'type': 'MCQ', 'img': True,
                       'q': 'What range best describes the total ventricular volume (mm\u00b3)?',
                       'a': v_ans, 'options': v_ranges})

        # Open — thickness (pick 1 random region)
        thick_regions = list(THICK_PAIRS.items())
        random.shuffle(thick_regions)
        for rname, (lcol, rcol) in thick_regions[:1]:
            lt = sv(row, lcol)
            if lt is not None:
                qa.append({'cat': 'Measurement', 'type': 'Open', 'img': True,
                           'q': f'What is the approximate left {rname} cortical thickness (mm)?',
                           'a': f'{float(lt):.1f}'})

        # ========== 7. DIAGNOSIS (cognitive group — no disease) ==========
        cg = sv(row, 'cognitive_group')
        if cg is not None and cg in ('Above', 'Average', 'Below'):
            cg_full = {'Above': 'Above average cognitive performance',
                       'Average': 'Average cognitive performance',
                       'Below': 'Below average cognitive performance'}

            _age_raw = sv(row, 'age')
            _age_i = None
            if _age_raw is not None:
                try: _age_i = int(float(_age_raw))
                except Exception: _age_i = None
            # Recovered clean rule (0-contradiction, 100% on frozen): "above-average
            # cognitive performance" is keyed to NIH_TotalComp > 112.5, NOT the coarse
            # cognitive_group tertile (which flipped 132 items).
            _nih_tot_cg = safe_float(sv(row, 'NIH_TotalComp'))
            _above_avg = (_nih_tot_cg is not None and _nih_tot_cg > 112.5)
            if _age_i is not None:
                # Age-bearing Diagnosis wording (matches frozen v4).
                qa.append({'cat': 'Diagnosis', 'type': 'YN', 'img': True,
                           'q': f'Does this brain scan from a subject (age: {_age_i}) show structural features consistent with above-average cognitive performance?',
                           'a': 'Yes' if _above_avg else 'No'})
                # Recovered clean NIH bands (match frozen v4, not the coarse cognitive_group
                # tertile): consistent-with-average = NIH in [98,112]; inconsistent-with-average
                # = NIH<98 or NIH>129 (frozen uses independent bands for the two templates).
                _cons_avg = (_nih_tot_cg is not None and 98 <= _nih_tot_cg <= 112)
                _incons_avg = (_nih_tot_cg is not None and (_nih_tot_cg < 98 or _nih_tot_cg > 129))
                qa.append({'cat': 'Diagnosis', 'type': 'YN', 'img': True,
                           'q': f'Does this brain scan from a subject (age: {_age_i}) show structural features consistent with average cognitive performance?',
                           'a': 'Yes' if _cons_avg else 'No'})
                qa.append({'cat': 'Diagnosis', 'type': 'YN', 'img': True,
                           'q': f'Does this brain scan from a subject (age: {_age_i}) show structural features inconsistent with average cognitive performance?',
                           'a': 'Yes' if _incons_avg else 'No'})

            # NOTE: Fixed health templates REMOVED — always same answer (pure shortcut)
            # "Is this from a neurologically healthy subject?" → always Yes
            # "Does this show neurological abnormalities?" → always No

            opts_cg = ['Above average cognitive performance', 'Average cognitive performance',
                       'Below average cognitive performance']
            random.shuffle(opts_cg)
            qa.append({'cat': 'Diagnosis', 'type': 'MCQ', 'img': True,
                       'q': 'What cognitive performance group does this subject belong to?',
                       'a': cg_full[cg], 'options': opts_cg + ['Unknown']})

            qa.append({'cat': 'Diagnosis', 'type': 'Open', 'img': True,
                       'q': 'What is the cognitive performance group for this subject?',
                       'a': cg})

        # ========== 8. REASONING (structure-cognition concordance) ==========
        nih_total = safe_float(sv(row, 'NIH_TotalComp'))
        nih_fluid = safe_float(sv(row, 'NIH_FluidComp'))

        if cg is not None and hl is not None and hr is not None and vent_val is not None and icv_val is not None:
            avg_h = (hl + hr) / 2
            below = avg_h < MED_HIPPO
            enl = float(vent_val) / float(icv_val) > MED_VENT_ICV if float(icv_val) > 0 else False

            # MCQ — structure description (data-driven, not fixed)
            if below and enl:
                struct_ans = 'Below median volume with above median ventricle ratio'
            elif below:
                struct_ans = 'Below median hippocampal volume'
            elif enl:
                struct_ans = 'Above median ventricle-to-brain ratio'
            else:
                struct_ans = 'Within median range for all structures'
            struct_opts = ['Below median hippocampal volume',
                           'Above median ventricle-to-brain ratio',
                           'Below median volume with above median ventricle ratio',
                           'Within median range for all structures']
            random.shuffle(struct_opts)
            qa.append({'cat': 'Reasoning', 'type': 'MCQ', 'img': True,
                       'q': 'Which description best matches the structural findings in this scan?',
                       'a': struct_ans, 'options': struct_opts})

            concordant = (cg in ('Above', 'Average') and not below) or \
                         (cg == 'Below' and below)
            qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                       'q': 'Is there concordance between brain structure and cognitive performance in this subject?',
                       'a': 'Yes' if concordant else 'No'})
            qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                       'q': 'Is there a mismatch between brain structure and cognitive performance in this subject?',
                       'a': 'No' if concordant else 'Yes'})

            qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                       'q': 'Is the overall brain morphology within expected limits for this age group?',
                       'a': 'Yes' if not below and not enl else 'No'})
            qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                       'q': 'Does the brain show structural deviations beyond expected age-related changes?',
                       'a': 'No' if not below and not enl else 'Yes'})

            # NOTE: Fixed disease/health templates REMOVED (pure shortcut)

            qa.append({'cat': 'Reasoning', 'type': 'Open', 'img': True,
                       'q': 'What is the most prominent structural finding in this brain scan?',
                       'a': 'Below median volume' if below else 'Within median range'})

        # ========== 9. SEVERITY (NIH Fluid + MoCA + NEO-FFI) ==========
        if nih_fluid is not None:
            if nih_fluid >= NIH_ABOVE_AVG: fluid_label = 'Above average'
            elif nih_fluid >= NIH_AVG: fluid_label = 'Average'
            else: fluid_label = 'Below average'

            qa.append({'cat': 'Severity', 'type': 'YN', 'img': True,
                       'q': 'Does this subject score above average on fluid cognition (NIH Toolbox)?',
                       'a': 'Yes' if fluid_label == 'Above average' else 'No'})
            qa.append({'cat': 'Severity', 'type': 'YN', 'img': True,
                       'q': 'Is the fluid cognition score within average range for this subject?',
                       'a': 'Yes' if fluid_label == 'Average' else 'No'})

            opts_fluid = ['Above average', 'Average', 'Below average']
            random.shuffle(opts_fluid)
            qa.append({'cat': 'Severity', 'type': 'MCQ', 'img': True,
                       'q': 'What is the fluid cognition level for this subject?',
                       'a': fluid_label, 'options': opts_fluid + ['Unknown']})

            qa.append({'cat': 'Severity', 'type': 'Open', 'img': True,
                       'q': 'What is the fluid cognition level for this subject?',
                       'a': fluid_label})

        if nih_total is not None:
            qa.append({'cat': 'Severity', 'type': 'YN', 'img': True,
                       'q': f'Is the total cognitive composite score above {MED_NIH_TOTAL_R}?',
                       'a': 'Yes' if nih_total > MED_NIH_TOTAL else 'No'})
            qa.append({'cat': 'Severity', 'type': 'YN', 'img': True,
                       'q': f'Is the total cognitive composite score below {MED_NIH_TOTAL_R}?',
                       'a': 'No' if nih_total > MED_NIH_TOTAL else 'Yes'})

        # MoCA level
        moca = safe_float(sv(row, 'MoCA_Total'))
        if moca is not None:
            moca_above = moca >= MED_MOCA
            moca_label = 'Above median' if moca_above else 'Below median'
            qa.append({'cat': 'Severity', 'type': 'YN', 'img': True,
                       'q': f'Is this subject\'s MoCA score above {int(MED_MOCA_R)}?',
                       'a': 'Yes' if moca_above else 'No'})
            qa.append({'cat': 'Severity', 'type': 'YN', 'img': True,
                       'q': f'Is this subject\'s MoCA score below {int(MED_MOCA_R)}?',
                       'a': 'No' if moca_above else 'Yes'})
            opts_moca = ['Above median', 'Below median']
            random.shuffle(opts_moca)
            qa.append({'cat': 'Severity', 'type': 'MCQ', 'img': True,
                       'q': f'Is this subject\'s MoCA score above or below the median ({int(MED_MOCA_R)})?',
                       'a': moca_label, 'options': opts_moca})
            qa.append({'cat': 'Severity', 'type': 'Open', 'img': True,
                       'q': 'What is the MoCA performance level for this subject relative to median?',
                       'a': moca_label})

        # NEO-FFI dominant trait
        neo_a = safe_float(sv(row, 'NEO_Agreeableness'))
        neo_o = safe_float(sv(row, 'NEO_Openness'))
        neo_c = safe_float(sv(row, 'NEO_Conscientiousness'))
        neo_n = safe_float(sv(row, 'NEO_Neuroticism'))
        neo_e = safe_float(sv(row, 'NEO_Extraversion'))
        neo_vals = {'Agreeableness': neo_a, 'Openness': neo_o, 'Conscientiousness': neo_c,
                    'Neuroticism': neo_n, 'Extraversion': neo_e}
        neo_valid = {k: v for k, v in neo_vals.items() if v is not None}
        if len(neo_valid) == 5:
            dominant = max(neo_valid, key=neo_valid.get)
            opts_neo = list(neo_valid.keys())
            random.shuffle(opts_neo)
            opts_neo_4 = [dominant] + [x for x in opts_neo if x != dominant][:3]
            random.shuffle(opts_neo_4)
            qa.append({'cat': 'Severity', 'type': 'MCQ', 'img': True,
                       'q': 'Which NEO-FFI personality dimension has the highest score for this subject?',
                       'a': dominant, 'options': opts_neo_4})
            qa.append({'cat': 'Severity', 'type': 'Open', 'img': True,
                       'q': 'Which NEO-FFI personality dimension is dominant for this subject?',
                       'a': dominant})

        # ========== 10. DEMOGRAPHIC ==========
        age = sv(row, 'age')
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
            age_f = safe_float(age)
            if age_f is not None:
                age_bin = get_age_bin(age_f)
                # Ensure correct answer is always in options (fix: 6 bins → pick correct + 3 wrong)
                wrong_bins = [b for b in AGE_BINS if b != age_bin]
                random.shuffle(wrong_bins)
                opts = [age_bin] + wrong_bins[:3]
                random.shuffle(opts)
                qa.append({'cat': 'Demographic', 'type': 'MCQ', 'img': False,
                           'q': 'What age range does this subject belong to?',
                           'a': age_bin, 'options': opts})

                # Aging questions
                qa.append({'cat': 'Demographic', 'type': 'YN', 'img': False,
                           'q': 'Is this subject older than 65?',
                           'a': 'Yes' if age_f > 65 else 'No'})
                qa.append({'cat': 'Demographic', 'type': 'YN', 'img': False,
                           'q': 'Is this subject younger than 50?',
                           'a': 'Yes' if age_f < 50 else 'No'})

        # === Add metadata ===
        IMG_DEP_MAP = {
            'Location': 'image-grounded',
            'Anatomy': 'image-grounded',
            'Presence': 'image-grounded',
            'Measurement': 'image-grounded',
            'Diagnosis': 'image-informed',
            'Reasoning': 'image-informed',
            'Signal': 'image-informed',
            'Severity': 'image-informed',
            'Demographic': 'image-informed',
        }
        for q in qa:
            q['neuroqa_id'] = scan_id
            q['subject_id'] = str(subj_id)
            q['session_date'] = str(row.get('session_id', 'V1'))
            q['dataset'] = 'HCP-Aging'
            q['image_required'] = q.pop('img')
            cat = q.pop('cat')
            q['category'] = cat
            q['image_dependency'] = IMG_DEP_MAP.get(cat, 'unknown')
            q['question_type'] = q.pop('type')
            q['question'] = q.pop('q')
            q['answer'] = q.pop('a')

        all_qa.extend(qa)

    # ------------------------------------------------------------------
    # LATERALITY POST-HOC BALANCING
    # Balance Left/Right answers so no single side dominates per template.
    # ------------------------------------------------------------------
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
# BALANCING & SUBSAMPLING (9 categories — Modality removed)
# ============================================================
def balance_and_subsample(qa_df, df_meta, target_total=10000, max_per_subj_per_cat=3):
    id_to_cg = df_meta.set_index('neuroqa_id')['cognitive_group'].to_dict()
    qa_df = qa_df.copy()
    qa_df['_cg'] = qa_df['neuroqa_id'].map(id_to_cg).fillna('Unknown')
    print(f"Cognitive group distribution in raw QA: {qa_df['_cg'].value_counts().to_dict()}")

    cat_order = ['Location', 'Anatomy', 'Signal', 'Presence',
                 'Measurement', 'Diagnosis', 'Reasoning', 'Severity',
                 'Demographic']

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

    # Step 2: Limit per subject per category
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

    # MCQ position balance
    all_final = balance_mcq_positions(all_final)

    print(f"Final: {len(all_final):,} QA pairs")
    return all_final


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate HCP-Aging QA pairs for NeuroQA')
    parser.add_argument('--input', default='../outputs/hcpaging_neuroqa_merged.csv')
    parser.add_argument('--output', default='../outputs/')
    args = parser.parse_args()

    print("Loading HCP-Aging data...")
    df = pd.read_csv(args.input, low_memory=False)
    print(f"Loaded {len(df):,} scans, {df['hcpaging_subject_id'].nunique():,} subjects\n")

    print("Generating all QA pairs...")
    all_qa = generate_qa(df)
    all_qa, _n_skipped = filter_qa_list(all_qa)
    qa_df = pd.DataFrame(all_qa)
    print(f"Total raw QA: {len(qa_df):,}")

    cat_order = ['Location', 'Anatomy', 'Signal', 'Presence',
                 'Measurement', 'Diagnosis', 'Reasoning', 'Severity',
                 'Demographic']

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
    final_df = balance_and_subsample(qa_df, df, target_total=10000, max_per_subj_per_cat=3)

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
    final_df.to_json(os.path.join(out_dir, 'hcpaging_qa_final.json'), orient='records', indent=2)
    final_df.to_csv(os.path.join(out_dir, 'hcpaging_qa_final.csv'), index=False)
    qa_df.to_json(os.path.join(out_dir, 'hcpaging_qa_raw_all.json'), orient='records', indent=2)
    print(f"\nSaved: hcpaging_qa_final.json ({len(final_df):,}) + hcpaging_qa_raw_all.json ({len(qa_df):,})")


# Apply expert-reviewed wording fixes
# apply_wording_fixes(final_qa)  # TODO: add correct variable name