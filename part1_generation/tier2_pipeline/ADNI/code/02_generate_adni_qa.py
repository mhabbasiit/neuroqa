#!/usr/bin/env python3
"""
NeuroQA — Step 2: ADNI Template-Based QA Generator

Generates question-answer pairs from ADNI structured metadata CSV.
All questions are template-based (no LLM calls). Anti-shortcut by design:
  - 50/50 YN balance via template negation
  - Zero metadata in question text (no age/DX/score leakage)
  - MCQ options shuffled per instance
  - DX-stratified subsampling (CN/MCI/Dementia/Unknown)
  - Two image-dependency tiers (grounded / informed)
  - Median-based thresholds computed from data (no fixed constants)
  - No Modality category (all T1 = pure shortcut)

PREREQUISITES:
  Run 01_build_adni_metadata.py first to produce adni_neuroqa_merged.csv.

USAGE:
  python 02_generate_adni_qa.py \\
    --input  ../outputs/adni_neuroqa_merged.csv \\
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
import sys; sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from skip_templates import filter_qa_list
from wording_fixes import apply_wording_fixes

random.seed(42)
np.random.seed(42)

# ============================================================
# THRESHOLDS — WMH only (structural thresholds computed from data)
# ============================================================
WMH_PRESENT = 1.0          # mm³
WMH_MODERATE = 5000
WMH_SEVERE = 15000

AGE_BINS = [(50, 65), (65, 75), (75, 85), (85, 105)]
AGE_LABELS = ["50-65", "65-75", "75-85", "85+"]

# ============================================================
# HELPERS
# ============================================================
def sv(row, col):
    """Safe value — returns None if NaN."""
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
# REGION MAPPINGS
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
# MEDIAN COMPUTATION — anti-shortcut: thresholds from data
# ============================================================
def compute_medians(df):
    """Compute median values from the dataset for use as thresholds."""
    medians = {}

    # Hippocampal volume — average of L+R
    hl = pd.to_numeric(df.get('FS_Vol_L_Hippocampus'), errors='coerce')
    hr = pd.to_numeric(df.get('FS_Vol_R_Hippocampus'), errors='coerce')
    avg_h = ((hl + hr) / 2).dropna()
    if len(avg_h) > 0:
        medians['hippo'] = int(round(avg_h.median() / 100) * 100)  # round to nearest 100
    else:
        medians['hippo'] = 3200  # fallback

    # Ventricle/ICV ratio
    vent = pd.to_numeric(df.get('FS_Ventricles'), errors='coerce')
    icv = pd.to_numeric(df.get('FS_ICV'), errors='coerce')
    valid = (vent.notna()) & (icv.notna()) & (icv > 0)
    if valid.sum() > 0:
        ratio = (vent[valid] / icv[valid])
        medians['vent_icv'] = round(ratio.median(), 4)
        medians['vent_icv_pct'] = round(ratio.median() * 100, 2)
    else:
        medians['vent_icv'] = 0.03
        medians['vent_icv_pct'] = 3.0

    # Entorhinal thickness — average of L+R
    el = pd.to_numeric(df.get('FS_Thick_L_Entorhinal'), errors='coerce')
    er = pd.to_numeric(df.get('FS_Thick_R_Entorhinal'), errors='coerce')
    avg_e = ((el + er) / 2).dropna()
    if len(avg_e) > 0:
        medians['entorhinal'] = round(avg_e.median(), 1)  # round to 0.1
    else:
        medians['entorhinal'] = 3.0  # fallback

    # Brain volume — use FS_WholeBrain if values are >100000, else TotalGM + WM or ICV
    wb = pd.to_numeric(df.get('FS_WholeBrain'), errors='coerce').dropna()
    if len(wb) > 0 and wb.median() > 100000:
        medians['brain'] = int(round(wb.median() / 10000) * 10000)  # round to nearest 10000
    else:
        tgm = pd.to_numeric(df.get('FS_Vol_TotalGM'), errors='coerce')
        twm = pd.to_numeric(df.get('FS_Vol_TotalWM'), errors='coerce') if 'FS_Vol_TotalWM' in df.columns else None
        if twm is not None:
            total = (tgm + twm).dropna()
        else:
            total = tgm.dropna()
        if len(total) > 0 and total.median() > 100000:
            medians['brain'] = int(round(total.median() / 10000) * 10000)
        elif len(icv.dropna()) > 0:
            medians['brain'] = int(round(icv.median() / 10000) * 10000)
        else:
            medians['brain'] = 1000000  # fallback

    return medians


# ============================================================
# QA GENERATOR — with balanced YN via template negation
# ============================================================
def generate_qa(df):
    all_qa = []

    # Compute medians from data (anti-shortcut: no fixed thresholds)
    MED = compute_medians(df)
    print(f"  Median hippocampal volume: {MED['hippo']} mm³")
    print(f"  Median vent/ICV ratio: {MED['vent_icv_pct']:.2f}%")
    print(f"  Median entorhinal thickness: {MED['entorhinal']:.1f} mm")
    print(f"  Median brain volume: {MED['brain']:,} mm³")

    MED_HIPPO = MED['hippo']
    MED_VENT_ICV = MED['vent_icv']
    MED_VENT_PCT = MED['vent_icv_pct']
    MED_ENTORH = MED['entorhinal']
    MED_BRAIN = MED['brain']

    # Pre-compute longitudinal
    subj_sessions = {}
    for subj, group in df.groupby('adni_subject_id'):
        subj_sessions[subj] = group.sort_values('session_date').index.tolist()

    for idx, row in df.iterrows():
        scan_id = row['neuroqa_id']
        subj_id = row['adni_subject_id']
        qa = []

        # ========== NO MODALITY — all T1, pure shortcut ==========

        # ========== 1. LOCATION ==========
        for rname, (lcol, rcol) in SUBCORT_PAIRS.items():
            lv, rv = sv(row, lcol), sv(row, rcol)
            if lv is None or rv is None:
                continue
            asym = abs(lv - rv) / max(lv, rv) * 100
            if asym < 5:
                continue
            smaller = 'left' if lv < rv else 'right'
            larger = 'right' if lv < rv else 'left'

            # YN pair — template negation for balance
            qa.append({'cat': 'Location', 'type': 'YN', 'img': True,
                       'q': f'Is the {smaller} {rname} smaller than the {larger} {rname} in this scan?',
                       'a': 'Yes'})
            qa.append({'cat': 'Location', 'type': 'YN', 'img': True,
                       'q': f'Is the {larger} {rname} smaller than the {smaller} {rname} in this scan?',
                       'a': 'No'})
            # Open
            qa.append({'cat': 'Location', 'type': 'Open', 'img': True,
                       'q': f'Which side has a smaller {rname} volume in this scan?',
                       'a': smaller.capitalize()})
            # MCQ — lobe (REMOVED: 100% same-answer anatomical knowledge shortcut;
            # the answer depends only on the region name, not the scan data)
            # for lobe, regions in LOBE_MAP.items():
            #     if rname in regions:
            #         others = [l for l in LOBE_MAP if l != lobe]
            #         opts = [lobe.capitalize()] + [l.capitalize() for l in random.sample(others, 3)]
            #         random.shuffle(opts)
            #         qa.append({'cat': 'Location', 'type': 'MCQ', 'img': True,
            #                    'q': f'In which brain region is the {rname} located?',
            #                    'a': lobe.capitalize(), 'options': opts})
            #         break

        # ========== 2. ANATOMY / MORPHOLOGY ==========
        hl, hr = sv(row, 'FS_Vol_L_Hippocampus'), sv(row, 'FS_Vol_R_Hippocampus')
        if hl is not None and hr is not None:
            avg_h = (hl + hr) / 2
            atrophy = avg_h < MED_HIPPO
            # YN pair — negated templates
            qa.append({'cat': 'Anatomy', 'type': 'YN', 'img': True,
                       'q': 'Is the hippocampal volume below normal in this scan?',
                       'a': 'Yes' if atrophy else 'No'})
            qa.append({'cat': 'Anatomy', 'type': 'YN', 'img': True,
                       'q': 'Is the hippocampal volume within normal range in this scan?',
                       'a': 'No' if atrophy else 'Yes'})
            # MCQ 1 — hippocampal morphology
            ans = 'Hippocampal atrophy' if atrophy else 'Normal hippocampal volume'
            opts = ['Hippocampal atrophy', 'Normal hippocampal volume',
                    'Hippocampal enlargement', 'Hippocampal calcification']
            random.shuffle(opts)
            qa.append({'cat': 'Anatomy', 'type': 'MCQ', 'img': True,
                       'q': 'What best describes the hippocampal morphology in this scan?',
                       'a': ans, 'options': opts})
            # MCQ 2 — most atrophied structure
            most_atr = 'Hippocampus' if avg_h < MED_HIPPO else 'None (all normal)'
            opts2 = ['Hippocampus', 'Caudate nucleus', 'Thalamus', 'None (all normal)']
            random.shuffle(opts2)
            qa.append({'cat': 'Anatomy', 'type': 'MCQ', 'img': True,
                       'q': 'Which subcortical structure shows the most volume reduction in this scan?',
                       'a': most_atr, 'options': opts2})
            # Open
            qa.append({'cat': 'Anatomy', 'type': 'Open', 'img': True,
                       'q': 'Is hippocampal atrophy present in this scan?',
                       'a': 'Yes' if atrophy else 'No'})

        # Ventricular
        vent, icv = sv(row, 'FS_Ventricles'), sv(row, 'FS_ICV')
        if vent is not None and icv is not None and icv > 0:
            ratio = vent / icv
            enlarged = ratio > MED_VENT_ICV
            # YN pair
            qa.append({'cat': 'Anatomy', 'type': 'YN', 'img': True,
                       'q': 'Are the ventricles enlarged relative to brain size in this scan?',
                       'a': 'Yes' if enlarged else 'No'})
            qa.append({'cat': 'Anatomy', 'type': 'YN', 'img': True,
                       'q': 'Are the ventricles normal in size relative to brain size in this scan?',
                       'a': 'No' if enlarged else 'Yes'})
            # Open
            qa.append({'cat': 'Anatomy', 'type': 'Open', 'img': True,
                       'q': 'Are the ventricles enlarged or normal in this scan?',
                       'a': 'Enlarged' if enlarged else 'Normal'})

        # Entorhinal thickness
        el, er = sv(row, 'FS_Thick_L_Entorhinal'), sv(row, 'FS_Thick_R_Entorhinal')
        if el is not None and er is not None:
            avg_e = (el + er) / 2
            thin = avg_e < MED_ENTORH
            # YN pair
            qa.append({'cat': 'Anatomy', 'type': 'YN', 'img': True,
                       'q': 'Is the entorhinal cortex thinner than normal in this scan?',
                       'a': 'Yes' if thin else 'No'})
            qa.append({'cat': 'Anatomy', 'type': 'YN', 'img': True,
                       'q': 'Is the entorhinal cortical thickness within normal range in this scan?',
                       'a': 'No' if thin else 'Yes'})

        # ========== 3. SIGNAL & IMAGING FEATURE (WMH only — no T1 contrast) ==========
        wmh = sv(row, 'WMH_Volume')
        wmh_fs = sv(row, 'FS_Vol_WMHypo')
        if wmh is not None or wmh_fs is not None:
            # Recovered clean rule (0-contradiction, 100% on frozen WM-severity):
            # WMH_Volume is on a small (mL) scale, FS_Vol_WMHypo on a large (mm^3) scale
            # — use whichever is present, each with its own thresholds.
            if wmh is not None:
                has_wmh = wmh >= 1.0
                sev = 'None' if wmh < 1.0 else 'Mild'
            else:
                has_wmh = wmh_fs >= 500
                sev = ('None' if wmh_fs < 500 else 'Mild' if wmh_fs < 5000
                       else 'Moderate' if wmh_fs < 12000 else 'Severe')
            # YN pair
            qa.append({'cat': 'Signal', 'type': 'YN', 'img': True,
                       'q': 'Does this scan show white matter hyperintensities?',
                       'a': 'Yes' if has_wmh else 'No'})
            qa.append({'cat': 'Signal', 'type': 'YN', 'img': True,
                       'q': 'Is the white matter signal homogeneous (no hyperintensities) in this scan?',
                       'a': 'No' if has_wmh else 'Yes'})
            # MCQ — severity
            opts = ['None', 'Mild', 'Moderate', 'Severe']
            random.shuffle(opts)
            qa.append({'cat': 'Signal', 'type': 'MCQ', 'img': True,
                       'q': 'What is the severity of white matter changes in this scan?',
                       'a': sev, 'options': opts})
            # MCQ 2 — WMH location (REMOVED: 100% "Periventricular" — anatomical
            # knowledge shortcut, answer does not depend on scan data)
            # if has_wmh:
            #     opts_loc = ['Periventricular', 'Deep white matter', 'Juxtacortical', 'Infratentorial']
            #     random.shuffle(opts_loc)
            #     qa.append({'cat': 'Signal', 'type': 'MCQ', 'img': True,
            #                'q': 'Where are white matter hyperintensities most commonly located in this type of scan?',
            #                'a': 'Periventricular', 'options': opts_loc})
            # Open
            qa.append({'cat': 'Signal', 'type': 'Open', 'img': True,
                       'q': 'Are white matter hyperintensities present in this scan?',
                       'a': 'Yes' if has_wmh else 'No'})
        if wmh is None and wmh_fs is not None:  # was an elif that could never run
            has = wmh_fs > 1000
            qa.append({'cat': 'Signal', 'type': 'YN', 'img': True,
                       'q': 'Does this scan show white matter signal abnormalities?',
                       'a': 'Yes' if has else 'No'})
            qa.append({'cat': 'Signal', 'type': 'YN', 'img': True,
                       'q': 'Is the white matter signal normal in this scan?',
                       'a': 'No' if has else 'Yes'})

        # ========== 4. PRESENCE / ABSENCE (data-driven per-subject findings) ==========
        if hl is not None and hr is not None:
            loss = min(hl, hr) < MED_HIPPO
            # YN pair
            qa.append({'cat': 'Presence', 'type': 'YN', 'img': True,
                       'q': 'Is hippocampal volume loss present in this scan?',
                       'a': 'Yes' if loss else 'No'})
            qa.append({'cat': 'Presence', 'type': 'YN', 'img': True,
                       'q': 'Are both hippocampi within normal volume range in this scan?',
                       'a': 'No' if loss else 'Yes'})

        if vent is not None and icv is not None and icv > 0:
            enl = vent / icv > MED_VENT_ICV
            qa.append({'cat': 'Presence', 'type': 'YN', 'img': True,
                       'q': 'Is ventricular enlargement present in this scan?',
                       'a': 'Yes' if enl else 'No'})
            qa.append({'cat': 'Presence', 'type': 'YN', 'img': True,
                       'q': 'Are the ventricles normal-sized in this scan?',
                       'a': 'No' if enl else 'Yes'})

            # MCQ — which finding present (data-driven per subject)
            findings = []
            absent = []
            if hl is not None and (hl + hr) / 2 < MED_HIPPO:
                findings.append('Hippocampal atrophy')
            else:
                absent.append('Hippocampal atrophy')
            if enl:
                findings.append('Ventricular enlargement')
            else:
                absent.append('Ventricular enlargement')
            if wmh is not None and wmh > WMH_PRESENT:
                findings.append('White matter changes')
            else:
                absent.append('White matter changes')

            if findings:
                ans = findings[0]
                distractors = absent + ['Tumor mass', 'Acute infarction', 'Cortical calcification']
                opts = [ans] + random.sample(distractors, min(3, len(distractors)))
                opts = opts[:4]
                random.shuffle(opts)
                qa.append({'cat': 'Presence', 'type': 'MCQ', 'img': True,
                           'q': 'Which of the following findings is present in this scan?',
                           'a': ans, 'options': opts})
            if absent:
                ans_absent = absent[0]
                present_list = findings + ['Tumor mass', 'Acute infarction', 'Cortical calcification', 'Hemorrhage']
                present_list = [x for x in present_list if x != ans_absent]
                opts_a = [ans_absent] + random.sample(present_list, min(3, len(present_list)))
                opts_a = opts_a[:4]
                random.shuffle(opts_a)
                qa.append({'cat': 'Presence', 'type': 'MCQ', 'img': True,
                           'q': 'Which of the following findings is NOT present in this scan?',
                           'a': ans_absent, 'options': opts_a})

        # (Removed: "Is the cerebellum visible?" — always Yes = pure shortcut)

        # ========== 5. MEASUREMENT / COUNT ==========
        # Open — volumes
        if hl is not None:
            qa.append({'cat': 'Measurement', 'type': 'Open', 'img': True,
                       'q': 'What is the approximate left hippocampal volume (mm³)?',
                       'a': str(round_vol(hl))})
        if hr is not None:
            qa.append({'cat': 'Measurement', 'type': 'Open', 'img': True,
                       'q': 'What is the approximate right hippocampal volume (mm³)?',
                       'a': str(round_vol(hr))})
        if vent is not None:
            qa.append({'cat': 'Measurement', 'type': 'Open', 'img': True,
                       'q': 'What is the approximate total ventricular volume (mm³)?',
                       'a': str(int(round(vent / 1000) * 1000))})
        tgm = sv(row, 'FS_Vol_TotalGM')
        if tgm is not None:
            qa.append({'cat': 'Measurement', 'type': 'Open', 'img': True,
                       'q': 'What is the approximate total gray matter volume (mm³)?',
                       'a': str(int(round(tgm / 1000) * 1000))})

        # YN — median-based threshold templates (anti-shortcut: thresholds from data)
        if hl is not None and hr is not None:
            avg_h = (hl + hr) / 2
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the average hippocampal volume below {MED_HIPPO:,} mm³ in this scan?',
                       'a': 'Yes' if avg_h < MED_HIPPO else 'No'})
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the average hippocampal volume above {MED_HIPPO:,} mm³ in this scan?',
                       'a': 'No' if avg_h < MED_HIPPO else 'Yes'})

        if vent is not None and icv is not None and icv > 0:
            ratio = vent / icv * 100
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the ventricle-to-brain ratio greater than {MED_VENT_PCT}% in this scan?',
                       'a': 'Yes' if ratio > MED_VENT_PCT else 'No'})
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the ventricle-to-brain ratio less than {MED_VENT_PCT}% in this scan?',
                       'a': 'No' if ratio > MED_VENT_PCT else 'Yes'})

        wb = sv(row, 'FS_WholeBrain')
        if wb is not None:
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the whole brain volume greater than {MED_BRAIN:,} mm³ in this scan?',
                       'a': 'Yes' if wb > MED_BRAIN else 'No'})
            qa.append({'cat': 'Measurement', 'type': 'YN', 'img': True,
                       'q': f'Is the whole brain volume less than {MED_BRAIN:,} mm³ in this scan?',
                       'a': 'No' if wb > MED_BRAIN else 'Yes'})

        # MCQ — ICV range
        if icv is not None:
            ranges = ['< 1,200,000', '1,200,000 – 1,400,000', '1,400,000 – 1,600,000', '> 1,600,000']
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
            h_ranges = ['< 2,500', '2,500 – 3,000', '3,000 – 3,500', '> 3,500']
            if avg_h < 2500: h_ans = h_ranges[0]
            elif avg_h < 3000: h_ans = h_ranges[1]
            elif avg_h < 3500: h_ans = h_ranges[2]
            else: h_ans = h_ranges[3]
            random.shuffle(h_ranges)
            qa.append({'cat': 'Measurement', 'type': 'MCQ', 'img': True,
                       'q': 'What range best describes the average hippocampal volume (mm³)?',
                       'a': h_ans, 'options': h_ranges})

        # MCQ — ventricular volume range
        if vent is not None:
            v_ranges = ['< 20,000', '20,000 – 40,000', '40,000 – 60,000', '> 60,000']
            if vent < 20000: v_ans = v_ranges[0]
            elif vent < 40000: v_ans = v_ranges[1]
            elif vent < 60000: v_ans = v_ranges[2]
            else: v_ans = v_ranges[3]
            random.shuffle(v_ranges)
            qa.append({'cat': 'Measurement', 'type': 'MCQ', 'img': True,
                       'q': 'What range best describes the total ventricular volume (mm³)?',
                       'a': v_ans, 'options': v_ranges})

        # Open — thickness (pick 1 random region)
        thick_regions = list(THICK_PAIRS.items())
        random.shuffle(thick_regions)
        for rname, (lcol, rcol) in thick_regions[:1]:
            lt = sv(row, lcol)
            if lt is not None:
                qa.append({'cat': 'Measurement', 'type': 'Open', 'img': True,
                           'q': f'What is the approximate left {rname} cortical thickness (mm)?',
                           'a': f'{lt:.1f}'})

        # ========== 6. DIAGNOSIS / CLASSIFICATION (varies: CN/MCI/Dementia) ==========
        dx = sv(row, 'DX')
        if dx is not None:
            dx_map = {'CN': 'Cognitively Normal', 'MCI': 'Mild Cognitive Impairment', 'Dementia': 'Dementia'}
            dx_full = dx_map.get(dx, dx)

            # YN pairs — balanced by template negation (answer varies per subject)
            qa.append({'cat': 'Diagnosis', 'type': 'YN', 'img': True,
                       'q': 'Is this brain scan from a cognitively normal subject?',
                       'a': 'Yes' if dx == 'CN' else 'No'})
            qa.append({'cat': 'Diagnosis', 'type': 'YN', 'img': True,
                       'q': 'Does this brain scan show signs of cognitive impairment?',
                       'a': 'No' if dx == 'CN' else 'Yes'})
            qa.append({'cat': 'Diagnosis', 'type': 'YN', 'img': True,
                       'q': 'Does this brain scan show features consistent with dementia?',
                       'a': 'Yes' if dx == 'Dementia' else 'No'})
            qa.append({'cat': 'Diagnosis', 'type': 'YN', 'img': True,
                       'q': 'Is this brain scan inconsistent with a dementia diagnosis?',
                       'a': 'No' if dx == 'Dementia' else 'Yes'})

            # MCQ 1 — classification
            opts = ['Cognitively Normal', 'Mild Cognitive Impairment',
                    'Alzheimer\'s Dementia', 'Frontotemporal Dementia']
            random.shuffle(opts)
            qa.append({'cat': 'Diagnosis', 'type': 'MCQ', 'img': True,
                       'q': 'What is the most likely clinical classification based on this brain scan?',
                       'a': dx_full if dx != 'Dementia' else "Alzheimer's Dementia",
                       'options': opts})
            # MCQ 2 — disease type (data-driven: answer varies by DX group)
            if dx == 'Dementia':
                opts2 = ['Neurodegenerative disease', 'Vascular disease', 'Traumatic injury', 'Normal aging']
                random.shuffle(opts2)
                qa.append({'cat': 'Diagnosis', 'type': 'MCQ', 'img': True,
                           'q': 'What type of condition is most consistent with the findings in this scan?',
                           'a': 'Neurodegenerative disease', 'options': opts2})
            elif dx == 'CN':
                opts2 = ['Normal aging', 'Neurodegenerative disease', 'Vascular disease', 'Traumatic injury']
                random.shuffle(opts2)
                qa.append({'cat': 'Diagnosis', 'type': 'MCQ', 'img': True,
                           'q': 'What type of condition is most consistent with the findings in this scan?',
                           'a': 'Normal aging', 'options': opts2})
            # Open
            qa.append({'cat': 'Diagnosis', 'type': 'Open', 'img': True,
                       'q': 'What is the clinical diagnosis group for this subject?',
                       'a': dx})

        # ========== 7. DIAGNOSTIC REASONING (data-driven concordance only) ==========
        mmse = sv(row, 'MMSE')
        cdr = sv(row, 'CDR_SOB')
        if dx is not None and hl is not None and hr is not None and vent is not None and icv is not None:
            avg_h = (hl + hr) / 2
            atrophy = avg_h < MED_HIPPO
            enl = vent / icv > MED_VENT_ICV if icv > 0 else False

            # (Removed fixed per-DX MCQ — answer was deterministic per DX group)

            # YN pairs — data-driven concordance (both Yes and No cases)
            if mmse is not None:
                low_mmse = mmse < 24
                # Template 1: Does scan suggest neurodegeneration?
                if low_mmse and atrophy:
                    qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                               'q': 'Does the brain structure in this scan suggest neurodegenerative changes?',
                               'a': 'Yes'})
                elif not low_mmse and not atrophy:
                    qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                               'q': 'Does the brain structure in this scan suggest neurodegenerative changes?',
                               'a': 'No'})
                # Template 2: Is brain morphology within normal limits?
                if not low_mmse and not atrophy:
                    qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                               'q': 'Is the overall brain morphology within normal limits in this scan?',
                               'a': 'Yes'})
                elif low_mmse and atrophy:
                    qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                               'q': 'Is the overall brain morphology within normal limits in this scan?',
                               'a': 'No'})

                # Template 3: Does the structural pattern suggest impairment?
                qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                           'q': 'Does the structural pattern suggest cognitive impairment in this subject?',
                           'a': 'Yes' if (atrophy and low_mmse) else 'No'})

                # Template 4: Mismatch between structure and DX (data-driven)
                if dx == 'CN' and atrophy:
                    qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                               'q': 'Is there a mismatch between the hippocampal volume and the overall brain appearance in this scan?',
                               'a': 'Yes'})
                elif dx == 'Dementia' and not atrophy:
                    qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                               'q': 'Is there a mismatch between the hippocampal volume and the overall brain appearance in this scan?',
                               'a': 'Yes'})
                elif dx == 'CN' and not atrophy:
                    qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                               'q': 'Is there a mismatch between the hippocampal volume and the overall brain appearance in this scan?',
                               'a': 'No'})
                elif dx == 'Dementia' and atrophy:
                    qa.append({'cat': 'Reasoning', 'type': 'YN', 'img': True,
                               'q': 'Is there a mismatch between the hippocampal volume and the overall brain appearance in this scan?',
                               'a': 'No'})

            # MCQ — data-driven: structural concordance with DX (answer varies per subject)
            if atrophy and enl:
                conc_ans = 'Hippocampal atrophy with ventricular enlargement'
            elif atrophy and not enl:
                conc_ans = 'Hippocampal atrophy without ventricular enlargement'
            elif not atrophy and enl:
                conc_ans = 'Ventricular enlargement without hippocampal atrophy'
            else:
                conc_ans = 'No significant structural abnormality'
            conc_opts = ['Hippocampal atrophy with ventricular enlargement',
                         'Hippocampal atrophy without ventricular enlargement',
                         'Ventricular enlargement without hippocampal atrophy',
                         'No significant structural abnormality']
            random.shuffle(conc_opts)
            qa.append({'cat': 'Reasoning', 'type': 'MCQ', 'img': True,
                       'q': 'Which structural pattern best describes the findings in this scan?',
                       'a': conc_ans, 'options': conc_opts})

            # Open — data-driven
            qa.append({'cat': 'Reasoning', 'type': 'Open', 'img': True,
                       'q': 'What is the most prominent structural finding in this brain scan?',
                       'a': 'Atrophy' if atrophy else 'Normal'})

        # ========== 8. SEVERITY / STAGING ==========
        if cdr is not None:
            # YN pairs
            severe = cdr > 4
            qa.append({'cat': 'Severity', 'type': 'YN', 'img': True,
                       'q': 'Does this scan show features of severe cognitive impairment (CDR-SOB > 4)?',
                       'a': 'Yes' if severe else 'No'})
            qa.append({'cat': 'Severity', 'type': 'YN', 'img': True,
                       'q': 'Is this scan consistent with no or mild cognitive impairment (CDR-SOB ≤ 4)?',
                       'a': 'No' if severe else 'Yes'})

            mild = cdr <= 1 and cdr > 0
            qa.append({'cat': 'Severity', 'type': 'YN', 'img': True,
                       'q': 'Is the cognitive impairment at a mild stage (CDR-SOB 0.5-1) in this subject?',
                       'a': 'Yes' if mild else 'No'})
            qa.append({'cat': 'Severity', 'type': 'YN', 'img': True,
                       'q': 'Is the cognitive impairment beyond a mild stage in this subject?',
                       'a': 'No' if mild else 'Yes'})

            # MCQ 1 — severity level
            if cdr == 0: sev = 'No impairment'
            elif cdr <= 1: sev = 'Very mild'
            elif cdr <= 4: sev = 'Mild'
            else: sev = 'Moderate to severe'
            opts = ['No impairment', 'Very mild', 'Mild', 'Moderate to severe']
            random.shuffle(opts)
            qa.append({'cat': 'Severity', 'type': 'MCQ', 'img': True,
                       'q': 'What severity level is most consistent with this brain scan?',
                       'a': sev, 'options': opts})
            # MCQ 2 — CDR range
            if cdr == 0: cdr_label = 'CDR 0'
            elif cdr <= 0.5: cdr_label = 'CDR 0.5'
            elif cdr <= 1: cdr_label = 'CDR 0.5-1'
            elif cdr <= 4.5: cdr_label = 'CDR 1-4.5'
            else: cdr_label = 'CDR > 4.5'
            opts2 = ['CDR 0', 'CDR 0.5', 'CDR 0.5-1', 'CDR 1-4.5', 'CDR > 4.5']
            # Keep only 4
            if cdr_label in opts2:
                opts2_filtered = [cdr_label] + [o for o in opts2 if o != cdr_label][:3]
            else:
                opts2_filtered = opts2[:4]
            random.shuffle(opts2_filtered)
            qa.append({'cat': 'Severity', 'type': 'MCQ', 'img': True,
                       'q': 'Which CDR-SOB range is most consistent with this brain scan?',
                       'a': cdr_label, 'options': opts2_filtered})

        if mmse is not None:
            if mmse >= 27: mc = 'Normal'
            elif mmse >= 21: mc = 'Mild'
            elif mmse >= 10: mc = 'Moderate'
            else: mc = 'Severe'
            qa.append({'cat': 'Severity', 'type': 'Open', 'img': True,
                       'q': 'What cognitive severity level is most consistent with this scan?',
                       'a': mc})

        # ========== 9. LONGITUDINAL / PROGRESSION ==========
        sessions = subj_sessions.get(subj_id, [])
        if len(sessions) > 1:
            curr_pos = sessions.index(idx) if idx in sessions else -1
            if curr_pos > 0:
                prev = df.loc[sessions[curr_pos - 1]]
                # Hippocampal change
                ph = sv(prev, 'FS_Vol_L_Hippocampus')
                if hl is not None and ph is not None and ph > 0:
                    dec = hl < ph
                    pct = (hl - ph) / ph * 100
                    # YN pair
                    qa.append({'cat': 'Longitudinal', 'type': 'YN', 'img': True,
                               'q': 'Has the left hippocampal volume decreased compared to the previous scan?',
                               'a': 'Yes' if dec else 'No'})
                    qa.append({'cat': 'Longitudinal', 'type': 'YN', 'img': True,
                               'q': 'Has the left hippocampal volume remained stable or increased compared to the previous scan?',
                               'a': 'No' if dec else 'Yes'})
                    if abs(pct) > 1:
                        qa.append({'cat': 'Longitudinal', 'type': 'Open', 'img': True,
                                   'q': 'What is the approximate percentage change in left hippocampal volume?',
                                   'a': f'{pct:.0f}%'})

                # Ventricular change
                pv = sv(prev, 'FS_Ventricles')
                cv = sv(row, 'FS_Ventricles')
                if cv is not None and pv is not None:
                    inc = cv > pv
                    qa.append({'cat': 'Longitudinal', 'type': 'YN', 'img': True,
                               'q': 'Have the ventricles enlarged compared to the previous scan?',
                               'a': 'Yes' if inc else 'No'})
                    qa.append({'cat': 'Longitudinal', 'type': 'YN', 'img': True,
                               'q': 'Have the ventricles remained stable or decreased compared to the previous scan?',
                               'a': 'No' if inc else 'Yes'})

                # DX progression
                pdx = sv(prev, 'DX')
                cdx = sv(row, 'DX')
                if pdx is not None and cdx is not None:
                    dx_ord = {'CN': 0, 'MCI': 1, 'Dementia': 2}
                    if pdx in dx_ord and cdx in dx_ord:
                        if dx_ord[cdx] > dx_ord[pdx]: prog = 'Progressed'
                        elif dx_ord[cdx] < dx_ord[pdx]: prog = 'Improved'
                        else: prog = 'Stable'
                        opts = ['Progressed', 'Stable', 'Improved', 'Undetermined']
                        random.shuffle(opts)
                        # NOTE: "Stable" dominates (~93%) — this reflects actual ADNI
                        # cohort composition (most subjects remain at same DX between
                        # consecutive visits). Not fixable at template level without
                        # misrepresenting the data.
                        qa.append({'cat': 'Longitudinal', 'type': 'MCQ', 'img': True,
                                   'q': 'Has the clinical status progressed, remained stable, or improved compared to the previous visit?',
                                   'a': prog, 'options': opts})
                        qa.append({'cat': 'Longitudinal', 'type': 'Open', 'img': True,
                                   'q': 'What is the clinical progression status compared to the previous visit?',
                                   'a': prog})

        # ========== 10. DEMOGRAPHIC (age + sex only — image-informed) ==========
        age = sv(row, 'age')
        sex = sv(row, 'sex')

        if sex is not None:
            # YN pair
            qa.append({'cat': 'Demographic', 'type': 'YN', 'img': False,
                       'q': 'Is this subject male?',
                       'a': 'Yes' if sex == 'Male' else 'No'})
            qa.append({'cat': 'Demographic', 'type': 'YN', 'img': False,
                       'q': 'Is this subject female?',
                       'a': 'No' if sex == 'Male' else 'Yes'})
            qa.append({'cat': 'Demographic', 'type': 'Open', 'img': False,
                       'q': 'What is the sex of this subject?', 'a': sex})

        if age is not None:
            # YN pair
            qa.append({'cat': 'Demographic', 'type': 'YN', 'img': False,
                       'q': 'Is this subject older than 75 years?',
                       'a': 'Yes' if age > 75 else 'No'})
            qa.append({'cat': 'Demographic', 'type': 'YN', 'img': False,
                       'q': 'Is this subject younger than 75 years?',
                       'a': 'No' if age > 75 else 'Yes'})
            # MCQ
            age_ans = None
            for i, (lo, hi) in enumerate(AGE_BINS):
                if lo <= age < hi or (i == len(AGE_BINS) - 1 and age >= lo):
                    age_ans = AGE_LABELS[i]
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
            q['subject_id'] = subj_id
            q['dataset'] = 'ADNI'
            q['image_required'] = q.pop('img')
            cat = q.pop('cat')
            q['category'] = cat
            q['image_dependency'] = IMG_DEP_MAP.get(cat, 'unknown')
            q['question_type'] = q.pop('type')
            q['question'] = q.pop('q')
            q['answer'] = q.pop('a')

        all_qa.extend(qa)

    # ========== POST-HOC LATERALITY BALANCING ==========
    # For laterality questions (e.g. "Which side has a smaller X volume?"),
    # balance Left/Right answers per unique question template so that no
    # side dominates. Group by question text, find minority answer count,
    # remove templates with minority < 3, downsample majority to match minority.
    laterality_indices = defaultdict(lambda: defaultdict(list))
    for i, q in enumerate(all_qa):
        qtext = q['question']
        ans = q['answer']
        # Laterality YN: questions containing "smaller than"
        if 'smaller than' in qtext:
            laterality_indices[qtext][ans].append(i)
        # Laterality Open: answer is "Left" or "Right"
        elif q['question_type'] == 'Open' and ans in ('Left', 'Right'):
            laterality_indices[qtext][ans].append(i)

    remove_indices = set()
    rng_lat = random.Random(42)
    for qtext, ans_groups in laterality_indices.items():
        if len(ans_groups) < 2:
            # Only one answer value — skip (nothing to balance)
            continue
        counts = {a: len(idxs) for a, idxs in ans_groups.items()}
        min_count = min(counts.values())
        if min_count < 3:
            # Too few minority examples — remove this template entirely
            for idxs in ans_groups.values():
                remove_indices.update(idxs)
        else:
            # Downsample majority to match minority
            for ans_val, idxs in ans_groups.items():
                if len(idxs) > min_count:
                    keep = set(rng_lat.sample(idxs, min_count))
                    remove_indices.update(set(idxs) - keep)

    if remove_indices:
        n_before = len(all_qa)
        all_qa = [q for i, q in enumerate(all_qa) if i not in remove_indices]
        print(f"  Laterality balancing: removed {n_before - len(all_qa)} QA pairs "
              f"({n_before} → {len(all_qa)})")

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
def balance_and_subsample(qa_df, df_meta, target_total=8000, max_per_subj_per_cat=3):
    """
    1. Keep ALL subjects (including Unknown DX — important for anti-shortcut)
    2. Balance Yes/No 50/50 per category via subsampling min(Yes,No)
    3. Max N questions per subject per category
    4. Re-balance YN after limiting
    5. Subsample to target, stratified by category and DX (CN/MCI/Dementia/Unknown)
    """
    # Add DX info to qa_df for stratification
    id_to_dx = df_meta.set_index('neuroqa_id')['DX'].to_dict()
    qa_df = qa_df.copy()
    qa_df['_dx'] = qa_df['neuroqa_id'].map(id_to_dx)
    # Fill missing DX as 'Unknown' — do NOT drop them
    qa_df['_dx'] = qa_df['_dx'].fillna('Unknown')
    print(f"DX distribution in raw QA: {qa_df['_dx'].value_counts().to_dict()}")

    balanced = []
    cat_order = ['Location', 'Anatomy', 'Signal', 'Presence',
                 'Measurement', 'Diagnosis', 'Reasoning', 'Severity',
                 'Longitudinal', 'Demographic']

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

        # Non-YN
        non_yn = sub[sub['question_type'] != 'YN']

        cat_balanced = pd.concat([yn_balanced, non_yn])
        balanced.append(cat_balanced)

    all_balanced = pd.concat(balanced, ignore_index=True)
    print(f"After YN balancing: {len(all_balanced):,} QA pairs")

    # Step 2: Limit per subject per category
    limited = []
    for (subj, cat), group in all_balanced.groupby(['subject_id', 'category']):
        if len(group) > max_per_subj_per_cat:
            # Keep 1 of each type, then random fill
            sampled = []
            for qtype in ['YN', 'MCQ', 'Open']:
                type_q = group[group['question_type'] == qtype]
                if len(type_q) > 0:
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
    # Add _dx if not present
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

            # Stratify by DX within each category
            dx_groups = sub.groupby('_dx', dropna=False)
            dx_counts = dx_groups.size()
            # Equal allocation per DX group
            n_dx = len(dx_counts)
            per_dx = per_cat // max(n_dx, 1)

            dx_sampled = []
            for dx_val, dx_sub in dx_groups:
                n_take = min(per_dx, len(dx_sub))
                if n_take == 0:
                    continue
                # Within DX group, maintain type proportions + YN balance
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
    parser = argparse.ArgumentParser(description='Generate ADNI QA pairs for NeuroQA')
    parser.add_argument('--input', default='../outputs/adni_neuroqa_merged.csv',
                        help='Path to ADNI merged metadata CSV (from step 1)')
    parser.add_argument('--output', default='../outputs/',
                        help='Output directory for QA files')
    args = parser.parse_args()

    print("Loading ADNI data...")
    df = pd.read_csv(args.input, low_memory=False)
    df['session_date'] = pd.to_datetime(df['session_date'], errors='coerce')
    df = df.sort_values(['adni_subject_id', 'session_date']).reset_index(drop=True)
    print(f"Loaded {len(df):,} scans, {df['adni_subject_id'].nunique():,} subjects\n")

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
        n_pct = 100 - y_pct
        print(f"{cat:<25} {len(yn):>8,} {mcq:>8,} {opn:>8,} {len(s):>8,}  {y_pct:>4.0f}% {n_pct:>4.0f}%")

    # Balance & subsample
    print("\n" + "="*80)
    print("BALANCING & SUBSAMPLING")
    print("="*80)
    final_df = balance_and_subsample(qa_df, df, target_total=8000, max_per_subj_per_cat=3)

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
    # Get DX from merged df
    id_to_dx = df.set_index('neuroqa_id')['DX'].to_dict()
    final_df['_dx'] = final_df['neuroqa_id'].map(id_to_dx)
    dx_dist = final_df['_dx'].value_counts(dropna=False)
    for k, v in dx_dist.items():
        label = k if pd.notna(k) else 'Missing'
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
                print(f"  [{qtype}] Q: {s['question']}  →  A: {s['answer']}")

    # === TEMPLATE COUNT ===
    print("\n" + "="*80)
    print("UNIQUE QUESTION TEMPLATES")
    print("="*80)
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
    _recs = final_df.to_dict(orient='records')
    apply_wording_fixes(_recs)  # apply expert-reviewed wording fixes (in-place)
    final_df = pd.DataFrame(_recs)
    final_df.to_json(os.path.join(out_dir, 'adni_qa_final.json'), orient='records', indent=2)
    final_df.to_csv(os.path.join(out_dir, 'adni_qa_final.csv'), index=False)
    # Also save full raw
    qa_df.to_json(os.path.join(out_dir, 'adni_qa_raw_all.json'), orient='records', indent=2)
    print(f"\nSaved to {out_dir}/: adni_qa_final.json ({len(final_df):,} QA) + adni_qa_raw_all.json ({len(qa_df):,} QA)")


# Apply expert-reviewed wording fixes
# apply_wording_fixes(final_qa)  # TODO: add correct variable name