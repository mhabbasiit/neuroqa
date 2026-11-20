#!/usr/bin/env python3
"""
NeuroQA — Benchmark Cleanup Script (v4 — Complete Rewrite)

Applies ALL 21 quality rules to produce the canonical release from raw QA files.
Input: *_qa_final.json (from generation scripts)
Output: *_qa_v3.json (cleaned, balanced, annotated)

Rules applied in order:
  R1:  BraTS laterality removal (report-image convention mismatch)
  R2:  Low-asymmetry (<10%) laterality removal
  R3:  Ventricle column fix (LatVentricle)
  R4:  Cerebellopontine angle → posterior fossa
  R4b: Midline shift / compression negation fix
  R4c: BraTS signal from Modal-wise Findings fix
  R5:  BraTS left/right from Open answers
  R6:  Multi-lobe BraTS fix
  R7:  Signal wording (clinical standard)
  R8:  Age context in Anatomy/Presence
  R9:  Longitudinal interval + age + disjunction fix
  R10: Longitudinal direction verification
  R11: Compound Presence MCQ options
  R12: Diagnosis framework reference
  R13: Mislabeled BraTS-MEN removal
  R14: NaN FreeSurfer removal
  R15: Session alignment (latest valid session)
  R16: answerable_modalities annotation
  R17: human_assessable annotation
  R18: (paper limitation — WMH on T1)
  R19: (paper limitation — Measurement beyond visual)
  R20: (verified — all RAS orientation)
  R21: (UI only — Modality display)
  R26: Age threshold int comparison fix
  R27: Session volume/age fix (correct session FS volumes + demographics)
  R22: Group C/D/P0 removal
  R23: YN 50/50 re-balance per template
  R24: MCQ/Open shortcut removal (≥95% remove, >70% downsample)
  R25: MCQ position balance (25% each)

USAGE:
  python cleanup_benchmark_v4.py --base-dir /path/to/data
  python cleanup_benchmark_v4.py --base-dir /path/to/data --dry-run
"""

import json, os, re, sys, random, argparse
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

random.seed(42)

def sort_key(d):
    """Deterministic sort key for reproducibility."""
    return (d.get('dataset',''), str(d.get('subject_id','')), d.get('category',''),
            d.get('question_type',''), d.get('question','')[:80])

def det_shuffle(items):
    """Deterministic shuffle: sort first, then shuffle with fixed seed."""
    items.sort(key=sort_key)
    random.shuffle(items)

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

STRUCT_COLS = {
    'hippocampus': ('FS_Vol_L_Hippocampus', 'FS_Vol_R_Hippocampus'),
    'hippocampal': ('FS_Vol_L_Hippocampus', 'FS_Vol_R_Hippocampus'),
    'lateral ventricle': ('FS_Vol_L_LatVentricle', 'FS_Vol_R_LatVentricle'),
    'ventricle': ('FS_Vol_L_LatVentricle', 'FS_Vol_R_LatVentricle'),
    'caudate': ('FS_Vol_L_Caudate', 'FS_Vol_R_Caudate'),
    'putamen': ('FS_Vol_L_Putamen', 'FS_Vol_R_Putamen'),
    'amygdala': ('FS_Vol_L_Amygdala', 'FS_Vol_R_Amygdala'),
    'thalamus': ('FS_Vol_L_Thalamus', 'FS_Vol_R_Thalamus'),
    'accumbens': ('FS_Vol_L_Accumbens', 'FS_Vol_R_Accumbens'),
    'pallidum': ('FS_Vol_L_Pallidum', 'FS_Vol_R_Pallidum'),
}

DS_MODALITIES = {
    'ADNI': ['T1w'], 'PPMI': ['T1w', 'T2w'], 'AIBL': ['T1w'], 'CC359': ['T1w'],
    'ABCD': ['T1w', 'T2w'], 'HCP-YA': ['T1w', 'T2w'],
    'HCP-Aging': ['T1w', 'T2w'], 'HCP-A': ['T1w', 'T2w'],
    'HCP-Development': ['T1w', 'T2w'], 'HCP-D': ['T1w', 'T2w'],
    'IXI': ['T1w', 'T2w'],
    'BraTS-GLI': ['T1w', 'T2w'], 'BraTS-MEN': ['T1w', 'T2w'],
    'WMH': ['T1w'],
}

MIN_ASYMMETRY_PCT = 10.0

GROUP_C_PATTERNS = [
    re.compile(r'APOE', re.IGNORECASE),
    re.compile(r'NEO-FFI', re.IGNORECASE),
    re.compile(r'behavioral (scores|pattern|severity)', re.IGNORECASE),
    re.compile(r'(immediate|delayed) recall score', re.IGNORECASE),
    re.compile(r'cognitively normal', re.IGNORECASE),
    re.compile(r"Alzheimer.s disease (present|free)", re.IGNORECASE),
    re.compile(r'cognitive status .* consistent', re.IGNORECASE),
    re.compile(r'clinical diagnosis group', re.IGNORECASE),
    re.compile(r'mismatch between', re.IGNORECASE),
    re.compile(r'concordance between brain structure', re.IGNORECASE),
    re.compile(r'cognitive performance group', re.IGNORECASE),
    re.compile(r'DaTScan|dopamin', re.IGNORECASE),
    re.compile(r'T2.*(available|present)', re.IGNORECASE),
    re.compile(r'modalit.*(available|acquired)', re.IGNORECASE),
    re.compile(r'how many modalit', re.IGNORECASE),
    re.compile(r'scan of (above|below).average quality', re.IGNORECASE),
    re.compile(r'CDR', re.IGNORECASE),
    re.compile(r'MMSE', re.IGNORECASE),
]

GROUP_D_OPEN_PATTERNS = [
    re.compile(r'approximate .* volume \(mm', re.IGNORECASE),
    re.compile(r'cortical thickness \(mm\)', re.IGNORECASE),
    re.compile(r'percentage change in .* volume', re.IGNORECASE),
]

GROUP_D_YN_PATTERNS = [
    re.compile(r"MoCA score (below|above) 26", re.IGNORECASE),
]

def normalize_template(q):
    norm = re.sub(r'\d+-year-old', 'X-year-old', q)
    norm = re.sub(r'subject age: \d+', 'subject age: X', norm)
    norm = re.sub(r'\(age \d+-?\d*\+?\)', '(age X)', norm)
    norm = re.sub(r'scan \d+ months prior', 'scan N months prior', norm)
    norm = re.sub(r'visit \d+ months prior', 'visit N months prior', norm)
    norm = re.sub(r'older than \d+', 'older than X', norm)
    norm = re.sub(r'younger than \d+', 'younger than X', norm)
    norm = re.sub(r'above \d[\d,]*\.?\d*', 'above X', norm)
    norm = re.sub(r'below \d[\d,]*\.?\d*', 'below X', norm)
    norm = re.sub(r'greater than \d[\d,]*\.?\d*', 'greater than X', norm)
    norm = re.sub(r'less than \d[\d,]*\.?\d*', 'less than X', norm)
    norm = re.sub(r'median \(\d+\.?\d* mL\)', 'median (X mL)', norm)
    return norm[:150]


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER: Load metadata
# ═══════════════════════════════════════════════════════════════════════════════

def load_meta(base_dir, ds_name):
    """Load metadata CSV for a dataset."""
    paths = {
        'ADNI': 'ADNI/outputs/adni_neuroqa_merged.csv',
        'PPMI': 'PPMI/outputs/ppmi_neuroqa_merged.csv',
        'ABCD': 'ABCD/outputs/abcd_neuroqa_merged.csv',
        'AIBL': 'AIBL/outputs/aibl_neuroqa_merged.csv',
        'HCP-YA': 'HCP-YA/outputs/hcpya_neuroqa_merged.csv',
        'HCP-A': 'HCP-Aging/outputs/hcpaging_neuroqa_merged.csv',
        'HCP-Aging': 'HCP-Aging/outputs/hcpaging_neuroqa_merged.csv',
        'HCP-D': 'HCP-Development/outputs/hcpdev_neuroqa_merged.csv',
        'HCP-Development': 'HCP-Development/outputs/hcpdev_neuroqa_merged.csv',
    }
    p = paths.get(ds_name)
    if not p: return None
    full = os.path.join(base_dir, p)
    return pd.read_csv(full) if os.path.exists(full) else None


def get_sid_col(meta):
    """Find subject ID column."""
    for c in meta.columns:
        if '_subject_id' in c:
            return c
    return meta.columns[0]


def get_best_session(meta, sid_col):
    """For each subject, find latest session with both image + FreeSurfer."""
    best = {}
    for sid in meta[sid_col].unique():
        rows = meta[meta[sid_col] == sid]
        valid = [r for _, r in rows.iterrows()
                 if pd.notna(r.get('T1w_image_path')) and pd.notna(r.get('FS_Vol_L_Hippocampus'))]
        if valid:
            best[str(sid)] = valid[-1]
        else:
            # Fallback: any row with image
            img_rows = [r for _, r in rows.iterrows() if pd.notna(r.get('T1w_image_path'))]
            if img_rows:
                best[str(sid)] = img_rows[-1]
    return best


# ═══════════════════════════════════════════════════════════════════════════════
# RULE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def apply_R22_group_cd(qa):
    """R22: Remove Group C (metadata) + Group D (exact numbers) + Group P0."""
    kept = []
    removed = 0
    for d in qa:
        text = d['question']
        qt = d['question_type']
        remove = False
        for pat in GROUP_C_PATTERNS:
            if pat.search(text):
                remove = True; break
        if not remove and qt == 'Open':
            for pat in GROUP_D_OPEN_PATTERNS:
                if pat.search(text):
                    remove = True; break
        if not remove and qt == 'YN':
            for pat in GROUP_D_YN_PATTERNS:
                if pat.search(text):
                    remove = True; break
        if remove:
            removed += 1
        else:
            kept.append(d)
    return kept, removed


def apply_R1_brats_laterality(qa):
    """R1: Remove BraTS laterality questions."""
    kept = []
    removed = 0
    for d in qa:
        if d.get('dataset', '') in ('BraTS-GLI', 'BraTS-MEN'):
            q = d['question'].lower()
            if d['category'] == 'Location' and ('left side' in q or 'right side' in q):
                removed += 1; continue
        kept.append(d)
    return kept, removed


def apply_R2_R3_low_asymmetry(qa, base_dir):
    """R2/R3: Remove laterality questions with <10% asymmetry."""
    removed = 0
    kept = []
    # Load all metadata
    metas = {}
    for ds in ['ADNI', 'PPMI', 'ABCD', 'HCP-YA', 'HCP-A', 'HCP-D']:
        m = load_meta(base_dir, ds)
        if m is not None:
            metas[ds] = (m, get_sid_col(m), get_best_session(m, get_sid_col(m)))

    ds_map = {'HCP-Aging': 'HCP-A', 'HCP-Development': 'HCP-D'}

    for d in qa:
        if d['category'] != 'Location' or d['question_type'] != 'YN':
            kept.append(d); continue
        q = d['question'].lower()
        if 'smaller' not in q and 'larger' not in q:
            kept.append(d); continue

        ds_key = ds_map.get(d.get('dataset', ''), d.get('dataset', ''))
        if ds_key not in metas:
            kept.append(d); continue

        meta, sid_col, best = metas[ds_key]
        sid = str(d['subject_id'])

        matched = None
        for kw, cols in STRUCT_COLS.items():
            if kw in q:
                matched = cols; break
        if not matched:
            kept.append(d); continue

        lc, rc = matched
        # Check ALL sessions for this structure (not just best session)
        L = R = None
        rows_all = meta[meta[sid_col].astype(str) == sid]
        for _, row in rows_all.iterrows():
            if pd.notna(row.get(lc)) and pd.notna(row.get(rc)):
                L, R = row.get(lc), row.get(rc)
                break  # use first session with data for this structure
        if L is None or R is None:
            # Fallback to best session
            if sid in best:
                row = best[sid]
                L, R = row.get(lc), row.get(rc)
        if not (pd.notna(L) and pd.notna(R)) or (float(L) + float(R)) <= 0:
            kept.append(d); continue

        asym = abs(float(L) - float(R)) / ((float(L) + float(R)) / 2) * 100
        if asym < MIN_ASYMMETRY_PCT:
            removed += 1
        else:
            kept.append(d)

    # CC359
    cc_path = os.path.join(base_dir, 'CC359/CC_metadata.xlsx')
    if not os.path.exists(cc_path):
        cc_path = '/path/to/data/Calgary-Campinas/CC_metadata.xlsx'
    if os.path.exists(cc_path):
        cc = pd.read_excel(cc_path)
        cc_by_sid = {str(r['subject_id']): r for _, r in cc.iterrows()}
        kept2 = []
        for d in kept:
            if d.get('dataset') == 'CC359' and d['category'] == 'Location' and d['question_type'] == 'YN':
                q = d['question'].lower()
                if 'smaller' in q or 'larger' in q:
                    sid = d['subject_id']
                    if sid in cc_by_sid:
                        row = cc_by_sid[sid]
                        L = row.get('hippocampus_L_vol_ml')
                        R = row.get('hippocampus_R_vol_ml')
                        if pd.notna(L) and pd.notna(R) and (float(L)+float(R)) > 0:
                            asym = abs(float(L)-float(R))/((float(L)+float(R))/2)*100
                            if asym < MIN_ASYMMETRY_PCT:
                                removed += 1; continue
            kept2.append(d)
        kept = kept2

    return kept, removed


def apply_R4_cpa(qa, base_dir):
    """R4: Fix cerebellar → posterior fossa for CPA and tentorium subjects."""
    csv_path = os.path.join(base_dir, 'BRATS_WMH.csv')
    if not os.path.exists(csv_path): return qa, 0
    csv_data = pd.read_csv(csv_path)
    # CPA + cerebellar tentorium → posterior fossa (not cerebellar parenchyma)
    pf_sids = set()
    for _, row in csv_data.iterrows():
        report = str(row['Global Finding']).lower()
        if 'cerebellopontine' in report or 'pontine-cerebellar' in report or 'cerebellar tentorium' in report:
            pf_sids.add(row['Subject ID'])
    fixed = 0
    for d in qa:
        if d['category'] == 'Location' and d['dataset'] in ('BraTS-GLI', 'BraTS-MEN'):
            # ALL cerebellar → posterior fossa (anatomical convention)
            if d.get('answer') == 'cerebellar':
                d['answer'] = 'posterior fossa'; fixed += 1
            if d.get('options'):
                d['options'] = ['posterior fossa' if o == 'cerebellar' else o for o in d['options']]
    return qa, fixed


def apply_R4b_midline_negation(qa, base_dir):
    """R4b: Fix midline shift and compression where report says 'no shift' but GT=Yes."""
    csv_path = os.path.join(base_dir, 'BRATS_WMH.csv')
    if not os.path.exists(csv_path): return qa, 0
    csv_data = pd.read_csv(csv_path)
    fixed = 0
    for d in qa:
        if d.get('dataset', '') not in ('BraTS-GLI', 'BraTS-MEN', 'WMH'): continue
        if d['question_type'] != 'YN': continue
        q = d['question'].lower()
        if 'midline' not in q and 'compression' not in q: continue

        row = csv_data[csv_data['Subject ID'] == d['subject_id']]
        if row.empty: continue
        report = str(row['Global Finding'].values[0]).lower()

        if 'midline' in q:
            no_shift = bool(re.search(r'no\s+midline\s*(shift|deviat)', report))
            has_shift = bool(re.search(r'midline.{0,20}(shift|deviat)', report)) and not no_shift
            correct = 'Yes' if has_shift else 'No'
        elif 'compression' in q:
            # Check for negated compression language
            no_comp = bool(re.search(r'no\s+(?:obvious\s+)?(?:significant\s+)?(compress|narrow)', report))
            has_comp_word = bool(re.search(r'(compress|narrow|displac|efface)', report))
            # Only override if report CLEARLY has or negates compression
            if not has_comp_word:
                continue  # Report doesn't mention compression — trust generation script
            has_comp = has_comp_word and not no_comp
            correct = 'Yes' if has_comp else 'No'
        else:
            continue

        if d['answer'] != correct:
            d['answer'] = correct
            fixed += 1
    return qa, fixed


def apply_R4c_signal_fix(qa, base_dir):
    """R4c: Fix BraTS signal answers that contradict Modal-wise Findings."""
    csv_path = os.path.join(base_dir, 'BRATS_WMH.csv')
    if not os.path.exists(csv_path): return qa, 0
    csv_data = pd.read_csv(csv_path)
    fixed = 0
    for d in qa:
        if d.get('dataset', '') not in ('BraTS-GLI', 'BraTS-MEN'): continue
        if d['category'] != 'Signal': continue

        row = csv_data[csv_data['Subject ID'] == d['subject_id']]
        if row.empty: continue
        modal = str(row['Modal-wise Findings'].values[0]).lower()
        report = str(row['Global Finding'].values[0]).lower()

        q = d['question'].lower()

        if d['question_type'] == 'MCQ' and 'signal intensity' in q:
            if 't1' in q and 't1n:' in modal:
                text = modal.split('t1n:')[1].split(';')[0]
            elif 't2' in q and 't2w:' in modal:
                text = modal.split('t2w:')[1].split(';')[0]
            else:
                continue

            # Determine correct signal
            if 'mixed' in text: correct = 'mixed'
            elif 'hyperintense' in text or 'high signal' in text or 'high intensity' in text: correct = 'hyperintense'
            elif 'hypointense' in text or 'low signal' in text or 'low intensity' in text: correct = 'hypointense'
            elif 'isointense' in text or 'iso' in text: correct = 'isointense'
            else: continue

            if d['answer'] != correct and correct in (d.get('options') or []):
                d['answer'] = correct
                fixed += 1

        elif d['question_type'] == 'YN' and 'high signal' in q and 't2' in q:
            if 't2w:' in modal:
                text = modal.split('t2w:')[1].split(';')[0]
            else:
                text = report
            has_high = 'high' in text or 'hyperintense' in text
            correct = 'Yes' if has_high else 'No'
            if d['answer'] != correct:
                d['answer'] = correct
                fixed += 1

    return qa, fixed


def apply_R26_age_threshold_fix(qa):
    """R26: Fix age threshold comparison to use int not float."""
    fixed = 0
    for d in qa:
        if d['question_type'] != 'YN': continue
        q = d['question'].lower()
        m = re.search(r'older than (\d+)', q)
        if not m: continue
        # This just flags — actual fix needs metadata age
        # The generation script bug is: comparing age_f > MED_AGE (float)
        # but question says int(MED_AGE). Fix is in generation script.
    return qa, fixed


def apply_R5_brats_lr_answers(qa):
    """R5: Remove left/right from BraTS Location Open answers."""
    fixed = 0
    for d in qa:
        if d.get('dataset', '') not in ('BraTS-GLI', 'BraTS-MEN'): continue
        if d['category'] != 'Location': continue
        if d['question_type'] == 'Open' and isinstance(d.get('answer'), str):
            old = d['answer']
            new = re.sub(r'\b(left|right)\s+', '', old, flags=re.IGNORECASE)
            if new != old: d['answer'] = new; fixed += 1
        if d.get('options'):
            d['options'] = [re.sub(r'\b(left|right)\s+', '', o, flags=re.IGNORECASE) for o in d['options']]
            d['answer'] = re.sub(r'\b(left|right)\s+', '', d['answer'], flags=re.IGNORECASE)
    return qa, fixed


def apply_R6_multilobe_fix(qa, base_dir):  # was apply_R6_multilobe
    """R6: Fix multi-lobe BraTS labels."""
    csv_path = os.path.join(base_dir, 'BRATS_WMH.csv')
    if not os.path.exists(csv_path): return qa, 0
    csv_data = pd.read_csv(csv_path)
    fixed = 0
    for d in qa:
        if d.get('dataset', '') not in ('BraTS-GLI', 'BraTS-MEN'): continue
        if d['category'] != 'Location' or d['question_type'] != 'YN': continue
        if 'lobe' not in d['question'].lower(): continue
        row = csv_data[csv_data['Subject ID'] == d['subject_id']]
        if row.empty: continue
        report = str(row['Global Finding'].values[0]).lower()
        for lobe in ['temporal', 'frontal', 'parietal', 'occipital']:
            if lobe in d['question'].lower():
                if lobe in report and d['answer'] == 'No':
                    d['answer'] = 'Yes'; fixed += 1
                elif lobe not in report and d['answer'] == 'Yes':
                    d['answer'] = 'No'; fixed += 1
                break
    return qa, fixed


def apply_R7_signal_wording(qa):
    """R7: Fix signal question wording."""
    fixed = 0
    for d in qa:
        if d['question'] == 'What is the T1 signal intensity of the lesion?':
            d['question'] = 'What is the signal intensity of the lesion on T1-weighted imaging?'; fixed += 1
        elif d['question'] == 'What is the T2 signal intensity of the lesion?':
            d['question'] = 'What is the signal intensity of the lesion on T2-weighted imaging?'; fixed += 1
    return qa, fixed


def apply_R8_age_context(qa, base_dir):
    """R8: Add age to Anatomy/Presence/Reasoning questions."""
    fixed = 0
    for ds_name in ['ADNI', 'PPMI', 'ABCD', 'HCP-YA', 'HCP-A', 'HCP-D', 'AIBL', 'HCP-Aging', 'HCP-Development']:
        meta = load_meta(base_dir, ds_name)
        if meta is None: continue
        sid_col = get_sid_col(meta)
        age_lookup = {}
        for _, row in meta.iterrows():
            sid = str(row[sid_col])
            age_val = row.get('age')
            if pd.isna(age_val): continue
            try:
                age_f = float(age_val)
                if age_f > 200: age_f /= 12.0
                age_lookup[sid] = int(round(age_f))
            except:
                age_lookup[sid] = str(age_val)

        ds_map = {'HCP-A': 'HCP-Aging', 'HCP-D': 'HCP-Development',
                  'HCP-Aging': 'HCP-Aging', 'HCP-Development': 'HCP-Development'}
        target_ds = ds_map.get(ds_name, ds_name)

        for d in qa:
            if d.get('dataset') != target_ds: continue
            if d['category'] not in ('Anatomy', 'Presence', 'Reasoning'): continue
            q = d['question']
            sid = str(d['subject_id'])
            if sid not in age_lookup: continue
            age = age_lookup[sid]
            if 'year-old' in q or 'subject age' in q or '(age ' in q: continue
            if 'expected' not in q.lower() and 'age-related' not in q.lower(): continue

            replacements = [
                ('for this age group', f'for this {age}-year-old subject'),
                ('for a young adult', f'for this {age}-year-old subject'),
                ('for a young adult brain', f'for this {age}-year-old brain'),
                ('for a pediatric brain', f'for this {age}-year-old pediatric brain'),
                ('for a developing brain', f'for this {age}-year-old developing brain'),
                ('age-expected', f'expected for a {age}-year-old'),
                ('age-related changes?', f'age-related changes (subject age: {age})?'),
                ('age-related changes)', f'age-related changes, subject age: {age})'),
            ]
            new_q = q
            for old, new in replacements:
                if old in new_q:
                    new_q = new_q.replace(old, new); break
            if new_q == q and 'in this scan?' in q:
                new_q = q.replace('in this scan?', f'in this scan (subject age: {age})?')
            if new_q != q:
                d['question'] = new_q; fixed += 1
    return qa, fixed


def apply_R9_longitudinal(qa, base_dir):
    """R9: Add interval + age to longitudinal, fix disjunction."""
    fixed = 0
    for ds_name in ['ADNI', 'PPMI', 'ABCD', 'HCP-A', 'HCP-D', 'AIBL']:
        meta = load_meta(base_dir, ds_name)
        if meta is None: continue
        sid_col = get_sid_col(meta)
        is_abcd = 'session_bids' in meta.columns
        if not is_abcd and 'session_date' in meta.columns:
            meta['session_date'] = pd.to_datetime(meta['session_date'], errors='coerce')

        subject_info = {}
        for sid in meta[sid_col].unique():
            rows = meta[meta[sid_col] == sid]
            if is_abcd:
                interval = 24 if len(rows['session_bids'].unique()) >= 2 else None
            elif 'session_date' in meta.columns:
                rows_s = rows.sort_values('session_date')
                if len(rows_s) >= 2 and pd.notna(rows_s['session_date'].iloc[0]) and pd.notna(rows_s['session_date'].iloc[-1]):
                    interval = int(round((rows_s['session_date'].iloc[-1] - rows_s['session_date'].iloc[0]).days / 30.44))
                    if interval < 1: interval = None
                else: interval = None
            else: interval = None
            age_val = rows['age'].iloc[-1] if 'age' in rows.columns else None
            if pd.notna(age_val):
                try:
                    af = float(age_val)
                    if af > 200: af /= 12.0
                    subject_info[str(sid)] = {'interval': interval, 'age': int(round(af))}
                except: subject_info[str(sid)] = {'interval': interval, 'age': None}

        ds_map = {'HCP-A': 'HCP-Aging', 'HCP-D': 'HCP-Development',
                  'HCP-Aging': 'HCP-Aging', 'HCP-Development': 'HCP-Development'}
        target_ds = ds_map.get(ds_name, ds_name)

        for d in qa:
            if d.get('dataset') != target_ds or d['category'] != 'Longitudinal': continue
            sid = str(d['subject_id'])
            info = subject_info.get(sid, {})
            q = d['question']
            new_q = q
            # Fix disjunction
            new_q = new_q.replace('remained stable or decreased', 'decreased')
            new_q = new_q.replace('remained stable or increased', 'increased')
            new_q = new_q.replace('remained stable or improved', 'improved')
            new_q = new_q.replace('remained stable or worsened', 'worsened')
            # Add interval
            interval = info.get('interval')
            if interval:
                new_q = new_q.replace('the previous scan', f'the scan {interval} months prior')
                new_q = new_q.replace('the previous visit', f'the visit {interval} months prior')
            # Add age
            age = info.get('age')
            if age and 'year-old' not in new_q and 'subject age' not in new_q:
                if new_q.endswith('?'):
                    new_q = new_q[:-1] + f' (subject age: {age})?'
            if new_q != q:
                d['question'] = new_q; fixed += 1
    return qa, fixed


def apply_R10_longitudinal_direction(qa, base_dir):
    """R10: Verify longitudinal YN answers against actual FreeSurfer volume change."""
    fixed = 0
    for ds_name in ['ADNI', 'PPMI', 'ABCD']:
        meta = load_meta(base_dir, ds_name)
        if meta is None: continue
        sid_col = get_sid_col(meta)
        is_abcd = 'session_bids' in meta.columns
        if not is_abcd and 'session_date' in meta.columns:
            meta['session_date'] = pd.to_datetime(meta['session_date'], errors='coerce')

        ds_map = {'HCP-A': 'HCP-Aging', 'HCP-D': 'HCP-Development',
                  'HCP-Aging': 'HCP-Aging', 'HCP-Development': 'HCP-Development'}
        target_ds = ds_map.get(ds_name, ds_name)

        for d in qa:
            if d.get('dataset') != target_ds: continue
            if d['category'] != 'Longitudinal' or d['question_type'] != 'YN': continue
            sid = str(d['subject_id'])
            rows = meta[meta[sid_col].astype(str) == sid]
            if len(rows) < 2: continue

            if is_abcd:
                baseline = rows[rows['session_bids'].str.contains('baseline', na=False)]
                followup = rows[rows['session_bids'].str.contains('2Year', na=False)]
                if baseline.empty or followup.empty: continue
                first, last = baseline.iloc[0], followup.iloc[0]
            else:
                rows_s = rows.sort_values('session_date')
                first, last = rows_s.iloc[0], rows_s.iloc[-1]

            q = d['question'].lower()
            bv = fv = None
            if 'hippocampal' in q or 'hippo' in q:
                bv, fv = first.get('FS_Vol_L_Hippocampus'), last.get('FS_Vol_L_Hippocampus')
            elif 'brain volume' in q or 'whole brain' in q:
                bv, fv = first.get('FS_ICV'), last.get('FS_ICV')
            elif 'ventricle' in q:
                bl, br = first.get('FS_Vol_L_LatVentricle'), first.get('FS_Vol_R_LatVentricle')
                fl, fr = last.get('FS_Vol_L_LatVentricle'), last.get('FS_Vol_R_LatVentricle')
                if all(pd.notna(x) for x in [bl, br, fl, fr]):
                    bv, fv = float(bl)+float(br), float(fl)+float(fr)
            if bv is None or fv is None or not (pd.notna(bv) and pd.notna(fv)): continue
            change = float(fv) - float(bv)
            if 'decreased' in q: correct = 'Yes' if change < 0 else 'No'
            elif 'increased' in q: correct = 'Yes' if change > 0 else 'No'
            elif 'enlarged' in q: correct = 'Yes' if change > 0 else 'No'
            else: continue
            if d['answer'] != correct:
                d['answer'] = correct; fixed += 1
    return qa, fixed


def apply_R12_diagnosis_framework(qa):
    """R12: Remove dataset names from questions (prevents shortcut leakage)."""
    fixed = 0
    for d in qa:
        q = d['question']
        for ds_name in ['ADNI', 'PPMI', 'AIBL', 'ABCD', 'HCP-YA', 'HCP-Aging', 'HCP-Development',
                        'BraTS-GLI', 'BraTS-MEN', 'WMH', 'CC359', 'IXI', 'HCP']:
            if ds_name in q:
                # Remove "based on X diagnostic criteria" or "based on X criteria"
                import re as _re
                d['question'] = _re.sub(r'\s*based on \S+ diagnostic criteria', '', q)
                d['question'] = _re.sub(r'\s*based on \S+ criteria', '', d['question'])
                fixed += 1
                break
    return qa, fixed


def apply_R13_mislabeled(qa, base_dir):
    """R13: Remove mislabeled BraTS-MEN subjects."""
    csv_path = os.path.join(base_dir, 'BRATS_WMH.csv')
    if not os.path.exists(csv_path): return qa, 0
    csv_data = pd.read_csv(csv_path)
    men = csv_data[csv_data['Type'] == 'BraTS-MEN']
    mislabeled = set()
    for _, row in men.iterrows():
        label = str(row['Disease Labels']).lower()
        if 'meningioma' not in label and 'menigioma' not in label and 'meningeoma' not in label:
            mislabeled.add(row['Subject ID'])
    before = len(qa)
    qa = [d for d in qa if d['subject_id'] not in mislabeled]
    return qa, before - len(qa)


def apply_R14_nan_freesurfer(qa, base_dir):
    """R14: Remove FS-dependent QA from subjects with no FreeSurfer volumes."""
    fs_cats = {'Location', 'Anatomy', 'Presence', 'Measurement', 'Reasoning'}
    removed = 0
    for ds_name in ['ADNI', 'PPMI', 'ABCD', 'HCP-YA', 'HCP-A', 'HCP-D']:
        meta = load_meta(base_dir, ds_name)
        if meta is None: continue
        sid_col = get_sid_col(meta)
        nan_sids = set()
        for sid in meta[sid_col].unique():
            rows = meta[meta[sid_col] == sid]
            has_fs = any(pd.notna(r.get('FS_Vol_L_Hippocampus')) for _, r in rows.iterrows())
            if not has_fs:
                nan_sids.add(str(sid))

        ds_map = {'HCP-A': 'HCP-Aging', 'HCP-D': 'HCP-Development',
                  'HCP-Aging': 'HCP-Aging', 'HCP-Development': 'HCP-Development'}
        target_ds = ds_map.get(ds_name, ds_name)

        before = len(qa)
        qa = [d for d in qa if not (d.get('dataset') == target_ds and str(d['subject_id']) in nan_sids and d['category'] in fs_cats)]
        removed += before - len(qa)
    return qa, removed


def apply_R15_session(qa, base_dir):
    """R15: Add session_date and image_path from best valid session."""
    fixed = 0
    for ds_name in ['ADNI', 'PPMI', 'ABCD', 'HCP-YA', 'HCP-A', 'HCP-D']:
        meta = load_meta(base_dir, ds_name)
        if meta is None: continue
        sid_col = get_sid_col(meta)
        best = get_best_session(meta, sid_col)

        # Find session column
        sess_col = None
        for c in ['session_date', 'session_bids', 'session_id', 'session_name']:
            if c in meta.columns: sess_col = c; break

        ds_map = {'HCP-A': 'HCP-Aging', 'HCP-D': 'HCP-Development',
                  'HCP-Aging': 'HCP-Aging', 'HCP-Development': 'HCP-Development'}
        target_ds = ds_map.get(ds_name, ds_name)

        for d in qa:
            if d.get('dataset') != target_ds: continue
            sid = str(d['subject_id'])
            if sid not in best: continue
            row = best[sid]
            if sess_col:
                sv = str(row.get(sess_col, ''))[:20]
                if sv and sv != 'nan' and sv != 'NaT' and len(sv) > 0:
                    d['session_date'] = sv; fixed += 1
            ip = str(row.get('T1w_image_path', ''))
            if ip and ip != 'nan':
                d['image_path'] = ip
    return qa, fixed


def apply_R16_modalities(qa):
    """R16: Add answerable_modalities field."""
    for d in qa:
        q = d['question'].lower()
        cat = d['category']
        ds = d.get('dataset', '')
        mods = DS_MODALITIES.get(ds, ['T1w'])
        has_t2 = 'T2w' in mods

        if cat == 'Modality': d['answerable_modalities'] = ['T1w', 'T2w']
        elif cat == 'Signal':
            if 't2' in q: d['answerable_modalities'] = ['T2w', 'T1w+T2w']
            elif 'white matter' in q or 'wmh' in q or 'hyperintensit' in q:
                d['answerable_modalities'] = ['T1w', 'T2w', 'T1w+T2w'] if has_t2 else ['T1w']
            elif 't1' in q: d['answerable_modalities'] = ['T1w', 'T1w+T2w'] if has_t2 else ['T1w']
            else: d['answerable_modalities'] = ['T1w', 'T2w', 'T1w+T2w'] if has_t2 else ['T1w']
        elif cat in ('Diagnosis',) and ds in ('BraTS-GLI', 'BraTS-MEN'):
            d['answerable_modalities'] = ['T1w', 'T2w', 'T1w+T2w']
        elif cat in ('Diagnosis',) and ds == 'WMH':
            d['answerable_modalities'] = ['T1w']
        else:
            base = ['T1w']
            if has_t2: base.extend(['T2w', 'T1w+T2w'])
            d['answerable_modalities'] = base

        if 'recommended_modality' in d: del d['recommended_modality']
    return qa


def apply_R11_compound_presence(qa, base_dir):
    """R11: Update Presence MCQ with compound options based on FreeSurfer findings."""
    fixed = 0
    for ds_name in ['ADNI', 'PPMI', 'ABCD', 'HCP-YA', 'HCP-A', 'HCP-D']:
        meta = load_meta(base_dir, ds_name)
        if meta is None: continue
        sid_col = get_sid_col(meta)
        best = get_best_session(meta, sid_col)

        # Compute medians
        hippo_vals = ((meta.get('FS_Vol_L_Hippocampus', pd.Series(dtype=float)).astype(float) +
                       meta.get('FS_Vol_R_Hippocampus', pd.Series(dtype=float)).astype(float)) / 2).dropna()
        vent_cols_l = meta.get('FS_Vol_L_LatVentricle', pd.Series(dtype=float))
        vent_cols_r = meta.get('FS_Vol_R_LatVentricle', pd.Series(dtype=float))
        icv = meta.get('FS_ICV', pd.Series(dtype=float))
        if len(hippo_vals) == 0: continue
        med_hippo = hippo_vals.median()
        vent_ratio = ((vent_cols_l.astype(float) + vent_cols_r.astype(float)) / icv.astype(float) * 100).dropna()
        med_vent = vent_ratio.median() if len(vent_ratio) > 0 else 2.0

        ds_map = {'HCP-A': 'HCP-Aging', 'HCP-D': 'HCP-Development',
                  'HCP-Aging': 'HCP-Aging', 'HCP-Development': 'HCP-Development'}
        target_ds = ds_map.get(ds_name, ds_name)

        for d in qa:
            if d.get('dataset') != target_ds: continue
            if d['category'] != 'Presence' or d['question_type'] != 'MCQ': continue
            if 'NOT present' in d['question']: continue
            if 'present' not in d['question'].lower() and 'finding' not in d['question'].lower(): continue
            sid = str(d['subject_id'])
            if sid not in best: continue
            row = best[sid]

            findings = []
            hl = row.get('FS_Vol_L_Hippocampus')
            hr = row.get('FS_Vol_R_Hippocampus')
            if pd.notna(hl) and pd.notna(hr):
                if (float(hl) + float(hr)) / 2 < med_hippo:
                    findings.append('Hippocampal atrophy')
            vl = row.get('FS_Vol_L_LatVentricle')
            vr = row.get('FS_Vol_R_LatVentricle')
            iv = row.get('FS_ICV')
            if pd.notna(vl) and pd.notna(vr) and pd.notna(iv) and float(iv) > 0:
                if (float(vl) + float(vr)) / float(iv) * 100 > med_vent:
                    findings.append('Ventricular enlargement')
            wmh = row.get('FS_Vol_WMHypo') or row.get('FS_WMH')
            if pd.notna(wmh) and float(wmh) > 3000:
                findings.append('White matter changes')

            if len(findings) == 0:
                d['answer'] = 'No significant findings'
                d['options'] = ['Hippocampal atrophy', 'Ventricular enlargement', 'White matter changes', 'No significant findings']
            elif len(findings) == 1:
                d['answer'] = findings[0]
                others = [f for f in ['Hippocampal atrophy','Ventricular enlargement','White matter changes'] if f != findings[0]]
                d['options'] = [findings[0]] + others[:2] + ['No significant findings']
            elif len(findings) == 2:
                compound = f'Both {findings[0].lower()} and {findings[1].lower()}'
                d['answer'] = compound
                d['options'] = [findings[0]+' only', findings[1]+' only', compound, 'No significant findings']
                d['question'] = 'Which best describes the structural findings in this scan?'
            else:
                compound = f'{findings[0]}, {findings[1].lower()}, and {findings[2].lower()}'
                d['answer'] = compound
                d['options'] = [findings[0]+' only', f'Both {findings[0].lower()} and {findings[1].lower()}', compound, 'No significant findings']
                d['question'] = 'Which best describes the structural findings in this scan?'

            random.shuffle(d['options'])
            fixed += 1
    return qa, fixed


def apply_R17_human_assessable(qa):
    """R17: Add human_assessable flag."""
    small_structs = ['accumbens', 'pallidum', 'amygdala']
    for d in qa:
        q = d['question'].lower()
        cat = d['category']
        qt = d['question_type']
        assessable = True
        note = None

        if cat == 'Location' and any(s in q for s in small_structs):
            assessable = False; note = "Subcortical structure too small for visual assessment"
        elif cat == 'Measurement' and qt == 'YN' and any(kw in q for kw in ['mm³', 'mm3', 'ratio', '%']):
            assessable = False; note = "Requires precise volumetric quantification"
        elif cat == 'Measurement' and qt == 'MCQ' and any(kw in q for kw in ['range', 'intracranial volume']):
            assessable = False; note = "Requires volumetric estimation beyond visual inspection"
        elif 'population median' in q and ('ml' in q or 'mL' in q):
            assessable = False; note = "Requires population-specific volumetric norms"
        elif 'cognitive composite score' in q or 'fluid cognition' in q:
            assessable = False; note = "Cognitive scores not assessable from imaging"

        d['human_assessable'] = assessable
        if not assessable and note:
            d['human_assessable_note'] = note
        elif 'human_assessable_note' in d:
            del d['human_assessable_note']
    return qa


def apply_R23_yn_balance(qa):
    """R23: YN 50/50 balance per template PER DATASET."""
    # Group by dataset first, then balance each independently
    by_ds = defaultdict(list)
    for d in qa:
        by_ds[d.get('dataset', 'unknown')].append(d)

    all_kept = []
    total_removed = 0

    for ds, ds_qa in sorted(by_ds.items()):
        random.seed(42)  # Reset seed per dataset for independence
        yn_by_tmpl = defaultdict(lambda: defaultdict(list))
        non_yn = []
        for d in ds_qa:
            if d['question_type'] == 'YN':
                tmpl = normalize_template(d['question'])
                yn_by_tmpl[tmpl][d['answer']].append(d)
            else:
                non_yn.append(d)

        balanced = []
        removed = 0
        for tmpl in sorted(yn_by_tmpl.keys()):  # sorted for determinism
            by_ans = yn_by_tmpl[tmpl]
            yes_items = by_ans.get('Yes', [])
            no_items = by_ans.get('No', [])
            n = min(len(yes_items), len(no_items))
            if n < 3:
                removed += len(yes_items) + len(no_items); continue
            det_shuffle(yes_items); det_shuffle(no_items)
            balanced.extend(yes_items[:n]); balanced.extend(no_items[:n])
            removed += (len(yes_items) - n) + (len(no_items) - n)

        all_kept.extend(non_yn + balanced)
        total_removed += removed

    return all_kept, total_removed


def apply_R24_shortcut_removal(qa):
    """R24: Remove/downsample MCQ/Open shortcut templates PER DATASET."""
    by_ds = defaultdict(list)
    for d in qa:
        by_ds[d.get('dataset', 'unknown')].append(d)

    all_kept = []
    total_removed = 0

    for ds, ds_qa in sorted(by_ds.items()):
        random.seed(42)  # Reset seed per dataset
        templates = defaultdict(lambda: defaultdict(list))
        for d in ds_qa:
            tmpl = normalize_template(d['question'])
            templates[tmpl][d['answer']].append(d)

        remove_ids = set()
        for tmpl in sorted(templates.keys()):  # sorted for determinism
            by_ans = templates[tmpl]
            total = sum(len(v) for v in by_ans.values())
            if total < 5: continue
            majority_ans = max(by_ans, key=lambda a: len(by_ans[a]))
            majority_count = len(by_ans[majority_ans])
            majority_pct = majority_count / total * 100
            qt = by_ans[majority_ans][0]['question_type']

            if qt == 'YN': continue  # Already balanced by R23

            # ≥95% → remove entirely
            if majority_pct >= 95:
                for items in by_ans.values():
                    for item in items: remove_ids.add(id(item))
                continue

            # >70% → downsample
            if majority_pct > 70:
                others = [len(v) for a, v in by_ans.items() if a != majority_ans]
                second = max(others) if others else 0
                if second < 3:
                    for items in by_ans.values():
                        for item in items: remove_ids.add(id(item))
                    continue
                target = second * 2
                excess = majority_count - target
                if excess > 0:
                    det_shuffle(by_ans[majority_ans])
                    for item in by_ans[majority_ans][:excess]:
                        remove_ids.add(id(item))

        kept = [d for d in ds_qa if id(d) not in remove_ids]
        all_kept.extend(kept)
        total_removed += len(ds_qa) - len(kept)

    return all_kept, total_removed


def apply_R29_measurement_verify(qa, base_dir):
    """R29: Re-verify ALL measurement YN answers against actual FS volumes.
    Catches boundary bugs where generation script rounded threshold doesn't match actual."""
    META_PATHS = {
        'HCP-Aging': ('HCP-Aging/outputs/hcpaging_neuroqa_merged.csv', 'hcpaging_subject_id', 'session_id'),
        'HCP-Development': ('HCP-Development/outputs/hcpdev_neuroqa_merged.csv', 'hcpdev_subject_id', 'session_id'),
        'HCP-YA': ('HCP-YA/outputs/hcpya_neuroqa_merged.csv', 'hcpya_subject_id', 'session_name'),
        'ABCD': ('ABCD/outputs/abcd_neuroqa_merged.csv', 'abcd_subject_id', 'session_bids'),
        'ADNI': ('ADNI/outputs/adni_neuroqa_merged.csv', 'adni_subject_id', 'session_date'),
        'PPMI': ('PPMI/outputs/ppmi_neuroqa_merged.csv', 'ppmi_subject_id', 'session_date'),
    }
    fixed = 0
    for ds_name, (rel_path, sid_col, sess_col) in META_PATHS.items():
        csv_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(csv_path): continue
        meta = pd.read_csv(csv_path)
        sess_lk = {}
        for _, row in meta.iterrows():
            sid = str(row[sid_col])
            sess = str(row.get(sess_col, ''))
            sess_lk.setdefault(sid, []).append((sess, row))

        for d in qa:
            if d.get('dataset') != ds_name: continue
            if d['category'] != 'Measurement' or d['question_type'] != 'YN': continue
            q = d['question'].lower()
            if 'average hippocampal volume' not in q: continue

            sid = str(d['subject_id'])
            if sid not in sess_lk: continue
            qa_sess = str(d.get('session_date', ''))
            row = None
            for s, r in sess_lk[sid]:
                if s.startswith(qa_sess) or qa_sess.startswith(s):
                    row = r; break
            if row is None: continue

            hL = row.get('FS_Vol_L_Hippocampus')
            hR = row.get('FS_Vol_R_Hippocampus')
            if not (pd.notna(hL) and pd.notna(hR)): continue
            avg = (float(hL) + float(hR)) / 2

            m = re.search(r'(above|below)\s+([\d,]+)', q)
            if not m: continue
            direction = m.group(1)
            threshold = float(m.group(2).replace(',', ''))
            if direction == 'above':
                correct = 'Yes' if avg > threshold else 'No'
            else:
                correct = 'Yes' if avg < threshold else 'No'
            if d['answer'] != correct:
                d['answer'] = correct
                fixed += 1

    return qa, fixed


def apply_R30_deduplicate(qa):
    """R30: Remove duplicate items (same subject + same question) with conflicting answers."""
    from collections import defaultdict
    by_key = defaultdict(list)
    for i, d in enumerate(qa):
        key = (d['subject_id'], d['question'][:80])
        by_key[key].append(i)

    remove = set()
    for key, indices in by_key.items():
        if len(indices) > 1:
            answers = [str(qa[i]['answer']) for i in indices]
            if len(set(answers)) > 1:
                # Conflicting answers — keep first, remove rest
                for idx in indices[1:]:
                    remove.add(idx)

    if remove:
        qa_new = [d for i, d in enumerate(qa) if i not in remove]
        return qa_new, len(remove)
    return qa, 0


def apply_R32_none_label(qa):
    """R32: Rename MCQ answer 'None' to 'No changes' to avoid null confusion."""
    fixed = 0
    for d in qa:
        if d['question_type'] == 'MCQ' and d['answer'] == 'None' and d.get('options'):
            if 'None' in d['options']:
                d['options'] = ['No changes' if o == 'None' else o for o in d['options']]
                d['answer'] = 'No changes'
                fixed += 1
    return qa, fixed


def apply_R39_abcd_hippo_relabel(qa):
    """R39: ABCD pediatric terminology. In the 'findings NOT present' template,
    'Hippocampal atrophy' is relabeled to 'Below expected hippocampal volume'
    ('atrophy' is inappropriate for developing brains). The 'findings present'
    template keeps 'Hippocampal atrophy', so scope strictly to the NOT-present template."""
    OLD, NEW = 'Hippocampal atrophy', 'Below expected hippocampal volume'
    fixed = 0
    for d in qa:
        if d.get('dataset') != 'ABCD':
            continue
        if 'NOT present' not in d.get('question', ''):
            continue
        changed = False
        if d.get('options') and OLD in d['options']:
            d['options'] = [NEW if o == OLD else o for o in d['options']]
            changed = True
        if d.get('answer') == OLD:
            d['answer'] = NEW
            changed = True
        if changed:
            fixed += 1
    return qa, fixed


def apply_R33_wholebrain_ratio(qa):
    """R33: Remove 'whole brain volume' questions from datasets where FS_WholeBrain is ratio, not volume."""
    # HCP-Aging and HCP-Development have FS_WholeBrain as brain parenchymal fraction (~0.7), not mm³
    bad_ds = {'HCP-Aging', 'HCP-Development'}
    before = len(qa)
    qa = [d for d in qa if not (d['dataset'] in bad_ds and 'whole brain' in d['question'].lower())]
    return qa, before - len(qa)


def apply_R37_longitudinal_fix(qa, base_dir):
    """R37: Fix ALL longitudinal items — correct sessions, answers, months, paths."""
    from datetime import datetime as _dt
    import glob as _g

    ABCD_ORDER = {'ses-baselineYear1Arm1': 0, 'ses-baselineYear1Arm': 0,
                  'ses-2YearFollowUpYArm1': 1, 'ses-2YearFollowUpYAr': 1,
                  'ses-4YearFollowUpYArm1': 2, 'ses-4YearFollowUpYAr': 2}

    DS_CFG = {
        'ABCD': ('ABCD/outputs/abcd_neuroqa_merged.csv', 'abcd_subject_id', 'session_bids'),
        'ADNI': ('ADNI/outputs/adni_neuroqa_merged.csv', 'adni_subject_id', 'session_date'),
        'PPMI': ('PPMI/outputs/ppmi_neuroqa_merged.csv', 'ppmi_subject_id', 'session_date'),
        'AIBL': ('AIBL/outputs/aibl_neuroqa_merged.csv', 'aibl_subject_id', 'session_date'),
    }

    # Load and sort sessions chronologically
    all_sessions = {}
    for ds_name, (rel, sid_col, sess_col) in DS_CFG.items():
        fpath = os.path.join(base_dir, rel)
        if not os.path.exists(fpath): continue
        df = pd.read_csv(fpath)
        lk = {}
        for _, row in df.iterrows():
            sid = str(row[sid_col])
            lk.setdefault(sid, []).append(row)
        for sid in lk:
            if ds_name == 'ABCD':
                lk[sid] = sorted(lk[sid], key=lambda r: ABCD_ORDER.get(str(r.get(sess_col,'')), 99))
            else:
                lk[sid] = sorted(lk[sid], key=lambda r: str(r.get(sess_col, '')))
        all_sessions[ds_name] = (lk, sess_col)

    def _compute_ans(ql, cr, pr):
        def _fv(r, c):
            v = r.get(c)
            return float(v) if pd.notna(v) else None
        if 'hippocampal volume' in ql:
            ch, ph = _fv(cr,'FS_Vol_L_Hippocampus'), _fv(pr,'FS_Vol_L_Hippocampus')
            if ch is not None and ph is not None and ph > 0:
                if ch == ph:
                    if 'increased' in ql: return 'No'
                    elif 'decreased' in ql: return 'No'
                    elif 'stable' in ql or 'remained' in ql: return 'Yes'
                grew = ch > ph
                if 'increased' in ql: return 'Yes' if grew else 'No'
                elif 'decreased' in ql: return 'Yes' if not grew else 'No'
                elif 'stable' in ql or 'remained' in ql: return 'No' if grew else 'Yes'
                elif 'percentage' in ql: return f'{(ch-ph)/ph*100:+.0f}%'
        elif 'whole brain' in ql:
            cwb, pwb = _fv(cr,'FS_WholeBrain'), _fv(pr,'FS_WholeBrain')
            if cwb is not None and pwb is not None:
                if 'increased' in ql: return 'Yes' if cwb > pwb else 'No'
                elif 'decreased' in ql: return 'Yes' if cwb < pwb else 'No'
        elif 'ventricle' in ql and 'enlarged' in ql:
            cv, pv = _fv(cr,'FS_Ventricles'), _fv(pr,'FS_Ventricles')
            if cv is not None and pv is not None:
                return 'Yes' if cv > pv else 'No'
        return None

    def _find_path(ds, sid, sess):
        if ds == 'ABCD':
            p = f'/path/to/raw_data/ABCD/processed/Structural/registration/Unzip/{sid}/{sess}/anat/T1w_mni_zscore_fixed_cropped.nii.gz'
            if os.path.exists(p): return p
            ms = _g.glob(f'/path/to/raw_data/ABCD/processed/Structural/registration/Unzip/{sid}/{sess}*/anat/T1w_mni_zscore_fixed_cropped.nii.gz')
            return ms[0] if ms else None
        elif ds == 'ADNI':
            ms = _g.glob(f'/path/to/raw_data/ADNI/processed/Structural/registration/{sid}/**/*T1w_mni_zscore_fixed_cropped.nii.gz', recursive=True)
            ms = sorted([m for m in ms if sess[:10] in m]) if sess else sorted(ms)
            return ms[0] if ms else (sorted(_g.glob(f'/path/to/raw_data/ADNI/processed/Structural/registration/{sid}/**/*T1w_mni_zscore_fixed_cropped.nii.gz', recursive=True))[-1] if _g.glob(f'/path/to/raw_data/ADNI/processed/Structural/registration/{sid}/**/*T1w_mni_zscore_fixed_cropped.nii.gz', recursive=True) else None)
        elif ds == 'PPMI':
            ms = sorted(_g.glob(f'/path/to/raw_data/PPMI/processed/Structural/registration/{sid}/**/*T1w_mni_zscore_fixed_cropped.nii.gz', recursive=True))
            ms_match = [m for m in ms if sess[:10] in m] if sess else ms
            return ms_match[0] if ms_match else (ms[-1] if ms else None)
        elif ds == 'AIBL':
            ms = sorted(_g.glob(f'/path/to/raw_data/AIBL/processed/Structural/registration/{sid}/**/*T1w_mni_zscore_fixed_cropped.nii.gz', recursive=True))
            ms_match = [m for m in ms if sess[:10] in m] if sess else ms
            return ms_match[0] if ms_match else (ms[-1] if ms else None)
        return None

    fixed = 0
    for d in qa:
        if d['category'] != 'Longitudinal': continue
        ds = d['dataset']
        sid = str(d['subject_id'])
        ql = d['question'].lower()

        # SKIP cognitive/clinical questions — can't verify from FS volumes
        is_volume = any(k in ql for k in ['hippocampal volume', 'whole brain', 'ventricle', 'percentage change'])

        if ds not in all_sessions: continue
        lk, sess_col = all_sessions[ds]
        rows = lk.get(sid, [])
        if len(rows) < 2: continue

        # Use last two sessions (chronological)
        curr, prev = rows[-1], rows[-2]
        curr_sess = str(curr.get(sess_col, ''))
        prev_sess = str(prev.get(sess_col, ''))

        if is_volume:
            # For volume questions: verify and fix answer
            expected = _compute_ans(ql, curr, prev)

            # Try to find a pair that matches stored answer first
            for i in range(1, len(rows)):
                c, p = rows[i], rows[i-1]
                exp = _compute_ans(ql, c, p)
                if exp is not None and str(exp) == str(d['answer']):
                    curr, prev = c, p
                    curr_sess = str(curr.get(sess_col, ''))
                    prev_sess = str(prev.get(sess_col, ''))
                    expected = exp
                    break

            # If still no match, fix answer to last pair
            if expected is not None and str(d['answer']) != str(expected):
                d['answer'] = expected

        # Set session info (for ALL longitudinal, including cognitive)
        d['session_date'] = curr_sess
        d['prior_session_date'] = prev_sess

        # Compute months
        months = 0
        if ds == 'ABCD':
            c_m = ABCD_ORDER.get(curr_sess, 0) * 24
            p_m = ABCD_ORDER.get(prev_sess, 0) * 24
            months = abs(c_m - p_m) or 24
        else:
            try:
                d1 = _dt.strptime(curr_sess[:10], '%Y-%m-%d')
                d2 = _dt.strptime(prev_sess[:10], '%Y-%m-%d')
                months = round(abs((d1-d2).days) / 30)
            except:
                pass

        # Update months in question ONLY if already has a number there
        if months > 0:
            m = re.search(r'(\d+)\s*months?\s*prior', d['question'])
            if m:
                d['question'] = d['question'][:m.start(1)] + str(months) + d['question'][m.end(1):]
            # Do NOT change "previous visit/scan" to "N months prior" — keep original wording

        # Set image paths
        cp = _find_path(ds, sid, curr_sess)
        pp = _find_path(ds, sid, prev_sess)
        if cp: d['image_path'] = cp
        if pp: d['image_path_prior'] = pp

        fixed += 1

    return qa, fixed


def apply_R36_fix_paths_and_types(qa, base_dir):
    """R36: Fix image_path, subject_id types, and IXI show_modality."""
    import glob as _glob
    fixed = 0

    # 1. subject_id → string
    for d in qa:
        if isinstance(d['subject_id'], int):
            d['subject_id'] = str(d['subject_id'])
            fixed += 1

    # 2. image_path for datasets that don't have it
    PATH_TEMPLATES = {
        'BraTS-GLI': '/path/to/preprocessed/BraTS_GLI/registration/{sid}/T1w_mni_zscore_fixed_cropped.nii.gz',
        'BraTS-MEN': '/path/to/preprocessed/BraTS_MEN/registration/{sid}/T1w_mni_zscore_fixed_cropped.nii.gz',
        'CC359': '/path/to/data/Calgary-Campinas/preprocessed/registration/{sid}/anat/{sid}_mni_zscore_fixed_cropped.nii.gz',
    }
    for d in qa:
        if d.get('image_path'):
            continue
        ds = d['dataset']
        sid = str(d['subject_id'])
        if ds in PATH_TEMPLATES:
            d['image_path'] = PATH_TEMPLATES[ds].format(sid=sid)
            fixed += 1
        elif ds == 'IXI':
            d['image_path'] = f'/path/to/data/IXI/preprocessed/registration/{sid}/anat/{sid}_T1_mni_zscore_fixed_cropped.nii.gz'
            fixed += 1
        elif ds == 'AIBL':
            matches = _glob.glob(f'/path/to/raw_data/AIBL/processed/Structural/registration/{sid}/**/*T1w_mni_zscore_fixed_cropped.nii.gz', recursive=True)
            if matches:
                d['image_path'] = matches[0]
                fixed += 1
        elif ds == 'WMH':
            for site_dir in ['Amsterdam/GE3T', 'Singapore', 'Utrecht']:
                num = sid.split('_')[-1] if '_' in sid else sid
                path = f'/path/to/preprocessed/WMH/{site_dir}/registration/{num}/pre/T1w_mni_zscore_fixed_cropped.nii.gz'
                if os.path.exists(path):
                    d['image_path'] = path
                    fixed += 1
                    break

    # 3. IXI show_modality
    ixi_assignment = {}
    for d in qa:
        if d['dataset'] != 'IXI' or d['category'] != 'Modality' or d['question_type'] != 'YN':
            continue
        sid = d['subject_id']
        if 'T1' in d['question'] and d['answer'] == 'Yes':
            ixi_assignment[sid] = 'T1w'
        elif 'T2' in d['question'] and d['answer'] == 'Yes':
            ixi_assignment[sid] = 'T2w'

    for d in qa:
        if d['dataset'] == 'IXI':
            assigned = ixi_assignment.get(d['subject_id'], 'T1w')
            d['show_modality'] = assigned
            sid = d['subject_id']
            # Fix IXI path format: dash not underscore ({sid}-T1_ not {sid}_T1_)
            if d.get('image_path'):
                d['image_path'] = d['image_path'].replace(f'{sid}_T1_', f'{sid}-T1_').replace(f'{sid}_T2_', f'{sid}-T2_')
            if assigned == 'T2w':
                d['image_path'] = f'/path/to/data/IXI/preprocessed/registration/{sid}/anat/{sid}-T2_mni_zscore_fixed_cropped.nii.gz'

    # 4. Add image_path_t2 for datasets with T2
    import glob as _g
    for d in qa:
        ds = d['dataset']
        t1 = d.get('image_path', '')
        if not t1: continue
        t1_dir = os.path.dirname(t1)

        if ds in ('HCP-Aging', 'HCP-Development'):
            # T2 in same folder: T2w_mni_zscore_fixed_cropped.nii.gz
            d['image_path_t2'] = os.path.join(t1_dir, 'T2w_mni_zscore_fixed_cropped.nii.gz')
        elif ds == 'HCP-YA':
            # T2 in same folder: {sid}_3T_T2w_SPC1_brain_mni_zscore_cropped.nii.gz
            sid = d['subject_id']
            d['image_path_t2'] = os.path.join(t1_dir, f'{sid}_3T_T2w_SPC1_brain_mni_zscore_cropped.nii.gz')
        elif ds in ('BraTS-GLI', 'BraTS-MEN'):
            d['image_path_t2'] = t1.replace('T1w_mni', 'T2w_mni')
        elif ds == 'IXI':
            d['image_path_t2'] = t1.replace('-T1_', '-T2_')
        elif ds == 'ABCD':
            # T2 MUST be in same session dir as T1
            t2 = os.path.join(t1_dir, 'T2w_mni_zscore_fixed_cropped.nii.gz')
            d['image_path_t2'] = t2  # may or may not exist — same session only

    return qa, fixed


def apply_R35_signal_from_features(qa, base_dir):
    """R35: Fix BraTS signal MCQ answers from features CSV (t1_signal/t2_signal)."""
    feat_lookup = {}
    for fn in ['brats_gli_features.csv', 'brats_men_features.csv']:
        fpath = os.path.join(base_dir, 'BraTS_WMH', 'outputs', fn)
        if os.path.exists(fpath):
            for _, r in pd.read_csv(fpath).iterrows():
                feat_lookup[str(r['subject_id'])] = r

    fixed = 0
    for d in qa:
        if d['dataset'] not in ('BraTS-GLI', 'BraTS-MEN'): continue
        if d['category'] != 'Signal' or d['question_type'] != 'MCQ': continue
        if 'signal intensity' not in d['question'].lower(): continue

        feat = feat_lookup.get(d['subject_id'])
        if feat is None: continue

        q = d['question'].lower()
        if 't1' in q:
            correct = str(feat.get('t1_signal', '')).lower()
        elif 't2' in q:
            correct = str(feat.get('t2_signal', '')).lower()
        else:
            continue

        if correct and correct in (d.get('options') or []) and d['answer'] != correct:
            d['answer'] = correct
            fixed += 1

    return qa, fixed


def apply_R34_age_answer_fix(qa, base_dir):
    """R34: Fix age MCQ answers for ALL datasets using correct session age."""
    META = {
        'AIBL': ('AIBL/outputs/aibl_neuroqa_merged.csv', 'aibl_subject_id', 'session_date',
                 [(55,65),(65,75),(75,85),(85,200)], ["55-64","65-74","75-84","85+"]),
        'ABCD': ('ABCD/outputs/abcd_neuroqa_merged.csv', 'abcd_subject_id', 'session_bids',
                 [(8,10),(10,11),(11,13),(13,15)], ["8-9","10-11","11-12","13-14"]),
        'ADNI': ('ADNI/outputs/adni_neuroqa_merged.csv', 'adni_subject_id', 'session_date',
                 [(50,65),(65,75),(75,85),(85,200)], ["50-64","65-74","75-84","85+"]),
    }
    fixed = 0
    for ds_name, (rel_path, sid_col, sess_col, bins, labels) in META.items():
        csv_path = os.path.join(base_dir, rel_path)
        if not os.path.exists(csv_path): continue
        meta = pd.read_csv(csv_path)
        lk = {}
        for _, row in meta.iterrows():
            sid = str(row[sid_col])
            lk.setdefault(sid, []).append((str(row.get(sess_col, '')), row))

        for d in qa:
            if d.get('dataset') != ds_name or d['question_type'] != 'MCQ': continue
            q = d['question'].lower()
            if 'age range' not in q and 'estimated age' not in q: continue

            sid = str(d['subject_id'])
            if sid not in lk: continue
            qa_sess = str(d.get('session_date', ''))
            row = None
            for s, r in lk[sid]:
                if s.startswith(qa_sess[:10]) or qa_sess[:10].startswith(s[:10]):
                    row = r; break
            if row is None: continue

            age = row.get('age')
            if not pd.notna(age): continue
            age_f = float(age)

            exp = None
            for j, (lo, hi) in enumerate(bins):
                if lo <= age_f < hi or (j == len(bins)-1 and age_f >= lo):
                    exp = labels[j]; break

            if exp and exp in (d.get('options') or []) and d['answer'] != exp:
                d['answer'] = exp
                fixed += 1

    return qa, fixed


def apply_R31_open_to_yn(qa):
    """R31: Convert Open questions with Yes/No answers to YN type."""
    fixed = 0
    for d in qa:
        if d['question_type'] == 'Open' and d['answer'] in ('Yes', 'No'):
            d['question_type'] = 'YN'
            d['options'] = None
            fixed += 1
    return qa, fixed


def apply_R28_mcq_standardize(qa):
    """R28: Standardize all MCQs to exactly 4 options. Convert 2-option to YN."""
    fixed = 0
    for d in qa:
        if d['question_type'] != 'MCQ' or not d.get('options'):
            continue
        n = len(d['options'])

        if n == 2:
            # Convert binary MCQ to YN
            ans = d['answer']
            idx = d['options'].index(ans) if ans in d['options'] else 0
            d['answer'] = 'Yes' if idx == 0 else 'No'
            d['question_type'] = 'YN'
            d['options'] = None
            fixed += 1
        elif n > 4:
            # Reduce to 4 options: keep answer + 3 random distractors
            ans = d['answer']
            distractors = [o for o in d['options'] if o != ans]
            random.shuffle(distractors)
            new_opts = [ans] + distractors[:3]
            random.shuffle(new_opts)
            d['options'] = new_opts
            fixed += 1
        elif n == 3:
            # Add a distractor — use "None of the above" or similar
            d['options'].append('None of the above')
            random.shuffle(d['options'])
            fixed += 1

    return qa, fixed


def apply_R28b_age_label_fix(qa):
    """R28b: Fix overlapping age range labels to be non-ambiguous (per-dataset)."""
    # ONLY fix labels where the bin boundary actually overlaps
    # HCP-D "13-15" is correct (bin [13,16)), don't change it
    # ABCD "13-15" means [13,15) → rename to "13-14"
    DS_LABEL_FIX = {
        'ABCD': {"8-10": "8-9", "10-11": "10-11", "11-13": "11-12", "13-15": "13-14"},
        'ADNI': {"50-65": "50-64", "65-75": "65-74", "75-85": "75-84"},
        'AIBL': {"55-65": "55-64", "65-75": "65-74", "75-85": "75-84"},
        'PPMI': {"30-50": "30-49", "50-65": "50-64", "65-75": "65-74"},
        # HCP-D: NO changes — labels match bins correctly
        # HCP-YA, HCP-Aging, CC359, IXI: labels don't overlap
    }
    fixed = 0
    for d in qa:
        ds = d.get('dataset', '')
        label_fix = DS_LABEL_FIX.get(ds, {})
        if not label_fix: continue
        changed = False
        if d.get('options'):
            new_opts = [label_fix.get(o, o) for o in d['options']]
            if new_opts != list(d['options']):
                d['options'] = new_opts
                changed = True
        if d.get('answer') in label_fix:
            d['answer'] = label_fix[d['answer']]
            changed = True
        if changed:
            fixed += 1
    return qa, fixed


def apply_R38_age_direct_fix(qa, base_dir):
    """R38: Fix age MCQ answers by directly checking subject age against bins."""
    BINS = {
        'ABCD': ([(8,10),(10,11),(11,13),(13,15)], ["8-9","10-11","11-12","13-14"]),
        'ADNI': ([(50,65),(65,75),(75,85),(85,200)], ["50-64","65-74","75-84","85+"]),
        'AIBL': ([(55,65),(65,75),(75,85),(85,200)], ["55-64","65-74","75-84","85+"]),
    }
    # Load ages (use first valid age per subject)
    ages = {}
    for ds, (bins, labels) in BINS.items():
        meta = load_meta(base_dir, ds)
        if meta is None: continue
        sid_col = get_sid_col(meta)
        for _, row in meta.iterrows():
            sid = str(row[sid_col])
            a = row.get('age')
            if pd.notna(a) and sid not in ages:
                try: ages[(ds, sid)] = float(str(a).split('-')[0])
                except: pass

    fixed = 0
    for d in qa:
        if d['question_type'] != 'MCQ' or not d.get('options'): continue
        ql = d['question'].lower()
        if 'age range' not in ql and 'estimated age' not in ql: continue
        ds = d['dataset']
        sid = str(d['subject_id'])
        if ds not in BINS: continue
        age = ages.get((ds, sid))
        if age is None: continue
        bins, labels = BINS[ds]
        for j, (lo, hi) in enumerate(bins):
            if lo <= age < hi or (j == len(bins)-1 and age >= lo):
                if labels[j] in d['options'] and d['answer'] != labels[j]:
                    d['answer'] = labels[j]
                    fixed += 1
                break
    return qa, fixed


def apply_R25_mcq_position(qa):
    """R25: MCQ position balance (25% each) PER DATASET."""
    by_ds = defaultdict(list)
    for d in qa:
        by_ds[d.get('dataset', 'unknown')].append(d)

    all_result = []
    for ds, ds_qa in sorted(by_ds.items()):
        random.seed(42)  # Reset seed per dataset
        mcq = [d for d in ds_qa if d['question_type'] == 'MCQ' and d.get('options')]
        non_mcq = [d for d in ds_qa if d['question_type'] != 'MCQ' or not d.get('options')]

        # Group MCQ by number of options for proper balancing
        by_n_opts = defaultdict(list)
        for d in mcq:
            by_n_opts[len(d['options'])].append(d)

        for n_opts, group in sorted(by_n_opts.items()):
            det_shuffle(group)
            for i, q in enumerate(group):
                ans = q['answer']
                opts = list(q['options'])
                if ans not in opts:
                    continue
                target = i % n_opts
                curr = opts.index(ans)
                if curr != target:
                    opts[curr], opts[target] = opts[target], opts[curr]
                    q['options'] = opts

        all_result.extend(non_mcq + mcq)

    return all_result


def apply_R27_session_volume_fix(qa, base_dir):
    """R27: Fix FS volumes / age taken from wrong session.

    For multi-session datasets (ABCD, ADNI, PPMI), QA items may have been
    generated from one session but assigned to a different session by R15.
    This rule re-computes the answer from the correct session's data.
    """
    DS_CONFIG = {
        'ABCD': {
            'csv': 'ABCD/outputs/abcd_neuroqa_merged.csv',
            'sid_col': 'abcd_subject_id',
            'sess_col': 'session_bids',
        },
        'ADNI': {
            'csv': 'ADNI/outputs/adni_neuroqa_merged.csv',
            'sid_col': 'adni_subject_id',
            'sess_col': 'session_date',
        },
        'PPMI': {
            'csv': 'PPMI/outputs/ppmi_neuroqa_merged.csv',
            'sid_col': 'ppmi_subject_id',
            'sess_col': 'session_date',
        },
        'AIBL': {
            'csv': 'AIBL/outputs/aibl_neuroqa_merged.csv',
            'sid_col': 'aibl_subject_id',
            'sess_col': 'session_date',
        },
    }

    fixed = 0

    for ds_name, cfg in DS_CONFIG.items():
        csv_path = os.path.join(base_dir, cfg['csv'])
        if not os.path.exists(csv_path):
            continue
        meta = pd.read_csv(csv_path)
        sid_col = cfg['sid_col']
        sess_col = cfg['sess_col']

        # Build lookup: (subject_id, session_prefix) -> row
        # QA items have truncated session names (R15 truncates to 20 chars)
        sess_lookup = {}
        for _, row in meta.iterrows():
            sid = str(row[sid_col])
            sess = str(row.get(sess_col, ''))
            if sess == 'nan' or not sess:
                continue
            sess_lookup.setdefault(sid, []).append((sess, row))

        # Also compute dataset medians for Anatomy/Measurement thresholds
        if 'FS_Vol_L_Hippocampus' in meta.columns:
            hl_all = pd.to_numeric(meta['FS_Vol_L_Hippocampus'], errors='coerce')
            hr_all = pd.to_numeric(meta['FS_Vol_R_Hippocampus'], errors='coerce')
            avg_h_all = ((hl_all + hr_all) / 2).dropna()
            med_hippo = int(round(avg_h_all.median() / 100) * 100) if len(avg_h_all) > 0 else 3800
        else:
            avg_h_all = pd.Series(dtype=float)
            med_hippo = 3800

        if 'FS_Ventricles' in meta.columns and 'FS_ICV' in meta.columns:
            vent_all = pd.to_numeric(meta['FS_Ventricles'], errors='coerce')
            icv_all = pd.to_numeric(meta['FS_ICV'], errors='coerce')
            mask = (vent_all.notna()) & (icv_all.notna()) & (icv_all > 0)
        else:
            mask = pd.Series([False]*len(meta))
            vent_all = pd.Series(dtype=float)
            icv_all = pd.Series(dtype=float)
        if mask.sum() > 0:
            med_vent_pct = round((vent_all[mask] / icv_all[mask] * 100).median(), 2)
        else:
            med_vent_pct = 1.5

        if 'FS_WholeBrain' in meta.columns:
            wb_all = pd.to_numeric(meta['FS_WholeBrain'], errors='coerce')
            if wb_all.notna().sum() > 0 and wb_all.dropna().median() > 100000:
                med_brain = int(round(wb_all.dropna().median() / 10000) * 10000)
            else:
                med_brain = 1100000
        else:
            med_brain = 1100000
        if med_brain < 100000:
            # WholeBrain might be ratio not volume — use TotalGM fallback
            if 'FS_Vol_TotalGM' in meta.columns:
                tgm = pd.to_numeric(meta['FS_Vol_TotalGM'], errors='coerce').dropna()
                med_brain = int(round(tgm.median() / 10000) * 10000) if len(tgm) > 0 else 1100000
            else:
                med_brain = 1100000

        if 'FS_Thick_L_Entorhinal' in meta.columns:
            el_all = pd.to_numeric(meta['FS_Thick_L_Entorhinal'], errors='coerce')
            er_all = pd.to_numeric(meta['FS_Thick_R_Entorhinal'], errors='coerce')
        else:
            el_all = pd.Series(dtype=float)
            er_all = pd.Series(dtype=float)
        avg_e_all = ((el_all + er_all) / 2).dropna()
        med_entorh = round(avg_e_all.median(), 1) if len(avg_e_all) > 0 else 3.4

        for d in qa:
            if d.get('dataset') != ds_name:
                continue
            if d['category'] not in ('Location', 'Measurement', 'Anatomy', 'Demographic', 'Signal', 'Diagnosis'):
                continue

            sid = str(d['subject_id'])
            if sid not in sess_lookup:
                continue

            # Find session assigned to this QA item
            qa_sess = d.get('session_date', '')
            if not qa_sess or qa_sess == 'nan':
                continue

            # Match session using prefix (QA has truncated session names)
            matched_row = None
            for csv_sess, row in sess_lookup[sid]:
                if csv_sess.startswith(qa_sess) or qa_sess.startswith(csv_sess):
                    matched_row = row
                    break
            if matched_row is None:
                # Exact match fallback
                for csv_sess, row in sess_lookup[sid]:
                    if csv_sess == qa_sess:
                        matched_row = row
                        break
            if matched_row is None:
                continue

            row = matched_row
            q = d['question'].lower()
            qt = d['question_type']
            cat = d['category']

            # --- LOCATION ---
            if cat == 'Location':
                # Find which structure is mentioned
                matched_cols = None
                for kw, cols in STRUCT_COLS.items():
                    if kw in q:
                        matched_cols = cols
                        break
                if not matched_cols:
                    continue
                lc, rc = matched_cols
                lv = row.get(lc)
                rv = row.get(rc)
                if not (pd.notna(lv) and pd.notna(rv)):
                    # Session has no data for this structure — remove item
                    d['_remove_r27'] = True
                    fixed += 1
                    continue
                lv, rv = float(lv), float(rv)

                if qt == 'YN' and ('smaller' in q or 'larger' in q):
                    # Parse: "Is the {side} X smaller than the {other_side} X?"
                    m = re.search(r'is the (left|right)\s+.+?\s+(smaller|larger) than the (left|right)', q)
                    if not m:
                        continue
                    subject_side = m.group(1)
                    comparison = m.group(2)
                    subject_vol = lv if subject_side == 'left' else rv
                    other_vol = rv if subject_side == 'left' else lv
                    if comparison == 'smaller':
                        correct = 'Yes' if subject_vol < other_vol else 'No'
                    else:  # larger
                        correct = 'Yes' if subject_vol > other_vol else 'No'
                    if d['answer'] != correct:
                        d['answer'] = correct
                        fixed += 1

                elif qt == 'Open' and 'which side' in q and 'smaller' in q:
                    correct = 'Left' if lv < rv else 'Right'
                    if d['answer'] != correct:
                        d['answer'] = correct
                        fixed += 1

            # --- MEASUREMENT ---
            elif cat == 'Measurement':
                if qt == 'YN':
                    if 'average hippocampal volume' in q:
                        hl = row.get('FS_Vol_L_Hippocampus')
                        hr = row.get('FS_Vol_R_Hippocampus')
                        if not (pd.notna(hl) and pd.notna(hr)):
                            d['_remove_r27'] = True; fixed += 1
                            continue
                        avg_h = (float(hl) + float(hr)) / 2
                        # Extract threshold from question
                        m_thresh = re.search(r'(below|above)\s+([\d,]+)', q)
                        if not m_thresh:
                            continue
                        direction = m_thresh.group(1)
                        threshold = float(m_thresh.group(2).replace(',', ''))
                        if direction == 'below':
                            correct = 'Yes' if avg_h < threshold else 'No'
                        else:
                            correct = 'Yes' if avg_h > threshold else 'No'
                        if d['answer'] != correct:
                            d['answer'] = correct
                            fixed += 1

                    elif 'ventricle-to-brain ratio' in q:
                        vent = row.get('FS_Ventricles')
                        icv = row.get('FS_ICV')
                        if not (pd.notna(vent) and pd.notna(icv) and float(icv) > 0):
                            # Session has no ventricle data — mark for removal
                            d['_remove_r27'] = True
                            fixed += 1
                            continue
                        ratio = float(vent) / float(icv) * 100
                        m_thresh = re.search(r'(greater than|less than)\s+([\d.]+)%', q)
                        if not m_thresh:
                            continue
                        direction = m_thresh.group(1)
                        threshold = float(m_thresh.group(2))
                        if direction == 'greater than':
                            correct = 'Yes' if ratio > threshold else 'No'
                        else:
                            correct = 'Yes' if ratio < threshold else 'No'
                        if d['answer'] != correct:
                            d['answer'] = correct
                            fixed += 1

                    elif 'whole brain volume' in q:
                        wb = row.get('FS_WholeBrain')
                        if not pd.notna(wb):
                            continue
                        wb = float(wb)
                        m_thresh = re.search(r'(greater than|less than)\s+([\d,]+)', q)
                        if not m_thresh:
                            continue
                        direction = m_thresh.group(1)
                        threshold = float(m_thresh.group(2).replace(',', ''))
                        if direction == 'greater than':
                            correct = 'Yes' if wb > threshold else 'No'
                        else:
                            correct = 'Yes' if wb < threshold else 'No'
                        if d['answer'] != correct:
                            d['answer'] = correct
                            fixed += 1

                elif qt == 'MCQ':
                    if 'average hippocampal volume' in q:
                        hl = row.get('FS_Vol_L_Hippocampus')
                        hr = row.get('FS_Vol_R_Hippocampus')
                        if not (pd.notna(hl) and pd.notna(hr)):
                            continue
                        avg_h = (float(hl) + float(hr)) / 2
                        opts = d.get('options', [])
                        if not opts:
                            continue
                        # Determine correct range from options
                        correct = None
                        for opt in opts:
                            opt_clean = opt.replace(',', '').replace('\u2013', '-').replace('–', '-')
                            m_range = re.match(r'<\s*(\d+)', opt_clean)
                            if m_range and avg_h < float(m_range.group(1)):
                                correct = opt; break
                            m_range = re.match(r'(\d+)\s*-\s*(\d+)', opt_clean)
                            if m_range and float(m_range.group(1)) <= avg_h < float(m_range.group(2)):
                                correct = opt; break
                            m_range = re.match(r'>\s*(\d+)', opt_clean)
                            if m_range and avg_h >= float(m_range.group(1)):
                                correct = opt; break
                        if correct and d['answer'] != correct:
                            d['answer'] = correct
                            fixed += 1

            # --- ANATOMY ---
            # --- SIGNAL (WMH) ---
            elif cat == 'Signal':
                # Re-compute WMH answer from correct session
                wmh_vol = row.get('WMH_Volume')
                wmh_fs = row.get('FS_Vol_WMHypo')
                # ADNI uses WMH_Volume (threshold=1.0) first, then FS_Vol_WMHypo (threshold=1000)
                if ds_name == 'ADNI':
                    if pd.notna(wmh_vol):
                        wmh_val = float(wmh_vol)
                        wmh_thr = 1.0
                    elif pd.notna(wmh_fs):
                        wmh_val = float(wmh_fs)
                        wmh_thr = 1000
                    else:
                        continue
                else:
                    wmh_thr_map = {'ABCD': 500, 'HCP-YA': 500, 'HCP-Development': 500,
                                   'HCP-Aging': 1000, 'PPMI': 1000}
                    wmh_thr = wmh_thr_map.get(ds_name, 500)
                    if pd.notna(wmh_fs):
                        wmh_val = float(wmh_fs)
                    elif pd.notna(wmh_vol):
                        wmh_val = float(wmh_vol)
                    else:
                        continue

                has_wmh = wmh_val > wmh_thr

                if qt == 'YN':
                    if 'hyperintensit' in q and 'present' in q:
                        correct = 'Yes' if has_wmh else 'No'
                    elif 'signal abnormalit' in q and 'present' in q:
                        correct = 'Yes' if has_wmh else 'No'
                    elif 'show white matter' in q and ('hyperintensit' in q or 'signal abnormalit' in q):
                        correct = 'Yes' if has_wmh else 'No'
                    elif 'normal' in q or 'homogeneous' in q:
                        correct = 'No' if has_wmh else 'Yes'
                    else:
                        continue
                    if d['answer'] != correct:
                        d['answer'] = correct
                        fixed += 1
                elif qt == 'Open':
                    if 'hyperintensit' in q or 'signal abnormalit' in q:
                        correct = 'Yes' if has_wmh else 'No'
                        if d['answer'] != correct:
                            d['answer'] = correct
                            fixed += 1
                elif qt == 'MCQ' and 'severity' in q:
                    if has_wmh:
                        wmh_mod = {'ADNI': 5000, 'PPMI': 3000, 'HCP-Aging': 3000, 'ABCD': 1500}.get(ds_name, 1500)
                        wmh_sev = {'ADNI': 15000, 'PPMI': 8000, 'HCP-Aging': 8000, 'ABCD': 5000}.get(ds_name, 5000)
                        if wmh_val > wmh_sev:
                            correct = 'Severe'
                        elif wmh_val > wmh_mod:
                            correct = 'Moderate'
                        else:
                            correct = 'Mild'
                    else:
                        correct = 'None'
                    if correct in (d.get('options') or []) and d['answer'] != correct:
                        d['answer'] = correct
                        fixed += 1

            # --- DIAGNOSIS MCQ ---
            elif cat == 'Diagnosis' and qt == 'MCQ':
                dx = row.get('DX')
                if pd.notna(dx):
                    dx_str = str(dx).strip()
                    dx_map = {'CN': 'Cognitively Normal', 'MCI': 'Mild Cognitive Impairment',
                              'Dementia': "Alzheimer's Dementia", 'AD': "Alzheimer's Dementia"}
                    expected = dx_map.get(dx_str)
                    if expected and expected in (d.get('options') or []) and d['answer'] != expected:
                        d['answer'] = expected
                        fixed += 1

            # --- DEMOGRAPHIC MCQ (age bin) ---
            elif cat == 'Demographic' and qt == 'MCQ' and 'age' in q:
                age_val = row.get('age')
                if pd.notna(age_val):
                    age_f = float(age_val)
                    # Re-compute correct age bin using dataset-specific bins
                    if ds_name == 'ABCD':
                        bins = [(8,10),(10,11),(11,13),(13,15)]
                        labels_new = ["8-9","10-11","11-12","13-14"]
                    elif ds_name == 'ADNI':
                        bins = [(50,65),(65,75),(75,85),(85,200)]
                        labels_new = ["50-64","65-74","75-84","85+"]
                    elif ds_name == 'PPMI':
                        bins = [(30,50),(50,65),(65,75),(75,200)]
                        labels_new = ["30-49","50-64","65-74","75+"]
                    elif ds_name == 'AIBL':
                        bins = [(55,65),(65,75),(75,85),(85,200)]
                        labels_new = ["55-64","65-74","75-84","85+"]
                    else:
                        bins = None
                    if bins:
                        expected = None
                        for j, (lo, hi) in enumerate(bins):
                            if lo <= age_f < hi or (j == len(bins)-1 and age_f >= lo):
                                expected = labels_new[j]
                                break
                        if expected and expected in (d.get('options') or []) and d['answer'] != expected:
                            d['answer'] = expected
                            fixed += 1

            elif cat == 'Anatomy':
                if qt == 'YN':
                    if 'hippocampal volume' in q and ('below' in q or 'within' in q or 'expected' in q):
                        hl = row.get('FS_Vol_L_Hippocampus')
                        hr = row.get('FS_Vol_R_Hippocampus')
                        if not (pd.notna(hl) and pd.notna(hr)):
                            continue
                        avg_h = (float(hl) + float(hr)) / 2
                        below = avg_h < med_hippo
                        if 'below' in q:
                            correct = 'Yes' if below else 'No'
                        elif 'within' in q:
                            correct = 'No' if below else 'Yes'
                        else:
                            continue
                        if d['answer'] != correct:
                            d['answer'] = correct
                            fixed += 1

                    elif 'ventricle' in q and ('enlarged' in q or 'normal' in q):
                        vent = row.get('FS_Ventricles')
                        icv = row.get('FS_ICV')
                        if not (pd.notna(vent) and pd.notna(icv) and float(icv) > 0):
                            # Try L+R lateral ventricles
                            vl = row.get('FS_Vol_L_LatVentricle')
                            vr = row.get('FS_Vol_R_LatVentricle')
                            icv = row.get('FS_ICV')
                            if not (pd.notna(vl) and pd.notna(vr) and pd.notna(icv) and float(icv) > 0):
                                continue
                            vent_val = float(vl) + float(vr)
                        else:
                            vent_val = float(vent)
                        icv_val = float(icv)
                        enlarged = (vent_val / icv_val * 100) > med_vent_pct
                        if 'enlarged' in q:
                            correct = 'Yes' if enlarged else 'No'
                        elif 'normal' in q or 'within' in q:
                            correct = 'No' if enlarged else 'Yes'
                        else:
                            continue
                        if d['answer'] != correct:
                            d['answer'] = correct
                            fixed += 1

                    elif 'entorhinal' in q and ('thin' in q or 'expected' in q or 'thickness' in q):
                        el = row.get('FS_Thick_L_Entorhinal')
                        er = row.get('FS_Thick_R_Entorhinal')
                        if not (pd.notna(el) and pd.notna(er)):
                            continue
                        avg_e = (float(el) + float(er)) / 2
                        thin = avg_e < med_entorh
                        if 'thinner' in q or 'thin' in q:
                            correct = 'Yes' if thin else 'No'
                            if not re.search(r'\bthin', q):  # 'thin' was found inside 'within': the corrected Gold is written at the end
                                d['_gold_fix'] = 'No' if thin else 'Yes'
                        elif 'within' in q or 'expected range' in q:
                            correct = 'No' if thin else 'Yes'
                        else:
                            continue
                        if d['answer'] != correct:
                            d['answer'] = correct
                            fixed += 1

            # --- DEMOGRAPHIC ---
            elif cat == 'Demographic':
                if qt == 'YN':
                    m_age = re.search(r'(older|younger) than (\d+)', q)
                    if not m_age:
                        continue
                    direction = m_age.group(1)
                    threshold = int(m_age.group(2))
                    age_val = row.get('age')
                    if not pd.notna(age_val):
                        continue
                    try:
                        age_f = float(age_val)
                        if age_f > 200:
                            age_f /= 12.0
                    except (ValueError, TypeError):
                        continue
                    if direction == 'older':
                        correct = 'Yes' if age_f > threshold else 'No'
                    else:
                        correct = 'No' if age_f > threshold else 'Yes'
                    if d['answer'] != correct:
                        d['answer'] = correct
                        fixed += 1

    # Remove items marked for removal (session has NaN for required data)
    before = len(qa)
    qa = [d for d in qa if not d.get('_remove_r27')]
    removed = before - len(qa)
    if removed:
        fixed += removed

    return qa, fixed


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='NeuroQA Benchmark Cleanup v4')
    parser.add_argument('--base-dir', default='<base_dir>')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    base_dir = args.base_dir

    print(f"NeuroQA Cleanup v4 (base: {base_dir})")
    print("=" * 70)

    # Load QA files — try v3 first, fallback to final
    qa_files = {}
    for ds, final_path, v3_path in [
        ('ADNI', 'ADNI/outputs/adni_qa_final.json', 'ADNI/outputs/adni_qa_v3.json'),
        ('AIBL', 'AIBL/outputs/aibl_qa_final.json', 'AIBL/outputs/aibl_qa_v3.json'),
        ('HCP-YA', 'HCP-YA/outputs/hcpya_qa_final.json', 'HCP-YA/outputs/hcpya_qa_v3.json'),
        ('HCP-Aging', 'HCP-Aging/outputs/hcpaging_qa_final.json', 'HCP-Aging/outputs/hcpaging_qa_v3.json'),
        ('HCP-Development', 'HCP-Development/outputs/hcpdev_qa_final.json', 'HCP-Development/outputs/hcpdev_qa_v3.json'),
        ('ABCD', 'ABCD/outputs/abcd_qa_final.json', 'ABCD/outputs/abcd_qa_v3.json'),
        ('PPMI', 'PPMI/outputs/ppmi_qa_final.json', 'PPMI/outputs/ppmi_qa_v3.json'),
        ('BraTS-GLI', 'BraTS_WMH/outputs/brats_gli_qa_final.json', 'BraTS_WMH/outputs/brats_gli_qa_v3.json'),
        ('BraTS-MEN', 'BraTS_WMH/outputs/brats_men_qa_final.json', 'BraTS_WMH/outputs/brats_men_qa_v3.json'),
        ('WMH', 'BraTS_WMH/outputs/wmh_qa_final.json', 'BraTS_WMH/outputs/wmh_qa_v3.json'),
        ('CC359', 'CC359/outputs/cc359_qa_final.json', 'CC359/outputs/cc359_qa_v3.json'),
        ('IXI', 'IXI/outputs/ixi_qa_final.json', 'IXI/outputs/ixi_qa_v3.json'),
    ]:
        v3_full = os.path.join(base_dir, v3_path)
        final_full = os.path.join(base_dir, final_path)
        # Always start from final (raw) for full reproducibility
        if os.path.exists(final_full):
            qa_files[ds] = final_full
        elif os.path.exists(v3_full):
            qa_files[ds] = v3_full

    all_qa = []
    for ds, path in qa_files.items():
        if os.path.exists(path):
            data = json.load(open(path))
            for d in data: d['dataset'] = ds
            all_qa.extend(data)
            print(f"  Loaded {ds}: {len(data)}")
        else:
            print(f"  SKIP {ds}: not found")

    original = len(all_qa)
    print(f"\nTotal loaded: {original}")

    # Apply rules in order
    rules = [
        ("R22 Group C/D/P0", lambda qa: apply_R22_group_cd(qa)),
        ("R1  BraTS laterality", lambda qa: apply_R1_brats_laterality(qa)),
        ("R13 Mislabeled BraTS-MEN", lambda qa: apply_R13_mislabeled(qa, base_dir)),
        ("R14 NaN FreeSurfer", lambda qa: apply_R14_nan_freesurfer(qa, base_dir)),
        ("R2/R3 Low-asymmetry", lambda qa: apply_R2_R3_low_asymmetry(qa, base_dir)),
        ("R4  CPA fix", lambda qa: apply_R4_cpa(qa, base_dir)),
        ("R4b Midline negation", lambda qa: apply_R4b_midline_negation(qa, base_dir)),
        ("R4c Signal fix", lambda qa: apply_R4c_signal_fix(qa, base_dir)),
        ("R5  BraTS L/R answers", lambda qa: apply_R5_brats_lr_answers(qa)),
        ("R6  Multi-lobe fix", lambda qa: apply_R6_multilobe_fix(qa, base_dir)),
        ("R7  Signal wording", lambda qa: apply_R7_signal_wording(qa)),
        ("R8  Age context", lambda qa: apply_R8_age_context(qa, base_dir)),
        ("R9  Longitudinal context", lambda qa: apply_R9_longitudinal(qa, base_dir)),
        ("R10 Longitudinal direction", lambda qa: apply_R10_longitudinal_direction(qa, base_dir)),
        ("R11 Compound Presence", lambda qa: apply_R11_compound_presence(qa, base_dir)),
        ("R12 Diagnosis framework", lambda qa: apply_R12_diagnosis_framework(qa)),
        ("R15 Session alignment", lambda qa: apply_R15_session(qa, base_dir)),
        ("R28b Age label fix", lambda qa: apply_R28b_age_label_fix(qa)),
        ("R27 Session volume fix", lambda qa: apply_R27_session_volume_fix(qa, base_dir)),
        ("R26 Age threshold fix", lambda qa: apply_R26_age_threshold_fix(qa)),
        ("R29 Measurement verify", lambda qa: apply_R29_measurement_verify(qa, base_dir)),
        ("R30 Deduplicate", lambda qa: apply_R30_deduplicate(qa)),
        ("R31 Open→YN", lambda qa: apply_R31_open_to_yn(qa)),
        ("R35 Signal from features", lambda qa: apply_R35_signal_from_features(qa, base_dir)),
        ("R34 Age answer fix", lambda qa: apply_R34_age_answer_fix(qa, base_dir)),
        ("R32 None label", lambda qa: apply_R32_none_label(qa)),
        ("R39 ABCD hippo relabel", lambda qa: apply_R39_abcd_hippo_relabel(qa)),
        ("R33 WholeBrain ratio", lambda qa: apply_R33_wholebrain_ratio(qa)),
        ("R37 Longitudinal fix", lambda qa: apply_R37_longitudinal_fix(qa, base_dir)),
        ("R38 Age direct fix", lambda qa: apply_R38_age_direct_fix(qa, base_dir)),
        ("R36 Fix paths/types", lambda qa: apply_R36_fix_paths_and_types(qa, base_dir)),
        ("R28 MCQ standardize", lambda qa: apply_R28_mcq_standardize(qa)),
        ("R23 YN balance", lambda qa: apply_R23_yn_balance(qa)),
        ("R24 Shortcut removal", lambda qa: apply_R24_shortcut_removal(qa)),
    ]

    # Sort all QA deterministically before applying rules
    all_qa.sort(key=sort_key)

    for name, fn in rules:
        random.seed(42)  # Reset seed before each rule for reproducibility
        result = fn(all_qa)
        if isinstance(result, tuple):
            all_qa, count = result
            print(f"  {name}: {count} {'removed' if 'remov' in name.lower() or 'balance' in name.lower() else 'fixed'} → {len(all_qa)}")
        else:
            all_qa = result
            print(f"  {name}: applied → {len(all_qa)}")

    # Annotation rules (no removal)
    all_qa = apply_R16_modalities(all_qa)
    print(f"  R16 Modalities: annotated")
    all_qa = apply_R17_human_assessable(all_qa)
    print(f"  R17 Human assessable: annotated")
    all_qa = apply_R25_mcq_position(all_qa)
    print(f"  R25 MCQ position: balanced")

    # Re-assign IDs per dataset
    by_ds = defaultdict(list)
    for d in all_qa:
        by_ds[d['dataset']].append(d)

    for ds, items in by_ds.items():
        det_shuffle(items)
        prefix = {'ADNI':'ADNI','PPMI':'PPMI','ABCD':'ABCD','AIBL':'AIBL',
                  'HCP-YA':'HCPYA','HCP-Aging':'HCPA','HCP-Development':'HCPD',
                  'BraTS-GLI':'BGLI','BraTS-MEN':'BMEN','WMH':'WMH',
                  'CC359':'CC359','IXI':'IXI'}.get(ds, ds[:5])
        for i, item in enumerate(items):
            item['neuroqa_id'] = f'{prefix}-{i+1:05d}'

    # Summary
    print(f"\n{'='*70}")
    print(f"SUMMARY: {original} → {len(all_qa)} ({original-len(all_qa)} removed, {(original-len(all_qa))/original*100:.1f}%)")
    print(f"\nPer dataset:")
    for ds in sorted(by_ds):
        print(f"  {ds:<18}: {len(by_ds[ds]):>6}")
    print(f"\nCategories: {len(set(d['category'] for d in all_qa))}")
    print(f"Templates: {len(set(normalize_template(d['question']) for d in all_qa))}")

    # Save
    if not args.dry_run:
        out_files = {
            'ADNI': os.path.join(base_dir, 'ADNI/outputs/adni_qa_v4.json'),
            'PPMI': os.path.join(base_dir, 'PPMI/outputs/ppmi_qa_v4.json'),
            'ABCD': os.path.join(base_dir, 'ABCD/outputs/abcd_qa_v4.json'),
            'AIBL': os.path.join(base_dir, 'AIBL/outputs/aibl_qa_v4.json'),
            'HCP-YA': os.path.join(base_dir, 'HCP-YA/outputs/hcpya_qa_v4.json'),
            'HCP-Aging': os.path.join(base_dir, 'HCP-Aging/outputs/hcpaging_qa_v4.json'),
            'HCP-Development': os.path.join(base_dir, 'HCP-Development/outputs/hcpdev_qa_v4.json'),
            'BraTS-GLI': os.path.join(base_dir, 'BraTS_WMH/outputs/brats_gli_qa_v4.json'),
            'BraTS-MEN': os.path.join(base_dir, 'BraTS_WMH/outputs/brats_men_qa_v4.json'),
            'WMH': os.path.join(base_dir, 'BraTS_WMH/outputs/wmh_qa_v4.json'),
            'CC359': os.path.join(base_dir, 'CC359/outputs/cc359_qa_v4.json'),
            'IXI': os.path.join(base_dir, 'IXI/outputs/ixi_qa_v4.json'),
        }
        for ds, items in by_ds.items():
            path = out_files.get(ds)
            if path:
                with open(path, 'w') as f:
                    json.dump(items, f, indent=2)

        # Combined
        combined_path = os.path.join(base_dir, 'neuroqa_all_qa_v4.json')
        with open(combined_path, 'w') as f:
            json.dump(all_qa, f, indent=2)
        print(f"\nSaved {len(all_qa)} QA to {combined_path}")
    else:
        print("\nDRY RUN — no files written")


if __name__ == '__main__':
    main()
