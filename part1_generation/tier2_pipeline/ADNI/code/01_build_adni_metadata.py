#!/usr/bin/env python3
"""
NeuroQA — Step 1: Build ADNI Metadata CSV

Merges ADNI T1 scan metadata with clinical scores, demographics, volumetric
measures, and biomarkers from multiple ADNI study files into a single CSV.

PREREQUISITES:
  You must have ADNI data access (https://adni.loni.usc.edu/).
  Download the following from LONI:
    - T1 scan metadata CSV (from your processing pipeline)
    - PTDEMOG*.csv          (participant demographics)
    - ADNIMERGE*.csv        (merged clinical/biomarker data)
    - UCSDVOL*.csv          (UCSD volumetric measures)
    - UCSFSNTVOL*.csv       (UCSF SNT hippocampal volumes)
    - BSI*.csv              (boundary shift integral volumes)
    - UCD_ADNI1_WMH*.csv   (white matter hyperintensity volumes)

USAGE:
  python 01_build_adni_metadata.py \\
    --input     <path/to/ADNI_T1_metadata.csv> \\
    --ptdemog   <path/to/PTDEMOG.csv> \\
    --extracted <path/to/extracted_study_files/> \\
    --output    <path/to/adni_neuroqa_merged.csv>
"""

import csv
import os
import argparse
from collections import defaultdict, Counter
from datetime import datetime, timedelta

# === ARGUMENT PARSING ===
parser = argparse.ArgumentParser(description='Build ADNI NeuroQA metadata CSV')
parser.add_argument('--input', required=True, help='Path to ADNI T1 scan metadata CSV')
parser.add_argument('--ptdemog', required=True, help='Path to PTDEMOG CSV')
parser.add_argument('--extracted', required=True, help='Path to extracted ADNI study files directory')
parser.add_argument('--output', default='../outputs/adni_neuroqa_merged.csv', help='Output CSV path')
args = parser.parse_args()

INPUT_CSV = args.input
PTDEMOG_CSV = args.ptdemog
EXT = args.extracted
MERGE_CSV = os.path.join(EXT, 'ADNIMERGE_02Apr2025.csv')
UCSDVOL_CSV = os.path.join(EXT, 'UCSDVOL_02Apr2025.csv')
UCSFSNTVOL_CSV = os.path.join(EXT, 'UCSFSNTVOL_02Apr2025.csv')
BSI_CSV = os.path.join(EXT, 'BSI_02Apr2025.csv')
WMH_CSV = os.path.join(EXT, 'UCD_ADNI1_WMH_02Apr2025.csv')
OUTPUT_CSV = args.output

# === EXCLUDE LIST (manually curated low-quality scans) ===
# The original list contains ADNI subject IDs identified during quality review.
# To respect the ADNI Data Use Agreement, these IDs are not redistributed here.
# To use the same exclusion list, populate `exclude_list.csv` in this scripts/
# directory with rows of  subject_id,scan_type  (no header).
EXCLUDE_LIST = []
import csv as _csv
_excl_path = os.path.join(os.path.dirname(__file__), 'exclude_list.csv')
if os.path.exists(_excl_path):
    with open(_excl_path) as _f:
        for _row in _csv.reader(_f):
            if _row and not _row[0].startswith('#'):
                EXCLUDE_LIST.append((_row[0].strip(), _row[1].strip() if len(_row) > 1 else ''))
exclude_set = set(EXCLUDE_LIST)

# === HELPER: closest date match ===
def find_closest(records, target_date_str, max_days=180):
    try:
        target = datetime.strptime(target_date_str, '%Y-%m-%d')
    except:
        return None
    best, best_diff = None, timedelta(days=9999)
    for rec in records:
        try:
            d = datetime.strptime(rec['EXAMDATE'], '%Y-%m-%d')
            diff = abs(target - d)
            if diff < best_diff:
                best_diff = diff
                best = rec
        except:
            pass
    return best if best and best_diff <= timedelta(days=max_days) else None

# === LOAD PTDEMOG ===
ptdemog = {}
with open(PTDEMOG_CSV) as f:
    for row in csv.DictReader(f):
        if row['PTID'] not in ptdemog:
            ptdemog[row['PTID']] = row
print(f"PTDEMOG: {len(ptdemog)} subjects")

# === LOAD ADNIMERGE (RID->PTID map + clinical) ===
rid_to_ptid = {}
merge_by_ptid = defaultdict(list)
with open(MERGE_CSV) as f:
    for row in csv.DictReader(f):
        rid_to_ptid[row['RID']] = row['PTID']
        merge_by_ptid[row['PTID']].append(row)
print(f"ADNIMERGE: {len(merge_by_ptid)} subjects")

# === LOAD UCSDVOL (L/R hippocampus, ventricles, entorhinal) ===
ucsdvol_by_ptid = defaultdict(list)
with open(UCSDVOL_CSV) as f:
    for row in csv.DictReader(f):
        ptid = rid_to_ptid.get(row['RID'], '')
        if ptid:
            ucsdvol_by_ptid[ptid].append(row)
print(f"UCSDVOL: {len(ucsdvol_by_ptid)} subjects")

# === LOAD UCSFSNTVOL (L/R hippocampus SNT) ===
snt_by_ptid = defaultdict(list)
with open(UCSFSNTVOL_CSV) as f:
    for row in csv.DictReader(f):
        ptid = rid_to_ptid.get(row['RID'], '')
        if ptid:
            snt_by_ptid[ptid].append(row)
print(f"UCSFSNTVOL: {len(snt_by_ptid)} subjects")

# === LOAD BSI (brain/ventricle volumes + atrophy) ===
bsi_by_ptid = defaultdict(list)
with open(BSI_CSV) as f:
    for row in csv.DictReader(f):
        ptid = rid_to_ptid.get(row['RID'], '')
        if ptid:
            bsi_by_ptid[ptid].append(row)
print(f"BSI: {len(bsi_by_ptid)} subjects")

# === LOAD WMH ===
wmh_by_ptid = defaultdict(list)
with open(WMH_CSV) as f:
    for row in csv.DictReader(f):
        ptid = rid_to_ptid.get(row['RID'], '')
        if ptid:
            wmh_by_ptid[ptid].append(row)
print(f"WMH: {len(wmh_by_ptid)} subjects")

# === MAPPINGS ===
sex_map = {'1': 'Male', '2': 'Female'}
hand_map = {'1': 'Right', '2': 'Left', '3': 'Ambidextrous'}
eth_map = {'1': 'Hispanic/Latino', '2': 'Not Hispanic/Latino', '3': 'Unknown'}
race_map = {'1': 'American Indian/Alaska Native', '2': 'Asian', '3': 'Native Hawaiian/Pacific Islander',
            '4': 'Black/African American', '5': 'White', '6': 'More than one race', '7': 'Unknown'}
marry_map = {'1': 'Married', '2': 'Widowed', '3': 'Divorced', '4': 'Never married', '5': 'Unknown'}

# === READ ALL SCANS, DEDUP PER (subject, session) ===
raw_rows = []
excluded_count = 0
with open(INPUT_CSV) as f:
    for row in csv.DictReader(f):
        try:
            qc = float(row['overall_qc_score'])
        except:
            qc = 0.0
        old_path = row['MNI_Z_Cropped']
        parts = old_path.split('/')
        seq_type = parts[-3] if len(parts) >= 3 else ''
        if (row['subjectId'], seq_type) in exclude_set:
            excluded_count += 1
            continue
        row['_qc'] = qc
        row['_seq_type'] = seq_type
        row['_new_path'] = os.path.join(os.path.dirname(old_path), 'T1w_mni_zscore_fixed_cropped.nii.gz')
        raw_rows.append(row)

print(f"\nTotal scans after exclude: {len(raw_rows)} (excluded: {excluded_count})")

# Dedup: best QC per (subject, session)
groups = defaultdict(list)
for r in raw_rows:
    groups[(r['subjectId'], r['sessionName'])].append(r)
scans = [max(g, key=lambda x: x['_qc']) for g in groups.values()]
scans.sort(key=lambda r: (r['subjectId'], r['sessionName']))
print(f"After dedup (1 per session): {len(scans)}")

# === BUILD OUTPUT ===
output_fields = [
    'neuroqa_id', 'adni_subject_id', 'session_date', 'sequence_type',
    'T1w_image_path',
    # Demographics
    'age', 'sex', 'handedness', 'ethnicity', 'race',
    'education_years', 'marital_status',
    'scanner_manufacturer', 'qc_score',
    # Diagnosis
    'research_group', 'DX',
    # Clinical scores (ADNIMERGE)
    'CDR_SOB', 'MMSE', 'MoCA', 'ADAS13', 'FAQ',
    'APOE4',
    # CSF biomarkers (ADNIMERGE)
    'ABETA', 'TAU', 'PTAU',
    # FreeSurfer from ADNIMERGE (total)
    'FS_Hippocampus', 'FS_Ventricles', 'FS_WholeBrain',
    'FS_Entorhinal', 'FS_Fusiform', 'FS_MidTemp', 'FS_ICV',
    # UCSD volumetric (L/R separate!)
    'UCSD_Left_Hippocampus', 'UCSD_Right_Hippocampus',
    'UCSD_Ventricles', 'UCSD_Brain', 'UCSD_EICV',
    'UCSD_Left_Entorhinal', 'UCSD_Right_Entorhinal',
    'UCSD_Left_Fusiform', 'UCSD_Right_Fusiform',
    'UCSD_Left_MidTemp', 'UCSD_Right_MidTemp',
    'UCSD_Left_InfLatVent', 'UCSD_Right_InfLatVent',
    # UCSF SNT hippocampus
    'SNT_Left_Hippocampus', 'SNT_Right_Hippocampus',
    # BSI atrophy
    'BSI_BrainVol', 'BSI_VentVol', 'BSI_BBSI', 'BSI_VBSI',
    # WMH
    'WMH_Volume',
]

rows_out = []
match_stats = Counter()

for i, r in enumerate(scans, 1):
    sid = r['subjectId']
    sdate = r['sessionName']
    demog = ptdemog.get(sid, {})

    out = {
        'neuroqa_id': f'ADNI-{i:05d}',
        'adni_subject_id': sid,
        'session_date': sdate,
        'sequence_type': r['_seq_type'],
        'T1w_image_path': r['MNI_Z_Cropped'],  # the file from your sMRI pipeline
        'age': r['age'],
        'sex': sex_map.get(str(r['sex']).strip(), r['sex']),
        'handedness': hand_map.get(str(demog.get('PTHAND', '')).strip(), ''),
        'ethnicity': eth_map.get(str(demog.get('PTETHCAT', '')).strip(), ''),
        'race': race_map.get(str(demog.get('PTRACCAT', '')).strip(), ''),
        'education_years': '',
        'marital_status': marry_map.get(str(demog.get('PTMARRY', '')).strip(), ''),
        'scanner_manufacturer': r['scanner_manufacturer'],
        'qc_score': f"{r['_qc']:.4f}",
        'research_group': r['research_group'],
    }
    edu = r.get('education', '')
    try:
        out['education_years'] = str(int(float(edu))) if edu else ''
    except:
        pass

    # --- ADNIMERGE ---
    m = find_closest(merge_by_ptid.get(sid, []), sdate)
    if m:
        match_stats['adnimerge'] += 1
        out['DX'] = m.get('DX', '')
        out['CDR_SOB'] = m.get('CDRSB', '')
        out['MMSE'] = m.get('MMSE', '')
        out['MoCA'] = m.get('MOCA', '')
        out['ADAS13'] = m.get('ADAS13', '')
        out['FAQ'] = m.get('FAQ', '')
        out['APOE4'] = m.get('APOE4', '')
        out['ABETA'] = m.get('ABETA', '')
        out['TAU'] = m.get('TAU', '')
        out['PTAU'] = m.get('PTAU', '')
        out['FS_Hippocampus'] = m.get('Hippocampus', '')
        out['FS_Ventricles'] = m.get('Ventricles', '')
        out['FS_WholeBrain'] = m.get('WholeBrain', '')
        out['FS_Entorhinal'] = m.get('Entorhinal', '')
        out['FS_Fusiform'] = m.get('Fusiform', '')
        out['FS_MidTemp'] = m.get('MidTemp', '')
        out['FS_ICV'] = m.get('ICV', '')
        if m.get('DX', '').strip():
            out['research_group'] = m['DX']

    # --- UCSDVOL (L/R) ---
    uv = find_closest(ucsdvol_by_ptid.get(sid, []), sdate)
    if uv:
        match_stats['ucsdvol'] += 1
        out['UCSD_Left_Hippocampus'] = uv.get('LHIPPOC', '')
        out['UCSD_Right_Hippocampus'] = uv.get('RHIPPOC', '')
        out['UCSD_Ventricles'] = uv.get('VENTRICLES', '')
        out['UCSD_Brain'] = uv.get('BRAIN', '')
        out['UCSD_EICV'] = uv.get('EICV', '')
        out['UCSD_Left_Entorhinal'] = uv.get('LENTORHIN', '')
        out['UCSD_Right_Entorhinal'] = uv.get('RENTORHIN', '')
        out['UCSD_Left_Fusiform'] = uv.get('LFUSIFORM', '')
        out['UCSD_Right_Fusiform'] = uv.get('RFUSIFORM', '')
        out['UCSD_Left_MidTemp'] = uv.get('LMIDTEMP', '')
        out['UCSD_Right_MidTemp'] = uv.get('RMIDTEMP', '')
        out['UCSD_Left_InfLatVent'] = uv.get('LINFLATVEN', '')
        out['UCSD_Right_InfLatVent'] = uv.get('RINFLATVEN', '')

    # --- UCSF SNT ---
    sn = find_closest(snt_by_ptid.get(sid, []), sdate)
    if sn:
        match_stats['snt'] += 1
        out['SNT_Left_Hippocampus'] = sn.get('LEFTHIPPO', '')
        out['SNT_Right_Hippocampus'] = sn.get('RIGHTHIPPO', '')

    # --- BSI ---
    bs = find_closest(bsi_by_ptid.get(sid, []), sdate)
    if bs:
        match_stats['bsi'] += 1
        out['BSI_BrainVol'] = bs.get('BRAINVOL', '')
        out['BSI_VentVol'] = bs.get('VENTVOL', '')
        out['BSI_BBSI'] = bs.get('BBSI', '')
        out['BSI_VBSI'] = bs.get('VBSI', '')

    # --- WMH ---
    wm = find_closest(wmh_by_ptid.get(sid, []), sdate)
    if wm:
        match_stats['wmh'] += 1
        out['WMH_Volume'] = wm.get('WHITMATHYP', '')

    # Fill missing with empty
    for f in output_fields:
        if f not in out:
            out[f] = ''
    rows_out.append(out)

# === WRITE ===
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=output_fields)
    writer.writeheader()
    writer.writerows(rows_out)

# === STATS ===
unique_subs = set(r['adni_subject_id'] for r in rows_out)
print(f"\n=== FINAL CSV ===")
print(f"Total rows: {len(rows_out)}")
print(f"Unique subjects: {len(unique_subs)}")
print(f"Columns: {len(output_fields)}")

print(f"\nMatch rates:")
for src, cnt in match_stats.most_common():
    print(f"  {src}: {cnt}/{len(rows_out)} ({100*cnt/len(rows_out):.1f}%)")

print(f"\nResearch groups:")
grp = Counter(r['research_group'] for r in rows_out)
for g, c in grp.most_common():
    print(f"  {g}: {c}")

print(f"\nFill rates (columns with <100%):")
for col in output_fields:
    filled = sum(1 for r in rows_out if r.get(col, '').strip())
    pct = 100*filled/len(rows_out)
    if pct < 100:
        print(f"  {col}: {filled} ({pct:.1f}%)")

print(f"\nSaved: {OUTPUT_CSV}")
