#!/usr/bin/env python3
"""
NeuroQA — Step 1: Build AIBL Metadata CSV

Merges AIBL T1 scan metadata with clinical scores (MMSE, CDR, memory),
demographics, APOE genotype, and diagnosis from multiple AIBL study files.

Session mapping: T1 metadata uses date-time (e.g. 2013-10-12_10_41_39.0),
clinical data uses VISCODE (bl, m18, m36, m54, m72). We use aibl_mri3meta
as the bridge: RID + VISCODE → EXAMDATE, then match by closest date.

NOTE: AIBL does NOT include FreeSurfer volumetric data. QA generation
uses clinical cognitive scores (MMSE, CDR, memory) instead of brain volumes.

PREREQUISITES:
  You must have AIBL data access (https://ida.loni.usc.edu/).
  Download the following from LONI IDA:
    - T1 scan metadata CSV (from your processing pipeline)
    - aibl_pdxconv*.csv      (diagnosis at each visit)
    - aibl_mmse*.csv         (MMSE scores)
    - aibl_cdr*.csv          (CDR scores)
    - aibl_neurobat*.csv     (neuropsychological battery — LIMMTOTAL, LDELTOTAL)
    - aibl_ptdemog*.csv      (demographics — gender, birth year)
    - aibl_apoeres*.csv      (APOE genotyping)
    - aibl_mri3meta*.csv     (MRI exam dates — VISCODE → date bridge)

USAGE:
  python 01_build_aibl_metadata.py \\
    --t1meta    <path/to/AIBL_T1_metadata.csv> \\
    --metadata  <path/to/aibl_metadata_dir/> \\
    --output    <path/to/aibl_neuroqa_merged.csv>
"""

import csv
import os
import argparse
from collections import defaultdict, Counter
from datetime import datetime, timedelta

# === ARGUMENT PARSING ===
parser = argparse.ArgumentParser(description='Build AIBL NeuroQA metadata CSV')
parser.add_argument('--t1meta', required=True, help='Path to AIBL T1 scan metadata CSV')
parser.add_argument('--t2meta', default='', help='Path to AIBL T2 scan metadata CSV (optional)')
parser.add_argument('--metadata', required=True, help='Path to AIBL metadata directory')
parser.add_argument('--output', default='../outputs/aibl_neuroqa_merged.csv', help='Output CSV path')
args = parser.parse_args()

INPUT_CSV = args.t1meta
T2_CSV = args.t2meta
META_DIR = args.metadata
OUTPUT_CSV = args.output

# Locate metadata files (handle date suffix in filenames)
def find_file(prefix):
    for f in os.listdir(META_DIR):
        if f.startswith(prefix) and f.endswith('.csv'):
            return os.path.join(META_DIR, f)
    return None

PDXCONV_CSV = find_file('aibl_pdxconv')
MMSE_CSV = find_file('aibl_mmse')
CDR_CSV = find_file('aibl_cdr')
NEUROBAT_CSV = find_file('aibl_neurobat')
PTDEMOG_CSV = find_file('aibl_ptdemog')
APOERES_CSV = find_file('aibl_apoeres')
MRI3META_CSV = find_file('aibl_mri3meta')

for name, path in [('pdxconv', PDXCONV_CSV), ('mmse', MMSE_CSV), ('cdr', CDR_CSV),
                    ('neurobat', NEUROBAT_CSV), ('ptdemog', PTDEMOG_CSV),
                    ('apoeres', APOERES_CSV), ('mri3meta', MRI3META_CSV)]:
    if path is None:
        print(f"WARNING: {name} file not found in {META_DIR}")
    else:
        print(f"Found {name}: {os.path.basename(path)}")

# === HELPER: date-to-VISCODE mapping via mri3meta ===
def parse_date(date_str, formats=None):
    """Parse date from various formats."""
    if not date_str or date_str == '-4':
        return None
    if formats is None:
        formats = ['%m/%d/%Y', '%Y-%m-%d']
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except:
            continue
    return None

def find_viscode(rid, scan_date_str, mri_meta_by_rid, max_days=180):
    """Map a T1 scan date to VISCODE using mri3meta as bridge."""
    try:
        target = datetime.strptime(scan_date_str[:10], '%Y-%m-%d')
    except:
        return None
    best_viscode, best_diff = None, timedelta(days=9999)
    for rec in mri_meta_by_rid.get(rid, []):
        if rec.get('MMCONDCT', '') == '0':  # MRI not conducted
            continue
        examdate = rec.get('EXAMDATE', '')
        d = parse_date(examdate)
        if d is None:
            continue
        diff = abs(target - d)
        if diff < best_diff:
            best_diff = diff
            best_viscode = rec.get('VISCODE', '')
            best = rec
    return best_viscode if best_viscode and best_diff <= timedelta(days=max_days) else None

def find_by_viscode(records, viscode):
    """Find record matching a specific VISCODE."""
    for rec in records:
        if rec.get('VISCODE', '') == viscode:
            return rec
    return None

def find_closest_date(records, target_date_str, max_days=365):
    """Fallback: match by closest EXAMDATE."""
    try:
        target = datetime.strptime(target_date_str[:10], '%Y-%m-%d')
    except:
        return None
    best, best_diff = None, timedelta(days=9999)
    for rec in records:
        d = parse_date(rec.get('EXAMDATE', ''))
        if d is None:
            continue
        diff = abs(target - d)
        if diff < best_diff:
            best_diff = diff
            best = rec
    return best if best and best_diff <= timedelta(days=max_days) else None

# === LOAD MRI3META (date → VISCODE bridge) ===
mri_meta_by_rid = defaultdict(list)  # RID → list of {VISCODE, EXAMDATE}
if MRI3META_CSV:
    with open(MRI3META_CSV) as f:
        for row in csv.DictReader(f):
            mri_meta_by_rid[row['RID']].append(row)
    print(f"MRI3META: {len(mri_meta_by_rid)} subjects")

# === LOAD CLINICAL DATA ===
def load_by_rid(csv_path, name):
    data = defaultdict(list)
    if csv_path is None:
        return data
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            data[row['RID']].append(row)
    print(f"{name}: {len(data)} subjects, {sum(len(v) for v in data.values())} records")
    return data

pdxconv_by_rid = load_by_rid(PDXCONV_CSV, 'PDXCONV')
mmse_by_rid = load_by_rid(MMSE_CSV, 'MMSE')
cdr_by_rid = load_by_rid(CDR_CSV, 'CDR')
neurobat_by_rid = load_by_rid(NEUROBAT_CSV, 'NEUROBAT')
apoeres_by_rid = load_by_rid(APOERES_CSV, 'APOERES')

# === LOAD DEMOGRAPHICS (baseline only) ===
ptdemog = {}
if PTDEMOG_CSV:
    with open(PTDEMOG_CSV) as f:
        for row in csv.DictReader(f):
            ptdemog[row['RID']] = row
    print(f"PTDEMOG: {len(ptdemog)} subjects")

# === LOAD T2 AVAILABILITY ===
t2_set = set()
if T2_CSV and os.path.exists(T2_CSV):
    with open(T2_CSV) as f:
        for row in csv.DictReader(f):
            sid = row.get('subjectId', '')
            sess = row.get('sessionName', '')
            t2_set.add((sid, sess))
    print(f"T2 metadata: {len(t2_set)} scans")

# === MAPPINGS ===
sex_map = {'1': 'Male', '2': 'Female'}
dx_map = {'1': 'Normal', '2': 'MCI', '3': 'AD'}

# === READ T1 SCANS, DEDUP PER (subject, session) ===
raw_rows = []
with open(INPUT_CSV) as f:
    for row in csv.DictReader(f):
        try:
            qc = float(row.get('overall_qc_score', '') or '0')
        except:
            qc = 0.0
        row['_qc'] = qc
        raw_rows.append(row)

print(f"\nTotal T1 scans: {len(raw_rows)}")

# Dedup: best QC per (subject, session)
groups = defaultdict(list)
for r in raw_rows:
    groups[(r['subjectId'], r['sessionName'])].append(r)
scans = [max(g, key=lambda x: x['_qc']) for g in groups.values()]
scans.sort(key=lambda r: (r['subjectId'], r['sessionName']))
print(f"After dedup (1 per session): {len(scans)}")

# === BUILD OUTPUT ===
output_fields = [
    'neuroqa_id', 'aibl_subject_id', 'session_date', 'T1w_image_path',
    'has_t2',
    # Demographics
    'age', 'sex', 'scanner_manufacturer', 'qc_score',
    # Diagnosis
    'DX', 'current_diagnosis',
    # Clinical scores
    'MMSE', 'CDR_Global',
    # Memory scores (neuropsychological battery)
    'LIMMTOTAL', 'LDELTOTAL',
    # APOE
    'APOE_genotype', 'APOE4_carrier',
    # Smoking
    'smoking_history',
]

rows_out = []
match_stats = Counter()

for i, r in enumerate(scans, 1):
    sid = r['subjectId']
    sdate = r['sessionName']
    demog = ptdemog.get(sid, {})

    out = {
        'neuroqa_id': f'AIBL-{i:05d}',
        'aibl_subject_id': sid,
        'session_date': sdate,
        'T1w_image_path': r.get('MNI_Z_Cropped', ''),
        'has_t2': 'Yes' if (sid, sdate) in t2_set else 'No',
        'age': r.get('age', ''),
        'sex': sex_map.get(str(r.get('sex', '')).strip(), r.get('sex', '')),
        'scanner_manufacturer': r.get('scanner_manufacturer', ''),
        'qc_score': f"{r['_qc']:.4f}" if r['_qc'] > 0 else '',
        'current_diagnosis': r.get('current_diagnosis', ''),
        'smoking_history': r.get('smoking_history', ''),
    }

    # --- Map scan date to VISCODE ---
    viscode = find_viscode(sid, sdate, mri_meta_by_rid, max_days=180)

    # --- DIAGNOSIS (pdxconv — uses VISCODE, no EXAMDATE) ---
    if viscode:
        m = find_by_viscode(pdxconv_by_rid.get(sid, []), viscode)
    else:
        m = None
    if m:
        match_stats['pdxconv'] += 1
        dxc = m.get('DXCURREN', '')
        out['DX'] = dx_map.get(dxc, '')
    else:
        # Fallback to current_diagnosis from T1 metadata
        cd = r.get('current_diagnosis', '')
        if cd in ['Normal', 'MCI', 'AD']:
            out['DX'] = cd
        else:
            out['DX'] = ''

    # --- MMSE (try VISCODE first, fallback to closest date) ---
    m = None
    if viscode:
        m = find_by_viscode(mmse_by_rid.get(sid, []), viscode)
    if m is None:
        m = find_closest_date(mmse_by_rid.get(sid, []), sdate, max_days=365)
    if m:
        match_stats['mmse'] += 1
        try:
            out['MMSE'] = str(int(float(m.get('MMSCORE', ''))))
        except:
            out['MMSE'] = ''
    else:
        out['MMSE'] = ''

    # --- CDR (try VISCODE first, fallback to closest date) ---
    m = None
    if viscode:
        m = find_by_viscode(cdr_by_rid.get(sid, []), viscode)
    if m is None:
        m = find_closest_date(cdr_by_rid.get(sid, []), sdate, max_days=365)
    if m:
        match_stats['cdr'] += 1
        out['CDR_Global'] = m.get('CDGLOBAL', '')
    else:
        out['CDR_Global'] = ''

    # --- Neuropsych battery (try VISCODE first, fallback to closest date) ---
    m = None
    if viscode:
        m = find_by_viscode(neurobat_by_rid.get(sid, []), viscode)
    if m is None:
        m = find_closest_date(neurobat_by_rid.get(sid, []), sdate, max_days=365)
    if m:
        match_stats['neurobat'] += 1
        out['LIMMTOTAL'] = m.get('LIMMTOTAL', '')
        out['LDELTOTAL'] = m.get('LDELTOTAL', '')
    else:
        out['LIMMTOTAL'] = ''
        out['LDELTOTAL'] = ''

    # --- APOE (baseline only) ---
    apoe_recs = apoeres_by_rid.get(sid, [])
    if apoe_recs:
        match_stats['apoeres'] += 1
        a = apoe_recs[0]  # Baseline
        g1 = a.get('APGEN1', '')
        g2 = a.get('APGEN2', '')
        if g1 and g2:
            try:
                g1, g2 = int(float(g1)), int(float(g2))
                out['APOE_genotype'] = f'e{min(g1,g2)}/e{max(g1,g2)}'
                out['APOE4_carrier'] = 'Yes' if 4 in (g1, g2) else 'No'
            except:
                out['APOE_genotype'] = ''
                out['APOE4_carrier'] = ''
        else:
            out['APOE_genotype'] = ''
            out['APOE4_carrier'] = ''
    else:
        out['APOE_genotype'] = ''
        out['APOE4_carrier'] = ''

    # Fill missing with empty
    for f in output_fields:
        if f not in out:
            out[f] = ''
    rows_out.append(out)

# === WRITE ===
os.makedirs(os.path.dirname(OUTPUT_CSV) or '.', exist_ok=True)
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=output_fields)
    writer.writeheader()
    writer.writerows(rows_out)

# === STATS ===
unique_subs = set(r['aibl_subject_id'] for r in rows_out)
print(f"\n=== FINAL CSV ===")
print(f"Total rows: {len(rows_out)}")
print(f"Unique subjects: {len(unique_subs)}")
print(f"Columns: {len(output_fields)}")

print(f"\nMatch rates:")
for src, cnt in match_stats.most_common():
    print(f"  {src}: {cnt}/{len(rows_out)} ({100*cnt/len(rows_out):.1f}%)")

print(f"\nDiagnosis groups:")
grp = Counter(r['DX'] for r in rows_out)
for g, c in grp.most_common():
    label = g if g else 'Unknown'
    print(f"  {label}: {c}")

print(f"\nSex distribution:")
for s in ['Male', 'Female']:
    n = sum(1 for r in rows_out if r['sex'] == s)
    print(f"  {s}: {n}")

print(f"\nScanner manufacturers:")
mfr = Counter(r['scanner_manufacturer'] for r in rows_out)
for m, c in mfr.most_common():
    print(f"  {m or 'Unknown'}: {c}")

print(f"\nAge: mean={sum(float(r['age']) for r in rows_out if r['age'])/len(rows_out):.1f}")

print(f"\nFill rates (columns with <100%):")
for col in output_fields:
    filled = sum(1 for r in rows_out if r.get(col, '').strip())
    pct = 100*filled/len(rows_out)
    if pct < 100:
        print(f"  {col}: {filled} ({pct:.1f}%)")

print(f"\nSaved: {OUTPUT_CSV}")
