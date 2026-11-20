#!/usr/bin/env python3
"""
NeuroQA — Step 1: Build HCP-Aging Metadata CSV

Merges HCP-Aging T1 scan metadata with FreeSurfer stats (parsed from per-subject
aseg.stats and aparc.stats files), NIH Toolbox cognitive composites (cogcomp01.txt),
individual NIH tests, MoCA (moca01.txt), and NEO-FFI personality (nffi01.txt).

ID MAPPING:
  T1 metadata uses subjectId with _V1_MR suffix (e.g., <subject>_V1_MR).
  NDA files use src_subject_id without suffix (e.g., <subject>_V1_MR).
  FS dirs use the full ID with _V1_MR suffix.

DATA SOURCES:
  - T1 metadata (user-created): HCP-Aging_T1_metadata.csv
  - T2 metadata (user-created): HCP-Aging_T2_metadata.csv
  - FreeSurfer: Raw-Structural/.../T1w/{subj}/stats/aseg.stats + lh/rh.aparc.stats
  - Cognitive composites: cogcomp01.txt (NDA format)
  - Individual NIH tests: dccs01, flanker01, lswmt01, orrt01, pcps01, tpvt01, psm01
  - MoCA: moca01.txt (NDA format)
  - NEO-FFI: nffi01.txt (NDA format)
  - Demographics: ndar_subject01.txt (NDA format)

USAGE:
  python 01_build_hcpaging_metadata.py \\
    --t1meta    <path/to/processed/Structural/metadata/> \\
    --metadata  <path/to/HCP-Aging/metadata/> \\
    --fsraw     <path/to/Raw-Structural/fmriresults01/> \\
    --output    ../outputs/hcpaging_neuroqa_merged.csv
"""

import csv
import os
import re
import argparse
from collections import Counter

# ── Argument parsing ──
parser = argparse.ArgumentParser(description='Build HCP-Aging NeuroQA metadata CSV')
parser.add_argument('--t1meta', required=True, help='Path to processed Structural metadata dir')
parser.add_argument('--metadata', required=True, help='Path to HCP-Aging NDA metadata dir')
parser.add_argument('--fsraw', required=True, help='Path to Raw-Structural/fmriresults01/ dir')
parser.add_argument('--output', default='../outputs/hcpaging_neuroqa_merged.csv', help='Output CSV path')
args = parser.parse_args()

T1_DIR = args.t1meta
META_DIR = args.metadata
FS_RAW = args.fsraw
OUTPUT_CSV = args.output

# ── Helpers ──
sex_map = {'1': 'Male', '2': 'Female', 'M': 'Male', 'F': 'Female'}

def safe_float(v):
    if v is None or str(v).strip() == '' or str(v).strip() == '""':
        return None
    try:
        return float(str(v).strip().strip('"'))
    except (ValueError, TypeError):
        return None


def load_nda(filepath, key_field='src_subject_id'):
    """Load NDA tab-separated file (skip description row 2), return dict keyed by key_field."""
    if not os.path.exists(filepath):
        print(f"  WARNING: {filepath} not found!")
        return {}
    by_key = {}
    with open(filepath, encoding='utf-8') as f:
        lines = f.readlines()
    if len(lines) < 3:
        return {}
    header = lines[0].strip().split('\t')
    header = [h.strip('"') for h in header]
    # Skip row 2 (description), start from row 3
    for line in lines[2:]:
        parts = line.strip().split('\t')
        rec = {}
        for i, h in enumerate(header):
            rec[h] = parts[i].strip('"') if i < len(parts) else ''
        key = rec.get(key_field, '').strip()
        if key:
            by_key[key] = rec
    return by_key


# ── FreeSurfer Stats Parsing ──
ASEG_STRUCTURES = {
    'Left-Hippocampus': 'FS_Vol_L_Hippocampus',
    'Right-Hippocampus': 'FS_Vol_R_Hippocampus',
    'Left-Amygdala': 'FS_Vol_L_Amygdala',
    'Right-Amygdala': 'FS_Vol_R_Amygdala',
    'Left-Caudate': 'FS_Vol_L_Caudate',
    'Right-Caudate': 'FS_Vol_R_Caudate',
    'Left-Putamen': 'FS_Vol_L_Putamen',
    'Right-Putamen': 'FS_Vol_R_Putamen',
    'Left-Pallidum': 'FS_Vol_L_Pallidum',
    'Right-Pallidum': 'FS_Vol_R_Pallidum',
    'Left-Thalamus-Proper': 'FS_Vol_L_Thalamus',
    'Right-Thalamus-Proper': 'FS_Vol_R_Thalamus',
    'Left-Accumbens-area': 'FS_Vol_L_Accumbens',
    'Right-Accumbens-area': 'FS_Vol_R_Accumbens',
    'Left-Lateral-Ventricle': 'FS_Vol_L_LatVentricle',
    'Right-Lateral-Ventricle': 'FS_Vol_R_LatVentricle',
    'WM-hypointensities': 'FS_Vol_WMHypo',
}

ASEG_MEASURES = {
    'EstimatedTotalIntraCranialVol': 'FS_ICV',
    'BrainSegVol': 'FS_WholeBrain',
    'CerebralWhiteMatterVol': 'FS_Vol_CerebralWM',
    'TotalGrayVol': 'FS_Vol_TotalGM',
}

APARC_REGIONS = [
    'entorhinal', 'fusiform', 'precuneus', 'superiortemporal',
    'inferiortemporal', 'posteriorcingulate', 'superiorfrontal',
    'superiorparietal', 'isthmuscingulate', 'medialorbitofrontal',
]

APARC_NAME_MAP = {
    'entorhinal': 'Entorhinal',
    'fusiform': 'Fusiform',
    'precuneus': 'Precuneus',
    'superiortemporal': 'SupTemporal',
    'inferiortemporal': 'InfTemporal',
    'posteriorcingulate': 'PostCing',
    'superiorfrontal': 'SupFrontal',
    'superiorparietal': 'SupParietal',
    'isthmuscingulate': 'IsthmusCing',
    'medialorbitofrontal': 'MedOrbFrontal',
}


def parse_aseg(filepath):
    """Parse aseg.stats file, return dict of standardized column names to values."""
    result = {}
    if not os.path.exists(filepath):
        return result
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            # Parse measure lines (comments): # Measure BrainSegVol, ...
            if line.startswith('# Measure'):
                for measure_name, out_col in ASEG_MEASURES.items():
                    if measure_name in line:
                        # Format: # Measure Name, ..., value, unit
                        parts = line.split(',')
                        if len(parts) >= 4:
                            val = parts[-2].strip()
                            fv = safe_float(val)
                            if fv is not None:
                                result[out_col] = str(fv)
                        break
            # Parse data lines: Index SegId NVoxels Volume_mm3 StructName ...
            elif not line.startswith('#') and line.strip():
                parts = line.split()
                if len(parts) >= 5:
                    struct_name = parts[4]
                    vol = parts[3]  # Volume_mm3
                    if struct_name in ASEG_STRUCTURES:
                        fv = safe_float(vol)
                        if fv is not None:
                            result[ASEG_STRUCTURES[struct_name]] = str(fv)
    return result


def parse_aparc(filepath, hemi):
    """Parse lh/rh.aparc.stats file, return dict of thickness columns."""
    result = {}
    prefix = 'L' if hemi == 'lh' else 'R'
    if not os.path.exists(filepath):
        return result
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line.startswith('#') or not line:
                continue
            parts = line.split()
            if len(parts) >= 5:
                region = parts[0]
                if region in APARC_REGIONS:
                    thick = safe_float(parts[4])  # ThickAvg
                    if thick is not None:
                        col_name = f'FS_Thick_{prefix}_{APARC_NAME_MAP[region]}'
                        result[col_name] = str(thick)
    return result


# ── NDA Column Mappings ──
# Cognitive composites from cogcomp01.txt
COG_COLS = {
    'NIH_FluidComp': 'nih_fluidcogcomp_unadjusted',
    'NIH_CrystComp': 'nih_crycogcomp_unadjusted',
    'NIH_TotalComp': 'nih_totalcogcomp_unadjusted',
    'NIH_FluidComp_AgeAdj': 'nih_fluidcogcomp_ageadjusted',
    'NIH_TotalComp_AgeAdj': 'nih_totalcogcomp_ageadjusted',
}

# Individual NIH tests (file → (src_col, out_col))
NIH_INDIVIDUAL = {
    'dccs01.txt': ('nih_dccs_unadjusted', 'NIH_CardSort'),
    'flanker01.txt': ('nih_flanker_unadjusted', 'NIH_Flanker'),
    'lswmt01.txt': ('age_corrected_standard_score', 'NIH_ListSort'),
    'orrt01.txt': ('tbx_reading_score', 'NIH_ReadEng'),
    'pcps01.txt': ('nih_patterncomp_unadjusted', 'NIH_ProcSpeed'),
    'tpvt01.txt': ('tbx_vocab_theta', 'NIH_PicVocab'),
    'psm01.txt': ('nih_picseq_unadjusted', 'NIH_PicSeq'),
}

# NEO-FFI from nffi01.txt
NEO_COLS = {
    'NEO_Agreeableness': 'neo2_score_ag',
    'NEO_Conscientiousness': 'neo2_score_co',
    'NEO_Extraversion': 'neo2_score_ex',
    'NEO_Neuroticism': 'neo2_score_ne',
    'NEO_Openness': 'neo2_score_op',
}

# ── Load T1 metadata ──
print("Loading T1 scans...")
t1_path = os.path.join(T1_DIR, 'HCP-Aging_T1_metadata.csv')
with open(t1_path) as f:
    scans = list(csv.DictReader(f))
print(f"  T1: {len(scans)} scans")

# Load T2 for has_t2 flag + path
t2_by_subject = {}
t2_path = os.path.join(T1_DIR, 'HCP-Aging_T2_metadata.csv')
if os.path.exists(t2_path):
    with open(t2_path) as f:
        for r in csv.DictReader(f):
            t2_by_subject[r['subjectId']] = r
    print(f"  T2: {len(t2_by_subject)} subjects")
t2_set = set(t2_by_subject.keys())

# ── Load NDA files ──
print("\nLoading NDA metadata files...")
cog_by = load_nda(os.path.join(META_DIR, 'cogcomp01.txt'))
print(f"  cogcomp01: {len(cog_by)} subjects")

moca_by = load_nda(os.path.join(META_DIR, 'moca01.txt'))
print(f"  moca01: {len(moca_by)} subjects")

neo_by = load_nda(os.path.join(META_DIR, 'nffi01.txt'))
print(f"  nffi01: {len(neo_by)} subjects")

ndar_by = load_nda(os.path.join(META_DIR, 'ndar_subject01.txt'))
print(f"  ndar_subject01: {len(ndar_by)} subjects")

# Load individual NIH tests
nih_individual_by = {}
for fname, (src_col, out_col) in NIH_INDIVIDUAL.items():
    by = load_nda(os.path.join(META_DIR, fname))
    nih_individual_by[out_col] = (by, src_col)
    print(f"  {fname}: {len(by)} subjects")

# ── Output fields ──
output_fields = [
    'neuroqa_id', 'hcpaging_subject_id', 'session_id',
    'T1w_image_path', 'T2w_image_path', 'has_t2',
    # Demographics
    'age', 'sex', 'site',
    # Diagnosis (all healthy — cognitive group for stratification)
    'DX', 'cognitive_group',
    # MoCA
    'MoCA_Total',
    # NIH Toolbox composites
    'NIH_FluidComp', 'NIH_CrystComp', 'NIH_TotalComp',
    'NIH_FluidComp_AgeAdj', 'NIH_TotalComp_AgeAdj',
    # NIH individual tests
    'NIH_PicSeq', 'NIH_CardSort', 'NIH_Flanker', 'NIH_ReadEng',
    'NIH_PicVocab', 'NIH_ProcSpeed', 'NIH_ListSort',
    # NEO-FFI personality
    'NEO_Agreeableness', 'NEO_Openness', 'NEO_Conscientiousness',
    'NEO_Neuroticism', 'NEO_Extraversion',
    # FreeSurfer Volumes
    'FS_Vol_L_Hippocampus', 'FS_Vol_R_Hippocampus',
    'FS_Vol_L_Amygdala', 'FS_Vol_R_Amygdala',
    'FS_Vol_L_Caudate', 'FS_Vol_R_Caudate',
    'FS_Vol_L_Putamen', 'FS_Vol_R_Putamen',
    'FS_Vol_L_Pallidum', 'FS_Vol_R_Pallidum',
    'FS_Vol_L_Thalamus', 'FS_Vol_R_Thalamus',
    'FS_Vol_L_Accumbens', 'FS_Vol_R_Accumbens',
    'FS_Vol_L_LatVentricle', 'FS_Vol_R_LatVentricle',
    'FS_Ventricles', 'FS_ICV', 'FS_WholeBrain', 'FS_Vol_TotalGM',
    'FS_Vol_WMHypo', 'FS_Vol_CerebralWM',
    # FreeSurfer Cortical Thickness
    'FS_Thick_L_Entorhinal', 'FS_Thick_R_Entorhinal',
    'FS_Thick_L_Fusiform', 'FS_Thick_R_Fusiform',
    'FS_Thick_L_Precuneus', 'FS_Thick_R_Precuneus',
    'FS_Thick_L_SupTemporal', 'FS_Thick_R_SupTemporal',
    'FS_Thick_L_InfTemporal', 'FS_Thick_R_InfTemporal',
    'FS_Thick_L_PostCing', 'FS_Thick_R_PostCing',
    'FS_Thick_L_SupFrontal', 'FS_Thick_R_SupFrontal',
    'FS_Thick_L_SupParietal', 'FS_Thick_R_SupParietal',
    'FS_Thick_L_IsthmusCing', 'FS_Thick_R_IsthmusCing',
    'FS_Thick_L_MedOrbFrontal', 'FS_Thick_R_MedOrbFrontal',
]

# ── Build output ──
print("\nBuilding merged dataset...")
rows_out = []
match_stats = Counter()

for i, r in enumerate(scans, 1):
    sid_full = r['subjectId']  # e.g., <subject>_V1_MR
    sid_bare = sid_full.replace('_V1_MR', '')  # e.g., <subject>_V1_MR

    out = {
        'neuroqa_id': f'HCPA-{i:05d}',
        'hcpaging_subject_id': sid_full,
        'session_id': 'V1',
        'T1w_image_path': r.get('MNI_Z_Cropped', ''),
        'T2w_image_path': t2_by_subject[sid_full].get('MNI_Z_Cropped', '') if sid_full in t2_by_subject else '',
        'has_t2': 'Yes' if sid_full in t2_set else 'No',
        'age': r.get('age', ''),
        'sex': sex_map.get(str(r.get('sex', '')).strip(), r.get('sex', '')),
        'DX': 'Healthy',
    }

    # ── Site from ndar_subject01 ──
    ndar_rec = ndar_by.get(sid_bare)
    if ndar_rec:
        match_stats['ndar'] += 1
        out['site'] = ndar_rec.get('site', '')

    # ── FreeSurfer from raw stats files ──
    aseg_path = os.path.join(FS_RAW, sid_full, 'T1w', sid_full, 'stats', 'aseg.stats')
    lh_aparc_path = os.path.join(FS_RAW, sid_full, 'T1w', sid_full, 'stats', 'lh.aparc.stats')
    rh_aparc_path = os.path.join(FS_RAW, sid_full, 'T1w', sid_full, 'stats', 'rh.aparc.stats')

    aseg_data = parse_aseg(aseg_path)
    if aseg_data:
        match_stats['freesurfer'] += 1
        out.update(aseg_data)
        # Compute total ventricles
        lv = safe_float(out.get('FS_Vol_L_LatVentricle'))
        rv = safe_float(out.get('FS_Vol_R_LatVentricle'))
        if lv is not None and rv is not None:
            out['FS_Ventricles'] = str(lv + rv)

    lh_data = parse_aparc(lh_aparc_path, 'lh')
    rh_data = parse_aparc(rh_aparc_path, 'rh')
    if lh_data:
        match_stats['aparc_lh'] += 1
        out.update(lh_data)
    if rh_data:
        match_stats['aparc_rh'] += 1
        out.update(rh_data)

    # ── Cognitive composites from cogcomp01 ──
    cog_rec = cog_by.get(sid_bare)
    if cog_rec:
        match_stats['cogcomp'] += 1
        for out_col, src_col in COG_COLS.items():
            val = cog_rec.get(src_col, '')
            if val and val.strip():
                out[out_col] = val.strip()

    # ── Individual NIH tests ──
    for out_col, (by_dict, src_col) in nih_individual_by.items():
        rec = by_dict.get(sid_bare)
        if rec:
            val = rec.get(src_col, '')
            if val and val.strip():
                out[out_col] = val.strip()

    # ── MoCA ──
    moca_rec = moca_by.get(sid_bare)
    if moca_rec:
        match_stats['moca'] += 1
        val = moca_rec.get('moca_total', '')
        if val and val.strip():
            out['MoCA_Total'] = val.strip()

    # ── NEO-FFI ──
    neo_rec = neo_by.get(sid_bare)
    if neo_rec:
        match_stats['neo_ffi'] += 1
        for out_col, src_col in NEO_COLS.items():
            val = neo_rec.get(src_col, '')
            if val and val.strip():
                out[out_col] = val.strip()

    # ── Cognitive group from NIH_TotalComp ──
    # HCP-Aging: mean=105.3, sd=10.5
    # Standard 85/115 → 2.4% Below, 79.2% Avg, 18.4% Above
    # Adapted thresholds for balanced stratification
    total_comp = safe_float(out.get('NIH_TotalComp'))
    if total_comp is not None:
        if total_comp >= 113:
            out['cognitive_group'] = 'Above'
        elif total_comp >= 98:
            out['cognitive_group'] = 'Average'
        else:
            out['cognitive_group'] = 'Below'

    # Fill missing with empty
    for f in output_fields:
        if f not in out:
            out[f] = ''
    rows_out.append(out)

# ── Write ──
os.makedirs(os.path.dirname(OUTPUT_CSV) or '.', exist_ok=True)
with open(OUTPUT_CSV, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=output_fields)
    writer.writeheader()
    writer.writerows(rows_out)

# ── Stats ──
unique_subs = set(r['hcpaging_subject_id'] for r in rows_out)
print(f"\n{'='*60}")
print(f"HCP-Aging NeuroQA Merged CSV")
print(f"{'='*60}")
print(f"Total rows:      {len(rows_out)}")
print(f"Unique subjects: {len(unique_subs)}")
print(f"Columns:         {len(output_fields)}")

print(f"\nCognitive group distribution:")
cg = Counter(r['cognitive_group'] for r in rows_out if r['cognitive_group'])
for g, c in cg.most_common():
    print(f"  {g}: {c} ({100*c/len(rows_out):.1f}%)")

print(f"\nSex distribution:")
sx = Counter(r['sex'] for r in rows_out if r['sex'])
for s, c in sx.most_common():
    print(f"  {s}: {c} ({100*c/len(rows_out):.1f}%)")

print(f"\nAge distribution (bins):")
age_bins = {'36-45': 0, '46-55': 0, '56-65': 0, '66-75': 0, '76-85': 0, '86+': 0}
for r in rows_out:
    age = safe_float(r.get('age'))
    if age is not None:
        if age < 46: age_bins['36-45'] += 1
        elif age < 56: age_bins['46-55'] += 1
        elif age < 66: age_bins['56-65'] += 1
        elif age < 76: age_bins['66-75'] += 1
        elif age < 86: age_bins['76-85'] += 1
        else: age_bins['86+'] += 1
for ab, c in age_bins.items():
    print(f"  {ab}: {c} ({100*c/len(rows_out):.1f}%)")

print(f"\nSite distribution:")
st = Counter(r['site'] for r in rows_out if r['site'])
for s, c in st.most_common():
    print(f"  {s}: {c} ({100*c/len(rows_out):.1f}%)")

print(f"\nMatch rates:")
for src, cnt in sorted(match_stats.items(), key=lambda x: -x[1]):
    print(f"  {src}: {cnt}/{len(rows_out)} ({100*cnt/len(rows_out):.1f}%)")

print(f"\nFill rates (columns with <100%):")
for col in output_fields:
    filled = sum(1 for r in rows_out if r.get(col, '').strip())
    pct = 100 * filled / len(rows_out)
    if pct < 100:
        print(f"  {col}: {filled} ({pct:.1f}%)")

print(f"\nSaved: {OUTPUT_CSV}")
