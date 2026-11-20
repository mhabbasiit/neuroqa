#!/usr/bin/env python3
"""
NeuroQA — Step 1: Build ABCD Metadata CSV

Merges ABCD T1/T2 scan metadata with sMRI parcellation (FreeSurfer),
NIH Toolbox cognitive scores, CBCL behavioral scores, and anthropometrics.

ID MAPPING:
  BIDS: sub-NDARINV<XXXXXXXX>  →  NDA: NDAR_INV<XXXXXXXX>
  BIDS: ses-baselineYear1Arm1 →  NDA: baseline_year_1_arm_1

NDA TSV NOTE:
  NDA tabulated data has a description row on line 2 (after the header).
  This script skips that row automatically.

PREREQUISITES:
  ABCD data access via NDA (https://nda.nih.gov/).

USAGE:
  python 01_build_abcd_metadata.py \
    --t1meta    <path/to/ABCD_T1_metadata.csv> \
    --t2meta    <path/to/ABCD_T2_metadata.csv> \
    --behavdata <path/to/nda_tabulated_data/> \
    --output    ../outputs/abcd_neuroqa_merged.csv
"""

import csv
import os
import argparse
from collections import defaultdict, Counter

# ── Argument parsing ──
parser = argparse.ArgumentParser(description='Build ABCD NeuroQA metadata CSV')
parser.add_argument('--t1meta', required=True, help='Path to ABCD_T1_metadata.csv')
parser.add_argument('--t2meta', default='', help='Path to ABCD_T2_metadata.csv (optional)')
parser.add_argument('--behavdata', required=True, help='Path to NDA tabulated data folder')
parser.add_argument('--output', default='../outputs/abcd_neuroqa_merged.csv', help='Output CSV path')
args = parser.parse_args()

T1_CSV = args.t1meta
T2_CSV = args.t2meta
BD = args.behavdata
OUTPUT_CSV = args.output

# ── Exclude list (QC failures: 38 abnormal plot + 1 registration issue) ──
# Format: (subject_id_bids, session_bids) — curated from visual QC
# === EXCLUDE LIST (manually curated low-quality scans) ===
# The original list contains ABCD subject IDs (sub-NDARINV...) identified
# during quality review. To respect the ABCD Data Use Agreement (NIMH DAC),
# these IDs are not redistributed here.
# To use the same exclusion list, populate `exclude_list.csv` in this scripts/
# directory with rows of  subject_id,session  (no header).
EXCLUDE_LIST = []
import csv as _csv
_excl_path = os.path.join(os.path.dirname(__file__), 'exclude_list.csv')
if os.path.exists(_excl_path):
    with open(_excl_path) as _f:
        for _row in _csv.reader(_f):
            if _row and not _row[0].startswith('#'):
                EXCLUDE_LIST.append((_row[0].strip(), _row[1].strip() if len(_row) > 1 else ''))
exclude_set = set(EXCLUDE_LIST)

# ── BIDS → NDA ID mapping ──
SESSION_MAP = {
    'ses-baselineYear1Arm1': 'baseline_year_1_arm_1',
    'ses-2YearFollowUpYArm1': '2_year_follow_up_y_arm_1',
    'ses-4YearFollowUpYArm1': '4_year_follow_up_y_arm_1',
}

def bids_to_nda_subject(bids_id):
    """sub-NDARINV<XXXXXXXX> → NDAR_INV<XXXXXXXX>"""
    s = bids_id.replace('sub-', '')
    return s[:4] + '_' + s[4:]

def bids_to_nda_session(bids_session):
    """ses-baselineYear1Arm1 → baseline_year_1_arm_1"""
    return SESSION_MAP.get(bids_session, bids_session)

# ── Helpers ──
sex_map = {'M': 'Male', 'F': 'Female', '1': 'Male', '2': 'Female'}

def safe_float(v):
    if v is None or str(v).strip() == '':
        return None
    try:
        return float(str(v).strip())
    except (ValueError, TypeError):
        return None

def strip_quotes(s):
    """Remove surrounding double quotes from NDA TSV fields."""
    if s and len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        return s[1:-1]
    return s

def load_nda_tsv(filepath):
    """Load NDA tabulated data file (TSV), skipping the description row (line 2).
    NDA files have quoted fields — we strip quotes from all values."""
    rows = []
    if not os.path.exists(filepath):
        print(f"  WARNING: File not found: {filepath}")
        return [], []
    with open(filepath, 'r') as f:
        reader = csv.reader(f, delimiter='\t')
        header = [strip_quotes(h) for h in next(reader)]
        next(reader, None)  # Skip description row (line 2)
        for row in reader:
            if len(row) == len(header):
                rows.append(dict(zip(header, [strip_quotes(v) for v in row])))
    return header, rows

def load_nda_by_subject_event(filepath):
    """Load NDA TSV and index by (subjectkey, eventname)."""
    by_key = {}
    header, rows = load_nda_tsv(filepath)
    for row in rows:
        key = (row.get('subjectkey', ''), row.get('eventname', ''))
        by_key[key] = row
    return by_key

# ── sMRI column mapping (NDA → standardized FS names) ──
FS_VOL_MAP = {
    'FS_Vol_L_Hippocampus': 'smri_vol_scs_hpuslh',
    'FS_Vol_R_Hippocampus': 'smri_vol_scs_hpusrh',
    'FS_Vol_L_Amygdala': 'smri_vol_scs_amygdalalh',
    'FS_Vol_R_Amygdala': 'smri_vol_scs_amygdalarh',
    'FS_Vol_L_Caudate': 'smri_vol_scs_caudatelh',
    'FS_Vol_R_Caudate': 'smri_vol_scs_caudaterh',
    'FS_Vol_L_Putamen': 'smri_vol_scs_putamenlh',
    'FS_Vol_R_Putamen': 'smri_vol_scs_putamenrh',
    'FS_Vol_L_Pallidum': 'smri_vol_scs_pallidumlh',
    'FS_Vol_R_Pallidum': 'smri_vol_scs_pallidumrh',
    'FS_Vol_L_Thalamus': 'smri_vol_scs_tplh',
    'FS_Vol_R_Thalamus': 'smri_vol_scs_tprh',
    'FS_Vol_L_Accumbens': 'smri_vol_scs_aal',       # nucleus accumbens left
    'FS_Vol_R_Accumbens': 'smri_vol_scs_aar',       # nucleus accumbens right
    'FS_Vol_L_LatVentricle': 'smri_vol_scs_ltventriclelh',
    'FS_Vol_R_LatVentricle': 'smri_vol_scs_ltventriclerh',
    'FS_ICV': 'smri_vol_scs_intracranialv',
    'FS_WholeBrain': 'smri_vol_scs_wholeb',
    'FS_Vol_TotalGM': 'smri_vol_scs_allventricles',
    'FS_Vol_WMHypo': 'smri_vol_scs_wmhint',
    'FS_BrainStem': 'smri_vol_scs_bstem',
    'FS_Ventricles': 'smri_vol_scs_latventricles',
}

FS_THICK_MAP = {
    'FS_Thick_L_Entorhinal': 'smri_thick_cdk_ehinallh',
    'FS_Thick_R_Entorhinal': 'smri_thick_cdk_ehinalrh',
    'FS_Thick_L_Fusiform': 'smri_thick_cdk_fusiformlh',
    'FS_Thick_R_Fusiform': 'smri_thick_cdk_fusiformrh',
    'FS_Thick_L_Precuneus': 'smri_thick_cdk_pclh',
    'FS_Thick_R_Precuneus': 'smri_thick_cdk_pcrh',
    'FS_Thick_L_SupTemporal': 'smri_thick_cdk_sutmlh',
    'FS_Thick_R_SupTemporal': 'smri_thick_cdk_sutmrh',
    'FS_Thick_L_InfTemporal': 'smri_thick_cdk_iftmlh',
    'FS_Thick_R_InfTemporal': 'smri_thick_cdk_iftmrh',
    'FS_Thick_L_PostCing': 'smri_thick_cdk_ptcatelh',
    'FS_Thick_R_PostCing': 'smri_thick_cdk_ptcaterh',
    'FS_Thick_L_SupFrontal': 'smri_thick_cdk_sufrlh',
    'FS_Thick_R_SupFrontal': 'smri_thick_cdk_sufrrh',
    'FS_Thick_L_SupParietal': 'smri_thick_cdk_supllh',
    'FS_Thick_R_SupParietal': 'smri_thick_cdk_suplrh',
    'FS_Thick_L_IsthmusCing': 'smri_thick_cdk_ihcatelh',
    'FS_Thick_R_IsthmusCing': 'smri_thick_cdk_ihcaterh',
    'FS_Thick_L_MedOrbFrontal': 'smri_thick_cdk_mobfrlh',
    'FS_Thick_R_MedOrbFrontal': 'smri_thick_cdk_mobfrrh',
}

# ── NIH Toolbox columns ──
NIH_COLS = {
    'NIH_PicVocab': 'nihtbx_picvocab_uncorrected',
    'NIH_Flanker': 'nihtbx_flanker_uncorrected',
    'NIH_List': 'nihtbx_list_uncorrected',
    'NIH_CardSort': 'nihtbx_cardsort_uncorrected',
    'NIH_Pattern': 'nihtbx_pattern_uncorrected',
    'NIH_PicSeqMem': 'nihtbx_picture_uncorrected',
    'NIH_Reading': 'nihtbx_reading_uncorrected',
    'NIH_FluidComp': 'nihtbx_fluidcomp_uncorrected',
    'NIH_CrystComp': 'nihtbx_cryst_uncorrected',
    'NIH_TotalComp': 'nihtbx_totalcomp_uncorrected',
}

# ── CBCL columns ──
CBCL_COLS = {
    'CBCL_Internalizing': 'cbcl_scr_syn_internal_t',
    'CBCL_Externalizing': 'cbcl_scr_syn_external_t',
    'CBCL_TotalProblems': 'cbcl_scr_syn_totprob_t',
    'CBCL_Anxious': 'cbcl_scr_syn_anxdep_t',
    'CBCL_Withdrawn': 'cbcl_scr_syn_withdep_t',
    'CBCL_Somatic': 'cbcl_scr_syn_somatic_t',
    'CBCL_Social': 'cbcl_scr_syn_social_t',
    'CBCL_Thought': 'cbcl_scr_syn_thought_t',
    'CBCL_Attention': 'cbcl_scr_syn_attention_t',
    'CBCL_RuleBreak': 'cbcl_scr_syn_rulebreak_t',
    'CBCL_Aggressive': 'cbcl_scr_syn_aggressive_t',
}

# ── Load T1 scans ──
print("Loading T1 scans...")
with open(T1_CSV) as f:
    t1_scans = list(csv.DictReader(f))
print(f"  Raw: {len(t1_scans)} scans")

t1_scans = [r for r in t1_scans if (r['subjectId'], r['sessionName']) not in exclude_set]
print(f"  After exclude: {len(t1_scans)} scans")

groups = defaultdict(list)
for r in t1_scans:
    groups[(r['subjectId'], r['sessionName'])].append(r)
scans = [g[0] for g in groups.values()]
scans.sort(key=lambda r: (r['subjectId'], r['sessionName']))
print(f"  After dedup: {len(scans)} scans, {len(set(r['subjectId'] for r in scans))} subjects")

# ── Load T2 metadata (optional) ──
t2_lookup = {}
if T2_CSV and os.path.exists(T2_CSV):
    print("\nLoading T2 scans...")
    with open(T2_CSV) as f:
        for r in csv.DictReader(f):
            t2_lookup[(r['subjectId'], r['sessionName'])] = r.get('MNI_Z_Cropped', '')
    print(f"  {len(t2_lookup)} T2 scans loaded")

# ── Load behavioral / clinical data ──
print("\nLoading NDA tabulated data...")

smri_path = os.path.join(BD, 'abcd_smrip10201.txt')
smri_by = load_nda_by_subject_event(smri_path)
print(f"  sMRI parcellation: {len(smri_by)} records")

nihtb_path = os.path.join(BD, 'abcd_tbss01.txt')
nihtb_by = load_nda_by_subject_event(nihtb_path)
print(f"  NIH Toolbox: {len(nihtb_by)} records")

cbcl_path = os.path.join(BD, 'abcd_cbcls01.txt')
cbcl_by = load_nda_by_subject_event(cbcl_path)
print(f"  CBCL (scored): {len(cbcl_by)} records")

ant_path = os.path.join(BD, 'abcd_ant01.txt')
ant_by = load_nda_by_subject_event(ant_path)
print(f"  Anthropometrics: {len(ant_by)} records")

mri_path = os.path.join(BD, 'abcd_mri01.txt')
mri_by = load_nda_by_subject_event(mri_path)
print(f"  MRI scanner info: {len(mri_by)} records")

# ── Output fields ──
output_fields = [
    'neuroqa_id', 'abcd_subject_id', 'nda_subject_id', 'session_bids', 'session_nda',
    'T1w_image_path', 'T2w_image_path',
    # Demographics
    'age', 'sex',
    'height_cm', 'weight_kg',
    'scanner_manufacturer', 'scanner_model', 'magnetic_field_strength',
    # Diagnosis (all healthy — cognitive group for stratification)
    'DX', 'cognitive_group',
    # NIH Toolbox scores
    'NIH_PicVocab', 'NIH_Flanker', 'NIH_List', 'NIH_CardSort',
    'NIH_Pattern', 'NIH_PicSeqMem', 'NIH_Reading',
    'NIH_FluidComp', 'NIH_CrystComp', 'NIH_TotalComp',
    # CBCL behavioral T-scores
    'CBCL_Internalizing', 'CBCL_Externalizing', 'CBCL_TotalProblems',
    'CBCL_Anxious', 'CBCL_Withdrawn', 'CBCL_Somatic',
    'CBCL_Social', 'CBCL_Thought', 'CBCL_Attention',
    'CBCL_RuleBreak', 'CBCL_Aggressive',
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
    'FS_Vol_WMHypo', 'FS_BrainStem',
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
    # Longitudinal
    'prev_NIH_TotalComp', 'prev_cognitive_group',
]

# ── Build output ──
print("\nBuilding merged dataset...")
rows_out = []
match_stats = Counter()

prev_by_subject = {}

for i, r in enumerate(scans, 1):
    sid_bids = r['subjectId']
    session_bids = r['sessionName']
    sid_nda = bids_to_nda_subject(sid_bids)
    session_nda = bids_to_nda_session(session_bids)

    out = {
        'neuroqa_id': f'ABCD-{i:05d}',
        'abcd_subject_id': sid_bids,
        'nda_subject_id': sid_nda,
        'session_bids': session_bids,
        'session_nda': session_nda,
        'T1w_image_path': r.get('MNI_Z_Cropped', ''),
        'T2w_image_path': t2_lookup.get((sid_bids, session_bids), ''),
        'age': r.get('age', ''),
        'sex': sex_map.get(str(r.get('sex', '')).strip(), r.get('sex', '')),
        'DX': 'Healthy',  # ABCD is a healthy pediatric cohort
    }

    # NDA match key
    nda_key = (sid_nda, session_nda)

    # ── sMRI parcellation (FreeSurfer volumes + thickness) ──
    smri_rec = smri_by.get(nda_key)
    if smri_rec:
        match_stats['smri'] += 1
        # Volumes
        for out_col, nda_col in FS_VOL_MAP.items():
            val = smri_rec.get(nda_col, '')
            if val and val.strip():
                out[out_col] = val.strip()
        # Compute total ventricles if not directly available
        if not out.get('FS_Ventricles'):
            lv = safe_float(out.get('FS_Vol_L_LatVentricle'))
            rv = safe_float(out.get('FS_Vol_R_LatVentricle'))
            if lv is not None and rv is not None:
                out['FS_Ventricles'] = str(lv + rv)
        # Thickness
        for out_col, nda_col in FS_THICK_MAP.items():
            val = smri_rec.get(nda_col, '')
            if val and val.strip():
                out[out_col] = val.strip()

    # ── NIH Toolbox ──
    nihtb_rec = nihtb_by.get(nda_key)
    if nihtb_rec:
        match_stats['nihtb'] += 1
        for out_col, nda_col in NIH_COLS.items():
            val = nihtb_rec.get(nda_col, '')
            if val and val.strip():
                out[out_col] = val.strip()

        # Cognitive group from NIH_TotalComp (3 groups)
        total_comp = safe_float(out.get('NIH_TotalComp'))
        if total_comp is not None:
            if total_comp >= 95:
                out['cognitive_group'] = 'Above'
            elif total_comp >= 85:
                out['cognitive_group'] = 'Average'
            else:
                out['cognitive_group'] = 'Below'

    # ── CBCL ──
    cbcl_rec = cbcl_by.get(nda_key)
    if cbcl_rec:
        match_stats['cbcl'] += 1
        for out_col, nda_col in CBCL_COLS.items():
            val = cbcl_rec.get(nda_col, '')
            if val and val.strip():
                out[out_col] = val.strip()

    # ── Anthropometrics ──
    ant_rec = ant_by.get(nda_key)
    if ant_rec:
        match_stats['anthro'] += 1
        out['height_cm'] = ant_rec.get('anthroheightcalc', '')
        out['weight_kg'] = ant_rec.get('anthroweightcalc', '')

    # ── MRI scanner info ──
    mri_rec = mri_by.get(nda_key)
    if mri_rec:
        match_stats['mri_scanner'] += 1
        out['scanner_manufacturer'] = mri_rec.get('mri_info_manufacturer', '')
        out['scanner_model'] = mri_rec.get('mri_info_manufacturersmn', '')
        out['magnetic_field_strength'] = mri_rec.get('mri_info_magneticfieldstrength', '')

    # ── Longitudinal ──
    prev = prev_by_subject.get(sid_bids)
    if prev:
        out['prev_NIH_TotalComp'] = prev.get('NIH_TotalComp', '')
        out['prev_cognitive_group'] = prev.get('cognitive_group', '')

    prev_by_subject[sid_bids] = out.copy()

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
unique_subs = set(r['abcd_subject_id'] for r in rows_out)
print(f"\n{'='*60}")
print(f"ABCD NeuroQA Merged CSV")
print(f"{'='*60}")
print(f"Total rows:      {len(rows_out)}")
print(f"Unique subjects: {len(unique_subs)}")
print(f"Columns:         {len(output_fields)}")

print(f"\nSession distribution:")
sess = Counter(r['session_nda'] for r in rows_out)
for s, c in sess.most_common():
    print(f"  {s}: {c}")

print(f"\nCognitive group distribution:")
cg = Counter(r['cognitive_group'] for r in rows_out if r['cognitive_group'])
for g, c in cg.most_common():
    print(f"  {g}: {c} ({100*c/len(rows_out):.1f}%)")

print(f"\nSex distribution:")
sx = Counter(r['sex'] for r in rows_out if r['sex'])
for s, c in sx.most_common():
    print(f"  {s}: {c} ({100*c/len(rows_out):.1f}%)")

print(f"\nScanner manufacturers:")
mfr = Counter(r['scanner_manufacturer'] for r in rows_out if r['scanner_manufacturer'])
for m, c in mfr.most_common():
    print(f"  {m}: {c} ({100*c/len(rows_out):.1f}%)")

print(f"\nMagnetic field strength:")
mfs = Counter(r['magnetic_field_strength'] for r in rows_out if r['magnetic_field_strength'])
for m, c in mfs.most_common():
    print(f"  {m}T: {c}")

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
