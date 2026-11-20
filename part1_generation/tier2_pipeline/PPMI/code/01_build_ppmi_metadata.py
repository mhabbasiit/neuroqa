#!/usr/bin/env python3
"""NeuroQA step 1 for PPMI: builds the merged table (one row per subject and imaging session) that 02_generate_ppmi_qa.py reads.

Inputs:
  --t1meta     PPMI_T1_metadata.csv from the sMRI preprocessing pipeline (subjectId, sessionName, MNI_Z_Cropped, age, sex, research_group)
  --t2meta     PPMI_T2_metadata.csv from the same pipeline (optional; subjectId, sessionName, MNI_Z_Cropped)
  --studydata  the LONI PPMI "study data" folder with MDS-UPDRS_Part_III*.csv, Montreal_Cognitive_Assessment__MoCA_*.csv,
               FS7_ASEG_VOL*.csv and FS7_APARC_CTH*.csv
Rules:
  - Sessions are the rows of the T1 metadata; session_date is the first 10 characters of sessionName.
  - UPDRS-III total, Hoehn and Yahr and MoCA come from the visit nearest the session date, within 365 days (INFODT is month/year; ties keep file order).
  - FreeSurfer values come from the LONI FS7 tables, which hold one (baseline) row per subject; every session of a subject uses it.
  - FS_Ventricles is the left and right lateral, 3rd and 4th ventricle volumes added.
  - neuroqa_id numbers the rows in the order of subject ID (as text) and session date.
"""
import argparse, glob, os, sys
sys.dont_write_bytecode = True
import pandas as pd

ASEG = {'FS_Vol_L_Hippocampus': 'Left_Hippocampus', 'FS_Vol_R_Hippocampus': 'Right_Hippocampus',
        'FS_Vol_L_Amygdala': 'Left_Amygdala', 'FS_Vol_R_Amygdala': 'Right_Amygdala',
        'FS_Vol_L_Caudate': 'Left_Caudate', 'FS_Vol_R_Caudate': 'Right_Caudate',
        'FS_Vol_L_Putamen': 'Left_Putamen', 'FS_Vol_R_Putamen': 'Right_Putamen',
        'FS_Vol_L_Pallidum': 'Left_Pallidum', 'FS_Vol_R_Pallidum': 'Right_Pallidum',
        'FS_Vol_L_Thalamus': 'Left_Thalamus', 'FS_Vol_R_Thalamus': 'Right_Thalamus',
        'FS_Vol_L_Accumbens': 'Left_Accumbens_area', 'FS_Vol_R_Accumbens': 'Right_Accumbens_area',
        'FS_Vol_L_LatVentricle': 'Left_Lateral_Ventricle', 'FS_Vol_R_LatVentricle': 'Right_Lateral_Ventricle',
        'FS_ICV': 'EstimatedTotalIntraCranialVol', 'FS_WholeBrain': 'BrainSegVolNotVent', 'FS_Vol_TotalGM': 'TotalGrayVol',
        'FS_Vol_WMHypo': 'WM_hypointensities', 'FS_BrainStem': 'Brain_Stem'}
MAX_GAP_DAYS = 365  # a visit more than a year from the session gives no value
VENTRICLES = ['Left_Lateral_Ventricle', 'Right_Lateral_Ventricle', '3rd_Ventricle', '4th_Ventricle']
THICK = {'Entorhinal': 'entorhinal', 'Fusiform': 'fusiform', 'Precuneus': 'precuneus', 'SupTemporal': 'superiortemporal',
         'InfTemporal': 'inferiortemporal', 'PostCing': 'posteriorcingulate', 'SupFrontal': 'superiorfrontal',
         'SupParietal': 'superiorparietal', 'IsthmusCing': 'isthmuscingulate', 'MedOrbFrontal': 'medialorbitofrontal'}


def table(folder, prefix):
    files = sorted(glob.glob(os.path.join(folder, prefix + '*.csv')))
    if not files: sys.exit(f'missing {prefix}*.csv in {folder}')
    return pd.read_csv(files[-1], low_memory=False, encoding='latin-1')


def nearest(sessions, t, columns):
    """For each session, the row of t (same PATNO) whose visit date is nearest the session date."""
    t = t.copy(); t['_visit'] = pd.to_datetime(t.INFODT, format='%m/%Y', errors='coerce'); t['_row'] = range(len(t))
    j = sessions[['_i', 'ppmi_subject_id', '_date']].merge(t[['PATNO', '_visit', '_row', *columns]].dropna(subset=['_visit']),
                                                            left_on='ppmi_subject_id', right_on='PATNO')
    j['_gap'] = (j._date - j._visit).abs().dt.days
    j = j.sort_values(['_i', '_gap', '_row'], kind='stable').drop_duplicates('_i').set_index('_i')
    return {c: pd.to_numeric(j[c], errors='coerce').where(j._gap <= MAX_GAP_DAYS).reindex(sessions._i) for c in columns}


def main():
    a = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    a.add_argument('--t1meta', required=True); a.add_argument('--t2meta'); a.add_argument('--studydata', required=True)
    a.add_argument('--output', required=True)
    x = a.parse_args()
    t1 = pd.read_csv(x.t1meta, low_memory=False, dtype={'subjectId': str})
    s = pd.DataFrame({'ppmi_subject_id': t1.subjectId.astype(str), 'session_date': t1.sessionName.astype(str).str[:10],
                      'T1w_image_path': t1.MNI_Z_Cropped, 'age': t1.age,
                      'sex': t1.sex.map({1: 'Male', 2: 'Female', '1': 'Male', '2': 'Female', 'M': 'Male', 'F': 'Female'}),
                      'research_group': t1.research_group, '_session': t1.sessionName.astype(str)})
    s['T2w_image_path'] = ''
    if x.t2meta:
        t2 = pd.read_csv(x.t2meta, low_memory=False, dtype={'subjectId': str})
        t2 = dict(zip(zip(t2.subjectId.astype(str), t2.sessionName.astype(str).str[:10]), t2.MNI_Z_Cropped))
        s['T2w_image_path'] = [t2.get(k, '') for k in zip(s.ppmi_subject_id, s.session_date)]
    s = s.sort_values(['ppmi_subject_id', 'session_date'], kind='stable').reset_index(drop=True)
    s['neuroqa_id'] = [f'PPMI-{i:05d}' for i in range(1, len(s) + 1)]
    s['_i'] = range(len(s)); s['_date'] = pd.to_datetime(s.session_date)
    for t, cols in ((table(x.studydata, 'MDS-UPDRS_Part_III'), {'NP3TOT': 'UPDRS_III_total', 'NHY': 'hoehn_yahr'}),
                    (table(x.studydata, 'Montreal_Cognitive_Assessment'), {'MCATOT': 'MoCA'})):
        t['PATNO'] = t.PATNO.astype(str)
        for src, v in nearest(s, t, list(cols)).items(): s[cols[src]] = v.values
    aseg = table(x.studydata, 'FS7_ASEG_VOL'); cth = table(x.studydata, 'FS7_APARC_CTH')
    for t in (aseg, cth): t['PATNO'] = t.PATNO.astype(str)
    fs = pd.DataFrame({'ppmi_subject_id': aseg.PATNO, **{k: aseg[v] for k, v in ASEG.items()},
                       'FS_Ventricles': aseg[VENTRICLES].sum(axis=1, min_count=len(VENTRICLES))}).drop_duplicates('ppmi_subject_id')
    th = pd.DataFrame({'ppmi_subject_id': cth.PATNO, **{f'FS_Thick_{h}_{k}': cth[f'{p}_{v}'] for k, v in THICK.items() for h, p in (('L', 'lh'), ('R', 'rh'))}}).drop_duplicates('ppmi_subject_id')
    s = s.merge(fs, on='ppmi_subject_id', how='left').merge(th, on='ppmi_subject_id', how='left')
    s = s.drop(columns=['_i', '_date', '_session'])
    first = ['neuroqa_id', 'ppmi_subject_id', 'session_date', 'T1w_image_path', 'T2w_image_path', 'age', 'sex', 'research_group']
    s = s[first + [c for c in s.columns if c not in first]]
    s.to_csv(x.output, index=False)
    print(f'{len(s)} sessions, {s.ppmi_subject_id.nunique()} subjects; FreeSurfer found for {s.FS_Vol_L_Hippocampus.notna().sum()} sessions')


if __name__ == '__main__':
    main()
