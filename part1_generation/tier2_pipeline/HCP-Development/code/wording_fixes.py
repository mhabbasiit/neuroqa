"""
Wording fixes from expert feedback.
Import this module and call apply_wording_fixes(qa_list) after generation.
Only question text is changed — answers, options, IDs are untouched.
"""
import re

# Map: old template (with N placeholders) → (old_example, new_example)
WORDING_FIXES = {
    'Are the ventricles relatively enlarged for a pediatric brain?':
        ('Are the ventricles relatively enlarged for a pediatric brain?',
         'Are the ventricles relatively enlarged for a pediatric brain? (subject age: 9)'),

    'Are the ventricles relatively enlarged for a young adult brain?':
        ('Are the ventricles relatively enlarged for a young adult brain?',
         'Are the ventricles relatively enlarged for a young adult brain? (subject age: 25)'),

    'Are the ventricles relatively enlarged for this age group?':
        ('Are the ventricles relatively enlarged for this age group?',
         'Are the ventricles relatively enlarged for this age group (55-78 years old)?'),

    'Are the ventricles relatively enlarged or normal in this pediatric scan?':
        ('Are the ventricles relatively enlarged or normal in this pediatric scan?',
         'Are the ventricles relatively enlarged or normal in this pediatric scan? (subject age: 12)'),

    'Are the ventricles relatively enlarged or normal in this scan?':
        ('Are the ventricles relatively enlarged or normal in this scan?',
         'Are the ventricles relatively enlarged or normal in this scan? (subject age: 48)'),

    'Are the ventricles within expected for a N-year-old size in this scan?':
        ('Are the ventricles within expected for a 87-year-old size in this scan?',
         'Are the ventricles in this scan within expected size for an 87-year-old subject?'),

    'Are the ventricles within normal size for a developing brain?':
        ('Are the ventricles within normal size for a developing brain?',
         'Are the ventricles within normal size for a developing brain (9-10 years old)?'),

    'Are the ventricles within normal size for a young adult brain?':
        ('Are the ventricles within normal size for a young adult brain?',
         'Are the ventricles within normal size for a young adult brain (22-35 years old)?'),

    'Describe the signal characteristics of the lesion on TN-weighted imaging.':
        ('Describe the signal characteristics of the lesion on T1-weighted imaging.',
         'Describe the signal characteristics of the lesion on the T1-weighted imaging scan.'),

    'Do the brain measurements suggest any developmental concern?':
        ('Do the brain measurements suggest any developmental concern?',
         'Are there any structural abnormalities that suggest atypical development for a 10-year-old subject?'),

    'Do the brain measurements suggest normal development?':
        ('Do the brain measurements suggest normal development?',
         'Are the structural findings consistent with normal development for a 9 year-old subject?'),

    'Do the clinical findings suggest neurodegenerative changes in this subject?':
        ('Do the clinical findings suggest neurodegenerative changes in this subject?',
         'Do the structural findings suggest neurodegenerative changes in this subject?'),

    'Do the white matter lesions appear hypointense on TN-weighted imaging?':
        ('Do the white matter lesions appear hypointense on T1-weighted imaging?',
         'Do the white matter lesions appear hypointense on the T1-weighted imaging scan?'),

    'Does the brain show structural deviations from expected developmental norms?':
        ('Does the brain show structural deviations from expected developmental norms?',
         'Does the brain show structural deviations from expected developmental norms (subject age: 9)?'),

    'Does the brain show structural deviations from expected developmental range?':
        ('Does the brain show structural deviations from expected developmental range?',
         'Does the brain show structural deviations from expected development within the age group (5-22 years old)?'),

    'Does the brain show structural deviations from expected norms?':
        ('Does the brain show structural deviations from expected norms?',
         'Are there any structural findings that deviate from what is expected for a subject of this age (subject age: 31)?'),

    'Does the clinical pattern suggest cognitive impairment in this subject?':
        ('Does the clinical pattern suggest cognitive impairment in this subject?',
         'Do the structural patterns in this scan suggest findings associated with conditions that may affect cognitive function in this subject?'),

    'Does the lesion show high signal on TN-weighted imaging?':
        ('Does the lesion show high signal on T2-weighted imaging?',
         'Does the lesion show high signal intensity on the T2-weighted imaging scan?'),

    'Does the structural pattern suggest cognitive impairment in this subject?':
        ('Does the structural pattern suggest cognitive impairment in this subject?',
         'Are there any structural patterns consistent with findings that may suggest mild cognitive impairment (MCI) in this subject?'),

    'Does the structural pattern suggest motor-related brain changes?':
        ('Does the structural pattern suggest motor-related brain changes?',
         "Do the structural patterns in this scan suggest motor-related changes in the subject's brain?"),

    'Does the structural pattern suggest preserved motor function?':
        ('Does the structural pattern suggest preserved motor function?',
         "Do the structural patterns in this scan suggest preserved motor function in the subject's brain?"),

    "Does this brain scan show features consistent with Alzheimer's disease?":
        ("Does this brain scan show features consistent with Alzheimer's disease?",
         "Does this brain scan show structural abnormalities consistent with Alzheimer's disease?"),

    "Does this brain scan show features consistent with Parkinson's disease?":
        ("Does this brain scan show features consistent with Parkinson's disease?",
         "Does this brain scan show structural findings consistent with Parkinson's disease?"),

    'Does this brain scan show features consistent with dementia?':
        ('Does this brain scan show features consistent with dementia?',
         'Does this brain scan show structural changes consistent with dementia?'),

    "Does this brain scan show features inconsistent with Parkinson's disease?":
        ("Does this brain scan show features inconsistent with Parkinson's disease?",
         "Are there any structural findings that suggest an alternative diagnosis to idiopathic Parkinson's disease?"),

    'Does this brain scan show neurological abnormalities?':
        ('Does this brain scan show neurological abnormalities?',
         'Does this brain scan show any structural abnormalities?'),

    'Does this brain scan show signs of cognitive impairment?':
        ('Does this brain scan show signs of cognitive impairment?',
         'Are there any structural abnormalities in this brain scan that suggest conclusive evidence of cognitive dysfunction?'),

    'Does this case show features suggestive of high-grade pathology?':
        ('Does this case show features suggestive of high-grade pathology?',
         'Does this brain scan contain structural patterns suggestive of high-grade pathology?'),

    'Does this scan suggest cognitive decline?':
        ('Does this scan suggest cognitive decline?',
         'Does this scan contain structural findings that suggest findings consistent with mild cognitive impairment (MCI) in this subject?'),

    'Does this scan suggest cognitive impairment based on clinical assessment?':
        ('Does this scan suggest cognitive impairment based on clinical assessment?',
         'Are there any structural abnormalities in this scan that may suggest findings consistent with mild cognitive impairment (MCI) in this subject?'),

    'Does this subject score above average on fluid cognition (NIH Toolbox)?':
        ('Does this subject score above average on fluid cognition (NIH Toolbox)?',
         "Are there any structural findings in this scan that suggest reduced fluid cognition (NIH Toolbox) in the subject's brain?"),

    'Does this subject show any structural deviations from the cohort median?':
        ('Does this subject show any structural deviations from the cohort median?',
         "Does this subject's brain show any structural deviations from age-related changes (subject age: 57)?"),

    'Has motor function improved compared to the visit N months prior (subject age: N)?':
        ('Has motor function improved compared to the visit 12 months prior (subject age: 52)?',
         "Do structural findings in this scan suggest that the subject's motor function improved compared to the visit 12 months prior (subject age: 52)?"),

    'Has motor function worsened compared to the visit N months prior (subject age: N)?':
        ('Has motor function worsened compared to the visit 24 months prior (subject age: 54)?',
         "Do structural findings in this scan suggest that the subject's motor function worsened compared to the visit 24 months prior (subject age: 54)?"),

    'Has the clinical status progressed, remained stable, or improved compared to the previous visit (subject age: N)?':
        ('Has the clinical status progressed, remained stable, or improved compared to the previous visit (subject age: 69)?',
         "Has the subject's clinical status progressed, remained stable, or improved compared to the previous visit based on the structural findings in this scan (subject age: 69)?"),

    'Has the cognitive performance improved, remained stable, or declined compared to the visit N months prior (subject age: N)?':
        ('Has the cognitive performance improved, remained stable, or declined compared to the visit 24 months prior (subject age: 10)?',
         "Has the subject's cognitive performance improved, remained stable, or declined compared to the visit 24 months prior based on the structural findings in this scan (subject age: 10)?"),

    'Have the ventricles enlarged compared to the scan N months prior (subject age: N)?':
        ('Have the ventricles enlarged compared to the scan 1 months prior (subject age: 64)?',
         'Have the ventricles enlarged compared to the scan 1 month prior (subject age: 64)?'),

    'How would you classify the severity of mass effect?':
        ('How would you classify the severity of mass effect?',
         'How would you classify the severity of mass effect in this scan?'),

    'Is cognitive impairment present in this subject?':
        ('Is cognitive impairment present in this subject?',
         'Are there any structural changes consistent with mild cognitive impairment (MCI) in this subject?'),

    'Is the cognitive impairment beyond a mild stage in this subject?':
        ('Is the cognitive impairment beyond a mild stage in this subject?',
         "Do the structural findings in this scan suggest that the subject's cognitive impairment is beyond a mild stage?"),

    'Is the fluid cognition score within average range for this subject?':
        ('Is the fluid cognition score within average range for this subject?',
         'Are there structural patterns in the brain that suggest the fluid cognition score is within average range for this subject (subject age: 25)?'),

    "Is the hippocampal volume consistent with what would be expected for this subject's age group?":
        ("Is the hippocampal volume consistent with what would be expected for this subject's age group?",
         "Is the hippocampal volume consistent with what would be expected for this subject's age (subject age: 63)?"),

    'Is the overall cognitive function within normal limits for this subject?':
        ('Is the overall cognitive function within normal limits for this subject?',
         'Does the brain morphology suggest that the overall cognitive function is within normal limits for this subject?'),

    'Is the total cognitive composite score above N?':
        ('Is the total cognitive composite score above 106.0?',
         'Are there any structural findings in the scan that would be inconsistent with intact global cognitive function in a subject of this age (subject age: 36)?'),

    'Is the total cognitive composite score above average (> N)?':
        ('Is the total cognitive composite score above average (> 87.0)?',
         'Are there any structural findings in this scan consistent with below-average global cognitive function?'),

    'Is the total cognitive composite score below N?':
        ('Is the total cognitive composite score below 120.4?',
         "Does the subject's brain morphology suggest findings consistent with superior global cognitive function for their age (subject age: 25)?"),

    'Is the total cognitive composite score below average (< N)?':
        ('Is the total cognitive composite score below average (< 87.0)?',
         "Does the subject's brain scan show structural findings associated with below-average global cognitive function?"),

    'Is the total hippocampal volume above the population median?':
        ('Is the total hippocampal volume above the population median?',
         'Is the total hippocampal volume above the population median (3.37 mL)?'),

    'Is the ventricle-to-brain ratio greater than N% in this scan?':
        ('Is the ventricle-to-brain ratio greater than 1.77% in this scan?',
         'Is the ventricular-to-brain volume ratio greater than 1.77% in this scan?'),

    'Is the ventricle-to-brain ratio less than N% in this scan?':
        ('Is the ventricle-to-brain ratio less than 0.62% in this scan?',
         'Is the ventricular-to-brain volume ratio less than 0.62% in this scan?'),

    'Is the whole brain volume greater than N mmN in this scan?':
        ('Is the whole brain volume greater than 1,220,000 mm3 in this scan?',
         'Is the total brain volume greater than 1,220,000 mm^3 in this scan?'),

    'Is the whole brain volume greater than N mm³ in this scan?':
        ('Is the whole brain volume greater than 1,170,000 mm³ in this scan?',
         'Is the total brain volume greater than 1,170,000 mm^3 in this scan?'),

    'Is the whole brain volume less than N mmN in this scan?':
        ('Is the whole brain volume less than 1,220,000 mm3 in this scan?',
         'Is the total brain volume less than 1,220,000 mm^3 in this scan?'),

    'Is the whole brain volume less than N mm³ in this scan?':
        ('Is the whole brain volume less than 1,020,000 mm³ in this scan?',
         'Is the total brain volume less than 1,020,000 mm^3 in this scan?'),

    'Is this brain scan from a child with average cognitive performance?':
        ('Is this brain scan from a child with average cognitive performance?',
         'Does this brain scan from a pediatric subject (age: 10) show structural features consistent with above-average cognitive performance?'),

    'Is this brain scan from a child with below average cognitive performance?':
        ('Is this brain scan from a child with below average cognitive performance?',
         'Does this brain scan from a pediatric subject (age: 9) show structural features consistent with below-average cognitive performance?'),

    'Is this brain scan from a subject with above average cognitive performance?':
        ('Is this brain scan from a subject with above average cognitive performance?',
         'Does this brain scan from a subject (age: 25) show structural features consistent with above-average cognitive performance?'),

    'Is this brain scan from a subject with average cognitive performance?':
        ('Is this brain scan from a subject with average cognitive performance?',
         'Does this brain scan from a subject (age: 29) show structural features consistent with average cognitive performance?'),

    'Is this brain scan from a subject with below average cognitive performance?':
        ('Is this brain scan from a subject with below average cognitive performance?',
         'Does this brain scan from a subject (age: 31) show structural features inconsistent with average cognitive performance?'),

    'Is this brain scan inconsistent with a dementia diagnosis?':
        ('Is this brain scan inconsistent with a dementia diagnosis?',
         'Does this brain scan show structural findings inconsistent with a dementia diagnosis (subject age: 64)?'),

    "Is this brain scan inconsistent with an Alzheimer's disease diagnosis?":
        ("Is this brain scan inconsistent with an Alzheimer's disease diagnosis?",
         "Does this brain scan show structural findings inconsistent with an Alzheimer's disease diagnosis (subject age: 64)?"),

    'Is this scan consistent with normal cognitive function?':
        ('Is this scan consistent with normal cognitive function?',
         'Does this scan show structural findings consistent with normal cognitive function?'),

    'Is this scan from a subject with normal cognition, MCI, or AD?':
        ('Is this scan from a subject with normal cognition, MCI, or AD?',
         "Based on this scan, does the subject have normal cognition, mild cognitive impairment (MCI), or Alzheimer's disease (AD) (subject age: 73)?"),

    'Is this subject N years old or younger?':
        ('Is this subject 73 years old or younger?',
         "Is this subject's age likely to be 73 years old or younger?"),

    'Is this subject in the N-N age range?':
        ('Is this subject in the 22-25 age range?',
         'Is this subject likely a young adult (22-25 years old)?'),

    'Is this subject older than N years?':
        ('Is this subject older than 48 years?',
         'Is this subject older than 48 years old?'),

    'Is this subject younger than N years?':
        ('Is this subject younger than 65 years?',
         'Is this subject younger than 65 years old?'),

    'Is this subject younger than N?':
        ('Is this subject younger than 13?',
         'Is this subject younger than 13 years old?'),

    "Is this subject's MoCA score above or below the median (N)?":
        ("Is this subject's MoCA score above or below the median (26)?",
         "Does this subject's scan show structural findings consistent with cognitive screening performance below normal (MoCA < 26)?"),

    'What age range does this subject belong to?':
        ('What age range does this subject belong to?',
         'What age range does this subject likely belong to?'),

    'What imaging features support the diagnosis?':
        ('What imaging features support the diagnosis?',
         "What imaging features support the subject's likely diagnosis?"),

    'What is the MoCA performance level for this subject relative to median?':
        ('What is the MoCA performance level for this subject relative to median?',
         'Based on this scan, describe the expected cognitive screening performance of this subject relative to normative MoCA scores.'),

    'What is the clinical progression status compared to the previous visit (subject age: N)?':
        ('What is the clinical progression status compared to the previous visit (subject age: 70)?',
         "What is the clinical progression status of this subject's brain compared to the previous visit (subject age: 70)?"),

    'What is the clinical progression status compared to the visit N months prior (subject age: N)?':
        ('What is the clinical progression status compared to the visit 15 months prior (subject age: 84)?',
         "What is the clinical progression status of this subject's brain compared to the visit 15 months prior (subject age: 84)?"),

    'What is the cognitive trajectory compared to the visit N months prior (subject age: N)?':
        ('What is the cognitive trajectory compared to the visit 24 months prior (subject age: 10)?',
         "What is the cognitive trajectory of the subject's brain compared to the visit 24 months prior (subject age: 10)?"),

    'What is the fluid cognition level for this subject?':
        ('What is the fluid cognition level for this subject?',
         'Based on this scan, describe the expected fluid cognition level for this subject.'),

    'What severity level is most consistent with this brain scan?':
        ('What severity level is most consistent with this brain scan?',
         'What disease severity level is most consistent with this brain scan?'),

    'What severity level is most consistent with this subject?':
        ('What severity level is most consistent with this subject?',
         'What disease severity level is most consistent with this subject?'),

    'What stage of parkinsonism is most consistent with this brain scan?':
        ('What stage of parkinsonism is most consistent with this brain scan?',
         "What stage of parkinsonism is most consistent with this subject's brain scan?"),

    'Which side has a smaller accumbens volume in this scan?':
        ('Which side has a smaller accumbens volume in this scan?',
         'Which side has a smaller accumbens volume in this brain scan?'),

    'Which side has a smaller amygdala volume in this scan?':
        ('Which side has a smaller amygdala volume in this scan?',
         'Which side has a smaller amygdala volume in this brain scan?'),

    'Which side has a smaller caudate volume in this scan?':
        ('Which side has a smaller caudate volume in this scan?',
         'Which side has a smaller caudate volume in this brain scan?'),

    'Which side has a smaller hippocampus volume in this scan?':
        ('Which side has a smaller hippocampus volume in this scan?',
         'Which side has a smaller hippocampus volume in this brain scan?'),

    'Which side has a smaller lateral ventricle volume in this scan?':
        ('Which side has a smaller lateral ventricle volume in this scan?',
         'Which side has a smaller lateral ventricle volume in this brain scan?'),

    'Which side has a smaller pallidum volume in this scan?':
        ('Which side has a smaller pallidum volume in this scan?',
         'Which side has a smaller pallidum volume in this brain scan?'),

    'Which side has a smaller putamen volume in this scan?':
        ('Which side has a smaller putamen volume in this scan?',
         'Which side has a smaller putamen volume in this brain scan?'),

    'Which side has a smaller thalamus volume in this scan?':
        ('Which side has a smaller thalamus volume in this scan?',
         'Which side has a smaller thalamus volume in this brain scan?'),

    'Which subcortical structure shows the most volume deviation from expected range?':
        ('Which subcortical structure shows the most volume deviation from expected range?',
         'Which subcortical structure in this scan shows the most volume deviation from expected range?'),

}


# The new wording of these templates asks the opposite question, so the corrected Gold is the opposite answer.
REVERSED = {'Does this subject score above average on fluid cognition (NIH Toolbox)?',
            'Is the total cognitive composite score above average (> N)?',
            'Is the total cognitive composite score above N?',   # becomes "inconsistent with intact global cognitive function"
            'Is the total cognitive composite score below N?'}   # becomes "superior global cognitive function"
# The new wording asks "above-average" instead of "average": the corrected Gold is Yes only for the Above group.
ABOVE = {'Is this brain scan from a child with average cognitive performance?'}


def _tmpl(q):
    return re.sub(r'\d[\d,.]*', 'N', q)


def apply_wording_fixes(qa_list):
    """Apply wording fixes to a list of QA dicts. Modifies in-place."""
    fixed = 0
    for d in qa_list:
        t = _tmpl(d['question'])
        if t not in WORDING_FIXES:
            continue
        old_ex, new_ex = WORDING_FIXES[t]
        # 'mm^3' is a unit, not a number slot (otherwise a number is written into the exponent)
        actual_nums = re.findall(r'\d[\d,.]*', d['question'].replace('mm^3', 'mm^'))
        ex_new_nums = re.findall(r'\d[\d,.]*', new_ex.replace('mm^3', 'mm^'))
        new_tmpl = re.sub(r'\d[\d,.]*', '{NUM}', new_ex.replace('mm^3', 'mm^@')).replace('mm^@', 'mm^3')
        result = new_tmpl
        for num in actual_nums:
            result = result.replace('{NUM}', num, 1)
        for num in ex_new_nums:
            result = result.replace('{NUM}', num, 1)
        if '{NUM}' not in result:
            d['question'] = result
            fixed += 1
            if t in REVERSED: d['_gold_fix'] = 'No' if d['answer'] == 'Yes' else 'Yes'
            if t in ABOVE: d['_gold_fix'] = 'group'
    return fixed