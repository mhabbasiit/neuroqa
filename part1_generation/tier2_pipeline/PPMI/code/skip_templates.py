"""
NeuroSAB — Template Skip Filter (v2 Curation)

Filters out Group C (pure metadata) and Group D (exact numeric) templates
from the QA generation pipeline before balancing.

Group C (not inferable from brain images):
  C1: APOE genotype
  C2: NEO-FFI personality
  C3: CBCL behavioral scores
  C4: Recall scores (LIMMTOTAL, LDELTOTAL)
  C5: Cognitive group / AD classification
  C6: Structure-cognition mismatch

Group D (exact numeric answers):
  D1: Exact volume in mm³ (Open)
  D2: Exact cortical thickness in mm (Open)
  D3: Exact CDR score (Open)
  D4: Exact MMSE/MoCA score (Open + MoCA YN)
  D5: Percentage change in volume (Open)
"""

import re

# ── Group C patterns (any question type) ──────────────────────────────────

_GROUP_C = [
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
    # P0: DaTScan / dopamine transporter (SPECT, not visible in T1 MRI)
    re.compile(r'DaTScan|dopamin', re.IGNORECASE),
    # P0: T2/modality availability (pure metadata)
    re.compile(r'T2.*(available|present)', re.IGNORECASE),
    re.compile(r'modalit.*(available|acquired)', re.IGNORECASE),
    re.compile(r'how many modalit', re.IGNORECASE),
    # P0: Scan quality (subjective metadata)
    re.compile(r'scan of (above|below).average quality', re.IGNORECASE),
    # CDR score (clinical rating, not image-derived)
    re.compile(r'CDR', re.IGNORECASE),
    # MMSE score (cognitive test, not image-derived)
    re.compile(r'MMSE', re.IGNORECASE),
]

# ── Group D-Open patterns (only Open questions) ──────────────────────────

_GROUP_D_OPEN = [
    re.compile(r'approximate .* volume \(mm', re.IGNORECASE),
    re.compile(r'cortical thickness \(mm\)', re.IGNORECASE),
    re.compile(r'CDR global score', re.IGNORECASE),
    re.compile(r'MMSE score', re.IGNORECASE),
    re.compile(r'change in MMSE', re.IGNORECASE),
    re.compile(r'percentage change in .* volume', re.IGNORECASE),
]

# ── Group D-YN patterns (only YN questions) ──────────────────────────────

_GROUP_D_YN = [
    re.compile(r"MoCA score (below|above) 26", re.IGNORECASE),
]


def should_skip(q_text, q_type):
    """Return True if this question should be skipped (Group C or D)."""
    for pat in _GROUP_C:
        if pat.search(q_text):
            return True
    if q_type == 'Open':
        for pat in _GROUP_D_OPEN:
            if pat.search(q_text):
                return True
    if q_type == 'YN':
        for pat in _GROUP_D_YN:
            if pat.search(q_text):
                return True
    return False


def filter_qa_list(all_qa):
    """Filter a list of QA dicts, removing Group C and D items.

    Supports both raw keys ('q'/'type') and renamed keys ('question'/'question_type').
    Returns (filtered_list, removed_count).
    """
    filtered = []
    removed = 0
    for item in all_qa:
        q_text = item.get('question') or item.get('q', '')
        q_type = item.get('question_type') or item.get('type', '')
        if should_skip(q_text, q_type):
            removed += 1
        else:
            filtered.append(item)
    if removed:
        print(f"  [skip_templates] Removed {removed} Group C/D items "
              f"({len(filtered)} remaining)")
    return filtered, removed
