#!/usr/bin/env python3
"""NeuroQA generator for HCP-Aging. Builds its questions from your own licensed metadata. See README.md.

Steps: metadata, generator, all cleanup rules, session link fixes, final wording, counts per template, subject split, MCQ balance, v1.1 corrections.
The files in code/ are the release scripts; this script runs them in order and adds the listed steps.
"""
import argparse, collections, contextlib, glob, importlib.util, json, os, random, re, runpy, shutil, sys, tempfile, types
sys.dont_write_bytecode = True
if os.environ.get('PYTHONHASHSEED') != '0':
    os.execve(sys.executable, [sys.executable, '-B', *sys.argv], {**os.environ, 'PYTHONHASHSEED': '0', 'PYTHONUTF8': '1'})
import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import csr_matrix

CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code')
SEED = 42
SPLITS = ('train', 'val', 'test_public', 'test_private')
mask = lambda q: re.sub(r'\d[\d,.]*', 'N', q)
VOLUME_WORDS = ['hippocampal volume', 'whole brain', 'ventricle', 'percentage change']
DATASETS = {  # short name, session column, age-range bins (from rule R38)
    'ABCD': ('abcd', 'session_bids', (((8, 10), '8-9'), ((10, 11), '10-11'), ((11, 13), '11-12'), ((13, 15), '13-14'))),
    'ADNI': ('adni', 'session_date', (((50, 65), '50-64'), ((65, 75), '65-74'), ((75, 85), '75-84'), ((85, 200), '85+'))),
    'HCP-Aging': ('hcpaging', 'session_id', ()),
    'HCP-Development': ('hcpdev', 'session_id', ()),
    'PPMI': ('ppmi', 'session_date', ()),
}


def years(a):
    a = float(a)
    return a / 12 if a > 200 else a


def run_script(name, argv, log):
    sys.argv = [name, *argv]
    with open(log, 'a') as f, contextlib.redirect_stdout(f):
        runpy.run_path(os.path.join(CODE, name), run_name='__main__')


def add_adni_freesurfer(merged_csv, extracted):
    """Adds the regional FreeSurfer columns the ADNI generator needs: newest UCSFFSX table first, nearest exam date within 180 days."""
    columns = json.load(open(os.path.join(CODE, 'adni_freesurfer_columns.json')))
    m = pd.read_csv(merged_csv, low_memory=False)
    m['_rid'] = m.adni_subject_id.astype(str).str.split('_S_').str[1].astype(int); m['_d'] = pd.to_datetime(m.session_date); m['_i'] = range(len(m))
    found = {}
    for name in ('UCSFFSX7', 'UCSFFSX6', 'UCSFFSX51', 'UCSFFSX_'):
        t = pd.read_csv(sorted(glob.glob(os.path.join(extracted, name + '*.csv')))[0], low_memory=False)
        t['_d'] = pd.to_datetime(t.EXAMDATE, errors='coerce'); t = t.rename(columns={'RID': '_rid'})
        use = [c for c in set(columns.values()) if c in t]
        j = m[['_i', '_rid', '_d']].merge(t[['_rid', '_d', *use]], on='_rid', suffixes=('', '_t'))
        j['_gap'] = (j._d - j._d_t).abs().dt.days
        j = j[j._gap <= 180].sort_values(['_i', '_gap']).drop_duplicates('_i')
        for r in j.to_dict('records'): found.setdefault(r['_i'], r)
    for c, st in columns.items():
        m[c] = [pd.to_numeric(found.get(i, {}).get(st), errors='coerce') for i in m._i]
    m.drop(columns=['_rid', '_d', '_i']).to_csv(merged_csv, index=False)


def run_cleanup(base, questions, merged_csv, ds, short, session_col, log, select=True):
    """Runs all cleanup rules. With select=False the selection rules R23 and R24 are skipped, which gives every valid question."""
    out = os.path.join(base, ds, 'outputs'); os.makedirs(out); shutil.copy(merged_csv, out)
    json.dump(questions, open(os.path.join(out, f'{short}_qa_final.json'), 'w'))
    spec = importlib.util.spec_from_file_location('cleanup_benchmark', os.path.join(CODE, 'cleanup_benchmark.py'))
    cb = importlib.util.module_from_spec(spec); sys.modules['cleanup_benchmark'] = cb; spec.loader.exec_module(cb)
    by_scan = pd.read_csv(merged_csv, low_memory=False).set_index('neuroqa_id')
    def session_from_source(qa, _):  # R15 takes one session per subject; each question keeps the session it was generated from
        n = 0
        for d in qa:
            if d.get('_scan') in by_scan.index:
                d['session_date'] = str(by_scan.at[d['_scan'], session_col]); n += 1
        return qa, n
    cb.apply_R15_session = session_from_source
    # R23 groups questions by template and R30 removes the same question twice, but "(age: N)" was left out of the first and the 80-character key of the second cuts it short.
    template = cb.normalize_template
    cb.normalize_template = lambda q: template(re.sub(r'\(age: \d+\)', '(age: X)', q))
    def dedupe(qa):
        by_key = collections.defaultdict(list)
        for i, d in enumerate(qa): by_key[d['subject_id'], d['question']].append(i)
        remove = {i for v in by_key.values() if len({str(qa[j]['answer']) for j in v}) > 1 for i in v[1:]}
        return [d for i, d in enumerate(qa) if i not in remove], len(remove)
    cb.apply_R30_deduplicate = dedupe
    if not select:
        cb.apply_R23_yn_balance = lambda qa: (qa, 0)
        cb.apply_R24_shortcut_removal = lambda qa: (qa, 0)
    real = glob.glob; glob.glob = lambda p, **k: []  # the rules look for image files; this script does not use images
    sys.argv = ['cleanup_benchmark.py', '--base-dir', base]
    with open(log, 'a') as f, contextlib.redirect_stdout(f):
        cb.main()
    glob.glob = real
    return cb, json.load(open(os.path.join(base, 'neuroqa_all_qa_v4.json')))


def link_to_session(qa, merged, ds, session_col, bins):
    """Connects each question to one session. Volume change questions keep the session pair R37 checked against their answer. Every other question keeps the session it was generated from. Writes no image path."""
    sid_col = next(c for c in merged.columns if c.endswith('_subject_id'))
    rows = {(str(r[sid_col]), str(r[session_col])): r for r in merged.to_dict('records')}
    by_scan = merged.set_index('neuroqa_id')
    position = {'ses-baselineYear1Arm1': 0, 'ses-2YearFollowUpYArm1': 1, 'ses-4YearFollowUpYArm1': 2}
    in_order = lambda v: sorted(v, key=lambda x: (position.get(x, 9), x)) if ds == 'ABCD' else sorted(v)
    order = {str(k): in_order(g[session_col].astype(str).tolist()) for k, g in merged.groupby(sid_col)}
    for d in qa:
        sid = str(d['subject_id']); session = str(by_scan.at[d['_scan'], session_col]); d.pop('_months', None)
        if d['category'] == 'Longitudinal':
            q = d['question'].lower()
            if any(k in q for k in VOLUME_WORDS) and ('ventricle' not in q or 'enlarged' in q) and d.get('session_date') in order[sid]:  # R37 checks only these
                session = d['session_date']  # set by R37
            else:
                sessions = order[sid]; i = sessions.index(session)
                d['prior_session_date'] = sessions[i - 1] if i else None
                if 'ventricle' in q: d['answer'] = d['_generator_answer']  # R37 does not check this wording, so the answer stays the one of this pair
            if ds != 'ABCD' and d.get('prior_session_date'):  # the months in the text follow this pair
                d['_months'] = round(abs((pd.Timestamp(session[:10]) - pd.Timestamp(d['prior_session_date'][:10])).days) / 30)
        d['session_date'] = session
        own = lambda s, col='T1w_image_path': (lambda v: v if isinstance(v, str) and v else None)(rows.get((sid, str(s)), {}).get(col))  # from your own sMRI metadata
        d['image_path'] = own(session) or '#'
        d.pop('image_path_prior', None); d.pop('image_path_t2', None)
        if own(session, 'T2w_image_path'): d['image_path_t2'] = own(session, 'T2w_image_path')
        if d.get('prior_session_date'): d['image_path_prior'] = own(d['prior_session_date']) or '#'
        age = rows[sid, session].get('age'); d['_age'] = None
        if pd.notna(age):
            age = years(age); d['_age'] = int(round(age))
            if d['question_type'] == 'MCQ' and ('age range' in d['question'].lower() or 'estimated age' in d['question'].lower()):
                d['answer'] = next((b for (lo, hi), b in bins if lo <= age < hi and b in d['options']), d['answer'])
    return qa


def usable(qa, merged, session_col):
    """Leaves out questions a session cannot support. Their places are filled from the other valid questions. A score of 777 or 999 is a missing code, not a score."""
    sid_col = next(c for c in merged.columns if c.endswith('_subject_id'))
    rows = {(str(r[sid_col]), str(r[session_col])): r for r in merged.to_dict('records')}
    def ok(d):
        if d['category'] == 'Longitudinal' and not d.get('prior_session_date'): return False
        if d['category'] == 'Longitudinal':  # a volume change needs the volume at both sessions, or R37 cannot check its answer
            q = d['question'].lower()
            col = 'FS_Vol_L_Hippocampus' if 'hippocampal volume' in q else 'FS_WholeBrain' if 'whole brain' in q else 'FS_Ventricles' if 'ventricle' in q else None
            sid = str(d['subject_id'])
            if col and any(pd.isna(pd.to_numeric(rows.get((sid, s), {}).get(col), errors='coerce')) for s in (d['session_date'], d['prior_session_date'])): return False
        if 'fluid' in d['question'].lower():
            v = pd.to_numeric(rows[str(d['subject_id']), d['session_date']].get('NIH_FluidComp'), errors='coerce')
            if v in (777, 999): return False
        return True
    return [d for d in qa if ok(d)]


def session_text(qa):
    """Writes the session's age and months into the text. Runs after the final wording."""
    for d in qa:
        n, months = d.pop('_age', None), d.pop('_months', None); q = d['question']
        if n is not None:
            q = re.sub(r'subject age: \d+', f'subject age: {n}', q)
            q = re.sub(r'\(age: \d+\)', f'(age: {n})', q); q = re.sub(r'\(age \d+\)', f'(age {n})', q)
            q = re.sub(r'\b\d+-year-old', f'{n}-year-old', q)
        if months: q = re.sub(r'\d+(\s*months?\s*prior)', lambda m: f'{months}{m.group(1)}', q)
        d['question'] = q
    return qa


def fit_counts(qa, spare, t):
    """Makes the count of each template (and of Yes and No within it) equal to the target: extra questions are removed, missing ones come from the other valid questions."""
    rng = random.Random(SEED)
    want = {}
    for m, n in t['template_counts'].items():
        if m in t['yes_no_answers']:
            for a, k in t['yes_no_answers'][m].items(): want[m, a] = k
        else: want[m, None] = n
    group = lambda d: (mask(d['question']), d['answer'] if d['question_type'] == 'YN' else None)
    key = lambda d: (str(d['subject_id']), mask(d['question']), str(d.get('_scan')))  # one question of a template per session
    order = lambda d: (str(d['subject_id']), str(d.get('_scan')), d['question'])
    have, used = collections.defaultdict(list), {key(d) for d in qa}
    for d in sorted(qa, key=order): have[group(d)].append(d)
    per_subject = collections.Counter(str(d['subject_id']) for d in qa)
    spare_by = collections.defaultdict(list)
    for d in sorted(spare, key=order):
        if group(d) in want: spare_by[group(d)].append(d)
    out, added, removed, shortage = [], 0, 0, {}
    for g in sorted(set(want) | set(have), key=str):
        items = have[g]; n = want.get(g, 0); rng.shuffle(items)
        if len(items) > n:
            removed += len(items) - n
            for d in list(items):  # every subject keeps at least one question
                if len(items) > n and per_subject[str(d['subject_id'])] > 1:
                    items.remove(d); per_subject[str(d['subject_id'])] -= 1
            items = items[:n]
        else:
            c = spare_by[g]; rng.shuffle(c)
            for d in c:
                if len(items) == n: break
                if key(d) not in used:
                    used.add(key(d)); items.append(d); added += 1
            if len(items) < n: shortage[g] = n - len(items)
        out += items
    print(f'questions added from other valid questions: {added}; extra removed: {removed}')
    if shortage:
        print(f'SHORTAGE: {sum(shortage.values())} questions in {len(shortage)} template and answer groups have too few valid questions:')
        for (m, a), n in sorted(shortage.items(), key=str): print(f'  -{n} {a or ""} {m[:90]}')
    return out


def assign_splits(qa, t):
    """Subject-level split with the published count of each template in each split, solved exactly as an integer program over groups of subjects with the same question profile."""
    want = t['template_split_counts']; yes_no = set(t['yes_no_answers']); tests = SPLITS[1:]
    count = collections.defaultdict(collections.Counter)
    for d in qa:
        c = count[str(d['subject_id'])]; c[mask(d['question'])] += 1
        if d['question_type'] == 'YN': c['answer|' + d['answer']] += 1
    profiles = collections.defaultdict(list)
    for s in sorted(count): profiles[tuple(sorted(count[s].items()))].append(s)
    profiles = sorted(profiles.items()); keys = sorted({k for p, _ in profiles for k, _ in p})
    n = len(profiles) * 4; row = []; col = []; val = []; low = []; high = []; rows = 0
    def add(terms, lo, hi):
        nonlocal rows
        for c_, v in terms: row.append(rows); col.append(c_); val.append(v)
        low.append(lo); high.append(hi); rows += 1
    for g, (_, members) in enumerate(profiles): add([(g * 4 + k, 1) for k in range(4)], len(members), len(members))
    for k, s in enumerate(SPLITS):
        if s != 'train': add([(g * 4 + k, len(m)) for g, (_, m) in enumerate(profiles)][:0] + [(g * 4 + k, 1) for g in range(len(profiles))], t['subjects_per_split'][s], t['subjects_per_split'][s])
    cost = [0.0] * n; cells = []
    half = {s: sum(v for m, v in want[s].items() if m in yes_no) // 2 for s in SPLITS}
    for k, s in enumerate(SPLITS):
        if s == 'train': continue
        for key in keys:
            goal = half[s] if key.startswith('answer|') else want[s].get(key, 0)
            terms = [(g * 4 + k, dict(p).get(key, 0)) for g, (p, _) in enumerate(profiles) if dict(p).get(key)]
            terms += [(n + 2 * len(cells), 1), (n + 2 * len(cells) + 1, -1)]
            add(terms, goal, goal); cells.append((s, key))
    total = n + 2 * len(cells)
    bounds = Bounds([0] * total, [len(m) for _, m in profiles for _ in range(4)] + [np.inf] * (2 * len(cells)))
    r = milp(np.array(cost + [1.0] * (2 * len(cells))), constraints=LinearConstraint(csr_matrix((val, (row, col)), shape=(rows, total)), low, high),
             integrality=np.array([1] * n + [0] * (2 * len(cells))), bounds=bounds, options=dict(node_limit=t.get('node_limit', 3000), mip_rel_gap=0))
    if r.x is None: sys.exit(f'split solver failed: {r.message}')
    print(f'template x split difference from the solver: {int(round(sum(r.x[n:])))}')
    rng = random.Random(SEED); split = {}
    for g, (_, members) in enumerate(profiles):
        members = list(members); rng.shuffle(members); i = 0
        for k, s in enumerate(SPLITS):
            take = int(round(r.x[g * 4 + k]))
            for m in members[i:i + take]: split[m] = s
            i += take
    for d in qa: d['split'] = split[str(d['subject_id'])]


def repair_splits(qa, spare, t):
    """Removes a small difference left by the solver: a question moves between splits by removing it in one and adding another valid question of the same template and answer in the other."""
    rng = random.Random(SEED)
    want = t['template_split_counts']
    key = lambda d: (str(d['subject_id']), mask(d['question']), str(d.get('_scan')))  # one question of a template per session
    split_of = {str(d['subject_id']): d['split'] for d in qa}
    used = {key(d) for d in qa}; per_subject = collections.Counter(str(d['subject_id']) for d in qa)
    items = collections.defaultdict(list)
    for d in sorted(qa, key=lambda d: (str(d['subject_id']), d['question'])): items[d['split'], mask(d['question']), d['answer'] if d['question_type'] == 'YN' else None].append(d)
    extra = collections.defaultdict(list)
    for d in sorted(spare, key=lambda d: (str(d['subject_id']), str(d.get('_scan')), d['question'])):
        s = split_of.get(str(d['subject_id']))
        if s and key(d) not in used: extra[s, mask(d['question']), d['answer'] if d['question_type'] == 'YN' else None].append(d)
    for v in list(items.values()) + list(extra.values()): rng.shuffle(v)
    def swap(cell_out, cell_in):
        """Removes one question from cell_out and adds one from the spare questions of cell_in."""
        d = next((x for x in items[cell_out] if per_subject[str(x['subject_id'])] > 1), None)
        c = next((x for x in extra[cell_in] if key(x) not in used), None)
        if d is None or c is None: return False
        items[cell_out].remove(d); per_subject[str(d['subject_id'])] -= 1
        c = dict(c); c['split'] = cell_in[0]; extra[cell_in] = [x for x in extra[cell_in] if key(x) != key(c)]
        items[cell_in].append(c); per_subject[str(c['subject_id'])] += 1; used.add(key(c))
        return True
    def count(s, m): return sum(len(v) for (a, b, _), v in items.items() if a == s and b == m)
    def answers(s, a): return sum(len(v) for (x, _, y), v in items.items() if x == s and y == a)
    moves = 0
    for m in sorted(t['template_counts']):
        for s_to in SPLITS:
            while count(s_to, m) < want[s_to].get(m, 0):
                s_from = next((x for x in SPLITS if count(x, m) > want[x].get(m, 0)), None)
                if s_from is None: break
                order = sorted(('Yes', 'No') if m in t['yes_no_answers'] else (None,), key=lambda a: a and answers(s_to, a))
                if not any(swap((s_from, m, a), (s_to, m, a)) for a in order): break
                moves += 1
    # Equal Yes and No in each split: one template gives a Yes for a No in one split, and the reverse in the other.
    for _ in range(1000):
        high = [s for s in SPLITS if answers(s, 'Yes') > answers(s, 'No')]; low = [s for s in SPLITS if answers(s, 'Yes') < answers(s, 'No')]
        if not high or not low: break
        p, q = high[0], low[0]
        done = False
        for m in sorted(t['yes_no_answers']):
            if items[p, m, 'Yes'] and items[q, m, 'No'] and extra[p, m, 'No'] and extra[q, m, 'Yes'] and swap((p, m, 'Yes'), (p, m, 'No')):
                if swap((q, m, 'No'), (q, m, 'Yes')): done = True; moves += 2; break
        if not done: break
    print(f'questions exchanged to reach the exact split counts: {moves}')
    return [d for v in items.values() for d in v]


def corrections(qa, merged, session_col, ds):
    """Corrections of v1.1, written after selection, splits and the count checks, so the question set is the same.
    Gold: R27 "within" questions and reversed wording (T185, T186, T187, T190) get the opposite answer; "above-average" is Yes only for the Above group.
    Text: the cognitive-score threshold left after the age ("71.0" becomes "71"); HCP-Development does not get the HCP-Aging range "55-78 years old"."""
    sid_col = next(c for c in merged.columns if c.endswith('_subject_id'))
    rows = {(str(r[sid_col]), str(r[session_col])): r for r in merged.to_dict('records')}
    n = collections.Counter()
    for d in qa:
        f = d.pop('_gold_fix', None)
        if f == 'group': f = 'Yes' if rows.get((str(d['subject_id']), str(d['session_date'])), {}).get('cognitive_group') == 'Above' else 'No'
        if f and f != d['answer']: d['answer'] = f; n['Gold'] += 1
        q = re.sub(r'(subject age: \d+)\.0\)', r'\1)', d['question'])
        if ds == 'HCP-Development': q = q.replace(' (55-78 years old)', '')
        if q != d['question']: d['question'] = q; n['text'] += 1
    s = {x: tuple(sum(d['split'] == x and d['question_type'] == 'YN' and d['answer'] == a for d in qa) for a in ('Yes', 'No')) for x in SPLITS}
    print(f"corrections written after the count checks: Gold {n['Gold']}, text {n['text']}; Yes, No per split now {s}")
    return qa


def report(qa, t):
    bad = 0
    subj = collections.defaultdict(set)
    for d in qa: subj[str(d['subject_id'])].add(d['split'])
    overlap = sum(len(v) > 1 for v in subj.values())
    print(f'total {len(qa)}; subjects {len(subj)}; subjects in more than one split {overlap}')
    print(f"subjects per split: { {s: sum(s in v for v in subj.values()) for s in SPLITS} }; questions without session_date: {sum(not d.get('session_date') for d in qa)}")
    for s in SPLITS:
        items = [d for d in qa if d['split'] == s]
        yes, no = [sum(d['question_type'] == 'YN' and d['answer'] == a for d in items) for a in ('Yes', 'No')]
        pos = collections.Counter(d['options'].index(d['answer']) for d in items if d['question_type'] == 'MCQ' and d.get('options') and d['answer'] in d['options'])
        print(f'{s}: {len(items)} questions, Yes {yes}, No {no}, MCQ answer positions {sorted(pos.values())}')
        bad += yes != no
    diff = {}
    for s in SPLITS:
        c = collections.Counter(mask(d['question']) for d in qa if d['split'] == s)
        for m in {*c, *t['template_split_counts'][s]}:
            if c[m] != t['template_split_counts'][s].get(m, 0): diff[m[:70], s] = c[m] - t['template_split_counts'][s].get(m, 0)
    print(f'template x split cells with another count: {len(diff)}; total difference {sum(map(abs, diff.values()))}')
    for k, v in sorted(diff.items()): print('  ', k, f'{v:+d}')
    bad += bool(diff) + bool(overlap)
    print('ALL COUNT CHECKS PASSED' if not bad else 'WARNING: some counts differ from the published ones. The usual cause is another version of the source files.')


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--out', required=True, help='output JSON')
    x, source = a.parse_known_args()  # the other arguments go to the metadata step, see README.md
    if os.path.exists(x.out): sys.exit('refusing to overwrite --out')
    ds = 'HCP-Aging'; short, session_col, bins = DATASETS[ds]
    t = json.load(open(os.path.join(CODE, f'targets_{ds}.json')))
    sys.path.insert(0, CODE)
    # Intermediate files hold licensed data, so they stay beside the output and are removed at the end.
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(x.out))) as work:
        log = os.path.join(work, 'run.log'); gen_dir = os.path.join(work, 'gen'); os.makedirs(gen_dir); merged_csv = os.path.join(gen_dir, f'{short}_neuroqa_merged.csv')
        run_script(f'01_build_{short}_metadata.py', [*source, '--output', merged_csv], log)
        if ds == 'PPMI':  # the baseline UPDRS-III for motor severity comes from the same study data folder
            os.environ['PPMI_UPDRS3_RAW'] = sorted(glob.glob(os.path.join(source[source.index('--studydata') + 1], 'MDS-UPDRS_Part_III*.csv')))[-1]
        if ds == 'ADNI': add_adni_freesurfer(merged_csv, source[source.index('--extracted') + 1])
        # Wording is applied last, as in the release: selection follows the generator's original text.
        sys.modules['wording_fixes'] = types.SimpleNamespace(apply_wording_fixes=lambda q: 0)
        run_script(f'02_generate_{short}_qa.py', ['--input', merged_csv, '--output', gen_dir + os.sep], log)
        del sys.modules['wording_fixes']
        raw = json.load(open(os.path.join(gen_dir, f'{short}_qa_raw_all.json'))); final_file = os.path.join(gen_dir, f'{short}_qa_final.json')
        final = json.load(open(final_file)) if os.path.exists(final_file) else json.loads(json.dumps(raw))  # the PPMI generator leaves selection to the cleanup rules
        for d in final + raw: d['_scan'] = d['neuroqa_id']; d['_generator_answer'] = d['answer']
        cb, qa = run_cleanup(os.path.join(work, 'a'), final, merged_csv, ds, short, session_col, log)
        _, spare = run_cleanup(os.path.join(work, 'b'), raw, merged_csv, ds, short, session_col, log, select=False)
        merged = pd.read_csv(merged_csv, low_memory=False)
        from wording_fixes import apply_wording_fixes
        for part in (qa, spare):
            link_to_session(part, merged, ds, session_col, bins); apply_wording_fixes(part); session_text(part)
        before = len(qa); qa, spare = usable(qa, merged, session_col), usable(spare, merged, session_col)
        print(f'questions left out because their session cannot support them: {before - len(qa)}')
        qa = fit_counts(qa, spare, t)
        qa = sorted(qa, key=cb.sort_key)
        assign_splits(qa, t)
        qa = sorted(repair_splits(qa, spare, t), key=cb.sort_key)
        out = []
        for s in SPLITS: out += cb.apply_R25_mcq_position([d for d in qa if d['split'] == s])
        ids = list(range(1, len(out) + 1)); random.Random(SEED).shuffle(ids)
        for i, d in zip(ids, sorted(out, key=cb.sort_key)):
            d['neuroqa_id'] = f'{ds}-GEN-{i:05d}'; d['tier'] = 'dua'
        report(out, t)
        corrections(out, merged, session_col, ds)
        for d in out:
            d.pop('_scan', None); d.pop('_generator_answer', None)
        out.sort(key=lambda d: d['neuroqa_id'])
        held = sum(d['split'] == 'test_private' for d in out); out = [d for d in out if d['split'] != 'test_private']; print(f'test_private held out (not written): {held} questions')
        json.dump(out, open(x.out, 'x'), indent=1, sort_keys=True)
        print(f'{len(out)} questions written to {x.out}')


if __name__ == '__main__':
    main()
