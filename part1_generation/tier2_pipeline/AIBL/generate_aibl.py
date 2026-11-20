#!/usr/bin/env python3
"""NeuroQA AIBL generator. Builds the AIBL questions from AIBL source files. See README.md.

Steps: metadata, generator, all cleanup rules, session link fixes, final wording, subject split, count and balance step.
The files in code/ are unchanged except targets.json, which holds counts only; this script runs them in order and adds the listed steps.
"""
import argparse, collections, contextlib, copy, glob, importlib.util, json, math, os, random, re, runpy, shutil, sys, tempfile, types
sys.dont_write_bytecode = True
if os.environ.get('PYTHONHASHSEED') != '0':
    os.execve(sys.executable, [sys.executable, '-B', *(['-I'] if sys.flags.isolated else []), *sys.argv], {**os.environ, 'PYTHONHASHSEED': '0', 'PYTHONUTF8': '1'})
import pandas as pd

CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code')
SEED = 42
SPLITS = ('train', 'val', 'test_public', 'test_private')
AGE = re.compile(r'\(subject age: \d+\)')
AGE_BINS = ((55, 65, '55-64'), (65, 75, '65-74'), (75, 85, '75-84'), (85, 200, '85+'))
mask = lambda q: re.sub(r'\d[\d,.]*', 'N', q)
# Generator text (before wording fixes) of the questions whose answer uses that visit's MMSE.
USES_MMSE = {'Does this scan suggest cognitive impairment based on clinical assessment?', 'Is this scan consistent with normal cognitive function?',
             'What cognitive status is most consistent with this brain scan?', 'Does this scan suggest cognitive decline?',
             'Which combination of clinical findings and diagnosis best describes this subject?', 'Do the clinical findings suggest neurodegenerative changes in this subject?',
             'Is the overall cognitive function within normal limits for this subject?', 'Does the clinical pattern suggest cognitive impairment in this subject?',
             'Is there a mismatch between the cognitive scores and the clinical diagnosis?', 'What is the most prominent clinical finding for this subject?',
             'What cognitive severity level is most consistent with this subject?'}


def run_script(name, argv, log):
    sys.argv = [name, *argv]
    with open(log, 'a') as f, contextlib.redirect_stdout(f):
        runpy.run_path(os.path.join(CODE, name), run_name='__main__')


def load_cleanup():
    spec = importlib.util.spec_from_file_location('cleanup_benchmark', os.path.join(CODE, 'cleanup_benchmark.py'))
    cb = importlib.util.module_from_spec(spec); sys.modules['cleanup_benchmark'] = cb; spec.loader.exec_module(cb)
    return cb


def run_cleanup(cb, base, log):
    """Runs all cleanup rules. Also returns the questions as they were just before selection (R23)."""
    real_glob, real_r23, before = glob.glob, cb.apply_R23_yn_balance, []
    glob.glob = lambda p, **k: []  # the rules look for image files; this script does not use images
    def r23(qa):
        before.extend(copy.deepcopy(qa))
        return real_r23(qa)
    cb.apply_R23_yn_balance = r23
    sys.argv = ['cleanup_benchmark.py', '--base-dir', base]
    with open(log, 'a') as f, contextlib.redirect_stdout(f):
        cb.main()
    glob.glob, cb.apply_R23_yn_balance = real_glob, real_r23
    return json.load(open(os.path.join(base, 'neuroqa_all_qa_v4.json'))), before


def link_to_session(qa, spare_of, merged):
    """Connects each question to the session it was generated from, and replaces questions that session cannot support.

    R36 takes the first image of the subject, R37 the last two sessions, and R8, R9 and R38 the age of the subject's last row, so all can come from another session.
    No image is read and no image path is written: users match session_date to their own files.
    """
    by_scan = merged.set_index('neuroqa_id')
    def unusable(d):  # MMSE below zero is a LONI missing code, not a score.
        mmse = pd.to_numeric(by_scan.MMSE.get(d['_scan']), errors='coerce')
        uses_mmse = d['question'] in USES_MMSE or (d['question'] == 'Which of the following findings is present in this subject?' and d['answer'] == 'Below-median MMSE')
        return 'mmse' if mmse < 0 and uses_mmse else None

    key = lambda d: (str(d['subject_id']), mask(d['question']))
    kept, removed = [d for d in qa if not unusable(d)], [d for d in qa if unusable(d)]
    used = {key(d) for d in kept}
    spare = sorted((d for d in spare_of(removed) if key(d) not in used and not unusable(d)), key=lambda d: (key(d), str(d['_scan'])))
    random.Random(SEED).shuffle(spare)
    for d in removed:  # a question from the same template, with the same answer when there is one
        same = [c for c in spare if mask(c['question']) == mask(d['question']) and c['question_type'] == d['question_type'] and key(c) not in used]
        pick = next((c for c in same if c['answer'] == d['answer']), same[0] if same else None)
        if pick is None:
            print(f"no replacement for a question of: {d['question']}"); continue
        kept.append(pick); used.add(key(pick))
    print(f'questions replaced: {collections.Counter(unusable(d) for d in removed)}')

    for d in kept:
        row = by_scan.loc[d['_scan']]
        d['session_date'] = str(row.session_date)
        if d['category'] == 'Longitudinal':  # R37 sets the last two sessions, but Gold comes from this session and the one before it
            sessions = sorted(merged[merged.aibl_subject_id == row.aibl_subject_id].session_date.astype(str))
            d['prior_session_date'] = sessions[sessions.index(d['session_date']) - 1]
        else:
            if 'age range' in d['question'].lower():
                d['answer'] = next((b for lo, hi, b in AGE_BINS if lo <= float(row.age) < hi and b in d['options']), d['answer'])
        subject = merged[merged.aibl_subject_id == row.aibl_subject_id]
        own = lambda s, col='T1w_image_path': next((v for v in subject[subject.session_date.astype(str) == s].get(col, []) if isinstance(v, str) and v), None)  # from your own sMRI metadata
        d['image_path'] = own(d['session_date']) or '#'
        d.pop('image_path_prior', None); d.pop('image_path_t2', None)
        if own(d['session_date'], 'T2w_image_path'): d['image_path_t2'] = own(d['session_date'], 'T2w_image_path')
        if d.get('prior_session_date'): d['image_path_prior'] = own(d['prior_session_date']) or '#'
    return kept, by_scan


def build(x):
    """Runs every step up to the split and returns (cleanup module, questions)."""
    outputs = os.path.join(x.work, 'AIBL', 'outputs'); os.makedirs(outputs); log = os.path.join(x.work, 'run.log')
    merged_csv = os.path.join(outputs, 'aibl_neuroqa_merged.csv'); sys.path.insert(0, CODE)

    run_script('01_build_aibl_metadata.py', ['--t1meta', x.t1meta, '--t2meta', x.t2meta, '--metadata', x.metadata, '--output', merged_csv], log)
    # Wording is applied last, as in the release: selection follows the generator's original text.
    sys.modules['wording_fixes'] = types.SimpleNamespace(apply_wording_fixes=lambda q: 0)
    run_script('02_generate_aibl_qa.py', ['--input', merged_csv, '--output', outputs + os.sep], log)
    del sys.modules['wording_fixes']
    gen = os.path.join(outputs, 'aibl_qa_final.json'); g = json.load(open(gen))
    for d in g: d['_scan'] = d['neuroqa_id']
    json.dump(g, open(gen, 'w'))

    merged = pd.read_csv(merged_csv, low_memory=False)
    cb = load_cleanup(); qa, _ = run_cleanup(cb, x.work, log)
    def spare_of(removed):
        """All generator questions of the removed questions' templates, passed through the same cleanup rules up to selection."""
        base = os.path.join(x.work, 'spare'); out = os.path.join(base, 'AIBL', 'outputs'); os.makedirs(out); shutil.copy(merged_csv, out)
        texts = {d['question'] for d in removed}
        raw = [d for d in json.load(open(os.path.join(outputs, 'aibl_qa_raw_all.json'))) if d['question'] in texts]
        for d in raw: d['_scan'] = d['neuroqa_id']
        json.dump(raw, open(os.path.join(out, 'aibl_qa_final.json'), 'w'))
        return run_cleanup(cb, base, log)[1]
    qa, by_scan = link_to_session(qa, spare_of, merged)

    from wording_fixes import apply_wording_fixes
    apply_wording_fixes(qa)
    for d in qa:
        d['question'] = AGE.sub(f"(subject age: {int(round(float(by_scan.age[d['_scan']])))})", d['question'])
        d['tier'] = 'dua'
    return cb, sorted(qa, key=cb.sort_key)


def assign_splits(qa, t):
    """Subject-level split with the published count of each template in each split: a seeded shuffle, then subject swaps (simulated annealing) until the difference is zero."""
    rng = random.Random(SEED)
    want, yes_no = t['template_split_counts'], set(t['yes_no_templates'])
    half = {s: sum(n for m, n in want[s].items() if m in yes_no) // 2 for s in SPLITS}
    count = collections.defaultdict(collections.Counter)
    for d in qa:
        c = count[str(d['subject_id'])]; c[mask(d['question'])] += 1
        if d['question_type'] == 'YN': c['answer|' + d['answer']] += 1
    subjects = sorted(count); rng.shuffle(subjects)
    k = t['subjects_per_test_val']
    split = dict(zip(subjects, ['train'] * (len(subjects) - 3 * k) + ['val'] * k + ['test_public'] * k + ['test_private'] * k))
    total = {s: collections.Counter() for s in SPLITS}
    for s in subjects: total[split[s]].update(count[s])

    def miss(s, key, have):  # extra Yes/No questions are removed later, so only a shortage counts for them
        if key.startswith('answer|'): return max(0, half[s] - have)
        n = want[s].get(key, 0)
        return max(0, n - have) if key in yes_no else abs(have - n)
    now = sum(miss(s, key, total[s][key]) for s in SPLITS for key in {*want[s], *total[s], 'answer|Yes', 'answer|No'})
    start_temperature, end_temperature = 0.5, 0.2
    for i in range(t['max_swaps']):
        if not now: break
        a, b = rng.sample(subjects, 2)
        p, q = split[a], split[b]
        if p == q: continue
        delta = 0
        for key in {*count[a], *count[b]}:
            change = count[b][key] - count[a][key]
            delta += miss(p, key, total[p][key] + change) - miss(p, key, total[p][key]) + miss(q, key, total[q][key] - change) - miss(q, key, total[q][key])
        temperature = start_temperature + (end_temperature - start_temperature) * i / t['max_swaps']
        if delta <= 0 or rng.random() < math.exp(-delta / temperature):
            for key in {*count[a], *count[b]}:
                change = count[b][key] - count[a][key]
                total[p][key] += change; total[q][key] -= change
            split[a], split[b] = q, p; now += delta
    print(f'template x split difference after swaps: {now}')
    for d in qa: d['split'] = split[str(d['subject_id'])]


def choose_drops(qa, t):
    """Picks the extra Yes/No questions to remove: exact count per template and split, and Yes = No within each split."""
    want, yes_no = t['template_split_counts'], set(t['yes_no_templates'])
    rng = random.Random(SEED)
    need_cell, need_answer, groups = collections.Counter(), collections.Counter(), collections.defaultdict(list)
    per_subject = collections.Counter(str(d['subject_id']) for d in qa)
    for s in SPLITS:
        half = sum(n for m, n in want[s].items() if m in yes_no) // 2
        need_answer[s, 'Yes'] = need_answer[s, 'No'] = -half
        for m, n in want[s].items():
            if m in yes_no: need_cell[m, s] = -n
    for d in qa:
        if d['question_type'] == 'YN':
            m, s, a = mask(d['question']), d['split'], d['answer']
            need_cell[m, s] += 1; need_answer[s, a] += 1
            if per_subject[str(d['subject_id'])] > 1: groups[m, s, a].append(d)  # every subject keeps at least one question
    if any(v < 0 for v in need_cell.values()) or any(v < 0 for v in need_answer.values()):
        print('SHORTAGE: not enough valid Yes/No questions for the published counts; see the report at the end of the run')
        return set()
    todo = sorted(c for c, n in need_cell.items() for _ in range(n))
    def solve(i, taken):
        if i == len(todo): return True
        m, s = todo[i]
        for a in ('Yes', 'No'):
            if need_answer[s, a] > 0 and taken[m, s, a] < len(groups[m, s, a]):
                need_answer[s, a] -= 1; taken[m, s, a] += 1
                if solve(i + 1, taken): return True
                need_answer[s, a] += 1; taken[m, s, a] -= 1
        return False
    taken = collections.Counter()
    if not solve(0, taken) or any(need_answer.values()):
        print('SHORTAGE: exact counts are not reachable with these source files; see the report at the end of the run')
        return set()
    drop = set()
    for g, n in sorted(taken.items()):
        items = sorted(groups[g], key=lambda d: (str(d['subject_id']), str(d['_scan']))); rng.shuffle(items)
        for d in items:
            if n and per_subject[str(d['subject_id'])] > 1:
                drop.add(id(d)); per_subject[str(d['subject_id'])] -= 1; n -= 1
    return drop


def balance(cb, qa, t):
    drop = choose_drops(qa, t)
    out = []
    for s in SPLITS:
        out += cb.apply_R25_mcq_position([d for d in qa if d['split'] == s and id(d) not in drop])
    ids = list(range(1, len(out) + 1)); random.Random(SEED).shuffle(ids)
    for i, d in zip(ids, sorted(out, key=cb.sort_key)): d['neuroqa_id'] = f'AIBL-GEN-{i:05d}'  # a separate prefix: these are not the release question IDs
    return out


def report(qa, t):
    """Prints the checks to the screen. Any difference from the published counts is reported here."""
    bad = 0
    subj = collections.defaultdict(set)
    for d in qa: subj[str(d['subject_id'])].add(d['split'])
    overlap = sum(len(v) > 1 for v in subj.values())
    print(f'total {len(qa)}; subjects {len(subj)}; subjects in more than one split {overlap}')
    print(f"subjects per split: { {s: sum(s in v for v in subj.values()) for s in SPLITS} }; questions without session_date: {sum(not d.get('session_date') for d in qa)}")
    for s in SPLITS:
        items = [d for d in qa if d['split'] == s]
        c = collections.Counter(f"{d['category']}|{d['question_type']}" for d in items)
        diff = {k: c[k] - t['split_counts'][s].get(k, 0) for k in set(c) | set(t['split_counts'][s]) if c[k] != t['split_counts'][s].get(k, 0)}
        yes, no = [sum(d['question_type'] == 'YN' and d['answer'] == a for d in items) for a in ('Yes', 'No')]
        pos = collections.Counter(d['options'].index(d['answer']) for d in items if d['question_type'] == 'MCQ' and d.get('options') and d['answer'] in d['options'])
        print(f'{s}: {len(items)} questions, Yes {yes}, No {no}, MCQ answer positions {sorted(pos.values())}, count differences {diff or "none"}')
        bad += bool(diff) + (yes != no)
    c = collections.Counter(mask(d['question']) for d in qa)
    diff = {k: c[k] - v for k, v in t['template_counts'].items() if c[k] != v}; extra = [k for k in c if k not in t['template_counts']]
    print(f'templates with another count: {len(diff) + len(extra)}')
    bad += bool(diff or extra or overlap)
    shortage = {}
    for s in SPLITS:
        c = collections.Counter(mask(d['question']) for d in qa if d['split'] == s)
        for m in {*c, *t['template_split_counts'][s]}:
            if c[m] != t['template_split_counts'][s].get(m, 0): shortage[m[:60], s] = c[m] - t['template_split_counts'][s].get(m, 0)
    print(f'template x split cells with another count: {len(shortage)}; total difference {sum(map(abs, shortage.values()))}')
    for k, v in sorted(shortage.items()): print('  ', k, f'{v:+d}')
    bad += bool(shortage)
    print('ALL COUNT CHECKS PASSED' if not bad else 'WARNING: some counts differ from the published ones. The usual cause is another version of the source files.')


def write(qa, path):
    for d in qa: d.pop('_scan', None)
    qa.sort(key=lambda d: d['neuroqa_id'])
    json.dump(qa, open(path, 'w'), indent=1, sort_keys=True)
    print(f'{len(qa)} questions written to {path}')


def main():
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument('--t1meta', required=True, help='AIBL_T1_metadata.csv')
    a.add_argument('--t2meta', required=True, help='AIBL_T2_metadata.csv')
    a.add_argument('--metadata', required=True, help='folder with the aibl_*.csv LONI tables')
    a.add_argument('--out', required=True, help='output JSON')
    x = a.parse_args()
    if os.path.exists(x.out): sys.exit('refusing to overwrite --out')
    t = json.load(open(os.path.join(CODE, 'targets.json')))
    # Intermediate files hold licensed data, so they stay beside the output and are removed at the end.
    with tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(x.out))) as temp:
        x.work = os.path.join(temp, 'work')
        cb, qa = build(x)
        assign_splits(qa, t)
        qa = balance(cb, qa, t)
        report(qa, t)
        for d in qa: d.pop('_gold_fix', None)
        held = sum(d['split'] == 'test_private' for d in qa); qa = [d for d in qa if d['split'] != 'test_private']; print(f'test_private held out (not written): {held} questions')
        write(qa, x.out)


if __name__ == '__main__':
    main()
