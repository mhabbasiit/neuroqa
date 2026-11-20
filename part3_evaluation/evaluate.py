"""NeuroQA v1.1 evaluation.
Scores every predictions file in --pred_dir against the v1.1 test_public answers.
  predictions file: <Model>__<condition>.jsonl (or <Model>__<condition>__seed<k>.jsonl, averaged over seeds);
                    one JSON object per line with "neuroqa_id" and "response"
  closed questions (YN, MCQ): the answer is read from the response (see extract_answer) and compared with the gold answer
  open questions: exact match after normalization and removal of a single trailing period (on both sides)
Every model is scored on every item of the gold file; an item without a response counts as wrong.
Text-only reference: held-out majority per template, restricted to each item's own options (see majority_floor).
Usage: python evaluate.py --gold test.json --train train.json --pred_dir predictions --out results"""
import argparse, collections, glob, json, os, re, statistics

# ---------------------------------------------------------------- normalization (exact match; same as the released NeuroQA evaluate.py)
def normalize(text):
    if text is None: return ''
    text = str(text).lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return re.sub(r'[^\w\s%.,\-]', '', text)


def strip_period(t):
    return t[:-1].rstrip() if t.endswith('.') else t


def options(q):
    o = q.get('options'); return o if isinstance(o, list) else (json.loads(o) if isinstance(o, str) and o.startswith('[') else [])


# ---------------------------------------------------------------- answer extraction for closed questions
LEAD_YN = re.compile(r'^\W*(?:answer\W*)?(yes|no)\b', re.I)
LEAD_LETTER = re.compile(r'^\W*(?:(?:final\s+)?answer(?:\s+is)?\W*)?\(?([A-D])(?:\)|\.|:|,|\s|$)', re.I)
STATED = re.compile(r'(?:(?:best|correct|final)\s+answer\s+is|answer\s*:)\W*\(?(yes|no|[A-D])\b', re.I)
ANSWER_LINE = re.compile(r'^[\W_]*(?:(?:final\s+)?answer\s*(?:is)?[\W_]*)?\(?([A-D]|yes|no)\)?(?:[.:)]\s*(.*?))?[\W_]*$', re.I)
BOLD = re.compile(r'\*\*(.+?)\*\*', re.S)
BOLD_ANSWER = re.compile(r'^\W*(?:(?:final\s+)?answer\s*(?:is)?\W*)?\(?([A-D]|yes|no)\)?(?:[.:)]\s*(.*?))?\W*$', re.I)


def _choice(q, tok, rest):
    if q['question_type'] == 'YN':
        return tok.capitalize() if tok.lower() in ('yes', 'no') and not rest else None
    if tok.lower() in ('yes', 'no'): return None
    o = options(q); i = ord(tok.upper()) - 65
    if i >= len(o): return None
    if rest and normalize(rest).rstrip('.') != normalize(o[i]).rstrip('.'): return None
    return o[i]


def extract_answer(q, response):
    """The answer stated in the response, or None if there is no clear answer. In order:
    1. the response starts with the answer (Yes/No, or an option letter), or states it ("answer: X", "the final answer is X"),
       or is exactly one option's text
    2. the last line of the response is only the answer (e.g. "**C**", "Answer: B", "Yes")
    3. bold answer markers (e.g. **C**, **Yes**) that all name the same choice; markers naming different choices -> no answer"""
    s = str(response).strip(); stated = STATED.findall(s); stated = stated[-1] if stated else None
    if q['question_type'] == 'YN':
        m = LEAD_YN.match(s)
        if m: return m.group(1).capitalize()
        if stated and stated.lower() in ('yes', 'no'): return stated.capitalize()
    else:
        o = options(q); m = LEAD_LETTER.match(s)
        let = m.group(1) if m else (stated if stated and len(stated) == 1 else None)
        if let and ord(let.upper()) - 65 < len(o): return o[ord(let.upper()) - 65]
        for oo in o:
            if normalize(s).rstrip('.') == normalize(oo).rstrip('.'): return oo
    lines = [l for l in s.splitlines() if l.strip()]
    if lines:
        m = ANSWER_LINE.match(lines[-1].strip())
        c = _choice(q, m.group(1), (m.group(2) or '').strip()) if m else None
        if c is not None: return c
    picks = set()
    for b in BOLD.findall(s):
        m = BOLD_ANSWER.match(b.strip())
        c = _choice(q, m.group(1), (m.group(2) or '').strip()) if m else None
        if c is not None: picks.add(c)
    return picks.pop() if len(picks) == 1 else None


def correct(q, response):
    if response is None: return 0.0
    if q['question_type'] == 'Open': return float(strip_period(normalize(response)) == strip_period(normalize(q['answer'])))
    a = extract_answer(q, response)
    return 0.0 if a is None else float(normalize(a) == normalize(q['answer']))


# ---------------------------------------------------------------- tables
GROUPS = {'YN': lambda q: q['question_type'] == 'YN', 'MCQ': lambda q: q['question_type'] == 'MCQ', 'Closed': lambda q: q['question_type'] != 'Open',
          'Closed, image-grounded': lambda q: q['question_type'] != 'Open' and q['image_dependency'] == 'image-grounded',
          'Closed, image-informed': lambda q: q['question_type'] != 'Open' and q['image_dependency'] == 'image-informed',
          'Open EM': lambda q: q['question_type'] == 'Open'}


def scores(gold, pred):
    r = {g: 100 * statistics.mean(correct(q, pred.get(k)) for k, q in gold.items() if f(q)) for g, f in GROUPS.items()}
    cats = sorted({q['category'] for q in gold.values() if q['question_type'] != 'Open'})
    r['by_category'] = {c: 100 * statistics.mean(correct(q, pred.get(k)) for k, q in gold.items() if q['question_type'] != 'Open' and q['category'] == c) for c in cats}
    return r


def majority_floor(fit, items):
    """Held-out text-only reference. Template = question with numbers replaced by N. For each closed item, predict the item's own
    option (YN: Yes/No; MCQ: its options) with the highest count among TRAIN answers of the same template; ties go to the
    alphabetically first option; no answer position is used. Fallback (never needed on v1.1): counts over the item's
    (category, question_type), then the first option."""
    tmpl = lambda q: re.sub(r'\d[\d,.]*', 'N', q)
    c = collections.defaultdict(collections.Counter); cf = collections.defaultdict(collections.Counter)
    for d in fit: c[tmpl(d['question'])][str(d['answer'])] += 1; cf[(d['category'], d['question_type'])][str(d['answer'])] += 1
    def pred(d):
        opts = ['Yes', 'No'] if d['question_type'] == 'YN' else [str(o) for o in options(d)]
        for cnt in (c[tmpl(d['question'])], cf[(d['category'], d['question_type'])]):
            if any(cnt[o] for o in opts): return sorted(opts, key=lambda o: (-cnt[o], o))[0]
        return opts[0]
    return 100 * statistics.mean(pred(d) == str(d['answer']) for d in items)


if __name__ == '__main__':
    a = argparse.ArgumentParser(); a.add_argument('--gold', required=True); a.add_argument('--train', required=True)
    a.add_argument('--pred_dir', required=True); a.add_argument('--out', required=True); x = a.parse_args()
    os.makedirs(x.out, exist_ok=False)
    gold = {q['neuroqa_id']: q for q in json.load(open(x.gold))}
    closed = [q for q in gold.values() if q['question_type'] != 'Open']
    ref = {'Random': 100 * statistics.mean(0.5 if q['question_type'] == 'YN' else 1 / len(options(q)) for q in closed),
           'Text-only (held-out majority per template, option-constrained)': majority_floor([q for q in json.load(open(x.train)) if q['question_type'] != 'Open'], closed),
           'Text-only (majority answer per template, in-sample)': majority_floor(closed, closed)}
    runs = collections.defaultdict(list)
    for fp in sorted(glob.glob(os.path.join(x.pred_dir, '*.jsonl'))):
        model, cond = os.path.basename(fp)[:-6].split('__')[:2]
        runs[(model, cond)].append(scores(gold, {json.loads(l)['neuroqa_id']: json.loads(l)['response'] for l in open(fp)}))
    res = []
    for (model, cond), S in runs.items():
        r = {'model': model, 'condition': cond, 'seeds': len(S)}
        for g in GROUPS: r[g] = round(statistics.mean(s[g] for s in S), 1); r[g + ' sd'] = round(statistics.stdev(s[g] for s in S), 1) if len(S) > 1 else None
        r['by_category'] = {c: round(statistics.mean(s['by_category'][c] for s in S), 1) for c in S[0]['by_category']}
        res.append(r)
    json.dump({'n_items': {g: sum(f(q) for q in gold.values()) for g, f in GROUPS.items()}, 'reference': {k: round(v, 1) for k, v in ref.items()}, 'models': res},
              open(os.path.join(x.out, 'results.json'), 'w'), indent=1)
    fmt = lambda r, g: f"{r[g]}" + (f" ± {r[g + ' sd']}" if r[g + ' sd'] is not None else '')
    L = ['| Baseline | Closed |', '|---|---:|'] + [f'| {k} | {v:.1f} |' for k, v in ref.items()] + ['',
         '| Model | Input | ' + ' | '.join(GROUPS) + ' |', '|---|---|' + '---:|' * len(GROUPS)]
    L += [f"| {r['model']} | {r['condition']} | " + ' | '.join(fmt(r, g) for g in GROUPS) + ' |' for r in res]
    cats = list(res[0]['by_category']) if res else []
    L += ['', '| Model | Input | ' + ' | '.join(cats) + ' |', '|---|---|' + '---:|' * len(cats)]
    L += [f"| {r['model']} | {r['condition']} | " + ' | '.join(str(r['by_category'][c]) for c in cats) + ' |' for r in res]
    open(os.path.join(x.out, 'results.md'), 'w').write('\n'.join(L) + '\n'); print('\n'.join(L))
