"""Longer day summaries for the three hourly day cards.

Reads one day's hourly rows and tells, in a few sentences, which weather is
forecast, when it starts and ends, and how one kind of weather turns into the
next (rain -> clearing, jugo -> bura, ...). Deterministic: same input, same
text. Row hour T covers T..T+1 (precipitation is start-of-hour aligned).
"""
import math

RAIN_MM = 0.1          # an hour is wet above this
MERGE_GAP = 2          # dry gaps up to this many hours stay inside one episode
MIN_SKY_RUN = 3        # sky runs shorter than this merge into a neighbour
MAX_EVENT_SENTENCES = 5

_PARTS = (('noc', 0, 6), ('jutro', 6, 12), ('popodne', 12, 18), ('vece', 18, 24))
_PART_LOC = {'noc': 'noću', 'jutro': 'ujutru', 'popodne': 'popodne', 'vece': 'uveče'}
_PART_GEN = {'noc': 'noći', 'jutro': 'jutra', 'popodne': 'popodneva', 'vece': 'večeri'}


def _num(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(v) else v


def _cap(s):
    return s[:1].upper() + s[1:] if s else s


def _mm(total):
    if total < 1:
        return ''
    return f" (oko {total:.0f} mm)" if total >= 5 else f" (oko {total:.1f} mm)"


def _span_label(h0, h1, win0, win1):
    """Time-of-day words for hours h0..h1-1 inside the window win0..win1-1."""
    if h0 <= win0 and h1 >= win1 and (win1 - win0) >= 12:
        return 'tokom cijelog dana'
    parts = [p for p, a, b in _PARTS if min(h1, b) - max(h0, a) >= 2]
    if not parts:
        parts = [next(p for p, a, b in _PARTS if a <= h0 < b)]
    if len(parts) == 1:
        return _PART_LOC[parts[0]]
    if len(parts) == 2:
        return f"tokom {_PART_GEN[parts[0]]} i {_PART_GEN[parts[1]]}"
    return 'tokom većeg dijela dana'


def _hour_kind(pr, wc):
    wc = wc or 0
    wet = pr is not None and pr > RAIN_MM
    if wc >= 95:
        return 'thunder'
    if wet:
        if 71 <= wc <= 77 or 85 <= wc <= 86:
            return 'snow'
        if 80 <= wc <= 82:
            return 'shower'
        return 'rain'
    if 45 <= wc <= 48:
        return 'fog'
    return 'dry'


def _sky_word(c):
    if c is None:
        return None
    if c < 30:
        return 'vedro'
    if c < 50:
        return 'pretežno vedro'
    if c < 65:
        return 'djelimično oblačno'
    if c < 85:
        return 'pretežno oblačno'
    return 'oblačno'


def _sky_group(c):
    if c is None:
        return None
    return 0 if c < 50 else (1 if c < 65 else 2)


# --- precipitation episodes ------------------------------------------------

# noun, gender-agreeing pieces: start verb, ongoing verb, end verb,
# relative pronoun, "possible" predicate
_NOUNS = {
    'thunder': ('grmljavina s pljuskovima', 'počinje', 'traje', 'smiruje se', 'koja', 'moguća je'),
    'snow':    ('snijeg', 'počinje', 'pada', 'prestaje', 'koji', 'moguć je'),
    'shower':  ('povremeni pljuskovi', 'počinju', 'se javljaju', 'prestaju', 'koji', 'mogući su'),
    'rain':    ('kiša', 'počinje', 'pada', 'prestaje', 'koja', 'moguća je'),
}


def _episodes(hours, kinds, precip):
    """Group wet hours into episodes; short dry gaps stay inside an episode."""
    wet_idx = [i for i, k in enumerate(kinds) if k in _NOUNS]
    groups = []
    for i in wet_idx:
        if groups and hours[i] - hours[groups[-1][-1]] <= MERGE_GAP + 1:
            groups[-1].append(i)
        else:
            groups.append([i])
    eps = []
    for g in groups:
        ks = [kinds[i] for i in g]
        prs = [precip[i] or 0.0 for i in g]
        if 'snow' in ks:
            kind = 'snow'
        elif 'thunder' in ks:
            kind = 'thunder'
        elif ks.count('shower') * 2 >= len(ks):
            kind = 'shower'
        else:
            kind = 'rain'
        peak_i = g[max(range(len(g)), key=lambda j: prs[j])]
        eps.append({
            'start': hours[g[0]], 'end': hours[g[-1]] + 1,
            'kind': kind, 'total': sum(prs), 'peak': max(prs),
            'peak_hour': hours[peak_i],
            'gappy': (hours[g[-1]] - hours[g[0]] + 1) > len(g),
            'thunder_hours': [hours[i] for i in g if kinds[i] == 'thunder'],
        })
    return eps


def _episode_noun(ep):
    noun, start_v, cont_v, end_v, rel, maybe = _NOUNS[ep['kind']]
    if ep['kind'] == 'rain':
        if ep['peak'] >= 4 or ep['total'] >= 10:
            noun = 'jaka kiša'
        elif ep['peak'] < 0.5 and ep['total'] < 1:
            noun = 'slaba kiša'
    elif ep['kind'] == 'shower' and ep['peak'] >= 4:
        noun = 'jaki pljuskovi'
    return noun, start_v, cont_v, end_v, rel, maybe


def _episode_sentence(ep, win0, win1):
    noun, start_v, cont_v, end_v, rel, maybe = _episode_noun(ep)
    s, e = ep['start'], ep['end']
    mm = _mm(ep['total'])
    gaps = ' s prekidima' if ep['gappy'] and (e - s) >= 4 else ''
    till_end = 'do kraja dana' if win1 >= 24 else 'do kraja prognoze'
    if s <= win0 and e >= win1:
        txt = f"{_cap(noun)}{gaps} {cont_v} {_span_label(s, e, win0, win1)}{mm}."
    elif s <= win0:
        txt = f"{_cap(noun)}{gaps} {cont_v} {_span_label(s, e, win0, win1)} i {end_v} oko {e} h{mm}."
    elif e >= win1:
        txt = f"Oko {s} h {start_v} {noun}, {rel} {cont_v} {till_end}{mm}."
    elif e - s <= 2:
        txt = f"Između {s} i {e} h {maybe} {noun}{mm}."
    else:
        txt = f"Od oko {s} do {e} h {cont_v} {noun}{gaps}{mm}."
    # strongest burst inside a long episode
    if (e - s) >= 4 and ep['peak'] >= 2 and ep['kind'] != 'snow':
        most = 'najjači' if ep['kind'] == 'shower' else 'najjača'
        txt = txt[:-1] + f", {most} oko {ep['peak_hour']} h."
    # thunder that only sits in part of a longer rain episode
    th = ep['thunder_hours']
    if ep['kind'] == 'thunder' and th and (e - s) >= 6 and (th[-1] - th[0] + 1) <= (e - s) / 2:
        txt += f" Grmljavina je najvjerovatnija između {th[0]} i {th[-1] + 1} h."
    return txt


# --- dry spells: sky runs ---------------------------------------------------

def _sky_runs(hs, clouds):
    """Runs of the coarse sky group over consecutive dry hours, smoothed."""
    runs = []
    for h, c in zip(hs, clouds):
        g = _sky_group(c)
        if g is None:
            continue
        if runs and runs[-1]['g'] == g:
            runs[-1]['end'] = h + 1
            runs[-1]['cc'].append(c)
        else:
            runs.append({'g': g, 'start': h, 'end': h + 1, 'cc': [c]})
    changed = True
    while changed and len(runs) > 1:
        changed = False
        for i, r in enumerate(runs):
            if r['end'] - r['start'] < MIN_SKY_RUN:
                j = i - 1 if i > 0 else i + 1
                tgt = runs[j]
                tgt['start'] = min(tgt['start'], r['start'])
                tgt['end'] = max(tgt['end'], r['end'])
                tgt['cc'] = tgt['cc'] + r['cc']
                runs.pop(i)
                changed = True
                break
        # re-join neighbours that now share a group
        k = 0
        while k < len(runs) - 1:
            if runs[k]['g'] == runs[k + 1]['g']:
                runs[k]['end'] = runs[k + 1]['end']
                runs[k]['cc'] += runs[k + 1]['cc']
                runs.pop(k + 1)
            else:
                k += 1
    for r in runs:
        r['word'] = _sky_word(sum(r['cc']) / len(r['cc']))
    return runs


def _sky_change(prev, cur, again=False):
    """Clause (no capital, no full stop) for a change between two sky runs."""
    h = cur['start']
    rep = 'ponovo ' if again else ''
    if cur['g'] > prev['g']:
        return (f"oko {h} h se {rep}naoblačuje" if cur['g'] == 2
                else f"od oko {h} h oblačnost se {rep}povećava")
    return (f"oko {h} h se {rep}razvedrava" if cur['g'] == 0
            else f"od oko {h} h oblačnost se smanjuje")


def _after_rain(run):
    if run['g'] == 0:
        return "potom se razvedrava"
    if run['g'] == 1:
        return "potom suvo, uz djelimičnu oblačnost"
    return "potom suvo, ali i dalje oblačno"


def _join_clauses(clauses):
    if len(clauses) == 1:
        return _cap(clauses[0]) + '.'
    head, rest = clauses[0], clauses[1:]
    if len(rest) == 1:
        return f"{_cap(head)}, a {rest[0]}."
    return f"{_cap(head)}, {', '.join(rest[:-1])}, a {rest[-1]}."


# --- wind ---------------------------------------------------------------------

# name, gender ('m','f','n'), genitive, accusative
_WIND_NAMES = {
    'bura': ('bura', 'f', 'bure', 'buru'),
    'tramontana': ('tramontana', 'f', 'tramontane', 'tramontanu'),
    'jugo': ('jugo', 'n', 'juga', 'jugo'),
    'lebić': ('lebić', 'm', 'lebića', 'lebić'),
    'maestral': ('maestral', 'm', 'maestrala', 'maestral'),
}
_DIR8 = ['sjeverni', 'sjeveroistočni', 'istočni', 'jugoistočni',
         'južni', 'jugozapadni', 'zapadni', 'sjeverozapadni']


def _wind_key(deg, hour):
    d = deg % 360
    if d >= 337.5 or d < 22.5:
        return 'tramontana'
    if d < 67.5:
        return 'bura'
    if 112.5 <= d < 202.5:
        return 'jugo'
    if 202.5 <= d < 247.5:
        return 'lebić'
    if 292.5 <= d < 337.5 and 10 <= hour <= 19:
        return 'maestral'
    return _DIR8[int(((d + 22.5) % 360) // 45)]


def _wind_forms(key):
    if key in _WIND_NAMES:
        return _WIND_NAMES[key]
    return (f"{key} vjetar", 'm', f"{key[:-1]}og vjetra", f"{key} vjetar")


def _strength(peak, gender):
    if peak >= 14:
        base = 'vrlo jak'
    elif peak >= 10:
        base = 'jak'
    elif peak >= 7:
        base = 'umjereno jak'
    else:
        base = 'umjeren'
    end = {'m': '', 'f': 'a', 'n': 'o'}[gender]
    words = base.split()
    words[-1] += end
    return ' '.join(words)


def _wind_sentences(hs, ws, wg, wd, win0, win1):
    out = []
    peak = max((w for w in ws if w is not None), default=0.0)
    gust = max((g for g in wg if g is not None), default=0.0)
    if peak < 5:
        if gust >= 12:
            gh = hs[max(range(len(hs)), key=lambda i: wg[i] or 0)]
            out.append(f"Mogući udari vjetra do {gust:.0f} m/s oko {gh} h.")
        return out

    # direction regimes over the hours with some wind
    runs = []
    for h, w, d in zip(hs, ws, wd):
        if w is None or d is None or w < 4:
            continue
        k = _wind_key(d, h)
        if runs and runs[-1]['k'] == k and h - runs[-1]['end'] <= 2:
            runs[-1]['end'] = h + 1
            runs[-1]['n'] += 1
            runs[-1]['peak'] = max(runs[-1]['peak'], w)
        else:
            runs.append({'k': k, 'start': h, 'end': h + 1, 'n': 1, 'peak': w})
    runs = [r for r in runs if r['n'] >= 3]

    gust_txt = f", s udarima do {gust:.0f} m/s" if gust >= 12 else ''

    if len(runs) >= 2 and runs[0]['k'] != runs[-1]['k']:
        # direction shift: describe the first regime, then what it turns into
        a, b = runs[0], runs[-1]
        na, ga, _, _ = _wind_forms(a['k'])
        nb, gb, _, acc_b = _wind_forms(b['k'])
        str_b = _strength(b['peak'], gb)
        if gb == 'f':
            str_b = str_b[:-1] + 'u'
        span_a = _span_label(a['start'], a['end'], win0, win1)
        out.append(f"{_cap(_strength(a['peak'], ga))} {na} {span_a}, a oko {b['start']} h "
                   f"vjetar okreće na {str_b} {acc_b}{gust_txt}.")
        return out

    main = max(runs, key=lambda r: r['peak']) if runs else None
    peak_i = max(range(len(hs)), key=lambda i: ws[i] or 0)
    peak_h = hs[peak_i]
    if main:
        name, gender, _, _ = _wind_forms(main['k'])
    else:
        name, gender = 'vjetar', 'm'
    strong = [h for h, w in zip(hs, ws) if w is not None and w >= max(5, 0.8 * peak)]
    span = _span_label(strong[0], strong[-1] + 1, win0, win1) if strong else ''
    most = {'m': 'najjači', 'f': 'najjača', 'n': 'najjače'}[gender]
    txt = f"{_cap(_strength(peak, gender))} {name} {span}"
    if strong and (strong[-1] - strong[0]) >= 3:
        txt += f", {most} oko {peak_h} h"
    txt += gust_txt
    out.append(txt.rstrip() + '.')
    return out


# --- temperature ----------------------------------------------------------------

def _temp_sentence(hs, tp):
    pts = [(h, t) for h, t in zip(hs, tp) if t is not None]
    if len(pts) < 6:
        return None
    hmax, tmax = max(pts, key=lambda p: p[1])
    _, tmin = min(pts, key=lambda p: p[1])
    # min/max already sit on the card; only say when it peaks
    txt = f"Najtoplije oko {hmax} h"
    if tmax >= 33:
        txt += ', uz jaku vrućinu'
    elif tmax >= 30:
        txt += ', toplo'
    if tmin <= -5:
        txt += ', uz jak mraz'
    elif tmin <= 0:
        txt += ', uz mraz'
    return txt + '.'


# --- public -----------------------------------------------------------------------

def build(hours, cloud, precip, wind, gust, temp, wcode, wdir):
    """Return a few-sentence summary of the day, or None if too few hours."""
    rows = sorted(
        (int(h), _num(c), _num(p), _num(w), _num(g), _num(t), _num(k), _num(d))
        for h, c, p, w, g, t, k, d in zip(hours, cloud, precip, wind, gust, temp, wcode, wdir)
    )
    if len(rows) < 4:
        return None
    hs = [r[0] for r in rows]
    cc = [r[1] for r in rows]
    pr = [r[2] for r in rows]
    ws = [r[3] for r in rows]
    wg = [r[4] for r in rows]
    tp = [r[5] for r in rows]
    wc = [r[6] for r in rows]
    wd = [r[7] for r in rows]
    win0, win1 = hs[0], hs[-1] + 1

    kinds = [_hour_kind(p, k) for p, k in zip(pr, wc)]
    eps = _episodes(hs, kinds, pr)

    # dry spells = window minus episodes
    spells, cur = [], win0
    for ep in eps:
        if ep['start'] > cur:
            spells.append((cur, ep['start']))
        cur = max(cur, ep['end'])
    if cur < win1:
        spells.append((cur, win1))

    def runs_in(a, b):
        idx = [i for i, h in enumerate(hs) if a <= h < b and kinds[i] in ('dry', 'fog')]
        return _sky_runs([hs[i] for i in idx], [cc[i] for i in idx])

    def fog_in(a, b):
        fh = [h for h, k in zip(hs, kinds) if a <= h < b and k == 'fog']
        return fh

    events = []   # (hour, order, sentence)
    prev_sky = {}  # rain-episode end hour -> sky group just before that episode
    for a, b in spells:
        runs = runs_in(a, b)
        if not runs:
            continue
        follows_rain = any(ep['end'] == a for ep in eps)
        fh = fog_in(a, b)
        clauses = []
        first = 1
        if follows_rain:
            if b - a < 3:
                continue
            prev_g = prev_sky.get(a)
            if runs[0]['g'] == 0 and prev_g == 0:
                clauses.append("potom ponovo suvo i vedro")
            else:
                clauses.append(_after_rain(runs[0]))
        elif fh and fh[0] < a + 4:
            lift = fh[-1] + 1
            label = _span_label(a, lift, win0, win1)
            after = [r for r in runs if r['end'] > lift]
            if after:
                clauses.append(f"{label} magla, koja se diže oko {lift} h, "
                               f"a zatim {after[0]['word']}")
                runs = after
            else:
                clauses.append(f"{label} magla")
                runs = runs[-1:]
        else:
            label = _span_label(runs[0]['start'], runs[0]['end'], win0, win1)
            clauses.append(f"{label} {runs[0]['word']}")
        last_sky = runs[-1]['g']
        for ep in eps:
            if ep['start'] == b:
                prev_sky[ep['end']] = last_sky
        for i in range(first, len(runs)):
            again = i >= 2 and runs[i]['g'] == runs[i - 2]['g']
            clauses.append(_sky_change(runs[i - 1], runs[i], again))
        events.append((a, 1, _join_clauses(clauses)))

    for ep in eps:
        events.append((ep['start'], 0, _episode_sentence(ep, win0, win1)))

    events.sort(key=lambda e: (e[0], e[1]))
    sentences = [s for _, _, s in events][:MAX_EVENT_SENTENCES]

    total = sum(p or 0 for p in pr)
    if not eps and total > 0.2:
        sentences.append("Moguća je povremena slaba kiša.")

    sentences += _wind_sentences(hs, ws, wg, wd, win0, win1)
    t = _temp_sentence(hs, tp)
    if t:
        sentences.append(t)
    return ' '.join(sentences) if sentences else None
