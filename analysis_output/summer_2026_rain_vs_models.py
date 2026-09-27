"""Korigovana kisa naspram sirove i naspram globalnih modela, na ISTIM satima.

Stanica: satna kisa = WU precip_rate_mm iz wu_data/merged_observations.csv
(sat oznacen POCETKOM, [t, t+1)). Ranija verzija je citala precipitation_obs
iz budva_*_detailed.csv, a to je WU dnevni KUMULATIV (precip_accum_mm, resetuje
se u ponoc): poslije svakog pljuska svi sati do ponoci brojali su se kao kisni
(ljeto 2026: 114 "kisnih" sati umjesto 23 stvarna).

Poravnanje:
  objavljeni JSON  red t = kisa [t, t+1) (pipeline je vec pomjerio)  <-> stanica t
  Open-Meteo model red t = kisa [t-1, t]  -> model red t+1            <-> stanica t
"""
import csv, datetime, glob, json, os, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
PATH = "docs/forecast_data/forecast_48h.json"
START, END = "2026-06-01", "2026-08-19"
THR = 0.2


def hour_key(ts):
    return ts[:13].replace("T", " ")


def next_hour(key):
    t = datetime.datetime.strptime(key, "%Y-%m-%d %H") + datetime.timedelta(hours=1)
    return t.strftime("%Y-%m-%d %H")


obs = {}
with open("wu_data/merged_observations.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        try: obs[row["datetime"][:13]] = float(row["precip_rate_mm"])
        except (TypeError, ValueError): pass

shas = subprocess.run(["git", "log", f"--since={START}", f"--until={END}",
                       "--format=%H", "--", PATH], capture_output=True, text=True).stdout.split()
best = {}
for sha in shas:
    blob = subprocess.run(["git", "show", f"{sha}:{PATH}"], capture_output=True).stdout
    try: d = json.loads(blob.decode("utf-8", "replace"))
    except Exception: continue
    gen = d.get("generated", "")[:19]
    if not gen: continue
    for h in d.get("hourly_forecast", []):
        vt = hour_key(h.get("datetime", ""))
        if not vt: continue
        if vt in best and best[vt][0] >= gen: continue
        best[vt] = (gen, h.get("precipitation"), h.get("precipitation_raw"),
                    h.get("precipitation_pop"))


def score(pairs):
    hit = miss = fa = cn = 0
    for fp, op in pairs:
        if fp and op: hit += 1
        elif fp and not op: fa += 1
        elif op and not fp: miss += 1
        else: cn += 1
    tot = hit + miss + fa
    return dict(hit=hit, miss=miss, fa=fa,
                pod=hit / (hit + miss) if hit + miss else None,
                far=fa / (hit + fa) if hit + fa else None,
                csi=hit / tot if tot else None)


hours = sorted(k for k in best if k in obs and START <= k[:10] < END)
print("snapshot-a:", len(shas), " uporedivih sati:", len(hours),
      " kisnih (>= %.1f mm):" % THR, sum(obs[k] >= THR for k in hours))
corr = score([(best[k][1] >= THR, obs[k] >= THR) for k in hours if best[k][1] is not None])
raw = score([(best[k][2] >= THR, obs[k] >= THR) for k in hours if best[k][2] is not None])

rows = [("MOJ (korigovano)", corr), ("MOJ (sirovi blend)", raw)]
for p in (0.3, 0.4, 0.5, 0.6):
    pairs = [(best[k][3] >= p, obs[k] >= THR) for k in hours if best[k][3] is not None]
    if pairs: rows.append(("MOJ (POP>=%.0f%%)" % (p * 100), score(pairs)))

# globalni modeli na ISTIM satima: model red t+1 pokriva stanicni sat t
for path in sorted(glob.glob("budva_*_detailed.csv")):
    m = os.path.basename(path)[len("budva_"):-len("_detailed.csv")]
    if m == "METEOFRANCE": continue
    model = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try: model[row["datetime"][:13]] = float(row["precipitation_model"])
            except (TypeError, ValueError): pass
    pairs = [(model[next_hour(k)] >= THR, obs[k] >= THR)
             for k in hours if next_hour(k) in model]
    if len(pairs) > 500: rows.append((m, score(pairs)))

print()
print("%-22s %5s %5s %5s %7s %7s %7s" % ("", "hit", "miss", "FA", "POD", "FAR", "CSI"))
for name, s in sorted(rows, key=lambda r: -(r[1]["csi"] or 0)):
    print("%-22s %5d %5d %5d %7s %7s %7s" % (name, s["hit"], s["miss"], s["fa"],
        "%.3f" % s["pod"] if s["pod"] is not None else "-",
        "%.3f" % s["far"] if s["far"] is not None else "-",
        "%.3f" % s["csi"] if s["csi"] is not None else "-"))

# Provjera poravnanja: CSI objavljene prognoze naspram stanice pomjerene za
# -1/0/+1 sat. Vrh mora biti na 0; vrh na +1 znaci da prognoza kasni/zuri sat.
print("\nPomak stanice (h):   -1      0     +1")
for idx, name in ((1, "MOJ (korigovano)"), (2, "MOJ (sirovi blend)")):
    csis = []
    for shift in (-1, 0, 1):
        pairs = []
        for k in hours:
            t = datetime.datetime.strptime(k, "%Y-%m-%d %H") + datetime.timedelta(hours=shift)
            o = obs.get(t.strftime("%Y-%m-%d %H"))
            if best[k][idx] is not None and o is not None:
                pairs.append((best[k][idx] >= THR, o >= THR))
        csis.append(score(pairs)["csi"] or 0.0)
    print("%-18s %6.3f %6.3f %6.3f" % (name, *csis))
json.dump({k: v for k, v in rows}, open("analysis_output/summer_2026_rain_vs_models.json", "w"), indent=2)
