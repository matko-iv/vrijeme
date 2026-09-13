"""Korigovana kisa naspram sirove i naspram globalnih modela, na ISTIM satima."""
import csv, json, math, os, subprocess, collections
os.chdir(r"C:\Users\Matija\Documents\GitHub\vrijeme")
PATH = "docs/forecast_data/forecast_48h.json"
THR = 0.2

# osmotreno: isti izvor koji su koristili brojevi za globalne modele
obs = {}
with open("budva_ECMWF_IFS025_detailed.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        dt = row["datetime"][:13]
        try: obs[dt] = float(row["precipitation_obs"])
        except (TypeError, ValueError): pass

shas = subprocess.run(["git","log","--since=2026-06-01","--until=2026-08-19",
                       "--format=%H","--",PATH], capture_output=True, text=True).stdout.split()
best = {}
for sha in shas:
    blob = subprocess.run(["git","show",f"{sha}:{PATH}"], capture_output=True).stdout
    try: d = json.loads(blob.decode("utf-8","replace"))
    except Exception: continue
    gen = d.get("generated","")[:19]
    if not gen: continue
    for h in d.get("hourly_forecast", []):
        vt = h.get("datetime","")[:13].replace("T"," ")
        if not vt: continue
        if vt in best and best[vt][0] >= gen: continue
        best[vt] = (gen, h.get("precipitation"), h.get("precipitation_raw"),
                    h.get("precipitation_pop"))

def score(pairs):
    hit=miss=fa=cn=0
    for f,o in pairs:
        fp, op = f, o
        if fp and op: hit+=1
        elif fp and not op: fa+=1
        elif op and not fp: miss+=1
        else: cn+=1
    tot=hit+miss+fa
    return dict(hit=hit,miss=miss,fa=fa,
                pod=hit/(hit+miss) if hit+miss else None,
                far=fa/(hit+fa) if hit+fa else None,
                csi=hit/tot if tot else None)

hours = sorted(k for k in best if k in obs)
print("uporedivih sati:", len(hours))
corr = score([(best[k][1] is not None and best[k][1] >= THR, obs[k] >= THR) for k in hours
              if best[k][1] is not None])
raw  = score([(best[k][2] is not None and best[k][2] >= THR, obs[k] >= THR) for k in hours
              if best[k][2] is not None])

rows = [("MOJ (korigovano)", corr), ("MOJ (sirovi blend)", raw)]
# POP na vise pragova
for p in (0.3, 0.4, 0.5, 0.6):
    pairs=[(best[k][3] is not None and best[k][3] >= p, obs[k] >= THR) for k in hours if best[k][3] is not None]
    if pairs: rows.append(("MOJ (POP>=%.0f%%)" % (p*100), score(pairs)))

# globalni modeli na ISTIM satima
for path in sorted(__import__("glob").glob("budva_*_detailed.csv")):
    m = os.path.basename(path)[len("budva_"):-len("_detailed.csv")]
    if m == "METEOFRANCE": continue
    pairs=[]
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dt=row["datetime"][:13]
            if dt not in obs or dt not in best: continue
            try: v=float(row["precipitation_model"])
            except (TypeError,ValueError): continue
            pairs.append((v>=THR, obs[dt]>=THR))
    if len(pairs) > 500: rows.append((m, score(pairs)))

print()
print("%-22s %5s %5s %5s %7s %7s %7s" % ("","hit","miss","FA","POD","FAR","CSI"))
for name,s in sorted(rows, key=lambda r: -(r[1]["csi"] or 0)):
    print("%-22s %5d %5d %5d %7s %7s %7s" % (name,s["hit"],s["miss"],s["fa"],
        "%.3f"%s["pod"] if s["pod"] is not None else "-",
        "%.3f"%s["far"] if s["far"] is not None else "-",
        "%.3f"%s["csi"] if s["csi"] is not None else "-"))
json.dump({k:v for k,v in rows}, open("analysis_output/summer_2026_rain_vs_models.json","w"), indent=2)
