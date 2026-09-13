"""Ljeto 2026: greska sirovih modela naspram stanice IBUDVA5."""
import csv, glob, math, os, json
from collections import defaultdict

START, END = "2026-06-01", "2026-08-19"   # osmatranja idu do 18.08
os.chdir(r"C:\Users\Matija\Documents\GitHub\vrijeme")

def num(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None

PAIRS = {
    "temperature_2m": ("temperature_2m_model", "temperature_2m_obs"),
    "wind_speed_10m": ("wind_speed_10m_model", "wind_speed_10m_obs"),
    "wind_gusts_10m": ("wind_gusts_10m_model", "wind_gusts_10m_obs"),
    "relative_humidity_2m": ("relative_humidity_2m_model", "relative_humidity_2m_obs"),
    "dew_point_2m": ("dew_point_2m_model", "dew_point_2m_obs"),
}

results = {}
precip = {}
hours_seen = set()

for path in sorted(glob.glob("budva_*_detailed.csv")):
    model = os.path.basename(path)[len("budva_"):-len("_detailed.csv")]
    acc = defaultdict(lambda: {"n": 0, "abs": 0.0, "sum": 0.0, "sq": 0.0})
    # kisa: model kaze >=0.2mm/h, stanica kaze >=0.2mm/h
    hit = miss = fa = correct_neg = 0
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            dt = row.get("datetime", "")
            if not (START <= dt[:10] < END):
                continue
            hours_seen.add(dt)
            for var, (mc, oc) in PAIRS.items():
                m, o = num(row.get(mc)), num(row.get(oc))
                if m is None or o is None:
                    continue
                a = acc[var]
                d = m - o
                a["n"] += 1
                a["abs"] += abs(d)
                a["sum"] += d
                a["sq"] += d * d
            pm, po = num(row.get("precipitation_model")), num(row.get("precipitation_obs"))
            if pm is not None and po is not None:
                mp, op = pm >= 0.2, po >= 0.2
                if mp and op: hit += 1
                elif mp and not op: fa += 1
                elif op and not mp: miss += 1
                else: correct_neg += 1
    if not acc:
        continue
    results[model] = {
        v: {"n": a["n"], "mae": a["abs"] / a["n"], "bias": a["sum"] / a["n"],
            "rmse": math.sqrt(a["sq"] / a["n"])}
        for v, a in acc.items() if a["n"]
    }
    tot = hit + miss + fa
    precip[model] = {
        "hit": hit, "miss": miss, "false_alarm": fa, "correct_negative": correct_neg,
        "pod": hit / (hit + miss) if (hit + miss) else None,
        "far": fa / (hit + fa) if (hit + fa) else None,
        "csi": hit / tot if tot else None,
    }

print("Sati u uzorku:", len(hours_seen), " (%s .. %s)" % (min(hours_seen)[:10], max(hours_seen)[:10]))
print()
print("TEMPERATURA — sirovi modeli, ljeto 2026")
print("%-22s %6s %7s %7s %7s" % ("model", "n", "MAE", "bias", "RMSE"))
for m, r in sorted(results.items(), key=lambda kv: kv[1].get("temperature_2m", {}).get("mae", 9e9)):
    t = r.get("temperature_2m")
    if t:
        print("%-22s %6d %7.3f %+7.3f %7.3f" % (m, t["n"], t["mae"], t["bias"], t["rmse"]))
print()
print("VJETAR (10m) — MAE / bias m/s")
for m, r in sorted(results.items(), key=lambda kv: kv[1].get("wind_speed_10m", {}).get("mae", 9e9)):
    w = r.get("wind_speed_10m")
    if w:
        print("%-22s %6d %7.3f %+7.3f" % (m, w["n"], w["mae"], w["bias"]))
print()
print("KISA >=0.2 mm/h — POD (pogodak) / FAR (lazna uzbuna) / CSI")
print("%-22s %6s %6s %6s %7s %7s %7s" % ("model", "hit", "miss", "FA", "POD", "FAR", "CSI"))
for m, p in sorted(precip.items(), key=lambda kv: -(kv[1]["csi"] or 0)):
    if p["pod"] is None and p["far"] is None:
        continue
    print("%-22s %6d %6d %6d %7s %7s %7s" % (
        m, p["hit"], p["miss"], p["false_alarm"],
        "%.3f" % p["pod"] if p["pod"] is not None else "-",
        "%.3f" % p["far"] if p["far"] is not None else "-",
        "%.3f" % p["csi"] if p["csi"] is not None else "-"))

json.dump({"results": results, "precip": precip, "hours": len(hours_seen)},
          open(r"C:\Users\Matija\Documents\GitHub\vrijeme\analysis_output\summer_2026_eval.json", "w"),
          indent=2)
print()
print("-> analysis_output/summer_2026_eval.json")
