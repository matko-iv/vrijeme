"""Korigovano vs sirovo, na prognozama koje su ZAISTA objavljene tog ljeta."""
import csv, json, math, os, subprocess, collections
os.chdir(r"C:\Users\Matija\Documents\GitHub\vrijeme")
PATH = "docs/forecast_data/forecast_48h.json"

shas = subprocess.run(["git","log","--since=2026-06-01","--until=2026-08-19",
                       "--format=%H","--",PATH], capture_output=True, text=True).stdout.split()
print("snapshot-a:", len(shas))

obs = {}
with open("wu_data/merged_observations.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        try: obs[row["datetime"][:13]] = float(row["temp_c"])
        except (TypeError, ValueError): pass
wind = {}
with open("wu_data/merged_observations.csv", encoding="utf-8") as f:
    for row in csv.DictReader(f):
        try: wind[row["datetime"][:13]] = float(row["wind_ms"])
        except (TypeError, ValueError): pass

# po (valid_sat, lead_bucket) uzmi NAJSVJEZIJU prognozu
best = {}
for i, sha in enumerate(shas):
    blob = subprocess.run(["git","show",f"{sha}:{PATH}"], capture_output=True).stdout
    try: d = json.loads(blob.decode("utf-8","replace"))
    except Exception: continue
    gen = d.get("generated","")[:19]
    for h in d.get("hourly_forecast", []):
        vt = h.get("datetime","")[:13].replace("T"," ")
        if not vt or not gen: continue
        try:
            lead = (int(h["datetime"][11:13]) - int(gen[11:13])) % 24 + 24*max(0,(int(h["datetime"][8:10])-int(gen[8:10])))
        except Exception: continue
        bucket = "0-24h" if lead <= 24 else "24-48h"
        key = (vt, bucket)
        if key in best and best[key][0] >= gen: continue
        best[key] = (gen, h.get("temperature_2m"), h.get("temperature_2m_raw"),
                     h.get("wind_speed_10m"), h.get("wind_speed_10m_raw"))
    if (i+1) % 150 == 0: print("  ...", i+1)

acc = collections.defaultdict(lambda: {"n":0,"c":0.0,"r":0.0,"cb":0.0,"rb":0.0})
accw = collections.defaultdict(lambda: {"n":0,"c":0.0,"r":0.0})
for (vt,bucket),(gen,tc,tr,wc,wr) in best.items():
    o = obs.get(vt)
    if o is not None and isinstance(tc,(int,float)) and isinstance(tr,(int,float)):
        a = acc[bucket]; a["n"]+=1
        a["c"]+=abs(tc-o); a["r"]+=abs(tr-o); a["cb"]+=tc-o; a["rb"]+=tr-o
    ow = wind.get(vt)
    if ow is not None and isinstance(wc,(int,float)) and isinstance(wr,(int,float)):
        b = accw[bucket]; b["n"]+=1; b["c"]+=abs(wc-ow); b["r"]+=abs(wr-ow)

print()
print("TEMPERATURA — objavljena prognoza naspram osmotrenog")
print("%-8s %6s %9s %9s %9s" % ("lead","n","korig.MAE","sirovo MAE","dobitak"))
for b in ("0-24h","24-48h"):
    a = acc[b]
    if a["n"]:
        c,r = a["c"]/a["n"], a["r"]/a["n"]
        print("%-8s %6d %9.3f %9.3f %8.1f%%" % (b,a["n"],c,r,100*(r-c)/r))
print()
print("VJETAR — objavljena prognoza naspram osmotrenog")
for b in ("0-24h","24-48h"):
    a = accw[b]
    if a["n"]:
        c,r = a["c"]/a["n"], a["r"]/a["n"]
        print("%-8s %6d %9.3f %9.3f %8.1f%%" % (b,a["n"],c,r,100*(r-c)/r))
json.dump({"temp":{k:dict(v) for k,v in acc.items()},"wind":{k:dict(v) for k,v in accw.items()},
           "snapshots":len(shas)}, open("analysis_output/summer_2026_oos.json","w"), indent=2)
