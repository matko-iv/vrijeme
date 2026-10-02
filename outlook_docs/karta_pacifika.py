# -*- coding: utf-8 -*-
"""Rasterska karta odstupanja temperature mora u Tihom okeanu za dugorocne-najave.html.

Isti postupak kao karta iz avgusta 2026: NOAA OISST v2.1 (dnevne anomalije u odnosu
na 1971-2000) sa CoastWatch ERDDAP-a, prosjek 2x2 celije od 0,25 deg na mrezu od
0,5 deg, 120 E do 290 E i 30 N do 30 S (341 x 121 piksela), vrijednosti zaokruzene
na 0,25 C i ogranicene na +-3 C, iste boje kao skala na stranici. Kopno se popuni
najblizom vrijednoscu mora; na stranici ga pokrivaju obrisi iz Natural Earth-a.

Pokretanje:  python karta_pacifika.py 2026-10-01
Ispisuje prosjek u kutiji Nino 3.4 i pravi karta_pacifika_<datum>.b64; taj tekst ide
u <image href="data:image/png;base64,..."> na karti u clanku o El Ninu.
"""

import base64
import io
import sys
import urllib.request
import warnings

import numpy as np
from PIL import Image

ERDDAP = ("https://coastwatch.pfeg.noaa.gov/erddap/griddap/ncdcOisst21NrtAgg.csv?"
          "anom[(%sT12:00:00Z)][(0.0)][(-30.375):1:(30.625)][(119.625):1:(290.625)]")
STOPS = np.array([-3, -2, -1, -0.5, 0, 0.5, 1, 2, 3.])
RGB = np.array([(33, 102, 172), (67, 147, 195), (146, 197, 222), (209, 229, 240), (247, 247, 247),
                (253, 219, 199), (244, 165, 130), (214, 96, 77), (178, 24, 43)], float)


def ucitaj(datum):
    tekst = urllib.request.urlopen(ERDDAP % datum, timeout=300).read().decode()
    redovi = [r.split(',') for r in tekst.strip().splitlines()[2:]]
    lat = np.array([float(r[2]) for r in redovi])
    lon = np.array([float(r[3]) for r in redovi])
    vr = np.array([float(r[4]) for r in redovi])
    lats, lons = np.unique(lat)[::-1], np.unique(lon)
    mreza = np.full((len(lats), len(lons)), np.nan)
    mreza[np.searchsorted(-lats, -lat), np.searchsorted(lons, lon)] = vr
    return lats, lons, mreza


def main(datum):
    lats, lons, g = ucitaj(datum)
    kutija = (np.abs(lats)[:, None] < 5) & (lons[None, :] > 190) & (lons[None, :] < 240)
    print("Nino 3.4 (5S-5N, 170W-120W), osnova 1971-2000: %+.2f C"
          % np.nanmean(np.where(kutija, g, np.nan)))

    # 0,5 deg: cvor (y, x) je prosjek celija y +- 0,125 i x +- 0,125
    red = {round(v, 3): i for i, v in enumerate(lats)}
    kol = {round(v, 3): i for i, v in enumerate(lons)}
    tlat, tlon = np.arange(30.0, -30.01, -0.5), np.arange(120.0, 290.01, 0.5)
    out = np.full((len(tlat), len(tlon)), np.nan)
    for r, y in enumerate(tlat):
        rr = [red[round(y + 0.125, 3)], red[round(y - 0.125, 3)]]
        for c, x in enumerate(tlon):
            blok = g[np.ix_(rr, [kol[round(x - 0.125, 3)], kol[round(x + 0.125, 3)]])]
            if np.isfinite(blok).any():
                out[r, c] = np.nanmean(blok)

    while np.isnan(out).any():           # kopno: najbliza vrijednost mora
        p = np.pad(out, 1, constant_values=np.nan)
        susjedi = np.stack([p[:-2, 1:-1], p[2:, 1:-1], p[1:-1, :-2], p[1:-1, 2:]])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            m = np.nanmean(susjedi, axis=0)
        prazno = np.isnan(out)
        out[prazno] = m[prazno]

    v = np.clip(np.round(out * 4) / 4, -3, 3)
    slika = np.zeros(v.shape + (4,), np.uint8)
    for k in range(3):
        slika[..., k] = np.round(np.interp(v, STOPS, RGB[:, k]))
    slika[..., 3] = 255
    buf = io.BytesIO()
    Image.fromarray(slika, "RGBA").save(buf, "PNG", optimize=True)
    ime = "karta_pacifika_%s.b64" % datum
    open(ime, "w").write(base64.b64encode(buf.getvalue()).decode())
    print("%s (%d bajtova PNG)" % (ime, len(buf.getvalue())))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "2026-10-01")
