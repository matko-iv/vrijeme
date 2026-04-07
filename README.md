# Budva Weather Forecast — ML Bias Correction

Korigovana 48-satna i 7-dnevna prognoza za Budvu, Crna Gora. Pipeline preuzima sirove prognoze sa 11 globalnih NWP modela, primjenjuje XGBoost/CatBoost/LightGBM korekciju i generiše JSON koji pokreće web stranicu.

## Kako radi

1. **Historijski podaci** — 6 godina satnih prognoza (2020–2026) sa Open-Meteo API-ja, upareni sa stvarnim mjerenjima sa [IBUDVA5](https://www.wunderground.com/dashboard/pws/IBUDVA5) Weather Underground stanice.
2. **Trening** — XGBoost, CatBoost i LightGBM modeli uče bias (sistematsku grešku) svakog NWP modela za temperaturu, vlažnost, vjetar, pritisak i padavine. Ensemble sa Ridge stacking-om kombinuje korekcije.
3. **Live prognoza** — Svaki sat GitHub Actions pokreće pipeline: preuzme najnovije prognoze, primijeni korekciju, generiše `forecast_48h.json`.
4. **Web prikaz** — Statična HTML stranica čita JSON i prikazuje kartice, grafikone i dugročnu prognozu.

## Modeli

| # | Model | Rezolucija |
|---|-------|-----------|
| 1 | ARPÈGE Europe | ~10 km |
| 2 | GFS Seamless | ~25 km |
| 3 | ICON Seamless | ~13 km |
| 4 | Météo-France | ~10 km |
| 5 | ECMWF IFS 0.25° | ~25 km |
| 6 | ItaliaMeteo ICON-2I | ~2.2 km |
| 7 | UKMO Seamless | ~10 km |
| 8 | BOM ACCESS | ~12 km |
| 9 | ECMWF IFS | ~9 km |
| 10 | KNMI Seamless | ~11 km |
| 11 | DMI Seamless | ~13 km |

## Struktura

```
forecast_48h_v2.py          # Glavni pipeline (fetch → train → correct → output)
wu_scraper.py               # Scraper za historijske obs sa Weather Underground
requirements.txt            # Python zavisnosti
.github/workflows/
  forecast.yml              # Hourly cron job
docs/
  forecast.html             # 48h + 7-day prognoza
  index.html                # Analiza tačnosti modela
  forecast_data/
    forecast_48h.json       # Generisani output
production_forecast_model/
  forecast_48h_v2.py        # Production snapshot
  *.joblib                  # Trenirani modeli
```

## Pokretanje

```bash
# Instalacija
pip install -r requirements.txt

# Puni pipeline (trening + prognoza) — ~5 min
python forecast_48h_v2.py

# Samo prognoza sa postojećim modelima — ~1 min
python forecast_48h_v2.py --skip-training
```

Output se zapisuje u `forecast_output/forecast_48h.json`, a GitHub Actions ga kopira u `docs/forecast_data/`.

## Šansa za padavine

`precip_probability` se računa kao procenat NWP modela (od 11) koji predviđaju >0.1mm padavina u bilo kom satu tog dana. Nije klasičan ensemble probabilistički pristup — to je "model consensus" metrika.

## Autor

Matija Ivanović · [@matko-iv](https://github.com/matko-iv)