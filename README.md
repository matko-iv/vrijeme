# ⛅ XGBoost korekcija vremenske prognoze

Sistem koji poboljšava vremenske prognoze koristeći XGBoost za korekciju grešaka iz 6 NWP modela, na osnovu istorijskih tabela biasa.

Trenutno radi za **Budvu, Crna Gora** — ali se lako prilagođava za bilo koju lokaciju sa Weather Underground ličnom meteorološkom stanicom.

**[Live demo →](https://matko-iv.github.io/vrijeme/forecast.html)**

---

## Kako radi

1. **Scrape opservacije** — satni podaci sa Weather Underground (WU) stanice (temperatura, vlažnost, vjetar, pritisak, padavine, solarna radijacija)
2. **Preuzmi istorijske prognoze** — iz 6 modela preko [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) za isti vremenski period
3. **Analiziraj greške** — MAE, RMSE, bias po modelu, po sezoni, po dobu dana, po vremenskim uslovima
4. **Treniraj XGBoost** — po jedan model za svaki parametar (temperatura, vlažnost, vjetar, pritisak, oblačnost, padavine, solarna radijacija, tačka rose, udari vjetra), koristeći ~475 feature-a: ensemble statistike, tabele biasa, vremenski feature-i, cross-parameter interakcije
5. **Generiši korigovanu prognozu** — preuzmi live prognoze svih 6 modela, primijeni korekciju, output JSON za frontend

Pipeline se pokreće automatski preko GitHub Actions (svakih sat) i objavljuje na GitHub Pages.

## Korišćeni modeli

| Model | Open-Meteo ID | Pokrivenost |
|-------|--------------|-------------|
| ECMWF IFS 0.25° | `ecmwf_ifs025` | Globalna |
| ICON Seamless | `icon_seamless` | Globalna |
| GFS Seamless | `gfs_seamless` | Globalna |
| Météo-France Seamless | `meteofrance_seamless` | Globalna |
| ARPEGE Europe | `arpege_europe` | Evropa |
| ItaliaMeteo ICON 2I | `italia_meteo_arpae_icon_2i` | **Italija + okolina** ⚠️ |

> ⚠️ **ItaliaMeteo ICON 2I** pokriva samo Italiju i neposrednu okolinu (jadransku obalu, Sloveniju, Hrvatsku, dijelove Austrije/Švajcarske). Ako je vaša lokacija van ovog područja, uklonite ga iz `MODELS` u oba skripta ili ćete dobiti prazne podatke.

## Prilagođavanje za drugu lokaciju

### 1. Nađite Weather Underground stanicu

Na [wunderground.com/wundermap](https://www.wunderground.com/wundermap) pronađite PWS blizu vaše lokacije. Treba vam station ID (npr. `IBUDVA5`, `KLAX23`, `IMILANO42`).

Dobre stanice imaju:
- Barem 1–2 godine kontinuiranih podataka (više = bolje)
- Sve uobičajene senzore (temp, vlažnost, vjetar, pritisak, kiša, solar)
- Konzistentan uptime — praznine su ok ali velike rupe škode

### 2. Scrapujte istorijske opservacije

Koristite `wu_scraper.py` za prikupljanje satnih opservacija:

```python
# wu_scraper.py — promijenite:
STATION_ID = "VAS_STATION_ID"
YEARS_BACK = 3  # koliko godina unazad
```

```bash
python wu_scraper.py
```

Kreira CSV u `wu_data/` sa satnim kolonama: `temp_c`, `humidity_pct`, `wind_ms`, `pressure_hpa`, `solar_wm2`, `precip_rate_mm`, itd.

### 3. Preuzmite istorijske prognoze modela

Editujte `advanced_model_analysis.py`:

```python
LAT, LON = 48.85, 2.35  # vaša lokacija (primjer: Pariz)
OBS_CSV = "wu_data/vasa_stanica_hourly.csv"
START_DATE = "2023-01-01"  # da odgovara početku vaših opservacija

MODELS = {
    "ECMWF_IFS025": "ecmwf_ifs025",
    "ICON_SEAMLESS": "icon_seamless",
    "GFS_SEAMLESS": "gfs_seamless",
    # "METEOFRANCE": "meteofrance_seamless",
    "ARPEGE_EUROPE": "arpege_europe",
    # "ITALIAMETEO_ICON2I": "italia_meteo_arpae_icon_2i",
}
```

```bash
python advanced_model_analysis.py
```

Kreira:
- `budva_{MODEL}_detailed.csv` po modelu (prefiks preimenujte kako želite)
- `budva_detailed_error_analysis.json` sa svim metrikama

> **Napomena:** Open-Meteo historical forecast API je besplatan ali ima rate limit. Skripta automatski ponavlja na 429 greškama sa 60s pauzom. Preuzimanje 2+ godine za 6 modela traje ~5–10 minuta.

### 4. Trenirajte i pokrenite forecast pipeline

Editujte `forecast_48h_v2.py`:

```python
LAT, LON = 48.85, 2.35  # iste koordinate

MODELS = ["ARPEGE_EUROPE", "GFS_SEAMLESS", "ICON_SEAMLESS",
          "METEOFRANCE", "ECMWF_IFS025"]  # uklonite ITALIAMETEO po potrebi

MODEL_IDS = {
    "ARPEGE_EUROPE": "arpege_europe",
    "GFS_SEAMLESS": "gfs_seamless",
    "ICON_SEAMLESS": "icon_seamless",
    "METEOFRANCE": "meteofrance_seamless",
    "ECMWF_IFS025": "ecmwf_ifs025",
}
```

Takođe ažurirajte ime lokacije u output sekciji (potražite `"Budva, Crna Gora"`) i ime stanice.

```bash
python forecast_48h_v2.py
```

Output ide u `forecast_output/forecast_48h.json` — to čita frontend.

### 5. GitHub Pages (opciono)

Folder `docs/` sadrži frontend:
- `index.html` — stranica sa analizom/metodologijom
- `forecast.html` — prikaz live prognoze
- `forecast_data/forecast_48h.json` — automatski ažuriran preko GitHub Actions

Za automatizaciju:
1. Push na GitHub
2. Uključite GitHub Pages iz `docs/` foldera
3. Workflow u `.github/workflows/forecast.yml` pokreće `forecast_48h_v2.py` svakih sat i commituje ažurirani JSON

### 6. Vizualizacija analize grešaka (opciono)

```bash
python visualize_analysis.py
```

Kreira grafikone u `plots_extended/` — poređenje modela, sezonska analiza, bias analiza, itd.

## Struktura projekta

```
├── forecast_48h_v2.py              # Glavni pipeline: trening + prognoza + output
├── advanced_model_analysis.py      # Istorijska analiza grešaka po modelu
├── visualize_analysis.py           # Matplotlib grafikoni iz rezultata analize
├── wu_scraper.py                   # WU scraper (requests/BS4)
├── wu_scraper_colab_3y_fast.py     # Brzi scraper za Google Colab (Playwright)
├── requirements.txt
├── .github/workflows/forecast.yml  # GitHub Actions cron (svakih sat)
├── docs/                           # GitHub Pages frontend
│   ├── index.html                  # Stranica analize
│   ├── forecast.html               # Prikaz live prognoze
│   └── forecast_data/
│       └── forecast_48h.json       # Automatski ažurirana prognoza
├── wu_data/                        # Scrapovane opservacije
├── trained_models_v2/              # Sačuvani XGBoost modeli + tabele biasa
└── forecast_output/                # Output pipeline-a (JSON + CSV)
```

## Zavisnosti

```
pip install -r requirements.txt
```

- pandas
- numpy
- xgboost
- scikit-learn
- requests

Python 3.10+

## Napomene

- **Training/test split** je hardkodiran na `2025-07-01` u `forecast_48h_v2.py` (`SPLIT_DATE`). Podesite tako da imate dovoljno podataka za trening (barem 6–12 mjeseci) prije tog datuma, i nešto test podataka poslije.
- **Oblačnost** se izvodi iz mjerenja solarne radijacije (ne iz direktnih opservacija oblačnosti). Ako vaša WU stanica nema solarni senzor, korekcija oblačnosti će biti ograničena.
- **Padavine** koriste `precip_rate_mm` sa WU. Neke stanice ovo prijavljuju loše — provjerite kvalitet podataka.
- **Detekcija bure** u analizi je specifična za jadransku obalu (SJ/SJI vjetar > 8 m/s). Ako ste drugdje, neće biti relevantna ali neće ni ništa pokvariti.
- Sistem najbolje radi na lokacijama sa **konzistentnim podacima stanice** i đe NWP modeli imaju poznate biase (priobalna područja, planinske doline, urbana toplija ostrva).

---

# ⛅ XGBoost Weather Forecast Correction

*[English version]*

Machine learning system that improves weather forecasts by correcting errors from 6 NWP models using XGBoost and historical bias tables.

Currently running for **Budva, Montenegro** — easily adaptable to any location with a Weather Underground personal weather station.

**[Live demo →](https://matko-iv.github.io/vrijeme/forecast.html)**

---

## How it works

1. **Scrape observations** — hourly data from a Weather Underground (WU) station (temperature, humidity, wind, pressure, precipitation, solar radiation)
2. **Fetch historical forecasts** — from 6 models via [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api) for the same period
3. **Analyze errors** — MAE, RMSE, bias per model, per season, per time of day, per weather condition
4. **Train XGBoost** — one model per parameter (temperature, humidity, wind, pressure, cloud cover, precipitation, solar radiation, dew point, gusts), using ~475 features: ensemble statistics, bias tables, temporal features, cross-parameter interactions
5. **Produce corrected forecast** — fetch live forecasts from all 6 models, apply correction, output JSON for the frontend

Runs automatically via GitHub Actions (hourly), publishes to GitHub Pages.

## Models

| Model | Open-Meteo ID | Coverage |
|-------|--------------|----------|
| ECMWF IFS 0.25° | `ecmwf_ifs025` | Global |
| ICON Seamless | `icon_seamless` | Global |
| GFS Seamless | `gfs_seamless` | Global |
| Météo-France Seamless | `meteofrance_seamless` | Global |
| ARPEGE Europe | `arpege_europe` | Europe |
| ItaliaMeteo ICON 2I | `italia_meteo_arpae_icon_2i` | **Italy + nearby** ⚠️ |

> ⚠️ **ItaliaMeteo ICON 2I** only covers Italy and immediate surroundings (Adriatic coast, Slovenia, Croatia, parts of Austria/Switzerland). If your location is outside this area, remove it from `MODELS` in both scripts or you'll get empty data.

## Adapt to your location

### 1. Find a Weather Underground station

Go to [wunderground.com/wundermap](https://www.wunderground.com/wundermap) and find a PWS near your target location. You need the station ID (e.g. `IBUDVA5`, `KLAX23`, `IMILANO42`).

Good stations have:
- At least 1–2 years of continuous data (more = better)
- All common sensors (temp, humidity, wind, pressure, rain, solar)
- Consistent uptime — gaps are fine but big holes hurt

### 2. Scrape historical observations

```python
# wu_scraper.py — change these:
STATION_ID = "YOUR_STATION_ID"
YEARS_BACK = 3  # how far back to scrape
```

```bash
python wu_scraper.py
```

Creates a CSV in `wu_data/` with hourly columns: `temp_c`, `humidity_pct`, `wind_ms`, `pressure_hpa`, `solar_wm2`, `precip_rate_mm`, etc.

> **Tip:** Weather Underground has rate limits. The scraper is slow on purpose. For faster scraping (e.g. 3 years), use `wu_scraper_colab_3y_fast.py` designed for Google Colab with Playwright.

### 3. Fetch historical model forecasts

Edit `advanced_model_analysis.py`:

```python
LAT, LON = 48.85, 2.35  # your location (Paris example)
OBS_CSV = "wu_data/your_station_hourly.csv"
START_DATE = "2023-01-01"  # match your observation data start

MODELS = {
    "ECMWF_IFS025": "ecmwf_ifs025",
    "ICON_SEAMLESS": "icon_seamless",
    "GFS_SEAMLESS": "gfs_seamless",
    "METEOFRANCE": "meteofrance_seamless",
    "ARPEGE_EUROPE": "arpege_europe",
    # Remove ITALIAMETEO if you're not near Italy:
    # "ITALIAMETEO_ICON2I": "italia_meteo_arpae_icon_2i",
}
```

```bash
python advanced_model_analysis.py
```

Creates:
- `budva_{MODEL}_detailed.csv` per model (rename prefix as you like)
- `budva_detailed_error_analysis.json` with all metrics

> **Note:** Open-Meteo's historical forecast API is free but rate-limited. The script retries on 429 errors with a 60s backoff. Fetching 2+ years of 6 models takes ~5–10 minutes.

### 4. Train & run the forecast pipeline

Edit `forecast_48h_v2.py`:

```python
LAT, LON = 48.85, 2.35  # same coordinates

MODELS = ["ARPEGE_EUROPE", "GFS_SEAMLESS", "ICON_SEAMLESS",
          "METEOFRANCE", "ECMWF_IFS025"]  # remove ITALIAMETEO if needed

MODEL_IDS = {
    "ARPEGE_EUROPE": "arpege_europe",
    "GFS_SEAMLESS": "gfs_seamless",
    "ICON_SEAMLESS": "icon_seamless",
    "METEOFRANCE": "meteofrance_seamless",
    "ECMWF_IFS025": "ecmwf_ifs025",
}
```

Also update the location name in the output section (search for `"Budva, Crna Gora"`) and the station name.

```bash
python forecast_48h_v2.py
```

Output goes to `forecast_output/forecast_48h.json` — this is what the frontend reads.

### 5. GitHub Pages (optional)

The `docs/` folder contains the frontend:
- `index.html` — analysis/methodology page
- `forecast.html` — live forecast display
- `forecast_data/forecast_48h.json` — auto-updated by GitHub Actions

To automate:
1. Push to GitHub
2. Enable GitHub Pages from `docs/` folder
3. The workflow in `.github/workflows/forecast.yml` runs `forecast_48h_v2.py` hourly and commits the updated JSON

### 6. Visualize error analysis (optional)

```bash
python visualize_analysis.py
```

Creates charts in `plots_extended/` — model comparison, seasonal breakdown, bias analysis, etc.

## Project structure

```
├── forecast_48h_v2.py              # Main pipeline: train + forecast + output
├── advanced_model_analysis.py      # Historical error analysis per model
├── visualize_analysis.py           # Matplotlib charts from analysis results
├── wu_scraper.py                   # WU station scraper (requests/BS4)
├── wu_scraper_colab_3y_fast.py     # Fast scraper for Google Colab (Playwright)
├── requirements.txt
├── .github/workflows/forecast.yml  # Hourly GitHub Actions cron
├── docs/                           # GitHub Pages frontend
│   ├── index.html                  # Analysis page
│   ├── forecast.html               # Live forecast display
│   └── forecast_data/
│       └── forecast_48h.json       # Auto-updated forecast
├── wu_data/                        # Scraped observations
├── trained_models_v2/              # Saved XGBoost models + bias tables
└── forecast_output/                # Pipeline output (JSON + CSV)
```

## Requirements

```
pip install -r requirements.txt
```

- pandas
- numpy
- xgboost
- scikit-learn
- requests

Python 3.10+

## Things to know

- **Training data split** is hardcoded at `2025-07-01` in `forecast_48h_v2.py` (`SPLIT_DATE`). Adjust so you have enough training data (at least 6–12 months) before it, and some test data after.
- **Cloud cover** is derived from solar radiation measurements (not direct cloud observations). If your WU station doesn't have a solar sensor, cloud cover correction will be limited.
- **Precipitation** uses `precip_rate_mm` from WU. Some stations report this poorly — check your data quality.
- **Bura detection** in the analysis is specific to the Adriatic coast (NNE/N wind > 8 m/s). If you're elsewhere, this won't be relevant but also won't break anything.
- The system works best in locations with **consistent station data** and where NWP models have known biases (coastal areas, mountain valleys, urban heat islands).

## License

MIT
