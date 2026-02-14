# ⛅ Budva Weather — XGBoost korekcija vremenske prognoze

Sistem koji koristi **XGBoost** za korekciju grešaka iz **8 NWP modela** na osnovu 6 godina istorijskih podataka, tabela biasa i revizija prognoza (Previous Runs). Trenira poseban model za svaki meteorološki parametar koristeći **1,006 feature-a**.

Trenutno radi za **Budvu, Crna Gora** (stanica ibudva5) — ali se lako prilagođava za bilo koju lokaciju sa Weather Underground ličnom meteorološkom stanicom.

**[Live prognoza →](https://matko-iv.github.io/vrijeme/forecast.html)** · **[Analiza i metodologija →](https://matko-iv.github.io/vrijeme/)**

---

## Rezultati (test period: jul 2025 — feb 2026)

| Parametar | XGBoost MAE | Najbolji model | Poboljšanje |
|-----------|-------------|----------------|:-----------:|
| **Temperatura** | **0.97°C** | 1.28°C (ARPÈGE) | **+24.3%** |
| Tačka rose | 1.17°C | 2.13°C (ECMWF) | +45.3% |
| Vlažnost | 6.37% | 9.03% (ECMWF) | +29.5% |
| **Brzina vjetra** | **0.45 m/s** | 0.72 m/s (ECMWF) | **+36.7%** |
| Udari vjetra | 0.60 m/s | 1.80 m/s (GFS) | +66.6% |
| **Pritisak** | **0.36 hPa** | 0.70 hPa (ItaliaMeteo) | **+49.0%** |
| Oblačnost | 9.89% | 27.39% (BOM) | +63.9% |
| Padavine | 1.61 mm | 1.52 mm (GFS) | −6.2% |
| Solar. radijacija | 21.45 W/m² | 35.17 W/m² (ItaliaMeteo) | +39.0% |

> Temperatura ispod **1°C MAE**. Oblačnost poboljšana za **64%**. Jedino padavine ostaju na nivou najboljeg modela — precipitacija je inherentno najteža varijabla.

---

## Kako radi

```
WU Stanica          8 NWP Modela              Previous Runs API
(opservacije)       (Open-Meteo)              (Day1/Day2 revizije)
     │                   │                          │
     ▼                   ▼                          ▼
  Feature Engineering (1,006 feature-a)               
  • Ensemble stats (mean, std, range, median)         
  • Tabele istorijskog biasa (mjesec × sat)           
  • Forecast revision (Day0−Day1, Day1−Day2)          
  • Cross-parameter interakcije                       
  • Ciklični vremenski feature-i                      
  • Lokalni signali (bura, jugo, maestral, more)      
                        │
                        ▼
              XGBoost (9 modela)
              po jedan za svaki parametar
                        │
                        ▼
              Korigovana prognoza
              + pametna korekcija vremenskog koda
              + JSON/CSV output za frontend
```

### Pipeline koraci

1. **Scrape opservacije** — satni podaci sa Weather Underground stanice (temperatura, vlažnost, vjetar, pritisak, padavine, solarna radijacija). 6 godina, ~50,000 sati.
2. **Preuzmi istorijske prognoze** — iz 8 modela preko [Open-Meteo Historical Forecast API](https://open-meteo.com/en/docs/historical-forecast-api)
3. **Preuzmi Previous Runs** — Day1/Day2 revizije prognoza iz [Previous Runs API](https://previous-runs-api.open-meteo.com) (od jan 2024)
4. **Feature engineering** — 1,006 feature-a: ensemble statistike, bias tabele, forecast revizije, meteorološki signali
5. **Treniraj XGBoost** — 9 modela sa `reg:absoluteerror`, early stopping, train/test split na jul 2025
6. **Generiši prognozu** — live prognoze + Previous Runs → korekcija → pametni vremenski kodovi → JSON za frontend

Pipeline se pokreće automatski preko GitHub Actions i objavljuje na GitHub Pages.

---

## Korišćeni modeli

| Model | Open-Meteo ID | Rezolucija | Pokrivenost |
|-------|--------------|:----------:|-------------|
| Météo-France Seamless | `meteofrance_seamless` | ~10 km | Evropa |
| ARPÈGE Europe | `arpege_europe` | ~10 km | Evropa |
| ItaliaMeteo ICON 2I | `italia_meteo_arpae_icon_2i` | ~2.2 km | Jadran/Italija |
| UKMO Seamless | `ukmo_seamless` | ~10 km | Globalna |
| GFS Seamless | `gfs_seamless` | ~25 km | Globalna |
| BOM ACCESS Global | `bom_access_global` | ~25 km | Globalna |
| ECMWF IFS 0.25° | `ecmwf_ifs025` | ~25 km | Globalna |
| ICON Seamless | `icon_seamless` | ~13 km | Globalna |

Sortirano po MAE za temperaturu (niži = bolji). Météo-France i ARPÈGE su najprecizniji za Budvu.

> ⚠️ **ItaliaMeteo ICON 2I** pokriva samo Italiju i neposrednu okolinu (jadransku obalu, Sloveniju, Hrvatsku). Ako je vaša lokacija van tog područja, uklonite ga iz `MODELS`.

---

## Forecast Revision Features

Korišcenje [Previous Runs API](https://previous-runs-api.open-meteo.com) donosi ~100 novih feature-a:

- **Day1/Day2 prognoze** za svaki od 7 modela × 8 varijabli
- **Revision (Day0 − Day1)** — koliko se model korigirao u zadnja 24h
- **Day1-to-Day2 revision** — trend korekcija
- **Ensemble revision stats** — prosjek, std, abs_mean revizija svih modela
- **Day0 vs Day1 ensemble spread**

Degradacija po forecast horizontu (prosjek svih modela, jan 2024 – feb 2026):

| Varijabla | Day0 MAE | Day1 MAE | Day2 MAE | Degradacija |
|-----------|----------|----------|----------|:-----------:|
| Temperatura | 1.685°C | 1.769°C | 1.812°C | +7.5% |
| Tačka rose | 2.122°C | 2.113°C | 2.181°C | +2.8% |
| Pritisak | 1.045 hPa | 1.027 hPa | 1.029 hPa | −1.6% |
| Padavine | 2.622 mm | 2.505 mm | 2.589 mm | −1.3% |

---

## Pametna korekcija vremenskog koda

Modeli često prijavljuju kišu (WC ≥ 51) tokom zimske oblačnosti kad zapravo ne pada. Funkcija `correct_weather_code_row()` koristi XGBoost predikcije padavina i oblačnosti da ispravi ovo:

- Ako XGBoost kaže precip < 0.1mm a kod je kiša → downgrade na oblačnost
- Ako XGBoost kaže precip > 0.5mm a kod je vedro → upgrade na kišu
- Koristi XGBoost cloud cover za određivanje vedro/djelimično/oblačno

---

## Prilagođavanje za drugu lokaciju

### 1. Nađite Weather Underground stanicu

Na [wunderground.com/wundermap](https://www.wunderground.com/wundermap) pronađite PWS blizu vaše lokacije. Treba vam station ID (npr. `IBUDVA5`, `KLAX23`, `IMILANO42`).

Dobre stanice imaju:
- Barem 1–2 godine kontinuiranih podataka (više = bolje, 3+ idealno)
- Sve uobičajene senzore (temp, vlažnost, vjetar, pritisak, kiša, solar)
- Konzistentan uptime — praznine su OK ali velike rupe škode

### 2. Scrapujte istorijske opservacije

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
LAT, LON = 42.32, 18.92  # vaša lokacija
OBS_CSV = "wu_data/vasa_stanica_hourly.csv"
START_DATE = "2023-01-01"  # početak vaših opservacija
```

```bash
python advanced_model_analysis.py
```

Kreira `budva_{MODEL}_detailed.csv` per model i `budva_detailed_error_analysis.json` sa metrikama.

> **Napomena:** Open-Meteo historical forecast API je besplatan ali ima rate limit. Skripta automatski retryuje na 429 sa 60s pauzom. Preuzimanje 6 godina za 8 modela traje ~10–20 minuta.

### 4. Preuzmite Previous Runs podatke (opciono ali preporučeno)

```bash
python forecast_horizon_analysis.py
```

Kreira `previous_runs_data/{MODEL}_previous_runs.csv` za 7 modela (od jan 2024). Dodaje ~100 feature-a za trening.

### 5. Trenirajte i pokrenite forecast pipeline

Editujte `forecast_48h_v2.py`:

```python
LAT, LON = 42.32, 18.92

MODELS = ["ARPEGE_EUROPE", "GFS_SEAMLESS", "ICON_SEAMLESS",
          "METEOFRANCE", "ECMWF_IFS025", "UKMO_SEAMLESS", "BOM_ACCESS"]

MODEL_IDS = {
    "ARPEGE_EUROPE": "arpege_europe",
    "GFS_SEAMLESS": "gfs_seamless",
    "ICON_SEAMLESS": "icon_seamless",
    "METEOFRANCE": "meteofrance_seamless",
    "ECMWF_IFS025": "ecmwf_ifs025",
    "UKMO_SEAMLESS": "ukmo_seamless",
    "BOM_ACCESS": "bom_access_global",
}
```

Ažurirajte `"Budva, Crna Gora"` i ime stanice u output sekciji.

```bash
python forecast_48h_v2.py
```

Output: `forecast_output/forecast_48h.json` + `forecast_output/forecast_48h.csv`

### 6. GitHub Pages (opciono)

`docs/` folder sadrži frontend:
- `index.html` — analiza i metodologija (sa interaktivnom kartom grešaka)
- `forecast.html` — prikaz live prognoze (responsive, mobilni-friendly)
- `forecast_data/forecast_48h.json` — automatski ažuriran

## Struktura projekta

```
├── forecast_48h_v2.py              # Glavni pipeline: trening + prognoza + output (v3)
├── forecast_horizon_analysis.py    # Previous Runs analiza degradacije
├── advanced_model_analysis.py      # Istorijska analiza grešaka po modelu
├── complete_analysis.py            # Cloud cover + weather code verifikacija
├── recompute_mae.py                # Svjež MAE proračun za svih 8 modela
├── wu_scraper.py                   # WU scraper (satni podaci)
├── requirements.txt
│
├── docs/                           # GitHub Pages frontend
│   ├── index.html                  # Analiza i metodologija
│   ├── forecast.html               # Live prognoza
│   ├── forecast_data/
│   │   └── forecast_48h.json
│   └── images/                     # Generisani plotovi za sajt
│
├── wu_data/                        # Opservacije sa WU stanice
│   ├── merged_observations.csv     # 49,964 redova (2020–2026)
│   ├── all_data.csv
│   └── ibudva5_hourly_*.csv
│
├── previous_runs_data/             # Day1/Day2 revizije (7 modela, jan 2024+)
│   ├── METEOFRANCE_previous_runs.csv
│   ├── ARPEGE_EUROPE_previous_runs.csv
│   ├── ECMWF_IFS025_previous_runs.csv
│   ├── GFS_SEAMLESS_previous_runs.csv
│   ├── ICON_SEAMLESS_previous_runs.csv
│   ├── UKMO_SEAMLESS_previous_runs.csv
│   └── BOM_ACCESS_previous_runs.csv
│
├── trained_models_v2/              # XGBoost modeli + metadata
│   ├── xgb_temperature_2m.json
│   ├── xgb_*.json                  # 9 modela ukupno
│   ├── feature_lists.json
│   ├── training_results.json
│   └── bias_tables.json
│
├── forecast_output/                # Output pipeline-a
│   ├── forecast_48h.json
│   └── forecast_48h.csv
│
├── budva_*_detailed.csv            # Istorijske prognoze po modelu (8 fajlova)
└── plots/ plots_extended/          # Generisani grafikoni
```

## Zavisnosti

```bash
pip install -r requirements.txt
```

- `pandas` + `numpy` 
- `xgboost` 
- `scikit-learn` 
- `requests`
- `matplotlib` — vizualizacija (opciono)

Python 3.10+

## Napomene

- **Train/test split** je na `2025-07-01` (`SPLIT_DATE`). Podesite tako da imate barem 6–12 mjeseci za trening prije tog datuma.
- **Oblačnost** se izvodi iz mjerenja solarne radijacije (nije direktna opservacija). Ako vaša stanica nema solarni senzor, korekcija oblačnosti će biti ograničena.
- **Padavine** koriste `precip_rate_mm` sa WU. Neke stanice ovo prijavljuju loše — provjerite kvalitet podataka.
- **Previous Runs** su dostupni od jan 2024. Stariji podaci nemaju ove feature-e (XGBoost korektno tretira NaN).
- Sistem najbolje radi na lokacijama sa **konzistentnim podacima stanice** i gdje NWP modeli imaju poznate biase (priobalna područja, planinske doline, urbana toplija ostrva).

## Autor

Matija Ivanović · [@matko-iv](https://github.com/matko-iv)