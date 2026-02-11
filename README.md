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
| Météo-France Seamless | `meteofrance_seamless` | Evropa |
| ARPEGE Europe | `arpege_europe` | Evropa |
| ItaliaMeteo ICON 2I | `italia_meteo_arpae_icon_2i` | Italija, Jonsko i Jadransko more |

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
LAT, LON = 42.32, 18.92  # vaša lokacija (primjer: Vrela)
OBS_CSV = "wu_data/vasa_stanica_hourly.csv"
START_DATE = "2023-01-01"  # da odgovara početku vaših opservacija (odnosno opservacija stanice)

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
LAT, LON = 42.32, 18.92

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

Output ide u `forecast_output/forecast_48h.json`

### 5. GitHub Pages (opciono)

Folder `docs/` sadrži frontend:
- `index.html` — stranica sa analizom/metodologijom
- `forecast.html` — prikaz live prognoze
- `forecast_data/forecast_48h.json` — automatski ažuriran preko GitHub Actions

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
├── requirements.txt
├── docs/                           # GitHub Pages frontend
│   ├── index.html                  # Stranica analize
│   ├── forecast.html               # Prikaz live prognoze
│   └── forecast_data/
│       └── forecast_48h.json       # Automatski ažurirana prognoza
├── wu_data/                        # Scrapovane opservacije
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

- **Training/test split** je hardkodiran na `2025-04-01` u `forecast_48h_v2.py` (`SPLIT_DATE`). Podesite tako da imate dovoljno podataka za trening (barem 6–12 mjeseci) prije tog datuma, i nešto test podataka poslije.
- **Oblačnost** se izvodi iz mjerenja solarne radijacije (ne iz direktnih opservacija oblačnosti). Ako vaša WU stanica nema solarni senzor, korekcija oblačnosti će biti ograničena.
- **Padavine** koriste `precip_rate_mm` sa WU. Neke stanice ovo prijavljuju loše — provjerite kvalitet podataka.
- Sistem najbolje radi na lokacijama sa **konzistentnim podacima stanice** i đe NWP modeli imaju poznate biase (priobalna područja, planinske doline, urbana toplija ostrva).