# Korekcija vremenske prognoze za Budvu

XGBoost model koji koriguje prognoze iz 11 vremenskih modela koristeći istorijske podatke sa lokalne meteorološke stanice. Treniran na ~6 godina satnih podataka za Budvu (stanica ibudva5 na Weather Underground).

Može se prilagoditi za bilo koju lokaciju koja ima WU stanicu.

**[Live prognoza](https://matko-iv.github.io/vrijeme/forecast.html)** · **[Kako radi](https://matko-iv.github.io/vrijeme/)**

---

## Rezultati

Test period: jul 2025 – feb 2026. Poređenje sa najboljim pojedinačnim modelom:

| Parametar | Korigovano | Najbolji model | Poboljšanje |
|-----------|------------|----------------|:-----------:|
| Temperatura | 0.86°C | 1.28°C (ARPÈGE) | +32% |
| Tačka rose | 1.05°C | 1.84°C (ECMWF) | +43% |
| Vlažnost | 6.1% | 9.0% (ECMWF) | +32% |
| Vjetar | 0.46 m/s | 0.72 m/s (ECMWF) | +36% |
| Udari vjetra | 0.61 m/s | 1.80 m/s (GFS) | +66% |
| Pritisak | 0.25 hPa | 0.70 hPa (ItaliaMeteo) | +64% |
| Oblačnost | 9.7% | 27.4% (BOM) | +65% |
| Padavine | 1.52 mm | 1.52 mm (GFS) | ~0% |
| Solarna rad. | 21.5 W/m² | 35.2 W/m² (ItaliaMeteo) | +39% |

Padavine su jedino što ostaje na nivou najboljeg modela — to je najteža varijabla za korekciju.

---

## Ukratko kako radi

1. Scrapujem satne opservacije sa WU stanice (~50k sati)
2. Preuzmem istorijske prognoze iz 11 modela preko [Open-Meteo](https://open-meteo.com/en/docs/historical-forecast-api)
3. Za svaki model računam bias tabele (mjesec × sat), ensemble statistike, i revizije prognoza (koliko se model koriguje iz dana u dan)
4. Od svega toga napravim ~1300 feature-a i treniram po jedan XGBoost model za svaki parametar
5. Za temperaturu i tačku rose koristim rezidualni pristup (model predviđa korekciju, ne apsolutnu vrijednost) sa Huber loss-om
6. Za padavine — dvostepenski pristup: prvo klasifikator (pada/ne pada), pa regressor za količinu
7. Sve to se vrti automatski preko GitHub Actions i rezultat ide na GitHub Pages

## Modeli

Météo-France, ARPÈGE, ItaliaMeteo ICON 2I, DMI HARMONIE, KNMI HARMONIE, UKMO, GFS, BOM ACCESS, ECMWF IFS 0.25°, ECMWF IFS 9km, ICON Seamless — svi preko Open-Meteo API-ja.

Za Budvu su Météo-France i ARPÈGE najprecizniji za temperaturu. ECMWF IFS 9km je odličan za pritisak.

> ItaliaMeteo ICON 2I pokriva samo Italiju i bliže jadransko primorje. Van tog područja treba ga izbaciti.

---

## Pokretanje

```bash
pip install -r requirements.txt

# puno treniranje + prognoza
python forecast_48h_v2.py

# samo nova prognoza (koristi spremljene modele)
python forecast_48h_v2.py --skip-training
```

Output ide u `forecast_output/forecast_48h.json` i `.csv`.

## Prilagođavanje za drugu lokaciju

1. Nađite WU stanicu na [wunderground.com/wundermap](https://www.wunderground.com/wundermap) — treba barem 2-3 godine podataka
2. U `wu_scraper.py` promijenite `STATION_ID` i pokrenite scraping
3. U `advanced_model_analysis.py` stavite svoje koordinate i pokrenite da preuzmete istorijske prognoze
4. U `forecast_48h_v2.py` ažurirajte `LAT`, `LON` i listu modela
5. Pokrenite `python forecast_48h_v2.py` — trenira i generiše prognozu

Opciono `forecast_horizon_analysis.py` za Previous Runs podatke (revizije prognoza, dostupno od jan 2024).

## Struktura

```
forecast_48h_v2.py          # glavni pipeline
wu_scraper.py               # scraping sa WU
advanced_model_analysis.py  # preuzimanje istorijskih prognoza
docs/                       # GitHub Pages frontend
wu_data/                    # opservacije
previous_runs_data/         # revizije prognoza (Day1/Day2)
trained_models_v2/          # spremljeni XGBoost modeli
forecast_output/            # output JSON/CSV
```

## Napomene

- Oblačnost se računa iz solarne radijacije — bez solarnog senzora nema korekcije oblačnosti
- Padavine zavise od kvaliteta kišomjera na stanici
- Train/test split je na `2025-07-01`, može se promijeniti u kodu
- Python 3.10+, XGBoost, pandas, scikit-learn, requests

## Autor

Matija Ivanović · [@matko-iv](https://github.com/matko-iv)