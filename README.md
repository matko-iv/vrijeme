# Budva ML Weather Forecast pipeline

Korigovana 48-satna i 7-dnevna prognoza za Budvu. Pipeline preuzima sirove
prognoze sa 10 globalnih NWP modela, uči njihov bias na 6 godina lokalnih
mjerenja i objavljuje JSON koji pokreće web stranicu.

## Kako radi

1. **Historijski podaci** — satne prognoze (2020–2026) sa Open-Meteo API-ja,
   uparene sa mjerenjima sa [IBUDVA5](https://www.wunderground.com/dashboard/pws/IBUDVA5)
   Weather Underground stanice.
2. **Trening** — produkcione korekcije koriste XGBoost direct/residual/MSE
   modele i validaciono izabran blend. LightGBM trenira kvantilne modele (CQR).
   Za količinu padavina se koriste focal loss, izotonička kalibracija i
   precision-first prag na odvojenim hronološkim blokovima; odluku da li pada
   kiša donosi multi-model klasifikator (vidi *Šansa za padavine*). CatBoost, dodatni LightGBM i Ridge
   meta-model dostupni su kao eksplicitna dijagnostika, ali nijesu kandidati za
   produkciju dok Ridge ne dobije temporalne OOF predikcije.
3. **Live prognoza** — GitHub Actions pokreće pipeline na svakih 5 sati
   (`--skip-training`): preuzme najnovije prognoze, primijeni korekciju i
   generiše `forecast_48h.json`. SKALA radar nowcast (poseban budva-radar
   projekat) blenduje se u prvih nekoliko sati padavina.
4. **Narativ** — dnevni opis generiše pravilo-bazirani generator; Gemini ga
   samo prefrazira, a deterministički guardrail (`gemini_narrative.validate`)
   odbacuje svaku halucinaciju i vraća sigurnu rečenicu.
5. **Web prikaz** — statična stranica (`docs/forecast.html`) čita JSON i
   prikazuje kartice, grafikone, marine prognozu i radar status.

## Modeli

| Model | Rezolucija |
|-------|-----------|
| ARPÈGE Europe | ~10 km |
| GFS Seamless | ~25 km |
| ICON Seamless | ~13 km |
| Météo-France | ~10 km |
| ECMWF IFS 0.25° | ~25 km |
| ItaliaMeteo ICON-2I | ~2.2 km |
| UKMO Seamless | ~10 km |
| ECMWF IFS | ~9 km |
| KNMI Seamless | ~11 km |
| DMI Seamless | ~13 km |

## Struktura

```
forecast_48h_v3.py          # Glavni pipeline (fetch → train → correct → output)
prob_forecast.py            # Probabilistički alati (EMOS, CQR, ECC, verifikacija)
gemini_narrative.py         # Gemini prefraziranje + guardrail
narrative_variants.py       # Pool fraza za dnevni narativ
wu_scraper.py               # Scraper historijskih obs (Weather Underground)
advanced_model_analysis.py  # Analiza grešaka po modelu
visualize_analysis*.py      # Grafikoni analize
trained_models_v2/          # Trenirani modeli + bias tabele
.github/workflows/
  forecast.yml              # Cron na svakih 5 sati
docs/
  forecast.html             # 48h + 7-dnevna prognoza
  index.html                # Analiza tačnosti modela
  forecast_data/forecast_48h.json
```

## Pokretanje

```bash
pip install -r requirements.txt

python forecast_48h_v3.py                  # puni pipeline (trening + prognoza)
python forecast_48h_v3.py --skip-training  # samo prognoza, postojeći modeli
python forecast_48h_v3.py --train-rain     # samo klasifikator kiše (~1 min, CPU)

# opciono: skupi CatBoost/LightGBM/Ridge dijagnostički kandidati
python forecast_48h_v3.py --gpu --aux-diagnostics

python test_gemini_narrative.py            # testovi guardrail-a
python test_narrative_variants.py

node test_nowcast_hourly.js                # SKALA NOWCAST gating u docs/forecast.html
node test_daily_conditions.js              # Usklađenost dnevnih ikonica, opisa i narativa
```

## GPU trening

Pipeline automatski koristi NVIDIA CUDA GPU kada ga XGBoost može stvarno
koristiti. `--gpu` uključuje fail-fast režim: proces se prekida umjesto da
neprimjetno nastavi na CPU-u.

```powershell
# Brza provjera (fit + predict za XGBoost, CatBoost i LightGBM)
python forecast_48h_v3.py --gpu --check-device

# Puni trening na GPU-u
python forecast_48h_v3.py --gpu

# Eksplicitni CPU fallback
python forecast_48h_v3.py --cpu
```

Isto se može podesiti varijablama `FC_DEVICE=auto|cuda|cpu` i `FC_GPU_ID=0`.
Na računarima sa više OpenCL platformi LightGBM se može odvojeno usmjeriti sa
`FC_LGB_GPU_PLATFORM_ID` i `FC_LGB_GPU_DEVICE_ID`.
XGBoost koristi CUDA, CatBoost koristi svoj CUDA backend, a LightGBM na Windowsu
koristi GPU preko OpenCL-a. Pandas feature engineering, Optuna orchestration i
kalibracija ostaju na CPU-u; focal-loss gradijenti su NumPy/CPU dok se XGBoost
stabla grade na GPU-u. Skupa pomoćna dijagnostika je podrazumijevano isključena;
uključuje se sa `--aux-diagnostics` ili `FC_AUX_DIAGNOSTICS=1`. GitHub-hosted
`ubuntu-latest` job ostaje CPU-only i pokreće samo `--skip-training`; modeli
trenirani lokalno na GPU-u mogu se normalno učitati tamo.

Puni retraining zahtijeva `wu_data/merged_observations.csv`. Canonical WU tabela
zamjenjuje stale observation kolone iz modelskih CSV fajlova (uključujući stari
mean-gust bug), a godišnji rain-label QA prekida trening ako coverage ili odnos
suvih/kišnih sati izgleda neispravno. NaN u canonical tabeli ostaje nepoznat —
nikad se ne pretvara automatski u suvi sat.

Output ide u `forecast_output/forecast_48h.json`; GitHub Actions ga kopira u
`docs/forecast_data/`. Gemini narativ traži `GEMINI_API_KEY` u okruženju;
bez ključa pipeline koristi pravilo-bazirane rečenice.

## Šansa za padavine

`precip_probability` na dnevnoj kartici je procenat NWP modela (od 10) koji
predviđaju >0.1 mm u bilo kom satu tog dana — consensus metrika.

Satnu odluku „pada kiša (≥ 0.2 mm)” donosi multi-model XGBoost klasifikator
(`train_rain_occurrence_model`). Vidi svih 10 modela, njihov ±1h vremenski
prozor, weather code-ove, vlažnost i oblačnost, a prag bira van uzorka
(walk-forward, maksimalan CSI). Od 6h do 30h lead-a prag se spušta na 0.6×
jer su starije prognoze manje oštre. `precipitation_pop` je vjerovatnoća tog
klasifikatora (više se ne gasi na 0 kad je ICON-2I suv), a količina u kišnom
satu dolazi iz posebnog regresora (medijana). U walk-forward testu 2023–2026
(`analysis_output/rain_classifier_backtest.py`) klasifikator je bio bolji od
svakog pojedinačnog modela, ansambla i ranijeg ICON-2I gate-a. Stari gate
ostaje samo kao rezerva kad bundle klasifikatora ne postoji.

Satni JSON i dalje izlaže `rain_signal` (konačna odluka),
`rain_signal_confidence`, SKALA support i ICON-2I dijagnostiku
(`italiameteo_precipitation`, `italiameteo_rain_signal`,
`italiameteo_rain_accepted` = ICON-2I se slaže sa konačnom odlukom). Tako
klijent ne mora da zaključuje signal kiše iz zaokružene količine padavina.

Verifikacija kiše: stanični sat je označen početkom (`precip_rate_mm` za
[t, t+1)), a Open-Meteo red t sadrži kišu za [t-1, t], pa se stanični sat t
upoređuje sa modelskim redom t+1 (objavljeni JSON je već pomjeren).
`precipitation_obs` u `budva_*_detailed.csv` je WU dnevni kumulativ
(resetuje se u ponoć) i nije satna kiša.

Odvojeni `rain_onset_signal` važi za svaki sat cijelog +48h horizonta. Uz njega
se objavljuju kalibrisani `rain_onset_hazard`, vjerovatnoća do tog sata,
validacionim onset/FAR/CSI skorovima izabrani prag i izvor. SKALA može potvrditi
signal samo u prva dva sata; nakon toga odluku nose ITALIAMETEO i XGBoost onset
model, bez prenošenja radarskog signala van njegovog provjerenog dometa.

## Autor

Matija Ivanović · [@matko-iv](https://github.com/matko-iv)
