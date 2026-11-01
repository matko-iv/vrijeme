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
3. **Live prognoza** — GitHub Actions pokreće pipeline svakog sata
   (`--skip-training`): preuzme najnovije prognoze, primijeni korekciju i
   generiše `forecast_48h.json`. SKALA radar nowcast (poseban budva-radar
   projekat) blenduje se u prvih nekoliko sati padavina.
4. **Narativ** — dnevni opis generiše pravilo-bazirani generator; Gemini ga
   samo prefrazira, a deterministički guardrail (`gemini_narrative.validate`)
   odbacuje svaku halucinaciju i vraća sigurnu rečenicu.
5. **Web prikaz** — statična stranica (`docs/forecast.html`) čita JSON i
   prikazuje kartice, grafikone, marine prognozu i radar status.
   Kad SKALA podaci zastare (pipeline ne radi), stranica sama čita DHMZ
   radarsku sliku (`vrijeme.hr/uljenje-stat.png`) u pregledaču, istom paletom
   i speckle filterom kao budva-radar, i kaže pada li kiša u Budvi i gdje je
   najbliža kiša. Za to vrijeme.hr mora dozvoliti CORS; ako ne dozvoljava,
   `cloudflare/dhmz_radar_proxy.js` je Worker koji ga dodaje (URL ide u
   `DHMZ_PROXY_URL`). Bez toga linija kaže da nema svježih podataka.

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
  forecast.yml              # Cron svakog sata
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
python forecast_48h_v3.py --train-rain     # klasifikator kiše + onset kalibracija (~1 min, CPU)

# opciono: skupi CatBoost/LightGBM/Ridge dijagnostički kandidati
python forecast_48h_v3.py --gpu --aux-diagnostics

python test_gemini_narrative.py            # testovi guardrail-a
python test_narrative_variants.py

node test_nowcast_hourly.js                # SKALA NOWCAST gating u docs/forecast.html
node test_daily_conditions.js              # Usklađenost dnevnih ikonica, opisa i narativa
node test_dhmz_radar.js                    # DHMZ radar sken u docs/forecast.html
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
(`train_rain_occurrence_model`). Vidi svih 10 modela sa tolerancijom na
vrijeme (±1–6h prozori, sati do najbližeg kišnog sata svakog modela,
Gauss-ponderisano slaganje modela, dnevni režim), weather code-ove, vlažnost,
oblačnost i pritisak, a prag bira van uzorka (walk-forward, maksimalan CSI).
Od 6h do 30h lead-a prag se spušta na 0.6× jer su starije prognoze manje
oštre. Iz istog razloga je sirova vjerovatnoća za sjutra preniska: u testu na
prognozama starim jedan dan (jun 2024 – feb 2026) sati sa 20–30% imali su kišu
u 48% slučajeva. Zato je `precipitation_pop` za iste lead-ove rekalibrisana
vjerovatnoća tog klasifikatora (`RAIN_OCC_DAY_AHEAD_PLATT`), pa kišni sat
nikad ne pokazuje PoP ispod praga klasifikatora (trenutno 0.28).
Više se ne gasi na 0 kad je ICON-2I suv. Količina u kišnom satu dolazi iz
posebnog regresora (medijana). U walk-forward testu 2023–2026
(`analysis_output/rain_classifier_backtest.py`) klasifikator je bio bolji od
svakog pojedinačnog modela, ansambla i ranijeg ICON-2I gate-a. Stari gate
ostaje samo kao rezerva kad bundle klasifikatora ne postoji.

Verzija 2 feature-a (`analysis_output/rain_v2_backtest.py`, forward-only, isti
sati kao verzija 1):

| Lead | CSI | FAR | Brier | AUC |
|------|-----|-----|-------|-----|
| isti dan (2024-09 – 2026-08) | 0.453 → 0.472 | 0.422 → 0.419 | 0.0393 → 0.0392 | 0.9595 → 0.9598 |
| ~24–30h (previous_day1) | 0.411 → 0.435 | 0.444 → 0.436 | 0.0530 → 0.0525 | 0.9387 → 0.9399 |
| ~48–54h (previous_day2) | 0.376 → 0.389 | 0.506 → 0.504 | 0.0586 → 0.0577 | 0.9213 → 0.9255 |

Granica: na kišnim danima ~750 sati za sjutra dobije PoP 30–60% i kiša padne u
otprilike polovini njih. To je informacija koju 10 determinističkih modela daje
za tajming kiše po satu. Nijedna druga proba nije bila bolja od verzije 2
(`analysis_output/rain_v2_rejected_variants.json`): trening na previous runs
sa lead feature-om izgubio je 6% Brier skill-a za sjutra, dvostepeni vremenski
model je bio lošiji na svim lead-ovima, a LightGBM i dublja/plića stabla
mijenjali su skorove za manje od 1%. Oštriji odgovor traži novu informaciju:
ansambl članove (ECMWF ENS, ICON-EPS), live stanicu (WU ključ u Actions) ili
radar.

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

Početak kiše (`rain_onset_signal`) je prvi sat u kom satna odluka postane
kišna nakon najmanje 3 suva sata, za svaki sat cijelog +48h horizonta. Zato
se početak i satne ikonice ne mogu razići. `rain_onset_probability_by_then`
je kalibrisana šansa da kiša počne (prvi kišni sat od sada) do tog sata;
logistička mapa maksimuma PoP-a, `1 − Π(1 − PoP)` i `log k`, fitovana pri
`--train-rain` na walk-forward OOF vjerovatnoćama. `rain_onset_hazard` je
šansa da kiša počne baš tog sata ako nije ranije. Blok `rain_onset` daje
najvjerovatniji sat i prozor (20–80% uslovne raspodjele početka). SKALA može
otvoriti početak samo u prva dva sata.

Raniji poseban hazard model (1396 feature-a, prag 0.03) je uklonjen. Na istim
slučajevima (2025-07 – 2026-08, iz svakog sata nakon 3 suva sata stanice,
narednih 24h) najavio je 2230 početaka za 1482 izmjerena, 45% lažnih, sa 48%
pogodaka u ±3h. Početak iz satne odluke ima 9–12% lažnih i 81–85% pogodaka u
±3h (timed CSI ±3h 0.19 → 0.46–0.53). Uz verziju 2 feature-a, ±3h pogodci
rastu sa 0.76 na 0.79 za isti dan i sa 0.62 na 0.66 za sjutra.

## Autor

Matija Ivanović · [@matko-iv](https://github.com/matko-iv)
