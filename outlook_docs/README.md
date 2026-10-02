# Dugoročne najave

The published page is `docs/dugorocne-najave.html`: a hand-maintained static page,
separate from the automated short-range forecast pipeline. It keeps the original
design (its own CSS, inline SVG figures, four articles: summer 2026, winter 2026/27,
El Niño, Balkan do 2030). JavaScript is only used for the sea-temperature tooltip;
every figure has its numbers in the text, an `aria-label` or a data table.

## Revision: 2 October 2026

Sources checked on 2026-10-02. The NOAA and C3S URLs are rolling and will later
show newer editions.

| Source edition | Values used | Source |
| --- | --- | --- |
| C3S, 10 September 2026 | DJF: above-average temperature virtually over all land; wetter than average in most of Europe away from its southern and northern edges | https://climate.copernicus.eu/seasonal-forecasts |
| NOAA CPC discussion, 10 September 2026 | August relative Niño 3.4 +1.8 °C (July +1.4); >90% chance of a very strong event; 75% chance of RONI ≥ +2.5 °C in OND (69% in August); next discussion 8 October | https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/ensodisc.shtml |
| NOAA CPC weekly indices, week centred on 23 September 2026 | Niño 3.4: +3.1 °C traditional (1991–2020), +2.2 °C relative | https://www.cpc.ncep.noaa.gov/data/indices/ (`wksst9120.for`, `rel_wksst9120.txt`) |
| NOAA CPC RONI (ERSSTv6) | Peaks: 1972/73 +2.0, 1982/83 +2.4, 1997/98 +2.3, 2015/16 +2.3; JJA 2026 +1.4 (latest values are provisional; table updated by the 5th of each month) | https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/ |
| NOAA CPC strengths, September 2026 | Probability table below | https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/strengths/ |
| WMO / Met Office, 28 May 2026 | Unchanged from August (86% chance at least one year in 2026–2030 beats 2024; annual range +1.3 to +1.9 °C) | https://wmo.int/resources/publication-series/wmo-global-annual-decadal-climate-update/global-annual-decadal-climate-update-2026-2035 |

The monthly NOAA discussion quotes the relative index; the classic anomaly is
0.7–0.9 °C higher from July to September 2026. Weekly values must not be compared with the
three-month RONI thresholds.

### Very strong El Niño probabilities (percent)

| Season | Any El Niño | Very strong (RONI ≥ +2.0) | Neutral | La Niña |
| --- | ---: | ---: | ---: | ---: |
| OND 2026 | 100 | 98 | 0 | 0 |
| NDJ 2026/27 | 100 | 93 | 0 | 0 |
| DJF 2026/27 | 100 | 75 | 0 | 0 |
| JFM 2027 | 100 | 39 | 0 | 0 |
| FMA 2027 | 97 | 5 | 3 | 0 |
| MAM 2027 | 82 | 0 | 18 | 0 |
| AMJ 2027 | 43 | 0 | 55 | 2 |

## Data behind each article

**Summer 2026** (scores the 15 May forecast in `docs/14_V_MMXXVI.pdf`):

- ERA5 0.25° (`models=era5`), cell 42.25 N 19.00 E, 1940–2026: summer mean
  27.97 °C, +1.88 °C on 1991–2020 (26.09 °C), second after 2024 (28.62 °C);
  June +2.13, July +0.97, August +2.56 °C; August 29.77 °C is the warmest of 87;
  59 days with Tmax ≥ 30 °C.
- Station IBUDVA5, 1 June – 31 August: mean 27.45 °C, warmest of 2020–2026 (2024:
  27.31 °C); rain 48.8 mm (2020–2025 summer mean 152 mm), ten rain days, one in
  August; 75 days with Tmax ≥ 30 °C and 67 nights with Tmin > 20 °C, from the 89 days
  with at least 20 hours of temperature data. `wu_data/` ends on 18 August at 15:00;
  19–31 August come from the WU daily history fetched on 27 September 2026, which is
  not yet in `wu_data/`. The station's maxima run warm (siting), so its hot-day counts
  are high in every year; the page shows them next to ERA5 rather than alone.
- ERA5 rain for the cell (160 mm) is not used as Budva rain: the 25 km cell includes
  the hills behind the coast (27 July: 68.8 mm in ERA5, 1.3 mm at the station).
- Sea: `docs/forecast_data/summer_2026.json` (Open-Meteo Marine, 42.29 N 18.79 E,
  24 hourly values per day). It is the same series as the August chart; the
  overlapping days match exactly. The JSON also holds the ERA5 request URLs,
  monthly summaries and daily rows.
- The ranking depends on the product: the default Open-Meteo archive series
  (≈9 km blend, 1950–) puts summer 2026 first. The page uses the homogeneous ERA5
  series for the rank.

**Winter 2026/27**: text updated from the C3S and NOAA rows above. The snow map is
unchanged from August and labelled as such. The local precipitation signal was moved
from "above average" to "no clear signal" because the C3S wet signal excludes
Europe's southern edge.

**El Niño**: the Pacific map is regenerated with `karta_pacifika.py 2026-10-01` (NOAA
OISST v2.1 via CoastWatch ERDDAP, 1971–2000 base, 0.5° grid, 0.25 °C colour steps,
same palette and extent as August). Niño 3.4 box mean: +3.35 °C (3 August: +2.66 °C).
The Indian Ocean Dipole sentence uses OISST weekly snapshots of (10 S–10 N, 50–70 E)
minus (10 S–0, 90–110 E): +0.07 to +0.34 °C in August, then +0.80, +1.01, +0.91 and
+1.02 °C on 10 September – 1 October (positive-event threshold +0.4 °C).

**Balkan do 2030**: unchanged except the hot-day chart and paragraph, which add 2026
up to 1 October: 80 days ≥ 30 °C at the Budva point and 100 at Podgorica, the most in
either series since 1950. Same request as `klimatologija.py` (Open-Meteo archive
default model, points 42.28 N 18.84 E and 42.44 N 19.26 E). That request still
reproduces the August averages (Budva 13 / 31 / 44, Podgorica 18 / 43 / 67).

## Updating

1. Read the latest primary-source editions first: NOAA discussion on 8 October 2026,
   C3S graphical products on 10 October 2026 at 12:00 UTC (14:00 in Montenegro),
   RONI by the 5th of each month.
2. Edit the HTML by hand. Keep the article `meta` dates, the figure captions and
   `aria-label`s, the data tables and the source list in step with the text.
3. For a new Pacific map run `python karta_pacifika.py YYYY-MM-DD` and paste the
   `.b64` into the map's `<image href>`. Update the box value, date and caption.
4. Check the page at desktop and phone widths, with and without JavaScript.
5. Keep the older PDFs as dated archive documents. `doc_*.py` and `patch_page.py`
   generated the August edition and should not be run for later updates.
