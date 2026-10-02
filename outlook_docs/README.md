# Dugoročne najave

The published page is `docs/dugorocne-najave.html`. It is a manually reviewed,
static outlook, separate from the automated short-range forecast pipeline.
Its charts and source dates remain readable without JavaScript. JavaScript only
switches the ENSO probability threshold.

## Revision: 2 October 2026

Sources checked on 2026-10-02, using Europe/Podgorica for publication dates.
The source URLs below are rolling publications and may later show newer data.

| Source edition | Values used | Source |
| --- | --- | --- |
| C3S, 10 September 2026 | Above-average European winter temperatures; wetter conditions in much of Europe, with exceptions at its southern and northern edges | https://climate.copernicus.eu/seasonal-forecasts |
| NOAA CPC weekly, 28 September 2026, slide 5 | Latest weekly Niño 3.4 SST anomaly: +2.2 °C | https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/lanina/enso_evolution-status-fcsts-web.pdf |
| NOAA CPC strengths, September 2026 | RONI category probabilities below | https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso/roni/strengths/ |
| NOAA CPC discussion, 10 September 2026 | Next discussion scheduled for 8 October | https://www.cpc.ncep.noaa.gov/products/analysis_monitoring/enso_advisory/ensodisc.shtml |
| WMO / Met Office, 28 May 2026 | Annual global anomalies +1.3 to +1.9 °C relative to 1850–1900 in 2026–2030; 86% chance at least one year exceeds the 2024 record | https://public.wmo.int/resources/publication-series/wmo-global-annual-decadal-climate-update/global-annual-decadal-climate-update-2026-2035 |

The local winter text is explicitly an interpretation of the European C3S
summary. No local rainfall totals, snowfall anomalies, January cold-spell dates,
wind frequencies or numerical confidence estimates are inferred from it.
The previous summer station ranking remains unverified and is not presented as an outcome.

## Summer 2026 retrospective

The summer article covers all 92 local days from 1 June through 31 August 2026.
Air temperature and rainfall use explicitly selected ERA5 data from Open-Meteo,
including a matching 1991–2020 baseline. The station archive in this checkout
ends on 18 August and therefore cannot establish a complete summer station mean.

The requested point is 42.28 N, 18.84 E. The returned ERA5 cell is 42.25 N,
19.00 E, with 0.25-degree resolution and API elevation 0 m. Coastal terrain and
the model grid affect these estimates; the article does not present them as
station observations or use them to verify the old all-time ranking claim.

Sea temperatures come from the Open-Meteo Marine archive, requested separately
for the full summer. The returned cell is 42.291664 N, 18.791672 E. These are
archived model values, not in-situ beach measurements. All 2,208 hourly values
are present. The chart shows daily means and the daily minimum-to-maximum range.

`docs/forecast_data/summer_2026.json` records the exact request URLs, source
coordinates, methods, monthly and seasonal summaries, and all 92 daily rows.
Both requests use Europe/Podgorica. Air means average daily means; rain totals
sum daily amounts. Monthly normals use the same calendar month over 30 years;
seasonal normals weight all 92 days. SST means use 24 hourly values per day.

The summer air mean is 27.9739 °C, versus 26.0898 °C for 1991–2020. Rain totals
160.1 mm, versus a 186.22 mm normal (85.97%). The summer SST mean is 25.9539 °C.
Values are rounded for display only. The two `summer-2026-sea*.svg` files are
Matplotlib exports of the daily SST data, with layouts for desktop and phones.

### Chart data (percent)

| Season | Any El Niño | Very strong El Niño | Neutral | La Niña |
| --- | ---: | ---: | ---: | ---: |
| OND 2026 | 100 | 98 | 0 | 0 |
| NDJ 2026/27 | 100 | 93 | 0 | 0 |
| DJF 2026/27 | 100 | 75 | 0 | 0 |
| JFM 2027 | 100 | 39 | 0 | 0 |
| FMA 2027 | 97 | 5 | 3 | 0 |
| MAM 2027 | 82 | 0 | 18 | 0 |
| AMJ 2027 | 43 | 0 | 55 | 2 |

Any El Niño sums the four positive RONI categories in the source table (>= +0.5 °C).
Very strong is a subset (>= +2.0 °C), not an additional probability category.
Seasons overlap. Published probabilities are rounded. Weekly SST anomalies
must not be compared directly with the relative three-month RONI thresholds.

## Updating

1. Read the latest dated primary-source editions before changing the page date.
2. Update the text, source editions, next publication dates, chart data attributes,
   initial bar widths/labels and accessible table together.
3. Check both chart thresholds, keyboard access, narrow screens, local PDF links
   and the page with JavaScript disabled.
4. Retain older PDFs as dated archive documents. The existing `doc_*.py` files and
   `patch_page.py` were written for the August edition; they are not the current
   page generator and should not be run as an October update.

C3S publishes graphical products on the 10th at 12:00 UTC (14:00 in Montenegro
on 10 October 2026). NOAA's next announced discussion is 8 October 2026.
These are source publication dates, not an automatic page-update schedule.
