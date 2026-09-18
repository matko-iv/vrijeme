"""Deterministic regression tests for forecast math and train/live parity."""

import os
import unittest

os.environ['FC_DEVICE'] = 'cpu'

import numpy as np
import pandas as pd

import forecast_48h_v3 as fc
import prob_forecast as pf


class _FakeDMatrix:
    def __init__(self, labels, weights=None):
        self._labels = np.asarray(labels, dtype=float)
        self._weights = (np.asarray(weights, dtype=float) if weights is not None
                         else np.array([], dtype=float))

    def get_label(self):
        return self._labels

    def get_weight(self):
        return self._weights


class _ConstantModel:
    def __init__(self, value):
        self.value = float(value)

    def predict(self, X):
        return np.full(len(X), self.value)


class ForecastCoreTests(unittest.TestCase):
    def test_focal_derivatives_and_sample_weights(self):
        gamma, alpha = 2.0, 0.25
        margins = np.array([-1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
        labels = np.array([0, 0, 0, 1, 1, 1], dtype=float)
        objective = fc.focal_loss_xgb_objective(gamma, alpha)
        grad, hess = objective(margins, _FakeDMatrix(labels))

        def loss(z):
            p = 1.0 / (1.0 + np.exp(-z))
            pt = np.where(labels == 1, p, 1 - p)
            at = np.where(labels == 1, alpha, 1 - alpha)
            return -at * np.power(1 - pt, gamma) * np.log(pt)

        eps = 1e-4
        grad_fd = (loss(margins + eps) - loss(margins - eps)) / (2 * eps)
        hess_fd = (loss(margins + eps) - 2 * loss(margins)
                   + loss(margins - eps)) / (eps ** 2)
        np.testing.assert_allclose(grad, grad_fd, rtol=2e-5, atol=2e-6)
        np.testing.assert_allclose(hess, hess_fd, rtol=2e-4, atol=2e-5)

        weights = np.array([0.7, 1.5, 1.0, 1.4, 0.8, 2.0])
        grad_w, hess_w = objective(margins, _FakeDMatrix(labels, weights))
        np.testing.assert_allclose(grad_w, grad * weights)
        np.testing.assert_allclose(hess_w, hess * weights)

        # A very hard positive has negative exact curvature; XGBoost must see
        # the documented positive floor, not abs(negative curvature).
        _, hard_hess = objective(np.array([-5.0]), _FakeDMatrix([1.0]))
        np.testing.assert_allclose(hard_hess, [1e-3])

        metric = fc.focal_loss_xgb_feval(gamma, alpha)
        _, weighted_value = metric(margins, _FakeDMatrix(labels, weights))
        self.assertAlmostEqual(weighted_value, float(np.average(loss(margins), weights=weights)))

    def test_meteorological_metric_boundaries(self):
        perfect = fc.meteorological_metrics([0, 1, 0, 1], [0, 1, 0, 1])
        self.assertGreater(perfect['sedi'], 0.999)
        self.assertEqual(perfect['precision'], 1.0)

        always_dry = fc.meteorological_metrics([0, 1, 0, 1], [0, 0, 0, 0])
        self.assertEqual(always_dry['precision'], 0.0)
        self.assertEqual(always_dry['far'], 0.0)

    def test_pit_uses_interior_midranks(self):
        members = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        pit = fc.pit_values(np.array([0.0, 4.0]), members)
        self.assertGreater(pit[0], 0.0)
        self.assertLess(pit[1], 1.0)
        self.assertAlmostEqual(pit[0] + pit[1], 1.0)

    def test_observation_only_features_are_excluded(self):
        frame = pd.DataFrame({
            'safe_model_feature': [1.0, 2.0],
            'dry_spell_length': [5.0, 6.0],
            'monthly_clim_rain_freq': [0.1, 0.2],
            '_derived_precip_obs': [0.0, 0.2],
            '_canonical_precip_rate_mm': [0.0, 0.2],
        })
        self.assertEqual(fc.get_feature_columns(frame), ['safe_model_feature'])

    def test_temporal_cv_groups_duplicate_valid_times(self):
        unique = pd.date_range('2026-01-01', periods=400, freq='h')
        duplicated = pd.Series(np.repeat(unique.values, 3))
        splits = fc._timestamp_grouped_cv_splits(duplicated)
        self.assertEqual(len(splits), 3)
        for train_idx, val_idx in splits:
            train_times = set(duplicated.iloc[train_idx])
            val_times = set(duplicated.iloc[val_idx])
            self.assertTrue(train_times.isdisjoint(val_times))
            self.assertLessEqual(
                max(train_times), min(val_times) - pd.Timedelta(hours=72)
            )

        q_times = pd.Series(np.repeat(
            pd.date_range('2025-01-01', periods=2200, freq='h').values, 3
        ))
        q_X = pd.DataFrame({'x': np.arange(len(q_times), dtype=float)})
        q_y = pd.Series(np.arange(len(q_times), dtype=float))
        folds = fc._quantile_clock_split(q_X, q_y, q_times)
        fit_times = set(folds['time_fit'])
        val_times = set(folds['time_val'])
        cal_times = set(folds['time_cal'])
        self.assertTrue(fit_times.isdisjoint(val_times | cal_times))
        self.assertTrue(val_times.isdisjoint(cal_times))
        self.assertGreaterEqual(folds['fit_val_gap'], pd.Timedelta(hours=72))
        self.assertGreaterEqual(folds['val_cal_gap'], pd.Timedelta(hours=72))

        precip_folds = fc._precip_clock_split(q_X, q_y, q_times)
        precip_sets = [set(precip_folds[f'time_{name}'])
                       for name in ('fit', 'val', 'cal', 'gate')]
        for i, left in enumerate(precip_sets):
            for right in precip_sets[i + 1:]:
                self.assertTrue(left.isdisjoint(right))
        for gap_name in ('fit_val_gap', 'val_cal_gap', 'cal_gate_gap'):
            self.assertGreaterEqual(
                precip_folds[gap_name], pd.Timedelta(hours=72)
            )

    def test_rain_consensus_excludes_missing_members_and_counts_hours(self):
        values = pd.DataFrame({
            # Exactly 0.1 mm is wet everywhere else in the rain contract.
            'a': [0.2, 0.1, np.nan, 0.0],
            'b': [0.3, np.nan, np.nan, 0.2],
            'c': [0.4, np.nan, np.nan, 0.0],
        })
        wet, available, agreement, wet_hour = fc._rain_consensus_stats(values)
        np.testing.assert_array_equal(wet.values, [3, 1, 0, 1])
        np.testing.assert_array_equal(available.values, [3, 1, 0, 3])
        np.testing.assert_allclose(
            agreement.values, [1.0, 1.0, np.nan, 1 / 3], equal_nan=True
        )
        np.testing.assert_allclose(
            wet_hour.values, [1.0, 1.0, np.nan, 0.0], equal_nan=True
        )

        frame = pd.DataFrame({
            'datetime': pd.date_range('2026-01-01', periods=4, freq='h'),
            'ARPEGE_EUROPE_precipitation_model': values['a'],
            'GFS_SEAMLESS_precipitation_model': values['b'],
            'ICON_SEAMLESS_precipitation_model': values['c'],
        })
        engineered = fc.engineer_features(frame)
        # Three wet model votes in hour 1 are one wet hour, not three.
        self.assertEqual(engineered['rain_hours_6h'].iloc[0], 1.0)
        self.assertEqual(engineered['rain_hours_6h'].iloc[1], 2.0)
        self.assertTrue(np.isnan(engineered['rain_agreement'].iloc[2]))

    def test_isolated_convection_requires_independent_support(self):
        model = fc.TRUSTED_RAIN_MODEL
        frame = pd.DataFrame({
            f'{model}_cape_model': [100.0, 800.0, 800.0, 800.0, 2000.0, 800.0],
            # Both CIN sign conventions represent the same inhibition strength.
            f'{model}_convective_inhibition_model': [200.0, 50.0, 150.0, -50.0, 0.0, np.nan],
            f'{model}_showers_model': [0.0, 0.3, 0.3, 0.0, 0.0, 0.0],
            f'{model}_lightning_potential_model': [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            f'{model}_nbr_fwet': [0.0, 0.0, 0.0, 0.2, 0.0, 0.0],
            'storm_wc_count': [0.0] * 6,
        })
        rescued = fc._convective_rescue_mask(frame, np.ones(6, dtype=bool))
        np.testing.assert_array_equal(
            rescued,
            # lightning; showers+CAPE+low CIN; high-CIN rejected;
            # neighborhood+CAPE+abs(CIN); CAPE-only and missing-CIN rejected.
            [True, True, False, True, False, False],
        )

    def test_cin_engineering_uses_inhibition_magnitude(self):
        frame = pd.DataFrame({
            'datetime': pd.date_range('2026-08-01', periods=2, freq='h'),
            'ITALIAMETEO_ICON2I_cape_model': [800.0, 800.0],
            'ITALIAMETEO_ICON2I_convective_inhibition_model': [50.0, -150.0],
        })
        engineered = fc.engineer_features(frame)
        np.testing.assert_allclose(engineered['cin_ens_mean'], [50.0, 150.0])
        np.testing.assert_array_equal(engineered['low_cin_indicator'], [1.0, 0.0])

    def test_precip_label_qa_rejects_wet_only_missing_dry_data(self):
        n = 20
        frame = pd.DataFrame({
            'datetime': pd.date_range('2025-01-01', periods=n, freq='h'),
            'temperature_2m_obs': np.full(n, 15.0),
            'relative_humidity_2m_obs': np.full(n, 70.0),
            'pressure_msl_obs': np.full(n, 1013.0),
        })
        good = pd.Series([0.0] * 18 + [0.2, 1.0])
        report = fc._validate_precip_observations(
            frame, good, 'unit-good', min_year_rows=10
        )
        self.assertEqual(report[0]['completeness'], 1.0)
        bad = pd.Series([np.nan] * 18 + [0.2, 1.0])
        with self.assertRaisesRegex(RuntimeError, 'wet-only'):
            fc._validate_precip_observations(
                frame, bad, 'unit-bad', min_year_rows=10
            )

    def test_onset_age_and_cdf_do_not_split_equal_age(self):
        frame = pd.DataFrame({
            '_derived_precip_obs': [0.0, 0.0, 0.0, 0.2, 0.2, 0.0],
            'datetime': pd.date_range('2026-01-01', periods=6, freq='h'),
            'feature': np.arange(6, dtype=float),
        })
        X, y, _ = fc.build_onset_person_period(frame, ['feature'])
        np.testing.assert_array_equal(X['dry_age'].values, [1, 2, 3, 3, 1])
        np.testing.assert_array_equal(y.values, [0, 0, 0, 1, 0])

        live_age = fc._forecast_onset_dry_age([0.0, 0.0, 0.0, 0.2])
        np.testing.assert_array_equal(live_age, [1, 2, 3, 3])
        cdf = fc._onset_cdf_from_hazard([0.1, 0.1, 0.1, 0.1], [1, 2, 3, 3])
        self.assertTrue(np.all(np.diff(cdf) > 0))
        self.assertAlmostEqual(cdf[-1], 1 - 0.9 ** 4)

        live_age_unknown = fc._forecast_onset_dry_age([0.0, np.nan, 0.0])
        np.testing.assert_allclose(live_age_unknown, [1.0, np.nan, 1.0], equal_nan=True)
        cdf_unknown = fc._onset_cdf_from_hazard(
            [0.1, 0.1, 0.1, 0.1], [1.0, np.nan, 1.0, 2.0]
        )
        np.testing.assert_allclose(cdf_unknown, [0.1, np.nan, 0.1, 0.19], equal_nan=True)

        # Capped age must not collapse a genuine two-hour timing miss to zero.
        pred_hours, obs_hours = fc._onset_event_hours(
            [0.01, 0.2, 0.01, 0.01], [0, 0, 0, 1], [47, 48, 48, 48],
            pd.date_range('2026-01-02', periods=4, freq='h'), threshold=0.1,
        )
        self.assertEqual(pred_hours, [1.0])
        self.assertEqual(obs_hours, [3.0])
        self.assertEqual(fc.onset_timing_metrics(pred_hours, obs_hours)['mae_hours'], 2.0)

        raw_features = pd.DataFrame({
            'datetime': pd.date_range('2026-01-01', periods=3, freq='h'),
            'humidity': [60.0, 70.0, 80.0],
            'agreement': [0.0, 0.5, 1.0],
            # Live previous-run features can arrive as integer columns.  The
            # shift adds a trailing NaN, which must widen rather than fail.
            'previous_run_humidity': [51, 48, 45],
        })
        aligned = fc._align_onset_features_to_display(raw_features)
        pd.testing.assert_series_equal(
            aligned['datetime'], raw_features['datetime'], check_names=False
        )
        np.testing.assert_allclose(
            aligned['humidity'], [70.0, 80.0, np.nan], equal_nan=True
        )
        np.testing.assert_allclose(
            aligned['agreement'], [0.5, 1.0, np.nan], equal_nan=True
        )
        np.testing.assert_allclose(
            aligned['previous_run_humidity'], [48.0, 45.0, np.nan], equal_nan=True
        )

    def test_onset_signal_covers_48h_and_uses_saved_threshold(self):
        class ConstantHazard:
            def predict_proba(self, X):
                p = np.full(len(X), 0.15)
                return np.column_stack([1.0 - p, p])

        now = fc.local_now().floor('h')
        frame = pd.DataFrame({
            'datetime': pd.date_range(now, periods=52, freq='h'),
            'precipitation_ens_mean': np.zeros(52),
        })
        info = fc.predict_onset_timing(frame, {
            'model': ConstantHazard(), 'calibrator': None,
            'features': ['dry_age'], 'declare_threshold': 0.14,
            'band_threshold': 0.07,
        })
        self.assertEqual(len(info['prob_by_hour']), 48)
        self.assertEqual(info['declare_threshold'], 0.14)
        self.assertFalse(info['prob_by_hour'][0]['eligible'])
        self.assertTrue(info['prob_by_hour'][2]['eligible'])
        self.assertTrue(info['prob_by_hour'][2]['signal'])

    def test_skala_only_supports_hourly_onset_in_first_two_hours(self):
        now = fc.local_now().floor('h')
        corrected = pd.DataFrame({
            'datetime': pd.date_range(now, periods=3, freq='h'),
            'rain_signal': [1.0, 1.0, 1.0],
            'rain_signal_skala_support': [1.0, 1.0, 1.0],
        })
        onset = {'prob_by_hour': [
            {'datetime': ts.isoformat(), 'lead_h': i + 1, 'eligible': True,
             'hazard': 0.08, 'p_by_then': 0.08, 'threshold': 0.10,
             'signal': False}
            for i, ts in enumerate(corrected['datetime'])
        ]}
        attached = fc.attach_hourly_onset_signal(corrected, onset)
        np.testing.assert_array_equal(
            attached['rain_onset_signal'].values, [1.0, 1.0, 0.0]
        )
        np.testing.assert_array_equal(
            attached['rain_onset_skala_support'].values, [1.0, 1.0, 0.0]
        )
        self.assertEqual(attached.iloc[0]['rain_onset_source'],
                         'ITALIAMETEO_XGB_ONSET+SKALA')

    def test_shared_stack_and_blend_dispatcher(self):
        X = pd.DataFrame({'x': [0.0, 1.0]})
        frame = pd.DataFrame({'baseline': [10.0, 20.0]})
        bundle = {
            'direct_model': _ConstantModel(1),
            'resid_model': _ConstantModel(2),
            'mse_model': _ConstantModel(3),
            'method': 'stacked',
            'stack_weights': (0.2, 0.3, 0.5),
        }
        stacked = fc._predict_nonprecip_bundle(bundle, X, frame, 'baseline')
        expected_stack = 0.2 * 1 + 0.3 * (frame['baseline'].values + 2) + 0.5 * 3
        np.testing.assert_allclose(stacked, expected_stack)

        bundle['method'] = 'blend'
        bundle['blend_alpha'] = 0.6
        blended = fc._predict_nonprecip_bundle(bundle, X, frame, 'baseline')
        np.testing.assert_allclose(
            blended, 0.6 * expected_stack + 0.4 * frame['baseline'].values
        )

        broken = dict(bundle)
        broken['resid_model'] = None
        with self.assertRaisesRegex(ValueError, 'resid_model'):
            fc._predict_nonprecip_bundle(broken, X, frame, 'baseline')

    def test_cloud_postprocessor_is_shared_and_deterministic(self):
        frame = pd.DataFrame({
            'datetime': pd.to_datetime([
                '2026-07-01 08:00', '2026-07-01 12:00',
                '2026-07-01 18:00', '2026-01-01 16:00'
            ]),
            'cloud_cover_ens_mean': [20.0, 5.0, 20.0, 95.0],
        })
        out = fc._postprocess_cloud_prediction([90.0, 90.0, 42.0, 10.0], frame)
        # 17-18h terrain shading and weak winter sunlight fall back to the
        # ensemble. Reliable daylight retains the ML correction and caps.
        np.testing.assert_allclose(out, [90.0, 35.0, 20.0, 95.0])
        legacy = fc._postprocess_cloud_prediction(
            [90.0, 90.0, 42.0, 10.0], frame, station_target_version=1
        )
        # Existing un-retrained artifact falls back at all reported bad hours.
        np.testing.assert_allclose(legacy, [90.0, 5.0, 20.0, 95.0])

    def test_station_cloud_proxy_calibrates_hour_specific_shading(self):
        times = pd.to_datetime([
            '2024-07-01 12:00', '2024-07-02 12:00',
            '2024-07-01 18:00', '2024-07-02 18:00',
            '2026-07-01 12:00', '2026-07-01 18:00',
        ])
        # The station's clear 18h reading is only 200 W/m2, versus 800 at noon.
        # Both held-out clear readings must map to the same near-clear sky state.
        solar = pd.Series([800.0, 760.0, 200.0, 190.0, 780.0, 195.0])
        fit = pd.Series([True, True, True, True, False, False])
        cloud, envelope = fc.derive_cloud_cover_from_station_solar(
            pd.Series(times), solar, fit_mask=fit, clear_quantile=1.0,
            min_station_transmission=0.0,
        )
        np.testing.assert_allclose(cloud.iloc[-2:], [2.5, 2.5], atol=0.1)

    def test_quantile_tail_exceedance_decays(self):
        qdf = pd.DataFrame({
            'q05': [1.0], 'q10': [1.2], 'q25': [1.5], 'q50': [2.0],
            'q75': [2.5], 'q90': [3.0], 'q95': [3.5],
        })
        p_at_q95 = pf.exceedance_from_quantiles(qdf, pf.DEFAULT_ALPHAS, 3.5)[0]
        p_10 = pf.exceedance_from_quantiles(qdf, pf.DEFAULT_ALPHAS, 10.0)[0]
        p_17 = pf.exceedance_from_quantiles(qdf, pf.DEFAULT_ALPHAS, 17.0)[0]
        self.assertAlmostEqual(p_at_q95, 0.05)
        self.assertGreater(p_10, p_17)
        self.assertLess(p_17, 1e-6)

        tied = qdf.copy()
        tied['q90'] = tied['q95']
        just_above = pf.exceedance_from_quantiles(
            tied, pf.DEFAULT_ALPHAS, float(tied['q95'].iloc[0]) + 1e-6
        )[0]
        farther = pf.exceedance_from_quantiles(tied, pf.DEFAULT_ALPHAS, 10.0)[0]
        self.assertAlmostEqual(just_above, 0.05, places=5)
        self.assertGreater(just_above, farther)

    def test_ecc_uses_integer_rank_indices(self):
        alphas = pf.DEFAULT_ALPHAS
        qdf = pd.DataFrame({
            f'q{int(round(a * 100)):02d}': [a * 10.0, a * 20.0]
            for a in alphas
        })
        raw = np.array([
            [1.0, 2.0, 3.0, 4.0],
            [4.0, 3.0, np.nan, 1.0],
        ])
        scenarios = pf.ecc_scenarios(qdf, alphas, raw)
        self.assertEqual(scenarios.shape, raw.shape)
        self.assertTrue(np.isfinite(scenarios).all())
        self.assertTrue(np.all(np.diff(scenarios[0]) > 0))
        self.assertGreater(scenarios[1, 0], scenarios[1, -1])

    def test_cqr_uses_conservative_order_statistic_and_crps_integral(self):
        X = pd.DataFrame({'x': np.arange(100, dtype=float)})
        models = {0.05: _ConstantModel(0.0), 0.95: _ConstantModel(0.0)}
        offsets = pf.cqr_calibrate(
            models, X, np.arange(100, dtype=float), pairs=((0.05, 0.95),)
        )
        # NumPy's conservative 'higher' selection at the finite-sample level.
        self.assertEqual(offsets[(0.05, 0.95)], 91.0)

        alphas = [0.05, 0.25, 0.50, 0.90]
        point = pd.DataFrame({
            f"q{int(round(a * 100)):02d}": [2.0, 2.0] for a in alphas
        })
        # A degenerate predictive distribution has CRPS == absolute error.
        self.assertAlmostEqual(
            pf.crps_from_quantiles([1.0, 3.0], point, alphas=alphas), 1.0
        )
        self.assertEqual(
            pf.crps_from_quantiles([2.0, 2.0], point, alphas=alphas), 0.0
        )

    def test_daily_rain_consensus_excludes_models_without_data(self):
        raw = pd.DataFrame({
            'ARPEGE_EUROPE_precipitation_model': [0.0, 0.0],
            'GFS_SEAMLESS_precipitation_model': [np.nan, np.nan],
            'ICON_SEAMLESS_precipitation_model': [np.nan, 0.2],
        })
        # One dry vote + one wet vote; the unavailable model is not a dry vote.
        self.assertEqual(fc._daily_model_rain_probability(raw), 50)
        self.assertIsNone(fc._daily_model_rain_probability(pd.DataFrame(index=[0, 1])))

    def test_dm_and_weather_direction_edge_cases(self):
        delta = np.array([-2, -1, 0, 1, 2, 2, 1, 0, -1, -2], dtype=float)
        dm, p = pf.dm_test_hln(10 + delta / 2, 10 - delta / 2, h=2)
        self.assertTrue(np.isfinite(dm) and np.isfinite(p))
        self.assertAlmostEqual(dm, 0.0)
        self.assertAlmostEqual(p, 1.0)

        alternating = np.array([0, 1] * 5, dtype=float)
        dm_degenerate, p_degenerate = pf.dm_test_hln(
            alternating, 1 - alternating, h=8
        )
        self.assertTrue(np.isnan(dm_degenerate) and np.isnan(p_degenerate))

        snow = pd.Series({
            'weather_code_raw': 71,
            'precipitation_xgb': 0.0,
            'cloud_cover_xgb': 10.0,
        })
        self.assertEqual(fc.correct_weather_code_row(snow), 71)

        directions = pd.DataFrame([[350.0, 10.0]])
        mean_dir = float(fc._circular_mean_degrees(directions).iloc[0])
        self.assertTrue(mean_dir < 1.0 or mean_dir > 359.0)
        cancelling = fc._circular_mean_degrees(pd.DataFrame([[0.0, 180.0]])).iloc[0]
        self.assertTrue(np.isnan(cancelling))


class MarineLogicTests(unittest.TestCase):
    def test_direction_capped_sea_state(self):
        # Jugo (SE) keeps the full Douglas scale.
        self.assertEqual(fc.adriatic_sea_state(2.0, 140.0), 4)
        # Bura (NE) chop never rates above "Talasasto" regardless of height.
        self.assertEqual(fc.adriatic_sea_state(3.0, 45.0), 4)
        # Waves "from" N or W come off the land: capped at "Blagi talasi",
        # so 0.8 m from those sectors can no longer show "Umjereni talasi".
        self.assertEqual(fc.adriatic_sea_state(0.8, 0.0), 2)
        self.assertEqual(fc.adriatic_sea_state(0.8, 270.0), 2)
        self.assertEqual(fc.adriatic_sea_state(0.8, 350.0), 2)
        # Unknown direction leaves the plain Douglas state; no height -> None.
        self.assertEqual(fc.adriatic_sea_state(0.8, None), 3)
        self.assertIsNone(fc.adriatic_sea_state(None, 140.0))

    def test_sailing_score_calm_end_unchanged(self):
        # The gentle rungs still key off Bft + height alone.
        self.assertEqual(fc.sailing_score(2, 0.2), ("green", "Idealno"))
        self.assertEqual(fc.sailing_score(3, 0.5), ("green", "Dobro"))
        self.assertEqual(fc.sailing_score(4, 0.8), ("yellow", "Prihvatljivo"))
        self.assertEqual(fc.sailing_score(None, 1.0), ("gray", "N/A"))

    def test_direction_capped_height_matches_the_douglas_pill(self):
        # West is capped at Douglas 2, whose upper bound is 0.5 m, so a 1.03 m
        # "wave" from 277° is worth 0.5 m to the score — exactly what the sea
        # state pill on the same card already claims.
        self.assertAlmostEqual(fc.direction_capped_height(1.03, 277.0), 0.5)
        self.assertAlmostEqual(fc.direction_capped_height(0.77, 275.0), 0.5)
        # Bura (NE) is capped at Douglas 4 -> 2.5 m.
        self.assertAlmostEqual(fc.direction_capped_height(3.0, 45.0), 2.5)
        # Southern fetch is uncapped: the height stands.
        self.assertAlmostEqual(fc.direction_capped_height(1.8, 178.0), 1.8)
        # Below the ceiling, nothing changes; unknown direction is left alone.
        self.assertAlmostEqual(fc.direction_capped_height(0.37, 290.0), 0.37)
        self.assertAlmostEqual(fc.direction_capped_height(1.2, None), 1.2)
        self.assertIsNone(fc.direction_capped_height(None, 180.0))

    def test_offshore_waves_no_longer_cost_a_rung(self):
        # Regression: 07-22 and 07-23 read "Prihvatljivo" purely because the
        # raw westerly height cleared 0.6 m in the calm ladder, while the very
        # same card called the sea "Blagi talasi".
        self.assertEqual(
            fc.sailing_score(3, 1.03, wave_direction=277.0,
                             wind_direction=47.0, wind_gusts_ms=9.5 * fc.MS_PER_KNOT),
            ("green", "Dobro"))
        self.assertEqual(
            fc.sailing_score(3, 0.77, wave_direction=275.0,
                             wind_direction=49.0, wind_gusts_ms=12.2 * fc.MS_PER_KNOT),
            ("green", "Dobro"))
        # Wind still governs on its own: a real blow is not "Dobro".
        self.assertEqual(
            fc.sailing_score(3, 0.37, wave_direction=290.0,
                             wind_direction=51.0, wind_gusts_ms=20.0 * fc.MS_PER_KNOT),
            ("yellow", "Prihvatljivo"))
        self.assertEqual(
            fc.sailing_score(5, 0.77, wave_direction=275.0,
                             wind_direction=49.0, wind_gusts_ms=8.0),
            ("yellow", "Prihvatljivo"))
        # Southern seas are untouched by the cap and still warn.
        self.assertEqual(
            fc.sailing_score(2, 1.74, wave_direction=178.0,
                             wind_direction=114.0, wind_gusts_ms=10.1 * fc.MS_PER_KNOT),
            ("orange", "Oprez"))

    def test_sailing_score_ignores_waves_off_the_land(self):
        # Regression: 2026-07-23 Bečići — 3 Bft Levanat, 4.3/7.9 m/s, and a
        # ~1 m "wave" out of 273°. The same page calls that wave "Blagi
        # talasi" (a 5.5 km-grid artifact off the land), so the badge must
        # not call it Oprez. Gusts are 15.4 kn -> the >15 kn rung, Prihvatljivo.
        self.assertEqual(
            fc.sailing_score(3, 1.05, wave_direction=273.0,
                             wind_direction=50.0, wind_gusts_ms=7.9),
            ("yellow", "Prihvatljivo"))
        # An absurd westerly height is not merely capped, it is disregarded:
        # clipped to the 0.5 m the direction can deliver, leaving the wind to
        # decide the day on its own. 3 m out of the W with a light breeze is a
        # grid artifact, and the card says "Blagi talasi" beside it.
        self.assertEqual(
            fc.sailing_score(3, 3.0, wave_direction=270.0,
                             wind_direction=300.0, wind_gusts_ms=5.0),
            ("green", "Dobro"))
        # ...but the same absurd wave with real wind is still no better than
        # Prihvatljivo, because Bft alone carries it there.
        self.assertEqual(
            fc.sailing_score(4, 3.0, wave_direction=270.0,
                             wind_direction=300.0, wind_gusts_ms=5.0),
            ("yellow", "Prihvatljivo"))

    def test_sailing_score_southern_waves_fire_on_height_alone(self):
        # Jugo/Sirocco/Lebić sector (112.5-247.5°) has the open-sea fetch, so
        # height alone counts even once the wind that built it has dropped.
        self.assertEqual(
            fc.sailing_score(2, 1.2, wave_direction=184.0,
                             wind_direction=50.0, wind_gusts_ms=5.0),
            ("orange", "Oprez"))
        self.assertEqual(
            fc.sailing_score(2, 2.0, wave_direction=188.0,
                             wind_direction=42.0, wind_gusts_ms=5.0),
            ("red", "Opasno"))
        # Just under the bar stays acceptable.
        self.assertEqual(
            fc.sailing_score(2, 0.95, wave_direction=184.0,
                             wind_direction=50.0, wind_gusts_ms=5.0),
            ("yellow", "Prihvatljivo"))

    def test_sailing_score_southern_wind_lowers_the_wave_bar(self):
        # A jugo still blowing at 3 Bft+ is actively building the sea, so the
        # thresholds drop from 1.0/1.8 m to 0.8/1.5 m.
        self.assertEqual(
            fc.sailing_score(3, 0.9, wave_direction=140.0,
                             wind_direction=140.0, wind_gusts_ms=6.0),
            ("orange", "Oprez"))
        self.assertEqual(
            fc.sailing_score(4, 1.6, wave_direction=140.0,
                             wind_direction=150.0, wind_gusts_ms=8.0),
            ("red", "Opasno"))
        # Same wave, but the wind is easterly: the higher bar applies.
        self.assertEqual(
            fc.sailing_score(3, 0.9, wave_direction=140.0,
                             wind_direction=70.0, wind_gusts_ms=6.0),
            ("yellow", "Prihvatljivo"))

    def _wind_day(self, gusts_by_hour, speed=4.0, direction=50.0):
        """One 24 h wind frame; gusts_by_hour maps hour -> gust m/s."""
        dts = pd.date_range('2026-07-24 00:00', periods=24, freq='h')
        return pd.DataFrame({
            'datetime': dts,
            'wind_speed_10m_mean': [speed] * 24,
            'wind_gusts_10m_mean': [gusts_by_hour.get(h, 3.0) for h in range(24)],
            'wind_direction_10m_mean': [direction] * 24,
        })

    def test_persistent_daytime_gust_ignores_isolated_hours(self):
        kn = fc.MS_PER_KNOT
        # Regression: 2026-07-24 Bečići peaked at 15.6 kn for exactly one hour,
        # at 07h, and that single hour dragged the whole day down a rung.
        one_spike = self._wind_day({7: 16.0 * kn, 14: 16.0 * kn})
        self.assertLess(fc.persistent_daytime_gust(one_spike) / kn, 15.0)
        # Night-long blow doesn't count: nobody is swimming at 02h.
        night = self._wind_day({h: 20.0 * kn for h in (0, 1, 2, 3, 4, 23)})
        self.assertLess(fc.persistent_daytime_gust(night) / kn, 15.0)
        # Four daytime hours is the bar, and the 4th-strongest sets the level.
        afternoon = self._wind_day({12: 24.0 * kn, 13: 22.0 * kn,
                                    14: 20.0 * kn, 15: 18.0 * kn})
        self.assertAlmostEqual(
            fc.persistent_daytime_gust(afternoon) / kn, 18.0, places=4)
        # A day with too few daytime rows can't establish persistence at all.
        short = self._wind_day({}).head(10)
        self.assertIsNone(fc.persistent_daytime_gust(short))

    def test_daily_badge_survives_a_single_gusty_hour(self):
        waves = {'2026-07-24': (0.37, 290.0)}  # westerly, cannot warn on its own
        kn = fc.MS_PER_KNOT

        spike = fc._build_wind_block(self._wind_day({7: 21.4 * kn}), {}, waves)
        day = spike['daily_summary'][0]
        self.assertEqual(day['sailing_score'], 'Dobro')
        self.assertLess(day['wind_gusts_persistent'] / kn, 15.0)
        # The card still reports the true peak; only the score ignores it.
        self.assertAlmostEqual(day['wind_gusts_max'], round(21.4 * kn, 1))

        blow = fc._build_wind_block(
            self._wind_day({h: 17.0 * kn for h in (11, 12, 13, 14, 15)}), {}, waves)
        self.assertEqual(blow['daily_summary'][0]['sailing_score'], 'Prihvatljivo')

    def test_sailing_score_gust_rungs_stand_alone(self):
        # Gusts are judged on their own, in knots, whatever the sea is doing.
        self.assertEqual(  # 23.3 kn
            fc.sailing_score(3, 0.2, wave_direction=270.0,
                             wind_direction=300.0, wind_gusts_ms=12.0),
            ("orange", "Oprez"))
        self.assertEqual(  # 16.5 kn floors an otherwise ideal day
            fc.sailing_score(2, 0.2, wave_direction=270.0,
                             wind_direction=300.0, wind_gusts_ms=8.5),
            ("yellow", "Prihvatljivo"))
        self.assertEqual(  # 7.8 kn leaves it alone
            fc.sailing_score(2, 0.2, wave_direction=270.0,
                             wind_direction=300.0, wind_gusts_ms=4.0),
            ("green", "Idealno"))
        self.assertEqual(  # gale-force backstop
            fc.sailing_score(3, 0.2, wave_direction=270.0,
                             wind_direction=300.0, wind_gusts_ms=18.0),
            ("red", "Opasno"))

    def test_hail_votes_from_model_codes(self):
        # WMO 96/99 votes per hour, shifted -1 to the start-of-hour convention
        # (Open-Meteo labels the preceding hour, same as precipitation).
        m1, m2 = fc.MODELS[0], fc.MODELS[1]
        dts = pd.date_range('2026-07-25 12:00', periods=4, freq='h')
        raw = pd.DataFrame({
            'datetime': dts,
            f'{m1}_weather_code_model': [95, 96, 96, 95],
            f'{m2}_weather_code_model': [95, 95, 99, 0],
        })
        votes = fc._hail_votes_by_datetime(raw)
        self.assertEqual(votes, {dts[0]: 1, dts[1]: 2})
        self.assertEqual(fc._hail_votes_by_datetime(None), {})
        self.assertEqual(fc._hail_votes_by_datetime(
            pd.DataFrame({'datetime': dts})), {})

    def test_daily_wind_consensus_keeps_model_peaks(self):
        # Ensemble-mean wind flattens to ~1.5 m/s, but each model's own daily
        # max is 4 and 6: the consensus field must carry their median so
        # long-range days stop all reading "slab".
        dts = pd.date_range('2026-07-25', periods=24, freq='h')
        grp = pd.DataFrame({
            'datetime': dts,
            'hour': list(range(24)),
            'temperature_2m_ensemble': [25.0] * 24,
            'wind_speed_10m_ensemble': [1.5] * 24,
            'wind_gusts_10m_ensemble': [3.0] * 24,
            'precipitation_ensemble': [0.0] * 24,
            'relative_humidity_2m_ensemble': [60.0] * 24,
            'pressure_msl_ensemble': [1013.0] * 24,
            'cloud_cover_ensemble': [20.0] * 24,
            'weather_code': [1.0] * 24,
        })
        m1, m2 = fc.MODELS[0], fc.MODELS[1]
        fc_raw = pd.DataFrame({
            'datetime': dts,
            f'{m1}_wind_speed_10m_model': [4.0] * 24,
            f'{m2}_wind_speed_10m_model': [6.0] * 24,
            f'{m1}_precipitation_model': [0.0] * 24,
        })
        ds = fc._build_daily_summary('2026-07-25', 'Saturday', grp, fc_raw=fc_raw)
        self.assertEqual(ds['wind_max'], 1.5)
        self.assertEqual(ds['wind_max_models'], 5.0)

    def test_daily_narrative_mentions_short_rain(self):
        # A clear summer day with a single 0.15 mm shower hour (total 0.2 mm,
        # below the old 0.2 mm gate) must still carry the rain note.
        precip = [0.0] * 24
        precip[14] = 0.15
        precip[15] = 0.05
        grp = pd.DataFrame({
            'hour': list(range(24)),
            'cloud_cover': [20.0] * 24,
            'precipitation': precip,
            'wind_speed_10m': [3.0] * 24,
            'wind_gusts_10m': [5.0] * 24,
            'temperature_2m': [26.0] * 24,
            'weather_code': [1.0] * 24,
            'wind_direction_10m': [180.0] * 24,
        })
        narrative = fc._daily_narrative(grp)['narrative'].lower()
        self.assertTrue(
            any(tok in narrative for tok in ('kiš', 'pljus', 'padavin')),
            narrative)


class BurstWindBoostTests(unittest.TestCase):
    def _burst_frames(self):
        """One 2h cell (5.8 then 19.4 mm/h) over an otherwise dead-calm bay."""
        n = 8
        idx = pd.date_range('2026-09-18 11:00', periods=n, freq='h')
        italia = [0.0] * n
        italia[4] = 5.8      # shift(-1) lands these on rows 3 and 4
        italia[5] = 19.4
        fc_df = pd.DataFrame({
            'datetime': idx,
            f'{fc.TRUSTED_RAIN_MODEL}_precipitation_model': italia,
        })
        gust = [3.3, 3.4, 4.8, 2.7, 2.1, 3.3, 1.7, 1.6]
        wind = [1.4, 1.5, 2.6, 1.4, 1.3, 2.2, 0.7, 0.6]
        corrected = pd.DataFrame({
            'datetime': idx,
            'wind_gusts_10m_xgb': gust,
            'wind_gusts_10m_ensemble': gust,
            'wind_gusts_10m_q90': [3.7, 3.7, 3.5, 3.5, 3.2, 2.7, 2.4, 2.2],
            'wind_speed_10m_xgb': wind,
            'wind_speed_10m_ensemble': wind,
            'wind_speed_10m_q90': [1.6, 1.9, 1.9, 1.9, 1.9, 1.6, 1.1, 0.9],
        })
        return corrected, fc_df

    def test_burst_wind_boost_is_bounded_by_the_ambient_gust_field(self):
        # A heavy cell over a calm bay used to saturate the rain-keyed floors
        # (gust 18.0, wind 8.0 m/s) for an hour whose own band topped out at
        # q90 = 3.2 m/s. Rain rate alone cannot conjure a downdraft.
        corrected, fc_df = self._burst_frames()
        fc._apply_burst_wind_boost(corrected, fc_df)

        gust = corrected['wind_gusts_10m_xgb'].values
        wind = corrected['wind_speed_10m_xgb'].values

        for i in (3, 4):
            self.assertLessEqual(
                gust[i],
                max(fc.BURST_GUST_FLOOR_BASE,
                    corrected['wind_gusts_10m_q90'].values[i] * fc.BURST_AMBIENT_GUST_FACTOR),
                f'gust boost at row {i} ignored the ambient band: {gust[i]}')
            self.assertLessEqual(
                wind[i],
                max(fc.BURST_WIND_FLOOR_BASE,
                    corrected['wind_speed_10m_q90'].values[i] * fc.BURST_AMBIENT_WIND_FACTOR),
                f'wind boost at row {i} ignored the ambient band: {wind[i]}')

        # The UI raises a wind warning at gusts >= 17 m/s; a calm-bay shower
        # must never reach it on rain rate alone.
        self.assertLess(gust.max(), 17.0)

    def test_burst_wind_boost_still_lifts_a_windy_cell(self):
        # The boost exists because models miss convective gusts. A cell in a
        # genuinely breezy field must still get lifted above its ambient band.
        corrected, fc_df = self._burst_frames()
        corrected['wind_gusts_10m_q90'] = [9.0] * 8
        corrected['wind_speed_10m_q90'] = [5.0] * 8
        fc._apply_burst_wind_boost(corrected, fc_df)
        self.assertGreater(corrected['wind_gusts_10m_xgb'].values[4], 9.0)

    def test_burst_wind_boost_survives_missing_quantile_columns(self):
        # Model bundles predating the quantile upgrade have no q90 band; the
        # boost must fall back to the ensemble point value, not to NaN.
        corrected, fc_df = self._burst_frames()
        corrected = corrected.drop(columns=['wind_gusts_10m_q90', 'wind_speed_10m_q90'])
        fc._apply_burst_wind_boost(corrected, fc_df)
        self.assertTrue(np.isfinite(corrected['wind_gusts_10m_xgb'].values).all())
        self.assertLess(corrected['wind_gusts_10m_xgb'].max(), 17.0)


if __name__ == '__main__':
    unittest.main(verbosity=2)
