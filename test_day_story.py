import unittest

import day_story

H = list(range(24))


def _build(cc, pr, ws, wd, wc=None, wg=None, tp=None):
    wc = wc or [1] * 24
    wg = wg or [w * 1.5 for w in ws]
    tp = tp or [10 + h / 4 for h in H]
    return day_story.build(H, cc, pr, ws, wg, tp, wc, wd)


class DayStoryTests(unittest.TestCase):
    def test_rain_onset_and_end_are_timed(self):
        pr = [0.0] * 13 + [2.0] * 6 + [0.0] * 5
        text = _build([30] * 13 + [95] * 11, pr, [3] * 24, [180] * 24)
        self.assertIn('Od oko 13 do 19 h', text)
        self.assertIn('kiša', text)
        self.assertIn('Potom', text)

    def test_short_shower_is_mentioned(self):
        pr = [0.0] * 24
        pr[14] = 0.15
        text = _build([20] * 24, pr, [3] * 24, [180] * 24)
        self.assertIn('Između 14 i 15 h', text)

    def test_wind_shift_jugo_to_bura(self):
        ws = [10] * 15 + [12] * 9
        text = _build([60] * 24, [0.0] * 24, ws, [140] * 15 + [40] * 9)
        self.assertIn('jugo', text)
        self.assertIn('okreće na jaku buru', text)

    def test_cloud_transition(self):
        text = _build([10] * 12 + [95] * 12, [0.0] * 24, [2] * 24, [0] * 24)
        self.assertIn('oko 12 h se naoblačuje', text)

    def test_too_few_hours(self):
        self.assertIsNone(day_story.build([22, 23], [50, 50], [0, 0], [1, 1],
                                          [2, 2], [10, 10], [1, 1], [0, 0]))

    def test_nan_inputs_do_not_crash(self):
        nan = float('nan')
        text = day_story.build(H, [nan] * 24, [nan] * 24, [nan] * 24,
                               [nan] * 24, [12.0] * 24, [nan] * 24, [nan] * 24)
        self.assertIn('Najtoplije', text)


if __name__ == '__main__':
    unittest.main()
