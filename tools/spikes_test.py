#!/usr/bin/env python3
"""Regression tests for spikes.py on judged results saved in spikes_test.json (no network).

  python3 tools/spikes_test.py
"""
import json
import unittest
from pathlib import Path

import spikes

RUNS = {r['id']: spikes.Run(**r) for r in json.loads(Path(__file__).with_name('spikes_test.json').read_text())}


def check(submission: int) -> tuple[dict[str, float], str]:
    run = RUNS[submission]
    others = [r for r in RUNS.values() if r.problem == run.problem and r.id != run.id]
    return spikes.spikes(run, others), spikes.report(run, others)


class Spikes(unittest.TestCase):
    def test_slow_cases_that_repeat_are_real(self):
        # min_plus_convolution_concave_arbitrary: monotone_01/02 take 26-31 ms in every run,
        # monotone_00/03 take 11-14 ms. The class median flagged 01/02 (clean 19 and 20 ms).
        for submission in (409349, 409355):
            found, text = check(submission)
            self.assertNotIn('monotone_01', found)
            self.assertNotIn('monotone_02', found)
            self.assertTrue(text.startswith(f'{submission}: judged 28 ms, clean 28 ms;'), text)

    def test_spike_far_above_its_class(self):
        # sqrt_of_formal_power_series: monomial_02 takes 20 ms here, 11 ms in 409316 (another source
        # that matches on the other monomial cases); 16 ms above its class median.
        found, text = check(409361)
        self.assertEqual(found.get('monomial_02'), 11)
        self.assertIn('  spike: monomial_02 20 (peers 11)', text.splitlines())
        self.assertTrue(text.startswith('409361: judged 20 ms, clean 12 ms;'), text)

    def test_heterogeneous_class_is_not_a_spike(self):
        # 409316: monomial_02 (11 ms) and lower_deg_zero_00 (10 ms) repeat in 409361.
        found, _ = check(409316)
        self.assertNotIn('monomial_02', found)
        self.assertNotIn('lower_deg_zero_00', found)

    def test_no_other_run_flags_nothing(self):
        self.assertEqual(spikes.spikes(RUNS[409349], []), {})


if __name__ == '__main__':
    unittest.main()
