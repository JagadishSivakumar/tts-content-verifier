import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tts_verifier.align import tag_errors  # noqa: E402
from tts_verifier.metrics import cohens_kappa, detection_metrics, grouped_bootstrap_ci  # noqa: E402
from tts_verifier.normalize import normalize  # noqa: E402
from tts_verifier.prompts import assign_splits, build_prompts  # noqa: E402


def same(a: str, b: str) -> bool:
    return normalize(a) == normalize(b)


class TestNormalize(unittest.TestCase):
    def test_currency(self):
        self.assertTrue(same("£25", "twenty-five pounds"))
        self.assertTrue(same("$3.50", "three dollars fifty cents"))
        self.assertTrue(same("£1,250", "one thousand two hundred and fifty pounds"))

    def test_number_and_optional(self):
        self.assertTrue(same("250", "two hundred fifty"))
        self.assertTrue(same("250", "two hundred and fifty"))

    def test_years_ordinals_times(self):
        self.assertTrue(same("in 1987", "in nineteen eighty seven"))
        self.assertTrue(same("in 2003", "in two thousand and three"))
        self.assertTrue(same("the 21st", "the twenty first"))
        self.assertTrue(same("at 9:05", "at nine oh five"))

    def test_percent_decimal(self):
        self.assertTrue(same("3.5%", "three point five percent"))

    def test_codes(self):
        self.assertTrue(same("QX7 49K", "Q X 7 49 K"))
        self.assertTrue(same("seat 14C", "seat 14 C"))

    def test_abbreviations_and_punctuation(self):
        self.assertTrue(same("Dr Okonkwo", "doctor okonkwo"))
        self.assertTrue(same("Mill St", "mill street"))
        self.assertTrue(same("No, no, I said", "no no i said"))   # 'no' not a number here

    def test_contractions_kept(self):
        self.assertEqual(normalize("o'clock, isn't it?"), ["o'clock", "isn't", "it"])


class TestAlign(unittest.TestCase):
    def tags(self, ref, hyp):
        res = tag_errors(normalize(ref), normalize(hyp))
        return sorted(e["type"] for e in res.errors), res

    def test_clean(self):
        t, res = self.tags("the cat sat on the mat", "The cat sat on the mat.")
        self.assertEqual(t, [])
        self.assertEqual(res.wer, 0)

    def test_omission(self):
        t, _ = self.tags("she opened the old wooden box", "she opened the box")
        self.assertEqual(t, ["omission"])

    def test_repetition(self):
        t, _ = self.tags("bring water and rope", "bring water water and rope")
        self.assertEqual(t, ["repetition"])

    def test_substitution(self):
        t, _ = self.tags("the red lorry", "the bed lorry")
        self.assertEqual(t, ["substitution"])

    def test_truncation(self):
        t, _ = self.tags("we walked back along the river path tonight", "we walked back along")
        self.assertEqual(t, ["truncation"])

    def test_max_deletion_run(self):
        _, res = self.tags("a b c d e f", "a f")
        self.assertEqual(res.max_deletion_run, 4)


class TestMetrics(unittest.TestCase):
    def test_detection(self):
        m = detection_metrics([1, 1, 0, 0], [1, 0, 1, 0])
        self.assertAlmostEqual(m["precision"], 0.5)
        self.assertAlmostEqual(m["recall"], 0.5)
        self.assertAlmostEqual(m["false_alarm_rate"], 0.5)

    def test_kappa(self):
        self.assertAlmostEqual(cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]), 1.0)
        self.assertAlmostEqual(cohens_kappa([1, 1, 0, 0], [1, 0, 1, 0]), 0.0)

    def test_grouped_bootstrap(self):
        rows = [{"g": i // 2, "y": i % 2} for i in range(20)]
        point, lo, hi = grouped_bootstrap_ci(rows, "g", lambda r: sum(x["y"] for x in r) / len(r), n_boot=200)
        self.assertAlmostEqual(point, 0.5)
        self.assertLessEqual(lo, point)
        self.assertGreaterEqual(hi, point)


class TestPrompts(unittest.TestCase):
    def test_counts_and_splits(self):
        prompts = build_prompts()
        self.assertEqual(len(prompts), 100)
        split = assign_splits(prompts)
        cats = {(p["category"], split[p["group"]]) for p in prompts}
        for c in {"ordinary", "numbers_names", "repetition_prone", "long"}:
            self.assertIn((c, "dev"), cats)
            self.assertIn((c, "test"), cats)

    def test_prompts_self_consistent(self):
        # Every prompt must normalise without error and be non-empty.
        for p in build_prompts():
            self.assertTrue(normalize(p["text"]), p["id"])


if __name__ == "__main__":
    unittest.main()
