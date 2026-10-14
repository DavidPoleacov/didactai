import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from src.exercise_bank import ExerciseBankError, load_practice_bank


ROOT = Path(__file__).resolve().parents[1]


class ExerciseBankTests(unittest.TestCase):
    def test_all_review_flags_are_applied_by_problem_hash_and_keep_original_index(self):
        source = pd.read_csv(ROOT / "data/processed/exercises_augmented.csv")
        flags_path = ROOT / "data/processed/exercise_review_flags.csv"
        flags = pd.read_csv(flags_path, dtype=str, keep_default_na=False)
        practice = load_practice_bank(source, flags_path)

        self.assertEqual(len(flags), 59)
        self.assertEqual(flags["problem_hash"].nunique(), 59)
        self.assertEqual(len(practice), 1285)
        self.assertFalse(practice["problem_hash"].isin(flags["problem_hash"]).any())
        self.assertEqual(practice.index.name, source.index.name)
        self.assertEqual(practice.loc[460, "Problema"], source.loc[460, "Problema"])

    def test_empty_statements_and_answer_keys_are_removed(self):
        source = pd.DataFrame({
            "Problema": ["valid", " ", None, "no key", "nan"],
            "Raspunsul": ["1", "2", "3", "  ", "None"],
            "problem_hash": ["a", "b", "c", "d", "e"],
        }, index=[4, 8, 12, 16, 20])
        with TemporaryDirectory() as directory:
            flags = Path(directory) / "flags.csv"
            flags.write_text("row_id,problem_hash,reason\n", encoding="utf-8")
            result = load_practice_bank(source, flags)

        self.assertEqual(result.index.tolist(), [4])

    def test_stale_review_hash_fails_instead_of_skipping_filter(self):
        source = pd.DataFrame({"Problema": ["p"], "Raspunsul": ["1"], "problem_hash": ["unreviewed"]})
        with self.assertRaisesRegex(ExerciseBankError, "do not match"):
            load_practice_bank(source, ROOT / "data/processed/exercise_review_flags.csv")

    def test_invalid_or_unreadable_flags_fail_clearly(self):
        source = pd.DataFrame({"Problema": ["p"], "Raspunsul": ["1"], "problem_hash": ["h"]})
        with self.assertRaisesRegex(ExerciseBankError, "Unable to load"):
            load_practice_bank(source, ROOT / "data/processed/not-a-review-file.csv")


if __name__ == "__main__":
    unittest.main()
