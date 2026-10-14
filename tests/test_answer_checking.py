import math
import unittest
from pathlib import Path

import pandas as pd

from src.exercise_bank import load_practice_bank
from src.pedagogical_engine import answer_result_is_verified, evaluate_answer


class AnswerCheckerTests(unittest.TestCase):
    def test_required_regressions(self):
        cases = [
            ("1", "12", "incorrect"),
            ("120", "12", "incorrect"),
            ("12", "-12", "incorrect"),
            ("12 or 99", "12", "needs_review"),
            ("1 2", "12", "needs_review"),
            ("0.5", "1/2", "correct"),
            ("0,5", "1/2", "correct"),
            ("2/4", "1/2", "correct"),
            ("15", "1,5", "incorrect"),
            ("x=-2,y=3", "x=3,y=-2", "needs_review"),
            ("[0;1]", "(0;1)", "needs_review"),
            ("12", "12 cm", "needs_review"),
            ("12 m", "12 cm", "needs_review"),
            ("", "12", "no_answer"),
            ("12", None, "invalid_key"),
            ("12", float("nan"), "invalid_key"),
            ("12", "", "invalid_key"),
            ("1/0", "1/2", "needs_review"),
            ("1/2", "1/0", "invalid_key"),
            ("2x", "3x", "needs_review"),
        ]
        for student, expected, status in cases:
            with self.subTest(student=student, expected=expected):
                result = evaluate_answer(student, expected)
                self.assertEqual(result["status"], status)
                self.assertEqual(result["verified"], status in {"correct", "incorrect"})
                self.assertEqual(answer_result_is_verified(result), result["verified"])

    def test_signed_values_and_complete_variable_labels(self):
        self.assertEqual(evaluate_answer("-4/6", "-2/3")["status"], "correct")
        self.assertEqual(evaluate_answer("x=-2", "x=-2")["status"], "correct")
        self.assertEqual(evaluate_answer("y=-2", "x=-2")["status"], "needs_review")

    def test_stored_answer_round_trips_check_format_compatibility_only(self):
        root = Path(__file__).resolve().parents[1]
        for filename in ("exercises_processed.csv", "exercises_augmented.csv"):
            frame = pd.read_csv(root / "data" / "processed" / filename)
            if filename == "exercises_augmented.csv":
                frame = load_practice_bank(
                    frame, root / "data" / "processed" / "exercise_review_flags.csv",
                )
            answers = frame["Raspunsul"]
            for row_number, answer in answers.items():
                with self.subTest(dataset=filename, row=row_number):
                    if isinstance(answer, float) and math.isnan(answer):
                        result = evaluate_answer(answer, answer)
                        self.assertEqual(result["status"], "invalid_key")
                    else:
                        result = evaluate_answer(answer, answer)
                    self.assertIn(
                        result["status"],
                        {"correct", "incorrect", "no_answer", "invalid_key", "needs_review"},
                    )
                    self.assertTrue(result["feedback"])


if __name__ == "__main__":
    unittest.main()