import unittest
from pathlib import Path

import pandas as pd

from src.exercise_bank import load_practice_bank
from src.pedagogical_engine import (
    assess_diagnostic_results,
    automatically_gradable_bank,
    choose_hint,
    evaluate_answer,
    generate_diagnostic_bank,
    recommend_next_exercise,
    tutor_answer_can_be_recorded,
)


class AnswerRegressions(unittest.TestCase):
    def test_invalid_identical_numeric_keys_do_not_pass(self):
        for malformed in ("1/0", "2..3", "NaN", "None", "<NA>", "null", "x=1/0", "1/0 cm", "--2"):
            with self.subTest(key=malformed):
                result = evaluate_answer(malformed, malformed)
                self.assertNotEqual(result["status"], "correct")
                self.assertFalse(result["verified"])
        self.assertEqual(evaluate_answer("", "1/0")["status"], "no_answer")

    def test_conservative_rational_and_label_comparison(self):
        for left, right in (
            ("0.5", "0,5"), ("0.5", "1/2"), ("0,5", "2/4"), ("2/4", "1/2"),
            ("x=0.5 cm", "x=1/2 cm"),
        ):
            self.assertEqual(evaluate_answer(left, right)["status"], "correct")
        for left, right in (("1", "12"), ("120", "12"), ("12", "-12")):
            self.assertEqual(evaluate_answer(left, right)["status"], "incorrect")
        self.assertEqual(evaluate_answer("y=1 cm", "x=1 cm")["status"], "needs_review")
        self.assertEqual(evaluate_answer("1 m", "1 cm")["status"], "needs_review")
        self.assertEqual(evaluate_answer("12 furlongs", "12 furlongs")["status"], "needs_review")
        self.assertEqual(evaluate_answer("x=12 furlongs", "x=12 furlongs")["status"], "needs_review")

    def test_huge_numbers_are_controlled_and_unverified(self):
        result = evaluate_answer("9" * 5000, "9" * 5000)
        self.assertEqual(result["status"], "invalid_key")
        self.assertFalse(result["verified"])

    def test_backend_guard_rejects_duplicate_completed_answer_and_unverified_answer(self):
        correct = evaluate_answer("12", "12")
        unsupported = evaluate_answer("12 or 1", "12")
        self.assertTrue(tutor_answer_can_be_recorded(correct, False))
        self.assertFalse(tutor_answer_can_be_recorded(correct, True))
        self.assertFalse(tutor_answer_can_be_recorded(unsupported, False))

    def test_first_hint_is_scaffold_then_every_stored_step_is_ordered(self):
        problem = "Fie f: R → R, f(x) = 4x + 12. Calculati f(0)."
        steps = "f(0) = 4·(0) + 12 = 0 + 12 = 12; Se efectuează înmulțirea; Rezultatul final este 12; Verifică înlocuirea."
        first = choose_hint(problem, steps, 0.55, 1)
        self.assertNotIn("12", first["hint"])
        self.assertEqual(first["shown_solution_steps"], 0)
        for request, expected in enumerate(
            ["f(0) = 4·(0) + 12 = 0 + 12 = 12", "Se efectuează înmulțirea", "Rezultatul final este 12", "Verifică înlocuirea"],
            start=2,
        ):
            hint = choose_hint(problem, steps, 0.55, request)
            self.assertIn(expected, hint["hint"])
            self.assertEqual(hint["shown_solution_steps"], request - 1)
        self.assertIn("Nu există alți pași", choose_hint(problem, steps, 0.55, 8)["hint"])
        self.assertIn("Nu există alți pași", choose_hint(problem, "", 0.55, 2)["hint"])


class DiagnosticAndRecommendationRegressions(unittest.TestCase):
    def test_weakest_domain_first_and_untested_domains_are_explicit(self):
        data = pd.DataFrame({
            "Domeniu": ["A", "B", "C"], "Tema_norm": ["a", "b", "c"],
            "Raspunsul": ["1", "2", "3"],
            "Dificultate": [1, 1, 1], "problem_words": [3, 3, 3],
        })
        profile = assess_diagnostic_results([
            {"domain": "A", "correct": True}, {"domain": "A", "correct": False},
            {"domain": "B", "correct": False}, {"domain": "B", "correct": False},
        ], data)
        self.assertEqual(profile["weak_domains"][:2], ["B", "A"])
        self.assertEqual(profile["domain_scores"]["C"]["status"], "untested")
        self.assertIsNone(profile["domain_scores"]["C"]["accuracy"])
        self.assertEqual(profile["domain_scores"]["B"]["priority"], 1.0)

    def test_diagnostic_round_robin_coverage_empty_and_deterministic(self):
        rows = []
        for domain in "ABCDEF":
            for number in range(3):
                rows.append({
                    "Domeniu": domain, "Problema": f"{domain} {number}", "Raspunsul": str(number),
                    "Tema_norm": domain, "Dificultate": number,
                })
        bank = generate_diagnostic_bank(pd.DataFrame(rows), 5, random_state=77)
        again = generate_diagnostic_bank(pd.DataFrame(rows), 5, random_state=77)
        self.assertEqual([item["Problema"] for item in bank], [item["Problema"] for item in again])
        self.assertEqual(len({item["Domeniu"] for item in bank}), 5)
        self.assertEqual(generate_diagnostic_bank(pd.DataFrame(), 5), [])
        self.assertEqual(generate_diagnostic_bank(pd.DataFrame(rows), 0), [])
        dirty = pd.DataFrame(rows + [{
            "Domeniu": "G", "Problema": None, "Raspunsul": "1", "Tema_norm": "G",
            "Dificultate": 1,
        }])
        self.assertNotIn(None, [item["Problema"] for item in generate_diagnostic_bank(dirty, 20)])

    def test_recommendation_stays_in_domain_with_small_fallback_and_exclusion(self):
        data = pd.DataFrame([
            {"Domeniu": "A", "Problema": "a1", "Raspunsul": "1", "Dificultate_group": "1"},
            {"Domeniu": "A", "Problema": "a2", "Raspunsul": "2", "Dificultate_group": "2"},
            {"Domeniu": "B", "Problema": "b1", "Raspunsul": "3", "Dificultate_group": "2"},
        ], index=[10, 20, 30])
        result = recommend_next_exercise(data, "A", "4", exclude_problem="a1")
        self.assertEqual(result.iloc[0]["Domeniu"], "A")
        self.assertEqual(result.iloc[0]["Problema"], "a2")
        self.assertTrue(recommend_next_exercise(data, "A", "4", exclude_problem="a1").index.tolist())
        self.assertTrue(recommend_next_exercise(data, "A", "2", exclude_problem="a1").index.tolist())
        self.assertTrue(recommend_next_exercise(data, "A", "2", exclude_problem="a2").index.tolist())
        self.assertTrue(recommend_next_exercise(data, "B", "2", exclude_problem="b1").empty)

    def test_unsupported_keys_are_excluded_from_diagnostics_and_recommendations(self):
        data = pd.DataFrame([
            {
                "Domeniu": "A", "Problema": "supported", "Raspunsul": "12",
                "Dificultate_group": "1", "Tema_norm": "scalar",
                "Dificultate": 1, "problem_words": 2,
            },
            {
                "Domeniu": "A", "Problema": "unsupported", "Raspunsul": "12 furlongs",
                "Dificultate_group": "1", "Tema_norm": "radical",
                "Dificultate": 1, "problem_words": 2,
            },
        ], index=[460, 24])

        self.assertEqual(automatically_gradable_bank(data).index.tolist(), [460])
        diagnostic = generate_diagnostic_bank(data, 10)
        self.assertTrue(diagnostic)
        self.assertEqual({item["Problema"] for item in diagnostic}, {"supported"})
        recommendation = recommend_next_exercise(data, "A", "1")
        self.assertEqual(recommendation.index.tolist(), [460])

    def test_audited_dataset_supported_subset_counts_and_stable_indices(self):
        root = Path(__file__).resolve().parents[1]
        source = pd.read_csv(root / "data/processed/exercises_augmented.csv")
        content = load_practice_bank(source, root / "data/processed/exercise_review_flags.csv")
        automatic = automatically_gradable_bank(content)
        browsing_only = content.loc[~content.index.isin(automatic.index)]

        self.assertEqual((len(source), len(content), len(automatic), len(browsing_only)), (1344, 1285, 1270, 15))
        self.assertIn(460, automatic.index)
        self.assertTrue(
            all(evaluate_answer(answer, answer)["status"] != "correct" for answer in browsing_only["Raspunsul"])
        )

    def test_recommendation_excludes_flagged_hashes(self):
        root = Path(__file__).resolve().parents[1]
        raw = pd.read_csv(root / "data/processed/exercises_augmented.csv")
        flags = pd.read_csv(root / "data/processed/exercise_review_flags.csv", dtype=str, keep_default_na=False)
        flagged = raw[raw["problem_hash"].isin(flags["problem_hash"])].iloc[[0]]
        eligible = raw[
            (raw["Domeniu"] == flagged.iloc[0]["Domeniu"])
            & ~raw["problem_hash"].isin(flags["problem_hash"])
        ].iloc[[0]]
        eligible = automatically_gradable_bank(eligible)
        if eligible.empty:
            eligible = automatically_gradable_bank(
                raw[
                    (raw["Domeniu"] == flagged.iloc[0]["Domeniu"])
                    & ~raw["problem_hash"].isin(flags["problem_hash"])
                ]
            ).iloc[[0]]
        candidate_pool = pd.concat([flagged, eligible])

        result = recommend_next_exercise(
            candidate_pool,
            str(flagged.iloc[0]["Domeniu"]),
            str(eligible.iloc[0]["Dificultate_group"]),
        )

        self.assertEqual(result.index.tolist(), eligible.index.tolist())
        self.assertNotIn(result.iloc[0]["problem_hash"], set(flags["problem_hash"]))


if __name__ == "__main__":
    unittest.main()
