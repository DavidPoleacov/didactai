import copy
import re
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.exercise_bank import load_practice_bank
from src.pedagogical_engine import (
    assess_diagnostic_results,
    automatically_gradable_bank,
    evaluate_answer,
    generate_diagnostic_bank,
    recommend_next_exercise,
)


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"


class TutorNavigationTests(unittest.TestCase):
    def test_manual_and_automatic_exercise_selection_respect_the_domain(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
        practice = load_practice_bank(
            pd.read_csv(ROOT / "data/processed/exercises_augmented.csv"),
            ROOT / "data/processed/exercise_review_flags.csv",
        )
        practice = automatically_gradable_bank(practice)
        functions = practice[practice["Domeniu"] == "Funcții"]
        explicitly_selected = int(functions.index[1])
        selector_key = "tutor_exercise_choice_Funcții_0"
        app.selectbox(key=selector_key).set_value(explicitly_selected).run()
        app.button(key="start_selected").click().run()
        self.assertEqual(app.session_state["selected_exercise_idx"], explicitly_selected)

        app.button(key="start_automatic").click().run()
        selected = app.session_state["selected_exercise_idx"]
        selector_key = f"tutor_exercise_choice_Funcții_{app.session_state['tutor_selection_generation']}"
        self.assertNotEqual(selected, explicitly_selected)
        self.assertEqual(practice.loc[selected, "Domeniu"], "Funcții")
        self.assertEqual(app.selectbox(key=selector_key).value, selected)

        other_domain = next(
            domain for domain in sorted(practice["Domeniu"].unique()) if domain != "Funcții"
        )
        app.selectbox(key="tutor_domain").set_value(other_domain).run()
        self.assertEqual(app.session_state["tutor_domain"], other_domain)
        domain_selector_key = (
            f"tutor_exercise_choice_{other_domain}_"
            f"{app.session_state['tutor_selection_generation']}"
        )
        self.assertIsNotNone(app.selectbox(key=domain_selector_key))
        app.selectbox(key="tutor_domain").set_value("Funcții").run()
        function_selector = app.selectbox(key=selector_key)
        self.assertIn(function_selector.value, functions.index)
        manually_selected_after_domain_change = next(
            int(index) for index in functions.index if int(index) != int(function_selector.value)
        )
        function_selector.set_value(manually_selected_after_domain_change).run()
        app.button(key="start_selected").click().run()
        self.assertEqual(app.session_state["selected_exercise_idx"], manually_selected_after_domain_change)

    def test_real_wrong_correct_continue_duplicate_guard_and_reset(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
        self.assertFalse(app.exception, [str(item.value) for item in app.exception])
        self.assertEqual(app.session_state["selected_exercise_idx"], None)
        self.assertTrue(any("Nu există încă răspunsuri verificate" in item.value for item in app.info))

        app.button(key="start_selected").click().run()
        self.assertEqual(app.session_state["selected_exercise_idx"], 460)
        index = 460
        run_id = app.session_state["tutor_run_id"]
        answer_key = f"tutor_answer_{run_id}_{index}"
        submit_key = f"tutor_submit_{run_id}"

        app.button(key=f"tutor_hint_{run_id}").click().run()
        self.assertEqual(app.session_state["current_exercise_hint_count"], 1)
        self.assertNotIn("12", app.session_state["tutor_latest_hint"]["hint"])
        self.assertTrue(any("Înlocuiește x" in message.value for message in app.info))
        app.run()
        self.assertIsNotNone(app.session_state["tutor_latest_hint"])

        app.text_area(key=answer_key).set_value("1")
        app.button(key=submit_key).click().run()
        self.assertEqual(app.session_state["tutor_feedback"]["status"], "incorrect")
        self.assertEqual(app.session_state["current_exercise_attempt_count"], 1)
        self.assertEqual(app.session_state["current_exercise_mistake_count"], 1)
        self.assertEqual(app.session_state["current_exercise_consecutive_errors"], 1)
        self.assertEqual(len(app.session_state["interaction_log"]), 1)
        self.assertEqual(app.session_state["tutor_latest_hint"]["shown_solution_steps"], 0)

        before_unverified_submission = {
            key: copy.deepcopy(app.session_state[key]) for key in (
                "mastery", "current_exercise_attempt_count", "current_exercise_mistake_count",
                "current_exercise_consecutive_errors", "interaction_log", "tutor_analysis",
            )
        }
        app.text_area(key=answer_key).set_value("12 or 99")
        app.button(key=submit_key).click().run()
        self.assertEqual(app.session_state["tutor_feedback"]["status"], "needs_review")
        for key, value in before_unverified_submission.items():
            self.assertEqual(app.session_state[key], value, key)

        tracked_before_preview = {
            key: copy.deepcopy(app.session_state[key]) for key in (
                "interaction_log", "mastery", "current_exercise_attempt_count",
                "current_exercise_hint_count", "current_exercise_mistake_count",
                "current_exercise_consecutive_errors", "current_exercise_completed",
                "selected_exercise_idx", "tutor_feedback", "tutor_analysis", "tutor_latest_hint",
            )
        }
        app.text_input(key="home_preview_460_answer").set_value("12")
        app.button(key="home_preview_460_hint_button").click().run()
        app.button(key="home_preview_460_evaluate").click().run()
        app.button(key="home_preview_460_evaluate").click().run()
        for key, value in tracked_before_preview.items():
            self.assertEqual(app.session_state[key], value, key)
        self.assertEqual(app.session_state["tutor_feedback"]["status"], "needs_review")
        self.assertIsNotNone(app.session_state["tutor_latest_hint"])

        app.text_area(key=answer_key).set_value("12")
        app.button(key=submit_key).click().run()
        self.assertEqual(app.session_state["tutor_feedback"]["status"], "correct")
        self.assertEqual(app.session_state["current_exercise_attempt_count"], 2)
        self.assertEqual(app.session_state["current_exercise_consecutive_errors"], 0)
        self.assertEqual(len(app.session_state["interaction_log"]), 2)
        self.assertEqual(
            [entry["is_correct"] for entry in app.session_state["interaction_log"]],
            [False, True],
        )
        self.assertTrue(app.session_state["interaction_log"][-1]["actual_domain"] == "Funcții")
        self.assertTrue(app.button(key=submit_key).disabled)
        self.assertTrue(app.button(key=submit_key).disabled)
        self.assertEqual(len(app.session_state["interaction_log"]), 2)
        self.assertEqual(app.session_state["current_exercise_attempt_count"], 2)
        app.run()
        self.assertEqual(len(app.session_state["interaction_log"]), 2)
        self.assertEqual(app.session_state["tutor_feedback"]["status"], "correct")

        recommendation = app.session_state["tutor_recommendation"]
        self.assertIsNotNone(recommendation)
        next_index = recommendation["index"]
        self.assertNotEqual(next_index, index)
        self.assertNotEqual(next_index, 37)
        practice = load_practice_bank(
            pd.read_csv(ROOT / "data/processed/exercises_augmented.csv"),
            ROOT / "data/processed/exercise_review_flags.csv",
        )
        self.assertIn(next_index, practice.index)
        self.assertEqual(practice.loc[next_index, "Domeniu"], "Funcții")
        app.button(key=f"tutor_continue_{run_id}").click().run()
        self.assertEqual(app.session_state["selected_exercise_idx"], next_index)
        self.assertEqual(app.session_state["current_exercise_attempt_count"], 0)
        self.assertEqual(app.session_state["current_exercise_hint_count"], 0)
        self.assertEqual(app.session_state["current_exercise_mistake_count"], 0)
        self.assertEqual(app.session_state["current_exercise_consecutive_errors"], 0)
        self.assertFalse(app.session_state["current_exercise_completed"])
        self.assertIsNone(app.session_state["tutor_latest_hint"])
        self.assertIsNone(app.session_state["tutor_analysis"])
        self.assertIsNone(app.session_state["tutor_recommendation"])
        self.assertIsNone(app.session_state["tutor_feedback"])
        self.assertEqual(len(app.session_state["interaction_log"]), 2)
        selector_key = (
            f"tutor_exercise_choice_Funcții_"
            f"{app.session_state['tutor_selection_generation']}"
        )
        self.assertEqual(app.selectbox(key=selector_key).value, next_index)
        app.button(key="start_selected").click().run()
        self.assertEqual(app.session_state["selected_exercise_idx"], next_index)
        self.assertEqual(app.selectbox(key=selector_key).value, next_index)
        self.assertEqual(app.text_area(key=f"tutor_answer_{app.session_state['tutor_run_id']}_{next_index}").value, "")
        self.assertEqual(len(app.session_state["interaction_log"]), 2)

        app.button(key="reset_progress").click().run()
        self.assertEqual(app.session_state["interaction_log"], [])
        self.assertEqual(app.session_state["mastery"], 0.55)
        self.assertIsNone(app.session_state["selected_exercise_idx"])
        self.assertTrue(any("Nu există încă răspunsuri verificate" in item.value for item in app.info))

    def test_diagnostic_preview_isolation_after_non_default_tutor_state(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
        app.button(key="start_selected").click().run()
        run_id = app.session_state["tutor_run_id"]
        answer_key = f"tutor_answer_{run_id}_460"
        app.text_area(key=answer_key).set_value("1")
        app.button(key=f"tutor_submit_{run_id}").click().run()
        tracked_before = {
            key: copy.deepcopy(app.session_state[key]) for key in (
                "interaction_log", "mastery", "current_exercise_attempt_count",
                "current_exercise_hint_count", "current_exercise_mistake_count",
                "current_exercise_consecutive_errors", "current_exercise_completed",
                "selected_exercise_idx", "tutor_feedback", "tutor_analysis", "tutor_latest_hint",
            )
        }

        practice = load_practice_bank(
            pd.read_csv(ROOT / "data/processed/exercises_augmented.csv"),
            ROOT / "data/processed/exercise_review_flags.csv",
        )
        quiz = generate_diagnostic_bank(practice, 5, random_state=42)
        self.assertTrue(quiz)
        app.session_state["diagnostic_started"] = True
        app.session_state["diagnostic_seed"] = 42
        app.session_state["diagnostic_generation"] = 7
        candidate = next(
            (
                (number, row) for number, row in enumerate(quiz)
                if evaluate_answer("0", row["Raspunsul"])["status"] == "incorrect"
            ),
            None,
        )
        self.assertIsNotNone(candidate)
        question_number, question = candidate
        app.run()
        app.text_input(key=f"diagnostic_answer_7_{question_number}").set_value("0")
        app.button(key="diagnostic_finish").click().run()
        self.assertTrue(app.session_state["diagnostic_results"]["weak_domains"])
        for key, value in tracked_before.items():
            self.assertEqual(app.session_state[key], value, key)

        diagnostic_preview_question = app.session_state["diagnostic_results"]["weak_domains"][0]
        self.assertEqual(question["Domeniu"], diagnostic_preview_question)
        supported_practice = automatically_gradable_bank(practice)
        recommended = recommend_next_exercise(
            supported_practice, diagnostic_preview_question, "2 - mediu", random_state=11,
        )
        preview_prefix = f"diagnostic_preview_7_{int(recommended.index[0])}"
        app.button(key=f"{preview_prefix}_hint_button").click().run()
        app.button(key=f"{preview_prefix}_hint_button").click().run()
        preview_hint_count = app.session_state[f"{preview_prefix}_hint_count"]
        self.assertEqual(preview_hint_count, 2)
        preview_answer = str(recommended.iloc[0]["Raspunsul"])
        app.text_input(key=f"{preview_prefix}_answer").set_value(preview_answer)
        app.button(key=f"{preview_prefix}_evaluate").click().run()
        app.button(key=f"{preview_prefix}_evaluate").click().run()
        self.assertIsNotNone(app.session_state.get(f"{preview_prefix}_estimate"))
        app.text_input(key=f"{preview_prefix}_answer").set_value(f"{preview_answer} or 99")
        app.button(key=f"{preview_prefix}_evaluate").click().run()
        self.assertEqual(app.session_state[f"{preview_prefix}_feedback"]["status"], "needs_review")
        self.assertNotIn(f"{preview_prefix}_estimate", app.session_state)
        self.assertNotIn(f"{preview_prefix}_next", app.session_state)
        for key, value in tracked_before.items():
            self.assertEqual(app.session_state[key], value, key)
        old_generation = app.session_state["diagnostic_generation"]
        old_widget = f"diagnostic_answer_{old_generation}_{question_number}"
        app.button(key="diagnostic_new").click().run()
        self.assertEqual(app.session_state["diagnostic_generation"], old_generation + 1)
        self.assertIsNone(app.session_state["diagnostic_results"])
        self.assertNotIn(old_widget, app.session_state)

    def test_home_preview_clears_estimate_and_recommendation_after_unverified_answer(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
        prefix = "home_preview_460"
        app.text_input(key=f"{prefix}_answer").set_value("12")
        app.button(key=f"{prefix}_evaluate").click().run()
        self.assertEqual(app.session_state[f"{prefix}_feedback"]["status"], "correct")
        self.assertIsNotNone(app.session_state.get(f"{prefix}_estimate"))
        self.assertIsNotNone(app.session_state.get(f"{prefix}_next"))

        tracked_before = {
            key: copy.deepcopy(app.session_state[key]) for key in (
                "interaction_log", "mastery", "current_exercise_attempt_count",
                "current_exercise_hint_count", "current_exercise_mistake_count",
                "current_exercise_consecutive_errors", "current_exercise_completed",
                "selected_exercise_idx", "tutor_feedback", "tutor_analysis",
            )
        }
        app.text_input(key=f"{prefix}_answer").set_value("12 or 99")
        app.button(key=f"{prefix}_evaluate").click().run()

        self.assertEqual(app.session_state[f"{prefix}_feedback"]["status"], "needs_review")
        self.assertNotIn(f"{prefix}_estimate", app.session_state)
        self.assertNotIn(f"{prefix}_next", app.session_state)
        for key, value in tracked_before.items():
            self.assertEqual(app.session_state[key], value, key)

        app.session_state[f"{prefix}_estimate"] = 0.73
        app.session_state[f"{prefix}_next"] = {"Problema": "previous"}
        app.text_input(key=f"{prefix}_answer").set_value("")
        app.button(key=f"{prefix}_evaluate").click().run()
        self.assertEqual(app.session_state[f"{prefix}_feedback"]["status"], "no_answer")
        self.assertNotIn(f"{prefix}_estimate", app.session_state)
        self.assertNotIn(f"{prefix}_next", app.session_state)

    def test_diagnostic_profile_function_used_by_app_exposes_unverified_separation(self):
        bank = automatically_gradable_bank(load_practice_bank(
            pd.read_csv(ROOT / "data/processed/exercises_augmented.csv"),
            ROOT / "data/processed/exercise_review_flags.csv",
        ))
        profile = assess_diagnostic_results(
            [{"domain": bank.iloc[0]["Domeniu"], "correct": False}],
            bank,
        )
        self.assertTrue(profile["weak_domains"])
        self.assertGreater(len(profile["domain_scores"]), 0)

    def test_diagnostic_profile_is_invalidated_when_answer_changes_before_refinalizing(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
        app.button(key="diagnostic_start").click().run()
        app.session_state["diagnostic_seed"] = 1
        app.run()
        generation = app.session_state["diagnostic_generation"]
        seed = app.session_state["diagnostic_seed"]
        practice = load_practice_bank(
            pd.read_csv(ROOT / "data/processed/exercises_augmented.csv"),
            ROOT / "data/processed/exercise_review_flags.csv",
        )
        quiz = generate_diagnostic_bank(automatically_gradable_bank(practice), 5, seed)
        self.assertTrue(quiz)

        wrong_submission = None
        for number, item in enumerate(quiz):
            expected = str(item["Raspunsul"]).strip()
            match = re.fullmatch(
                r"(?P<label>[a-z][a-z0-9_]*=)?[+-]?(?:\d+(?:[.,]\d*)?|[.,]\d+|"
                r"\d+/\d+)(?P<unit>\s+.*)?",
                expected,
                re.IGNORECASE,
            )
            if match is None:
                continue
            prefix = match.group("label") or ""
            suffix = match.group("unit") or ""
            for candidate_value in ("987654321", "-987654321", "314159265"):
                candidate = f"{prefix}{candidate_value}{suffix}"
                if evaluate_answer(candidate, expected)["status"] == "incorrect":
                    wrong_submission = number, candidate
                    break
            if wrong_submission is not None:
                break
        self.assertIsNotNone(wrong_submission, "Diagnostic quiz must contain a verifiable scalar key.")
        response_number, wrong_answer = wrong_submission
        answer_key = f"diagnostic_answer_{generation}_{response_number}"
        app.text_input(key=answer_key).set_value(wrong_answer)
        app.button(key="diagnostic_finish").click().run()
        previous_profile = copy.deepcopy(app.session_state["diagnostic_results"])
        quiz_item = quiz[response_number]
        self.assertEqual(previous_profile["domain_scores"][quiz_item["Domeniu"]]["total"], 1)
        self.assertTrue(previous_profile["weak_domains"])
        self.assertEqual(
            app.session_state["diagnostic_results_snapshot"][response_number],
            wrong_answer,
        )

        app.text_input(key=answer_key).set_value("12 or 99").run()
        self.assertIsNone(app.session_state["diagnostic_results"])
        self.assertIsNone(app.session_state["diagnostic_results_snapshot"])
        self.assertFalse(any(
            "Exercițiu sugerat pentru explorare" in str(item.value) for item in app.markdown
        ))

        app.button(key="diagnostic_finish").click().run()
        current_profile = app.session_state["diagnostic_results"]
        self.assertFalse(current_profile["weak_domains"])
        self.assertTrue(all(
            score["status"] == "untested" and score["total"] == 0
            for score in current_profile["domain_scores"].values()
        ))
        self.assertTrue(any(
            "toate domeniile disponibile rămân netestate" in item.value for item in app.info
        ))
        self.assertFalse(any(
            "Exercițiu sugerat pentru explorare" in str(item.value) for item in app.markdown
        ))

    def test_home_preview_verified_result_is_cleared_after_leave_and_return(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
        prefix = "home_preview_460"
        app.text_input(key=f"{prefix}_answer").set_value("12")
        app.button(key=f"{prefix}_evaluate").click().run()
        self.assertEqual(app.session_state[f"{prefix}_feedback"]["status"], "correct")
        self.assertIsNotNone(app.session_state[f"{prefix}_estimate"])
        self.assertIsNotNone(app.session_state[f"{prefix}_next"])

        app.selectbox(key="home_exercise_Toate_Toate").set_value(461).run()
        self.assertNotIn(f"{prefix}_answer", app.session_state)
        self.assertNotIn(f"{prefix}_feedback", app.session_state)
        self.assertNotIn(f"{prefix}_estimate", app.session_state)
        self.assertNotIn(f"{prefix}_next", app.session_state)

        app.selectbox(key="home_exercise_Toate_Toate").set_value(460).run()
        self.assertEqual(app.text_input(key=f"{prefix}_answer").value, "")
        self.assertNotIn(f"{prefix}_feedback", app.session_state)
        self.assertNotIn(f"{prefix}_estimate", app.session_state)
        self.assertNotIn(f"{prefix}_next", app.session_state)

    def test_home_and_diagnostic_preview_contexts_keep_independent_hint_state(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
        home_prefix = "home_preview_460"
        app.button(key=f"{home_prefix}_hint_button").click().run()
        home_hint = copy.deepcopy(app.session_state[f"{home_prefix}_latest_hint"])

        practice = load_practice_bank(
            pd.read_csv(ROOT / "data/processed/exercises_augmented.csv"),
            ROOT / "data/processed/exercise_review_flags.csv",
        )
        eligible = automatically_gradable_bank(practice)
        weak_domain = "Funcții"
        profile = assess_diagnostic_results(
            [{"domain": weak_domain, "correct": False}], practice,
        )
        recommended = recommend_next_exercise(
            eligible, weak_domain, "2 - mediu", random_state=11,
        )
        self.assertFalse(recommended.empty)
        generation = 7
        app.session_state["diagnostic_started"] = True
        app.session_state["diagnostic_generation"] = generation
        app.session_state["diagnostic_results"] = profile
        app.session_state["diagnostic_results_snapshot"] = ("", "", "", "", "")
        app.run()
        diagnostic_prefix = f"diagnostic_preview_{generation}_{int(recommended.index[0])}"
        self.assertEqual(app.session_state[f"{home_prefix}_latest_hint"], home_hint)

        app.button(key=f"{diagnostic_prefix}_hint_button").click().run()
        self.assertEqual(app.session_state[f"{home_prefix}_latest_hint"], home_hint)
        self.assertEqual(app.session_state[f"{diagnostic_prefix}_hint_count"], 1)


if __name__ == "__main__":
    unittest.main()
