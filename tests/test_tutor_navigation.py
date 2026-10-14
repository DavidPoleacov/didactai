import copy
import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.exercise_bank import load_practice_bank
from src.pedagogical_engine import (
    assess_diagnostic_results,
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
        functions = practice[practice["Domeniu"] == "Funcții"]
        explicitly_selected = int(functions.index[1])
        selector_key = "tutor_exercise_choice_Funcții"
        app.selectbox(key=selector_key).set_value(explicitly_selected).run()
        app.button(key="start_selected").click().run()
        self.assertEqual(app.session_state["selected_exercise_idx"], explicitly_selected)

        app.button(key="start_automatic").click().run()
        selected = app.session_state["selected_exercise_idx"]
        self.assertNotEqual(selected, explicitly_selected)
        self.assertEqual(practice.loc[selected, "Domeniu"], "Funcții")

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
        self.assertEqual(app.session_state["tutor_feedback"]["status"], "incorrect")
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
        self.assertIsNone(app.session_state["tutor_feedback"])
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
        recommended = recommend_next_exercise(
            practice, diagnostic_preview_question, "2 - mediu", random_state=11,
        )
        preview_prefix = f"diagnostic_preview_7_{int(recommended.index[0])}"
        app.button(key=f"{preview_prefix}_hint_button").click().run()
        app.button(key=f"{preview_prefix}_hint_button").click().run()
        preview_hint_count = app.session_state[f"{preview_prefix}_hint_count"]
        self.assertEqual(preview_hint_count, 2)
        app.text_input(key=f"{preview_prefix}_answer").set_value("12")
        app.button(key=f"{preview_prefix}_evaluate").click().run()
        app.button(key=f"{preview_prefix}_evaluate").click().run()
        for key, value in tracked_before.items():
            self.assertEqual(app.session_state[key], value, key)
        old_generation = app.session_state["diagnostic_generation"]
        old_widget = f"diagnostic_answer_{old_generation}_{question_number}"
        app.button(key="diagnostic_new").click().run()
        self.assertEqual(app.session_state["diagnostic_generation"], old_generation + 1)
        self.assertIsNone(app.session_state["diagnostic_results"])
        self.assertNotIn(old_widget, app.session_state)

    def test_diagnostic_profile_function_used_by_app_exposes_unverified_separation(self):
        bank = load_practice_bank(
            pd.read_csv(ROOT / "data/processed/exercises_augmented.csv"),
            ROOT / "data/processed/exercise_review_flags.csv",
        )
        profile = assess_diagnostic_results(
            [{"domain": bank.iloc[0]["Domeniu"], "correct": False}],
            bank,
        )
        self.assertTrue(profile["weak_domains"])
        self.assertGreater(len(profile["domain_scores"]), 0)


if __name__ == "__main__":
    unittest.main()
