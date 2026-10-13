import unittest
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.pedagogical_engine import answer_result_is_verified, evaluate_answer


class TutorAnswerFlowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        cls.app = AppTest.from_file(str(app_path), default_timeout=90).run()
        if cls.app.exception:
            raise AssertionError([str(error.value) for error in cls.app.exception])

    def test_unverified_answer_does_not_change_progress(self):
        tutor_tab = self.app.tabs[1]
        tutor_tab.button(key="start_new_exercise").click().run()
        exercise_index = self.app.session_state["selected_exercise_idx"]
        root = Path(__file__).resolve().parents[1]
        data = pd.read_csv(root / "data" / "processed" / "exercises_augmented.csv")
        expected = str(data.loc[exercise_index, "Raspunsul"])
        unverified_answer = f"{expected} or 999"
        result = evaluate_answer(unverified_answer, expected)
        self.assertEqual(result["status"], "needs_review")
        self.assertFalse(answer_result_is_verified(result))

        mastery_before = self.app.session_state["mastery"]
        attempt_count_before = self.app.session_state["current_exercise_attempt_count"]
        mistake_count_before = self.app.session_state["current_exercise_mistake_count"]
        consecutive_errors_before = self.app.session_state["current_exercise_consecutive_errors"]
        log_before = list(self.app.session_state["interaction_log"])

        self.app.text_area(key=f"tutor_answer_{exercise_index}").set_value(unverified_answer)
        verify_button = next(button for button in self.app.button if "Verifică răspunsul" in button.label)
        verify_button.click().run()

        self.assertEqual(self.app.session_state["mastery"], mastery_before)
        self.assertEqual(self.app.session_state["current_exercise_attempt_count"], attempt_count_before)
        self.assertEqual(self.app.session_state["current_exercise_mistake_count"], mistake_count_before)
        self.assertEqual(self.app.session_state["current_exercise_consecutive_errors"], consecutive_errors_before)
        self.assertEqual(list(self.app.session_state["interaction_log"]), log_before)
        self.assertTrue(any("Răspunsul nu poate fi comparat" in message.value for message in self.app.info))
        self.assertFalse(any(message.value for message in self.app.success))

        self.app.text_area(key=f"tutor_answer_{exercise_index}").set_value("")
        verify_button = next(button for button in self.app.button if "Verifică răspunsul" in button.label)
        verify_button.click().run()

        self.assertEqual(self.app.session_state["mastery"], mastery_before)
        self.assertEqual(self.app.session_state["current_exercise_attempt_count"], attempt_count_before)
        self.assertEqual(self.app.session_state["current_exercise_mistake_count"], mistake_count_before)
        self.assertEqual(self.app.session_state["current_exercise_consecutive_errors"], consecutive_errors_before)
        self.assertEqual(list(self.app.session_state["interaction_log"]), log_before)
        self.assertTrue(any("Nu ai introdus încă un răspuns" in message.value for message in self.app.warning))

        self.app.text_area(key=f"tutor_answer_{exercise_index}").set_value(expected)
        verify_button = next(button for button in self.app.button if "Verifică răspunsul" in button.label)
        verify_button.click().run()

        self.assertEqual(self.app.session_state["current_exercise_attempt_count"], attempt_count_before + 1)
        self.assertEqual(self.app.session_state["current_exercise_mistake_count"], mistake_count_before)
        self.assertEqual(len(self.app.session_state["interaction_log"]), len(log_before) + 1)
        self.assertTrue(self.app.session_state["interaction_log"][-1]["is_correct"])
        self.assertGreater(self.app.session_state["mastery"], mastery_before)
        self.assertTrue(any("Răspunsul se potrivește cu cheia" in message.value for message in self.app.success))


if __name__ == "__main__":
    unittest.main()