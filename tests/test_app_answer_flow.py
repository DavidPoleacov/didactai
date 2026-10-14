import copy
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest


class AppAnswerFlowTests(unittest.TestCase):
    def setUp(self):
        app_path = Path(__file__).resolve().parents[1] / "app.py"
        self.app = AppTest.from_file(str(app_path), default_timeout=120).run()
        self.assertFalse(self.app.exception, [str(item.value) for item in self.app.exception])
        self.app.button(key="start_selected").click().run()

    def test_unverified_answer_does_not_mutate_tutor_progress(self):
        index = self.app.session_state["selected_exercise_idx"]
        answer_key = f"tutor_answer_{self.app.session_state['tutor_run_id']}_{index}"
        submission_key = f"tutor_submit_{self.app.session_state['tutor_run_id']}"
        before = {
            "mastery": self.app.session_state["mastery"],
            "attempts": self.app.session_state["current_exercise_attempt_count"],
            "mistakes": self.app.session_state["current_exercise_mistake_count"],
            "errors": self.app.session_state["current_exercise_consecutive_errors"],
            "history": copy.deepcopy(self.app.session_state["interaction_log"]),
        }
        self.app.text_area(key=answer_key).set_value("12 or 999")
        self.app.button(key=submission_key).click().run()
        self.assertEqual(self.app.session_state["mastery"], before["mastery"])
        self.assertEqual(self.app.session_state["current_exercise_attempt_count"], before["attempts"])
        self.assertEqual(self.app.session_state["current_exercise_mistake_count"], before["mistakes"])
        self.assertEqual(self.app.session_state["current_exercise_consecutive_errors"], before["errors"])
        self.assertEqual(self.app.session_state["interaction_log"], before["history"])
        self.assertEqual(self.app.session_state["tutor_feedback"]["status"], "needs_review")

    def test_unverified_answer_labels_retained_verified_analysis(self):
        index = self.app.session_state["selected_exercise_idx"]
        run_id = self.app.session_state["tutor_run_id"]
        answer_key = f"tutor_answer_{run_id}_{index}"
        submission_key = f"tutor_submit_{run_id}"
        self.app.text_area(key=answer_key).set_value("1")
        self.app.button(key=submission_key).click().run()
        verified_state = {
            key: copy.deepcopy(self.app.session_state[key]) for key in (
                "mastery", "current_exercise_attempt_count", "current_exercise_mistake_count",
                "current_exercise_consecutive_errors", "interaction_log", "tutor_analysis",
            )
        }

        self.app.text_area(key=answer_key).set_value("12 or 99")
        self.app.button(key=submission_key).click().run()

        self.assertEqual(self.app.session_state["tutor_feedback"]["status"], "needs_review")
        for key, value in verified_state.items():
            self.assertEqual(self.app.session_state[key], value, key)
        self.assertTrue(any(
            "aparține ultimei încercări verificate" in item.value for item in self.app.caption
        ))


if __name__ == "__main__":
    unittest.main()
