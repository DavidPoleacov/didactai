import unittest
from pathlib import Path
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"


class SemanticEligibilityAppTests(unittest.TestCase):
    def test_open_choice_exercise_remains_browsable_but_not_selectable_for_scoring(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
        app.selectbox(key="home_exercise_Toate_Toate").set_value(37).run()

        self.assertTrue(any(
            "răspuns deschis este disponibil pentru explorare și verificare manuală" in item.value
            for item in app.warning
        ))
        tutor_exercises = app.selectbox(key="tutor_exercise_choice_Funcții_0")
        self.assertNotIn(37, tutor_exercises.options)

        app.button(key="start_automatic").click().run()
        self.assertNotEqual(app.session_state["selected_exercise_idx"], 37)


if __name__ == "__main__":
    unittest.main()
