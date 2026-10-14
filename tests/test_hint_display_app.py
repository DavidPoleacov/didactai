import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.exercise_bank import load_practice_bank
from src.pedagogical_engine import (
    assess_diagnostic_results,
    automatically_gradable_bank,
    generate_diagnostic_bank,
)

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"
SOURCE = pd.read_csv(
    ROOT / "data/processed/exercises_augmented.csv",
    dtype=str,
    keep_default_na=False,
)
PRACTICE = load_practice_bank(
    SOURCE, ROOT / "data/processed/exercise_review_flags.csv",
)


class HintDisplayAppTests(unittest.TestCase):
    def _assert_corrected_second_hint(self, hint):
        self.assertIn("Pasul 1: Δ = (-4)² - 4·1·3 = 4", hint)
        self.assertNotIn("Δ = -4²", hint)

    def test_home_preview_displays_corrected_second_hint_for_exercise_337(self):
        app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
        app.selectbox(key="home_exercise_Toate_Toate").set_value(337).run()

        prefix = "home_preview_337"
        app.button(key=f"{prefix}_hint_button").click().run()
        self.assertEqual(app.session_state[f"{prefix}_latest_hint"]["shown_solution_steps"], 0)
        app.button(key=f"{prefix}_hint_button").click().run()

        self._assert_corrected_second_hint(app.session_state[f"{prefix}_latest_hint"]["hint"])

    def test_tutor_hint_uses_the_shared_display_correction_policy(self):
        row = SOURCE.loc[337]
        app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
        app.selectbox(key="tutor_domain").set_value(row["Domeniu"]).run()
        selector_key = f"tutor_exercise_choice_{row['Domeniu']}_0"
        app.selectbox(key=selector_key).set_value(337).run()
        app.button(key="start_selected").click().run()

        run_id = app.session_state["tutor_run_id"]
        app.button(key=f"tutor_hint_{run_id}").click().run()
        self.assertEqual(app.session_state["current_exercise_hint_count"], 1)
        app.button(key=f"tutor_hint_{run_id}").click().run()

        self._assert_corrected_second_hint(app.session_state["tutor_latest_hint"]["hint"])

    def test_diagnostic_preview_uses_the_shared_display_correction_policy(self):
        eligible = automatically_gradable_bank(PRACTICE)
        domain = SOURCE.loc[337, "Domeniu"]
        quiz = generate_diagnostic_bank(eligible, 5, random_state=42)
        profile = assess_diagnostic_results(
            [{"domain": domain, "correct": False}], PRACTICE,
        )
        self.assertTrue(profile["weak_domains"])
        app = AppTest.from_file(str(APP_PATH), default_timeout=120).run()
        generation = 12
        app.session_state["diagnostic_started"] = True
        app.session_state["diagnostic_seed"] = 42
        app.session_state["diagnostic_generation"] = generation
        app.session_state["diagnostic_results"] = profile
        app.session_state["diagnostic_results_snapshot"] = tuple("" for _ in quiz)
        correction_row = SOURCE.loc[[337]].copy()

        with patch(
            "src.pedagogical_engine.recommend_next_exercise",
            return_value=correction_row,
        ):
            app.run()
            prefix = f"diagnostic_preview_{generation}_337"
            app.button(key=f"{prefix}_hint_button").click().run()
            app.button(key=f"{prefix}_hint_button").click().run()

        self._assert_corrected_second_hint(
            app.session_state[f"{prefix}_latest_hint"]["hint"],
        )


if __name__ == "__main__":
    unittest.main()
