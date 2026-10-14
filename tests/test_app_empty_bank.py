import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.exercise_bank import load_practice_bank as apply_content_flags


ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app.py"
SOURCE = pd.read_csv(ROOT / "data/processed/exercises_augmented.csv")


class EmptyTutorBankAppTests(unittest.TestCase):
    def _practice_fixture(self, rows, excluded_hashes=()):
        with tempfile.TemporaryDirectory() as directory:
            flags_path = Path(directory) / "flags.csv"
            lines = ["row_id,problem_hash,reason"]
            lines.extend(
                f"{number},{problem_hash},test exclusion"
                for number, problem_hash in enumerate(excluded_hashes)
            )
            flags_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            return apply_content_flags(rows, flags_path)

    def _template_row(self, index, problem_hash, answer, domain=None):
        row = SOURCE.loc[[460]].copy()
        row.index = [index]
        row.at[index, "problem_hash"] = problem_hash
        row.at[index, "Raspunsul"] = answer
        if domain is not None:
            row.at[index, "Domeniu"] = domain
        return row

    def _app_with_bank(self, bank):
        patcher = patch("src.exercise_bank.load_practice_bank", return_value=bank)
        patcher.start()
        self.addCleanup(patcher.stop)
        return AppTest.from_file(str(APP_PATH), default_timeout=120).run()

    def _assert_progress_history_retained(self, app):
        history = [{
            "bank_index": 460, "exercise_id": 460, "problem_text": "recorded",
            "actual_domain": "Funcții", "predicted_domain": "Funcții",
            "predicted_difficulty": "1 - bază", "time_spent_seconds": 1.0,
            "hint_count": 0, "attempt_count": 1, "mistake_count": 1,
            "consecutive_errors": 1, "is_correct": False,
            "predicted_learning_state": None, "timestamp": 1.0,
        }]
        app.session_state["interaction_log"] = copy.deepcopy(history)
        app.run()
        self.assertEqual(app.session_state["interaction_log"], history)
        self.assertTrue(any(item.value == "Progresul meu" for item in app.header))

    def test_all_content_excluded_shows_empty_tutor_and_preserves_progress(self):
        source = self._template_row(460, "excluded-hash", "12")
        bank = self._practice_fixture(source, ["excluded-hash"])
        self.assertTrue(bank.empty)
        app = self._app_with_bank(bank)

        self.assertFalse(app.exception, [str(item.value) for item in app.exception])
        self.assertTrue(any("Nu există exerciții cu chei compatibile" in item.value for item in app.info))
        self.assertNotIn("tutor_domain", [item.key for item in app.selectbox])
        self._assert_progress_history_retained(app)

    def test_nonempty_content_without_automatically_gradable_keys_is_safe(self):
        source = self._template_row(460, "manual-review-hash", "12 furlongs")
        bank = self._practice_fixture(source)
        self.assertEqual(len(bank), 1)
        app = self._app_with_bank(bank)

        self.assertFalse(app.exception, [str(item.value) for item in app.exception])
        self.assertTrue(any("Nu există exerciții cu chei compatibile" in item.value for item in app.info))
        self.assertTrue(any(
            "formatul cheii nu permite verificarea automată" in item.value for item in app.warning
        ))
        self.assertNotIn("start_selected", [item.key for item in app.button])
        self._assert_progress_history_retained(app)

    def test_invalid_preview_key_clears_stale_cached_results(self):
        source = self._template_row(460, "invalid-key-hash", "1/0")
        bank = self._practice_fixture(source)
        app = self._app_with_bank(bank)
        app.session_state["home_preview_460_estimate"] = 0.73
        app.session_state["home_preview_460_next"] = {"Problema": "old preview"}
        app.run()

        app.text_input(key="home_preview_460_answer").set_value("1")
        app.button(key="home_preview_460_evaluate").click().run()

        self.assertEqual(app.session_state["home_preview_460_feedback"]["status"], "invalid_key")
        self.assertNotIn("home_preview_460_estimate", app.session_state)
        self.assertNotIn("home_preview_460_next", app.session_state)

    def test_selected_content_domain_without_gradable_candidates_keeps_history(self):
        supported = self._template_row(460, "supported-hash", "12")
        browsing_only = self._template_row(1344, "manual-domain-hash", "12 furlongs", "Doar explorare")
        bank = self._practice_fixture(pd.concat([supported, browsing_only]))
        app = self._app_with_bank(bank)
        self.assertFalse(app.exception, [str(item.value) for item in app.exception])
        app.selectbox(key="tutor_domain").set_value("Doar explorare").run()

        self.assertFalse(app.exception, [str(item.value) for item in app.exception])
        self.assertTrue(any("Nu există exerciții care pot fi evaluate automat în acest domeniu" in item.value for item in app.info))
        self.assertFalse(any(
            item.key.startswith("tutor_exercise_choice_") for item in app.selectbox
        ))
        self._assert_progress_history_retained(app)

    def test_zero_answer_diagnostic_renders_content_domain_as_untested(self):
        supported = self._template_row(460, "supported-hash", "12", "Supported")
        browsing_only = self._template_row(1344, "manual-domain-hash", "12 furlongs", "Manual only")
        bank = self._practice_fixture(pd.concat([supported, browsing_only]))
        app = self._app_with_bank(bank)

        app.button(key="diagnostic_start").click().run()
        app.button(key="diagnostic_finish").click().run()

        profile = app.session_state["diagnostic_results"]
        self.assertEqual(set(profile["domain_scores"]), {"Supported", "Manual only"})
        self.assertTrue(all(
            score["status"] == "untested" and score["total"] == 0
            for score in profile["domain_scores"].values()
        ))
        self.assertTrue(any(
            "toate domeniile disponibile rămân netestate" in item.value for item in app.info
        ))
        self.assertTrue(any(
            "Manual only: netestat" in str(item.value) for item in app.markdown
        ))


if __name__ == "__main__":
    unittest.main()
