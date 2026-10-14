import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import neural_student_state_model


class NeuralModelStartupTests(unittest.TestCase):
    def test_missing_optional_model_does_not_train_automatically(self):
        with tempfile.TemporaryDirectory() as directory:
            with (
                patch.object(neural_student_state_model, "MODELS_DIR", Path(directory)),
                patch.object(neural_student_state_model, "tf", object()),
                patch.object(neural_student_state_model, "models", object()),
                patch.object(neural_student_state_model, "layers", object()),
                patch.object(neural_student_state_model, "EarlyStopping", object()),
                patch.object(neural_student_state_model, "train_neural_student_state_model") as train,
            ):
                result = neural_student_state_model.ensure_neural_model_exists()

        self.assertEqual(result["status"], "missing")
        self.assertIn("explicit training command", result["reason"])
        train.assert_not_called()

    @unittest.skipIf(neural_student_state_model.tf is None, "TensorFlow is optional")
    def test_saved_neural_model_real_inference_when_tensorflow_is_available(self):
        result = neural_student_state_model.predict_neural_student_state({
            "time_spent_seconds": 45.0, "hint_count": 1.0, "attempt_count": 2.0,
            "is_correct": 0.0, "mistake_count": 1.0, "exercise_difficulty_encoded": 2.0,
            "previous_mastery": 0.55, "consecutive_errors": 1.0, "help_level_requested": 1.0,
        })
        self.assertIn(result["predicted_state"], {
            "autonomie_buna", "blocaj", "progres", "supraincarcare",
        })


if __name__ == "__main__":
    unittest.main()