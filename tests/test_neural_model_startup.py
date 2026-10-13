import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from src import neural_student_state_model


@unittest.skipIf(neural_student_state_model.tf is None, "TensorFlow is optional")
class NeuralModelStartupTests(unittest.TestCase):
    def test_successful_automatic_training_marks_model_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "neural_student_state_model.keras"
            with (
                patch.object(neural_student_state_model, "MODELS_DIR", Path(directory)),
                patch.object(neural_student_state_model, "train_neural_student_state_model", return_value={"metrics": {"accuracy": 1.0}}) as train,
            ):
                result = neural_student_state_model.ensure_neural_model_exists()

        self.assertEqual(result["status"], "ready")
        self.assertEqual(result["model"], str(model_path))
        self.assertIn("report", result)
        train.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()