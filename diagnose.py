#!/usr/bin/env python
"""
Startup checks for the DidactAI Streamlit app.
Use the unittest suite for interactive behavior; neural inference is opt-in.
"""

import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def check_section(title):
    """Print section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

def test_imports(include_neural=False):
    """Test all required imports."""
    check_section("1. Testing Imports")
    
    imports_to_test = {
        "streamlit": "Streamlit web framework",
        "pandas": "Data manipulation",
        "numpy": "Numerical computing",
        "sklearn": "Scikit-learn ML library",
        "joblib": "Model serialization",
    }
    if include_neural:
        imports_to_test["tensorflow"] = "Optional neural network framework"
    
    failed = []
    for module_name, description in imports_to_test.items():
        try:
            __import__(module_name)
            print(f"  ✓ {module_name:15} - {description}")
        except ImportError as e:
            print(f"  ✗ {module_name:15} - FAILED: {e}")
            failed.append(module_name)
    
    return len(failed) == 0

def test_project_imports():
    """Test project-specific imports."""
    check_section("2. Testing Project Imports")
    
    sys.path.insert(0, str(ROOT))
    
    modules_to_test = {
        "src.pedagogical_engine": "Pedagogical decision engine",
        "src.model_utils": "ML model utilities",
        "src.neural_student_state_model": "Neural student state model",
    }
    
    failed = []
    for module_path, description in modules_to_test.items():
        try:
            __import__(module_path)
            print(f"  ✓ {module_path:40} - {description}")
        except Exception as e:
            print(f"  ✗ {module_path:40} - FAILED")
            print(f"      Error: {str(e)[:80]}")
            failed.append(module_path)
    
    return len(failed) == 0

def test_assets():
    """Test asset loading."""
    check_section("3. Testing Asset Loading")
    
    import pandas as pd
    
    sys.path.insert(0, str(ROOT))
    from src.model_utils import load_assets
    
    try:
        structured_model, unstructured_model, data, report = load_assets()
        print(f"  ✓ Structured model loaded: {type(structured_model).__name__}")
        print(f"  ✓ Unstructured model loaded: {type(unstructured_model).__name__}")
        print(f"  ✓ Data loaded: {data.shape[0]} exercises, {data.shape[1]} columns")
        print(f"  ✓ Report loaded with keys: {list(report.keys())}")
        
        # Check critical columns
        critical_cols = ["Domeniu", "Problema", "Raspunsul", "Dificultate_group"]
        missing = [c for c in critical_cols if c not in data.columns]
        if missing:
            print(f"  ✗ Missing columns: {missing}")
            return False
        print(f"  ✓ All critical columns present")
        from src.exercise_bank import load_practice_bank
        practice = load_practice_bank(data, ROOT / "data/processed/exercise_review_flags.csv")
        print(f"  ✓ Practice bank: {len(practice)} available, {len(data)-len(practice)} excluded for review")
        return True
    except Exception as e:
        print(f"  ✗ Asset loading failed: {e}")
        return False

def test_neural_model():
    """Test neural model availability."""
    check_section("4. Testing Neural Model")
    
    sys.path.insert(0, str(ROOT))
    from src.neural_student_state_model import ensure_neural_model_exists, predict_neural_student_state
    
    try:
        # Check model existence
        result = ensure_neural_model_exists()
        status = result.get("status")
        print(f"  ✓ Neural model status: {status}")
        
        if status != "ready":
            print(f"    Reason: {result.get('reason', 'Unknown')}")
            return False
        
        # Test prediction
        features = {
            "time_spent_seconds": 45.0,
            "hint_count": 1.0,
            "attempt_count": 2.0,
            "is_correct": 0.0,
            "mistake_count": 1.0,
            "exercise_difficulty_encoded": 2.0,
            "previous_mastery": 0.55,
            "consecutive_errors": 1.0,
            "help_level_requested": 1.0,
        }
        pred = predict_neural_student_state(features)
        print(f"  ✓ Neural prediction works: {pred['predicted_state']}")
        return True
    except Exception as e:
        print(f"  ✗ Neural model test failed: {e}")
        return False

def test_streamlit_config():
    """Test Streamlit configuration files."""
    check_section("5. Testing Streamlit Configuration")
    
    config_file = ROOT / ".streamlit" / "config.toml"
    if config_file.exists():
        print(f"  ✓ config.toml exists")
    else:
        print(f"  ✗ config.toml missing")
        return False
    
    secrets_file = ROOT / ".streamlit" / "secrets.toml"
    if secrets_file.exists():
        print(f"  ✓ secrets.toml exists")
    else:
        print("  - secrets.toml absent (no secrets are needed by this app)")
    
    return True

def test_data_files(include_neural=False):
    """Test that all required data files exist."""
    check_section("6. Testing Data Files")
    
    required_files = {
        "data/processed/exercises_augmented.csv": "Main dataset",
        "data/processed/exercise_review_flags.csv": "Practice exclusions",
        "models/structured_difficulty_model.joblib": "Difficulty model",
        "models/unstructured_domain_model.joblib": "Domain model",
    }
    if include_neural:
        required_files.update({
            "models/neural_student_state_model.keras": "Neural model",
            "models/neural_student_state_scaler.joblib": "Feature scaler",
            "models/neural_student_state_label_encoder.joblib": "Label encoder",
        })
    
    missing = []
    for file_path, description in required_files.items():
        full_path = ROOT / file_path
        if full_path.exists():
            print(f"  ✓ {file_path:45} - {description}")
        else:
            print(f"  ✗ {file_path:45} - MISSING")
            missing.append(file_path)
    
    return len(missing) == 0

def main():
    """Run all diagnostics."""
    print("\n" + "="*60)
    print("  DidactAI Streamlit App - Diagnostic Report")
    print("="*60)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--neural", action="store_true", help="Also run optional TensorFlow inference")
    args = parser.parse_args()
    results = {
        "Imports": test_imports(args.neural),
        "Project Imports": test_project_imports(),
        "Assets": test_assets(),
        "Neural Model": test_neural_model() if args.neural else None,
        "Streamlit Config": test_streamlit_config(),
        "Data Files": test_data_files(args.neural),
    }
    
    check_section("Summary")
    passed = sum(1 for v in results.values() if v)
    total = sum(v is not None for v in results.values())
    
    for test_name, result in results.items():
        status = "- SKIP" if result is None else "✓ PASS" if result else "✗ FAIL"
        print(f"  {status:8} - {test_name}")
    
    print(f"\n  Results: {passed}/{total} tests passed\n")
    
    if passed == total:
        print("  ✓ Mandatory startup checks passed. Run the behavioral tests before recording.")
        print("\n  To run the app, execute:")
        print("    streamlit run app.py\n")
        return 0
    else:
        print("  ✗✗✗ Some diagnostics failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
