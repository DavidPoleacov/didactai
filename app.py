from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data_prep import engineer_features
from src.exercise_bank import load_practice_bank
from src.model_utils import (
    load_assets,
    prepare_single_problem,
    predict_domain_from_text,
    predict_structured_difficulty,
    resolve_dataset_path,
)
try:
    from src.neural_student_state_model import ensure_neural_model_exists, predict_neural_student_state
except (ModuleNotFoundError, ImportError, Exception):
    ensure_neural_model_exists = None
    predict_neural_student_state = None

from src.pedagogical_engine import (
    METACOGNITIVE_QUESTIONS,
    assess_diagnostic_results,
    answer_result_is_verified,
    choose_hint,
    diagnose_learning_state,
    evaluate_answer,
    generate_diagnostic_bank,
    next_review_date,
    recommend_next_exercise,
    target_difficulty_from_mastery,
    update_mastery,
)
from src.train_models import train_all

ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Didact AI - Tutor adaptiv de matematică",
    page_icon="🧠",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main-card {padding: 1rem 1.2rem; border: 1px solid #E5E7EB; border-radius: 16px; background: #FFFFFF; box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);}
    .hero-card {padding: 1rem; border-radius: 16px; background: linear-gradient(135deg, #F8FAFC 0%, #EEF2FF 100%); border: 1px solid #E5E7EB;}
    .pill {display: inline-block; background: #EEF2FF; color: #3730A3; padding: 0.2rem 0.55rem; border-radius: 999px; font-size: 0.82rem; font-weight: 600;}
    .small-muted {color: #64748B; font-size: 0.92rem;}
    .rubric-good {background: #ECFDF5; color: #065F46; padding: 0.15rem 0.45rem; border-radius: 999px; font-weight: 600;}
    .rubric-warn {background: #FEF3C7; color: #92400E; padding: 0.15rem 0.45rem; border-radius: 999px; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


def _asset_signature(path: Path) -> str:
    try:
        stat = path.stat()
        return f"{path}:{stat.st_mtime_ns}:{stat.st_size}"
    except FileNotFoundError:
        return f"{path}:missing"


def display_answer_feedback(result: dict) -> None:
    if result["status"] == "correct":
        st.success(result["feedback"])
    elif result["status"] == "incorrect":
        st.error(result["feedback"])
    elif result["status"] in {"no_answer", "invalid_key"}:
        st.warning(result["feedback"])
    else:
        st.info(result["feedback"])


@st.cache_resource(show_spinner="Loading/training ML services...")
def cached_assets(dataset_signature: str, report_signature: str):
    required = [
        ROOT / "models" / "structured_difficulty_model.joblib",
        ROOT / "models" / "unstructured_domain_model.joblib",
        ROOT / "models" / "evaluation_report.json",
        ROOT / "data" / "processed" / "exercises_augmented.csv",
    ]
    dataset_path = resolve_dataset_path()
    report_path = ROOT / "models" / "evaluation_report.json"

    should_refresh = False
    if not all(p.exists() for p in required):
        should_refresh = True
    else:
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            should_refresh = int(report.get("dataset", {}).get("rows_total", 0)) != len(pd.read_csv(dataset_path))
        except Exception:
            should_refresh = True

    if should_refresh:
        train_all()
    return load_assets()


# Cache will persist by default. For development, use st.cache_resource.clear() manually if needed.

try:
    dataset_signature = _asset_signature(resolve_dataset_path())
    report_signature = _asset_signature(ROOT / "models" / "evaluation_report.json")
    structured_model, unstructured_model, data, report = cached_assets(dataset_signature, report_signature)
except Exception as e:
    st.error(f"Failed to load ML assets: {e}")
    st.info("Please check that all model files and data files are present in the models/ and data/processed/ directories.")
    st.stop()

try:
    practice_data = load_practice_bank(data, ROOT / "data" / "processed" / "exercise_review_flags.csv")
except (OSError, KeyError, ValueError) as e:
    st.error(f"Banca pentru practică nu poate fi verificată: {e}")
    st.stop()


def render_tutor_exercise(row: dict, exercise_idx: int, key_prefix: str) -> None:
    """Render the same tutor interaction used in the main demo."""
    st.markdown("### Problemă")
    st.markdown(f"<div class='main-card'>{row['Problema']}</div>", unsafe_allow_html=True)
    st.caption(f"Etichetă dataset: {row.get('Domeniu', '—')} · {row.get('Tema_norm', '—')} · {row.get('Dificultate_group', '—')}")

    student_answer = st.text_area(
        "Răspunsul elevului",
        placeholder="Scrie răspunsul aici",
        key=f"{key_prefix}_answer",
    )

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        time_seconds = st.number_input("Timp lucru (secunde)", min_value=5, max_value=3600, value=120, step=5, key=f"{key_prefix}_time")
    with col_b:
        attempts = st.number_input("Încercări", min_value=1, max_value=5, value=int(st.session_state.attempts), step=1, key=f"{key_prefix}_attempts")
    with col_c:
        hints_used = st.number_input("Indicii deja cerute", min_value=0, max_value=3, value=int(st.session_state.hints_used), step=1, key=f"{key_prefix}_hints")

    hint = choose_hint(row["Problema"], row.get("Pasii de rezolvare", ""), st.session_state.mastery, hints_used)
    with st.expander("Cere un indiciu gradual"):
        st.write(f"**Tip indiciu:** {hint['hint_type']}")
        st.write(hint["hint"])
        if st.button("Am folosit un indiciu", key=f"{key_prefix}_hint_used"):
            st.session_state.hints_used = min(3, int(hints_used) + 1)
            st.rerun()

    st.markdown("#### Întrebare de conștientizare")
    q_idx = (int(exercise_idx) + int(hints_used)) % len(METACOGNITIVE_QUESTIONS)
    st.info(METACOGNITIVE_QUESTIONS[q_idx])

    if st.button("Evaluează răspunsul și recomandă următorul pas", key=f"{key_prefix}_evaluate", type="primary"):
        result = evaluate_answer(student_answer, row["Raspunsul"])
        if not answer_result_is_verified(result):
            display_answer_feedback(result)
            return

        learning_state = diagnose_learning_state(result["correct"], int(hints_used), int(attempts), int(time_seconds))
        new_mastery = update_mastery(st.session_state.mastery, result["correct"], int(hints_used), int(attempts))
        target = target_difficulty_from_mastery(new_mastery, result["correct"])
        next_row = recommend_next_exercise(practice_data, row["Domeniu"], target, exclude_problem=row["Problema"], random_state=int(exercise_idx) + 1)

        st.session_state.mastery = new_mastery
        st.session_state.attempts = attempts
        st.session_state.hints_used = hints_used
        display_answer_feedback(result)
        st.write(f"**Stare estimată:** {learning_state['state']}")
        st.write(f"**Intervenție pedagogică:** {learning_state['intervention']}")
        st.write(f"**Indicator euristic de stăpânire:** {new_mastery:.2f}")
        st.write(f"**Reactivare spaced repetition:** {next_review_date(new_mastery, result['correct'])}")
        if not next_row.empty:
            nr = next_row.iloc[0]
            st.markdown("#### Recomandarea următoare")
            st.write(f"Țintă: **{target}**, domeniu: **{row['Domeniu']}**")
            st.markdown(f"<div class='main-card'>{nr['Problema']}</div>", unsafe_allow_html=True)
            st.caption(f"{nr.get('Tema_norm', '—')} · {nr.get('Dificultate_group', '—')}")


def process_demo_answer(result: dict, row: pd.Series, exercise_idx: int, hints_used: int, attempts: int, time_seconds: int) -> None:
    if not answer_result_is_verified(result):
        display_answer_feedback(result)
        return

    learning_state = diagnose_learning_state(result["correct"], hints_used, attempts, time_seconds)
    new_mastery = update_mastery(st.session_state.mastery, result["correct"], hints_used, attempts)
    target = target_difficulty_from_mastery(new_mastery, result["correct"])
    next_row = recommend_next_exercise(
        practice_data,
        row["Domeniu"],
        target,
        exclude_problem=row["Problema"],
        random_state=int(exercise_idx) + 1,
    )

    st.session_state.mastery = new_mastery
    st.session_state.attempts = attempts
    st.session_state.hints_used = hints_used
    display_answer_feedback(result)
    st.write(f"**Stare estimată:** {learning_state['state']}")
    st.write(f"**Intervenție pedagogică:** {learning_state['intervention']}")
    st.write(f"**Indicator euristic de stăpânire:** {new_mastery:.2f}")
    st.write(f"**Reactivare spaced repetition:** {next_review_date(new_mastery, result['correct'])}")
    if not next_row.empty:
        recommended_row = next_row.iloc[0]
        st.markdown("#### Recomandarea următoare")
        st.write(f"Țintă: **{target}**, domeniu: **{row['Domeniu']}**")
        st.markdown(f"<div class='main-card'>{recommended_row['Problema']}</div>", unsafe_allow_html=True)
        st.caption(f"{recommended_row.get('Tema_norm', '—')} · {recommended_row.get('Dificultate_group', '—')}")


# Fix dtypes after CSV load.
for col in ["Dificultate", "Itemul", "Sursa_year"]:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

# Initialize session state for learning tracking and interaction history
if "mastery" not in st.session_state:
    st.session_state.mastery = 0.55
if "hints_used" not in st.session_state:
    st.session_state.hints_used = 0
if "attempts" not in st.session_state:
    st.session_state.attempts = 1
if "diagnostic_started" not in st.session_state:
    st.session_state.diagnostic_started = False
if "diagnostic_results" not in st.session_state:
    st.session_state.diagnostic_results = None
if "diagnostic_seed" not in st.session_state:
    st.session_state.diagnostic_seed = int(time.time()) % 100000

# Neural model integration tracking
if "interaction_log" not in st.session_state:
    st.session_state.interaction_log = []
if "current_exercise_start_time" not in st.session_state:
    st.session_state.current_exercise_start_time = None
if "current_exercise_attempt_count" not in st.session_state:
    st.session_state.current_exercise_attempt_count = 0
if "current_exercise_hint_count" not in st.session_state:
    st.session_state.current_exercise_hint_count = 0
if "current_exercise_mistake_count" not in st.session_state:
    st.session_state.current_exercise_mistake_count = 0
if "current_exercise_consecutive_errors" not in st.session_state:
    st.session_state.current_exercise_consecutive_errors = 0
if "neural_available" not in st.session_state:
    st.session_state.neural_available = False
for key, default in {
    "tutor_feedback": None,
    "tutor_analysis": None,
    "tutor_recommendation": None,
    "tutor_last_hint": None,
    "tutor_exercise_completed": False,
    "exercise_selection_seed": 42,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# Check neural model availability once at startup
if ensure_neural_model_exists is not None:
    try:
        neural_check = ensure_neural_model_exists()
        st.session_state.neural_available = neural_check.get("status") == "ready"
    except Exception:
        st.session_state.neural_available = False

st.title("🧠 Didact AI")
st.subheader("Un tutor de matematică clar, practic și adaptat progresului tău.")
st.caption("Poți începe cu un diagnostic scurt, apoi primi exerciții potrivite acolo unde te blochezi cel mai mult.")

hero = st.columns(3)
with hero[0]:
    st.markdown("<div class='hero-card'><span class='pill'>Pasul 1</span><br><strong>Diagnostic scurt</strong><br>Un set de întrebări simple care arată unde ai nevoie de sprijin.</div>", unsafe_allow_html=True)
with hero[1]:
    st.markdown("<div class='hero-card'><span class='pill'>Pasul 2</span><br><strong>Exerciții potrivite</strong><br>Sistemul recomandă teme relevante, nu exerciții arbitrare.</div>", unsafe_allow_html=True)
with hero[2]:
    st.markdown("<div class='hero-card'><span class='pill'>Pasul 3</span><br><strong>Feedback clar</strong><br>Vezi ce ai făcut bine și unde merită să exersezi mai mult.</div>", unsafe_allow_html=True)

st.markdown("---")
st.markdown("### 🧭 Începe cu un diagnostic scurt")
st.info("Nu trebuie să fii perfect. Scrie răspunsul cât poți și sistemul va sugera apoi exerciții pe zonele care merită mai multă practică.")

if not st.session_state.diagnostic_started:
    st.caption("Acest pas te ajută să vezi rapid unde ai nevoie de sprijin, fără să te simți copleșit.")
    if st.button("Începe testul de diagnostic", type="primary"):
        st.session_state.diagnostic_started = True
        st.session_state.diagnostic_seed = int(time.time()) % 100000
        st.rerun()
else:
    diagnostic_bank = generate_diagnostic_bank(practice_data, n_questions=5, random_state=st.session_state.diagnostic_seed)
    st.caption("Ai 5 întrebări scurte. Răspunde natural și apoi primești o recomandare simplă asupra temelor de exersat.")
    diagnostic_answers = []
    for idx, row in enumerate(diagnostic_bank, start=1):
        st.markdown(f"<div class='main-card'><strong>{idx}.</strong> {row['Problema']}</div>", unsafe_allow_html=True)
        answer = st.text_area("Răspunsul tău", key=f"diag_{idx}", placeholder="Scrie răspunsul aici")
        if answer:
            result = evaluate_answer(answer, row.get("Raspunsul"))
            if answer_result_is_verified(result):
                diagnostic_answers.append({"problem": row["Problema"], "domain": row.get("Domeniu"), "correct": result["correct"]})
            elif result["status"] in {"needs_review", "invalid_key"}:
                display_answer_feedback(result)
    if st.button("Finalizează diagnostic și recomandă exerciții", type="primary"):
        if not diagnostic_answers:
            st.warning("Nu există răspunsuri pe care verificatorul să le poată evalua. Completează un răspuns verificabil sau cere ajutorul unui profesor.")
            st.stop()

        profile = assess_diagnostic_results(diagnostic_answers, practice_data)
        st.session_state.diagnostic_results = profile
        st.success("Diagnostic finalizat. Am identificat zonele unde merită să exersezi mai mult.")
        st.rerun()

if st.session_state.diagnostic_results:
    profile = st.session_state.diagnostic_results
    st.markdown("### Rezumatul tău de început")
    weak_domains = profile.get("weak_domains", [])
    if weak_domains:
        st.write("Punctele unde te-ai blocat cel mai mult sunt:")
        for domain in weak_domains[:3]:
            st.markdown(f"- **{domain}**")
    else:
        st.write("Nu am identificat o zonă clară de dificultate din răspunsurile de acum. Poți continua cu exerciții generale.")

    for item in profile.get("recommended_themes", [])[:3]:
        st.markdown(f"- **{item['domain']}** → teme sugerate: {', '.join(item['themes'])}")

    if profile.get("weak_domains"):
        weak_domain = profile["weak_domains"][0]
        rec = recommend_next_exercise(practice_data, weak_domain, "2 - mediu", random_state=11)
        if not rec.empty:
            st.markdown("### Exercițiul de început recomandat")
            st.markdown(f"Tema prioritară: **{weak_domain}**")
            try:
                exercise_idx = int(rec.iloc[0].get("Itemul", 0)) if pd.notna(rec.iloc[0].get("Itemul")) else 0
            except (TypeError, ValueError, OverflowError):
                exercise_idx = 0
            render_tutor_exercise(rec.iloc[0].to_dict(), exercise_idx=exercise_idx, key_prefix="diag_followup")

st.markdown("---")

with st.sidebar:
    st.header("Profil elev")
    name = st.text_input("Nume / poreclă", value="Alex")
    grade = st.selectbox("Clasa", ["V", "VI", "VII", "VIII", "IX"], index=4)
    st.info(
        "Profilul din demo folosește o actualizare simplă de tip knowledge tracing: corectitudine + indicii + încercări."
    )

# Top scorecard
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Exerciții disponibile", len(practice_data))
with col2:
    st.metric("Calitate model structură", f"{report['structured_model']['model']['macro_f1']:.3f}")
with col3:
    st.metric("Calitate model text", f"{report['unstructured_model']['model']['macro_f1']:.3f}")
with col4:
    st.metric("Interacțiuni în sesiune", len(st.session_state.interaction_log))


def show_probabilities(probabilities: dict | None, title: str):
    if not probabilities:
        st.caption("Modelul nu expune probabilități pentru această inferență.")
        return
    probs = pd.DataFrame(
        sorted(probabilities.items(), key=lambda kv: kv[1], reverse=True),
        columns=["clasă", "probabilitate"],
    )
    probs["probabilitate"] = probs["probabilitate"].astype(float)
    st.write(title)
    st.dataframe(probs, width="stretch", hide_index=True)
    st.bar_chart(probs.set_index("clasă"))


def encode_difficulty_to_number(difficulty_group: str) -> int:
    """Map difficulty label to numeric encoding for neural model."""
    mapping = {
        "1 - bază": 1,
        "2 - mediu": 2,
        "3 - consolidare": 3,
        "4 - avansat": 4,
    }
    return mapping.get(str(difficulty_group), 2)


def infer_help_level(hints_used: int) -> int:
    """Infer help level from hints used."""
    # 0 = no help, 1 = abstract hint, 2 = concrete hint, 3 = step-by-step hint
    return min(hints_used, 3)


def start_new_exercise(exercise_row: dict) -> None:
    """Initialize tracking for a new exercise."""
    st.session_state.current_exercise_start_time = time.time()
    st.session_state.current_exercise_attempt_count = 0
    st.session_state.current_exercise_hint_count = 0
    st.session_state.current_exercise_mistake_count = 0
    st.session_state.current_exercise_consecutive_errors = 0
    st.session_state.tutor_feedback = None
    st.session_state.tutor_analysis = None
    st.session_state.tutor_recommendation = None
    st.session_state.tutor_last_hint = None
    st.session_state.tutor_exercise_completed = False


def get_neural_prediction(row: dict, is_correct: bool, time_spent: float) -> dict | None:
    """Get neural model prediction for current exercise state."""
    if not st.session_state.neural_available or predict_neural_student_state is None:
        return None

    try:
        difficulty_encoded = encode_difficulty_to_number(row.get("Dificultate_group", "2 - mediu"))
        help_level = infer_help_level(st.session_state.current_exercise_hint_count)
        
        features = {
            "time_spent_seconds": float(time_spent),
            "hint_count": float(st.session_state.current_exercise_hint_count),
            "attempt_count": float(st.session_state.current_exercise_attempt_count),
            "is_correct": float(1.0 if is_correct else 0.0),
            "mistake_count": float(st.session_state.current_exercise_mistake_count),
            "exercise_difficulty_encoded": float(difficulty_encoded),
            "previous_mastery": float(st.session_state.mastery),
            "consecutive_errors": float(st.session_state.current_exercise_consecutive_errors),
            "help_level_requested": float(help_level),
        }
        return predict_neural_student_state(features)
    except Exception as e:
        st.warning(f"Neural prediction failed (reverting to rule-based): {str(e)[:100]}")
        return None


def record_interaction_to_log(
    row: dict,
    predicted_domain: str,
    predicted_difficulty: str,
    time_spent: float,
    is_correct: bool,
    predicted_state: str | None = None,
) -> None:
    """Record the interaction to session history."""
    entry = {
        "exercise_id": int(row.name),
        "bank_domain": str(row.get("Domeniu", "")),
        "problem_text": str(row.get("Problema", ""))[:200],
        "predicted_domain": predicted_domain,
        "predicted_difficulty": predicted_difficulty,
        "time_spent_seconds": float(time_spent),
        "hint_count": int(st.session_state.current_exercise_hint_count),
        "attempt_count": int(st.session_state.current_exercise_attempt_count),
        "mistake_count": int(st.session_state.current_exercise_mistake_count),
        "is_correct": bool(is_correct),
        "predicted_learning_state": predicted_state or "unknown",
        "timestamp": time.time(),
    }
    st.session_state.interaction_log.append(entry)


def get_neural_based_recommendation(predicted_state: str, row: dict) -> dict:
    """Get exercise recommendation based on neural state prediction."""
    domain = row.get("Domeniu", "Toate")
    
    if predicted_state == "blocaj":
        # Blocked state: recommend easier exercise + concrete hint
        target = "1 - bază"
    elif predicted_state == "progres":
        # Progress state: recommend slightly harder exercise
        target = "2 - mediu"
    elif predicted_state == "supraincarcare":
        # Overwhelmed state: recommend simple guided task
        target = "1 - bază"
    else:  # autonomie_buna
        # Good autonomy: offer challenge
        target = "3 - consolidare"
    
    next_ex = recommend_next_exercise(practice_data, domain, target, exclude_problem=row.get("Problema", ""), random_state=42)
    return {"target": target, "exercise": next_ex, "state": predicted_state}


def process_verified_tutor_answer(result: dict, current_row: pd.Series, exercise_idx: int) -> None:
    """Apply progress changes only after a definite checker verdict."""
    if not answer_result_is_verified(result) or st.session_state.tutor_exercise_completed:
        return

    st.session_state.current_exercise_attempt_count += 1
    is_correct = result["correct"]
    time_spent = time.time() - st.session_state.current_exercise_start_time
    if not is_correct:
        st.session_state.current_exercise_mistake_count += 1
        st.session_state.current_exercise_consecutive_errors += 1
    else:
        st.session_state.current_exercise_consecutive_errors = 0

    domain_pred = predict_domain_from_text(unstructured_model, str(current_row["Problema"]))
    feature_row = data.loc[[exercise_idx]][[
        "Itemul", "Sursa_year", "problem_chars", "problem_words", "steps_chars", "answer_chars",
        "n_digits", "n_math_symbols", "has_percent", "has_geometry_word", "has_equation_word",
        "has_radical", "has_function_word", "has_real_life_context", "Tema_norm", "Domeniu", "Sursa_type"
    ]]
    diff_pred = predict_structured_difficulty(structured_model, feature_row)
    neural_pred = get_neural_prediction(current_row, is_correct, time_spent)
    predicted_state = neural_pred["predicted_state"] if neural_pred else None

    record_interaction_to_log(
        current_row,
        domain_pred["prediction"],
        diff_pred["prediction"],
        time_spent,
        is_correct,
        predicted_state,
    )

    learning_state = diagnose_learning_state(
        is_correct,
        st.session_state.current_exercise_hint_count,
        st.session_state.current_exercise_attempt_count,
        int(time_spent),
    )
    new_mastery = update_mastery(
        st.session_state.mastery,
        is_correct,
        st.session_state.current_exercise_hint_count,
        st.session_state.current_exercise_attempt_count,
    )
    st.session_state.mastery = new_mastery
    st.session_state.tutor_exercise_completed = is_correct
    st.session_state.tutor_feedback = result
    st.session_state.tutor_analysis = {
        "time_spent": time_spent,
        "hints_used": st.session_state.current_exercise_hint_count,
        "mastery": new_mastery,
        "domain": domain_pred["prediction"],
        "difficulty": diff_pred["prediction"],
        "neural": neural_pred,
        "learning_state": learning_state,
        "review_date": next_review_date(new_mastery, is_correct),
    }
    st.session_state.tutor_recommendation = None

    if is_correct or st.session_state.current_exercise_attempt_count >= 3:
        if neural_pred and st.session_state.neural_available:
            recommendation = get_neural_based_recommendation(neural_pred["predicted_state"], current_row)
            target = recommendation["target"]
            next_exercise = recommendation["exercise"]
        else:
            target = target_difficulty_from_mastery(new_mastery, is_correct)
            next_exercise = recommend_next_exercise(
                practice_data,
                current_row["Domeniu"],
                target,
                exclude_problem=current_row["Problema"],
                random_state=42,
            )

        if not next_exercise.empty:
            st.session_state.tutor_recommendation = {
                "target": target,
                "exercise_idx": int(next_exercise.index[0]),
            }


def render_tutor_feedback() -> None:
    """Render saved results on every rerun, including navigation clicks."""
    result = st.session_state.tutor_feedback
    if result is None:
        return
    st.divider()
    display_answer_feedback(result)
    analysis = st.session_state.tutor_analysis
    if analysis is not None:
        with st.expander("📊 Analiza sistemului"):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Timp petrecut", f"{int(analysis['time_spent'])}s")
            with col_b:
                st.metric("Indicii folosite", analysis["hints_used"])
            with col_c:
                st.metric("Indicator de stăpânire", f"{analysis['mastery']:.2f}")
            st.caption("Indicator euristic pentru sesiunea curentă; nu este o probabilitate calibrată sau un diagnostic educațional.")
            st.markdown("**Predicții ML:**")
            st.write(f"- Domeniu estimat: **{analysis['domain']}**")
            st.write(f"- Dificultate estimată: **{analysis['difficulty']}**")
            if analysis["neural"]:
                neural = analysis["neural"]
                st.write(f"- Stare estimată (model neural): **{neural['predicted_state']}**")
                st.write(f"  Recomandare: {neural['recommended_action']}")
        learning_state = analysis["learning_state"]
        st.markdown("**Feedback pedagogic:**")
        st.write(f"- Stare estimată: {learning_state['state']}")
        st.write(f"- Intervenție: {learning_state['intervention']}")
        st.write(f"- Data sugerată pentru recapitulare: {analysis['review_date']}")

    recommendation = st.session_state.tutor_recommendation
    if recommendation is not None:
        next_idx = recommendation["exercise_idx"]
        st.markdown("### 🎯 Exercițiul următor recomandat")
        st.write(f"Țintă: **{recommendation['target']}** (pe baza progresului din această sesiune)")
        st.markdown(f"<div class='main-card'>{data.loc[next_idx, 'Problema']}</div>", unsafe_allow_html=True)
        if st.button("Continuă cu exercițiul următor ➜", type="primary", key="continue_next_exercise"):
            st.session_state.selected_exercise_idx = next_idx
            st.session_state[f"tutor_answer_{next_idx}"] = ""
            start_new_exercise(data.loc[next_idx])
            st.rerun()


tabs = st.tabs([
    "🏠 Acasă",
    "🤖 Tutor AI",
    "📈 Progresul meu",
])

with tabs[0]:
    st.header("Demo: elevul rezolvă, sistemul clasifică, estimează dificultatea și adaptează traseul")
    left, right = st.columns([1.2, 0.8])

    with left:
        domains = ["Toate"] + sorted(practice_data["Domeniu"].dropna().unique().tolist())
        selected_domain = st.selectbox("Domeniu", domains, index=0)
        filtered = practice_data.copy()
        if selected_domain != "Toate":
            filtered = filtered[filtered["Domeniu"] == selected_domain]
        topic_options = ["Toate"] + sorted(filtered["Tema_norm"].dropna().unique().tolist())
        selected_topic = st.selectbox("Temă", topic_options, index=0)
        if selected_topic != "Toate":
            filtered = filtered[filtered["Tema_norm"] == selected_topic]
        if filtered.empty:
            st.warning("Nu există exerciții pentru filtrele alese.")
            st.stop()

        exercise_idx = st.selectbox(
            "Alege exercițiul pentru demo",
            filtered.index.tolist(),
            format_func=lambda i: f"#{int(i)} · {str(data.loc[i, 'Tema_norm'])} · {str(data.loc[i, 'Dificultate_group'])} · {str(data.loc[i, 'Problema'])[:80]}...",
        )
        row = data.loc[exercise_idx]
        st.markdown("### Problemă")
        st.markdown(f"<div class='main-card'>{row['Problema']}</div>", unsafe_allow_html=True)
        st.caption(f"Etichetă dataset: {row['Domeniu']} · {row['Tema_norm']} · {row['Dificultate_group']}")

        student_answer = st.text_input("Răspunsul elevului", placeholder="Scrie răspunsul aici")
        col_a, col_b, col_c = st.columns(3)
        with col_a:
            time_seconds = st.number_input("Timp lucru (secunde)", min_value=5, max_value=3600, value=120, step=5)
        with col_b:
            st.session_state.attempts = st.number_input("Încercări", min_value=1, max_value=5, value=int(st.session_state.attempts), step=1)
        with col_c:
            st.session_state.hints_used = st.number_input("Indicii deja cerute", min_value=0, max_value=3, value=int(st.session_state.hints_used), step=1)

        hint = choose_hint(row["Problema"], row["Pasii de rezolvare"], st.session_state.mastery, st.session_state.hints_used)
        with st.expander("Cere un indiciu gradual"):
            st.write(f"**Tip indiciu:** {hint['hint_type']}")
            st.write(hint["hint"])
            if st.button("Am folosit un indiciu"):
                st.session_state.hints_used = min(3, int(st.session_state.hints_used) + 1)
                st.rerun()

        st.markdown("#### Întrebare de conștientizare")
        q_idx = (int(exercise_idx) + int(st.session_state.hints_used)) % len(METACOGNITIVE_QUESTIONS)
        st.info(METACOGNITIVE_QUESTIONS[q_idx])

        if st.button("Evaluează răspunsul și recomandă următorul pas", type="primary"):
            result = evaluate_answer(student_answer, row["Raspunsul"])
            process_demo_answer(
                result,
                row,
                exercise_idx,
                int(st.session_state.hints_used),
                int(st.session_state.attempts),
                int(time_seconds),
            )

    with right:
        st.markdown("### Inferențe live")
        domain_pred = predict_domain_from_text(unstructured_model, str(row["Problema"]))
        feature_row = data.loc[[exercise_idx]][
            [
                "Itemul", "Sursa_year", "problem_chars", "problem_words", "steps_chars", "answer_chars",
                "n_digits", "n_math_symbols", "has_percent", "has_geometry_word", "has_equation_word",
                "has_radical", "has_function_word", "has_real_life_context", "Tema_norm", "Domeniu", "Sursa_type"
            ]
        ]
        diff_pred = predict_structured_difficulty(structured_model, feature_row)
        st.markdown("<span class='rubric-good'>Serviciu ML nestructurat</span>", unsafe_allow_html=True)
        st.write(f"Predicție domeniu din text: **{domain_pred['prediction']}**")
        show_probabilities(domain_pred.get("probabilities"), "Probabilități domeniu")
        st.markdown("<span class='rubric-good'>Serviciu ML structurat</span>", unsafe_allow_html=True)
        st.write(f"Predicție dificultate din features tabelare: **{diff_pred['prediction']}**")
        show_probabilities(diff_pred.get("probabilities"), "Probabilități dificultate")

def render_tutor_tab() -> None:
    st.header("🤖 Tutor AI - Rezolvă exerciții și progresează")
    st.markdown("Exersează cu indicii graduale și feedback bazat pe reguli. Istoricul și progresul sunt păstrate doar în sesiunea curentă.")
    if "selected_exercise_idx" in st.session_state and st.session_state.selected_exercise_idx not in practice_data.index:
        del st.session_state.selected_exercise_idx
        st.warning("Exercițiul anterior necesită revizuire și a fost retras din practică. Alege un alt exercițiu.")
    
    if not st.session_state.neural_available:
        st.warning("Modelul opțional de estimare a stării elevului nu este disponibil. Predicțiile respective vor fi omise; feedbackul pedagogic bazat pe reguli rămâne activ.")
    else:
        st.info("Modelul opțional poate estima o stare pe baza caracteristicilor unei interacțiuni; această predicție nu măsoară învățarea reală.")

    # Exercise selector
    domains = sorted(practice_data["Domeniu"].dropna().unique().tolist())
    selected_domain = st.selectbox("Alege domeniul pentru a continua", domains, key="tutor_domain")
    
    filtered_by_domain = practice_data[practice_data["Domeniu"] == selected_domain] if selected_domain else practice_data
    if filtered_by_domain.empty:
        st.warning("Nu există exerciții în acest domeniu.")
        return

    requested_idx = st.selectbox(
        "Alege un exercițiu (opțional)",
        [None] + filtered_by_domain.index.tolist(),
        format_func=lambda i: "Selecție automată" if i is None else f"#{int(i)} · {str(data.loc[i, 'Problema'])[:100]}",
        key="tutor_requested_exercise",
    )

    # Start new exercise
    if st.button("▶ Începe exercițiu nou", type="primary", key="start_new_exercise"):
        if requested_idx is not None:
            selected_idx = requested_idx
        else:
            current_idx = st.session_state.get("selected_exercise_idx")
            candidates = filtered_by_domain.drop(index=[current_idx], errors="ignore")
            if candidates.empty:
                candidates = filtered_by_domain
            selected_idx = candidates.sample(1, random_state=st.session_state.exercise_selection_seed).index[0]
            st.session_state.exercise_selection_seed += 1
        st.session_state.selected_exercise_idx = selected_idx
        st.session_state[f"tutor_answer_{selected_idx}"] = ""
        start_new_exercise(data.loc[selected_idx])

    # Check if we have a current exercise
    if "selected_exercise_idx" not in st.session_state:
        st.info("Apasă butonul 'Începe exercițiu nou' pentru a selecta un exercițiu.")
        return

    exercise_idx = st.session_state.selected_exercise_idx
    current_row = data.loc[exercise_idx]
    
    st.divider()
    st.markdown("### 📝 Problemă")
    st.markdown(f"<div class='main-card'>{current_row['Problema']}</div>", unsafe_allow_html=True)
    st.caption(f"Domeniu: **{current_row.get('Domeniu', '—')}** · Temă: **{current_row.get('Tema_norm', '—')}** · Nivel: **{current_row.get('Dificultate_group', '—')}**")
    question_idx = (int(exercise_idx) + st.session_state.current_exercise_hint_count) % len(METACOGNITIVE_QUESTIONS)
    st.info(METACOGNITIVE_QUESTIONS[question_idx])

    # Student answer and tracking
    student_answer = st.text_area(
        "Scrie răspunsul tău",
        placeholder="Introdu răspunsul aici...",
        key=f"tutor_answer_{exercise_idx}",
    )

    col_hints, col_submit = st.columns([1, 1])
    
    with col_hints:
        if st.button("💡 Cere un indiciu", key="tutor_hint", disabled=st.session_state.tutor_exercise_completed):
            st.session_state.current_exercise_hint_count += 1
            st.session_state.tutor_last_hint = choose_hint(
                current_row["Problema"],
                current_row.get("Pasii de rezolvare", ""),
                st.session_state.mastery,
                st.session_state.current_exercise_hint_count
            )
            st.rerun()

    with col_submit:
        if st.button("✓ Verifică răspunsul", type="primary", key="tutor_verify", disabled=st.session_state.tutor_exercise_completed):
            result = evaluate_answer(student_answer, current_row.get("Raspunsul", ""))
            if not answer_result_is_verified(result):
                st.session_state.tutor_feedback = result
                st.session_state.tutor_analysis = None
            else:
                process_verified_tutor_answer(result, current_row, exercise_idx)
                st.rerun()

    hint = st.session_state.tutor_last_hint
    if hint is not None:
        st.markdown(f"**Tip indiciu: {hint['hint_type']}**")
        st.info(hint["hint"])
        if hint.get("shown_solution_steps"):
            st.caption("Ai ajuns la ultimul pas al soluției salvate; acesta poate include răspunsul final.")
    render_tutor_feedback()


with tabs[1]:
    render_tutor_tab()

with tabs[2]:
    st.header("📈 Progresul meu")
    
    if not st.session_state.interaction_log:
        st.info("Încă nu ai rezolvat exerciții. Mergi la **Tutor AI** și începe cu un exercițiu nou!")
        st.stop()

    log_df = pd.DataFrame(st.session_state.interaction_log)

    # Overall statistics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Încercări evaluate", len(log_df))
    with col2:
        accuracy = (log_df["is_correct"].sum() / len(log_df) * 100) if len(log_df) > 0 else 0
        st.metric("Acuratețe", f"{accuracy:.1f}%")
    with col3:
        avg_hints = log_df["hint_count"].mean()
        st.metric("Indicii în medie", f"{avg_hints:.1f}")
    with col4:
        avg_time = log_df["time_spent_seconds"].mean()
        st.metric("Timp mediu până la evaluare", f"{int(avg_time)}s")

    st.divider()

    # Domain breakdown
    st.markdown("### Progres pe domenii")
    domain_stats = log_df.groupby("bank_domain").agg({
        "is_correct": ["sum", "count"],
        "time_spent_seconds": "mean",
    }).round(2)
    domain_stats.columns = ["Corecte", "Total", "Timp mediu (s)"]
    if len(domain_stats) > 0:
        st.dataframe(domain_stats, width="stretch")

    # Learning state distribution
    known_states = log_df["predicted_learning_state"].dropna()
    known_states = known_states[known_states != "unknown"]
    if not known_states.empty:
        st.markdown("### Stări estimate de modelul experimental")
        state_counts = known_states.value_counts()
        st.bar_chart(state_counts)
        
        st.markdown("**Interpretare:**")
        st.write("- **blocaj**: Ai nevoie de exerciții mai ușoare și indicii concrete")
        st.write("- **progres**: Mergi înainte - încearcă exerciții puțin mai grele")
        st.write("- **supraincarcare**: Ai prea mult - hai la ceva mai simplu")
        st.write("- **autonomie_buna**: Gata! Poți face exerciții mai dificile singur")

    # Recent interactions
    st.markdown("### Istoric recent")
    recent = log_df.tail(10)[["problem_text", "predicted_difficulty", "hint_count", "attempt_count", "is_correct", "time_spent_seconds"]].copy()
    recent["Rezultat"] = recent["is_correct"].map({True: "✓ Corect", False: "✗ Incorect"})
    recent = recent.drop("is_correct", axis=1)
    st.dataframe(recent, width="stretch")

    # Reset progress button
    if st.button("🔄 Resetează progresul", key="reset_progress"):
        st.session_state.interaction_log = []
        st.session_state.mastery = 0.55
        if "selected_exercise_idx" in st.session_state:
            del st.session_state.selected_exercise_idx
        st.session_state.tutor_feedback = None
        st.session_state.tutor_analysis = None
        st.session_state.tutor_recommendation = None
        st.session_state.tutor_last_hint = None
        st.session_state.tutor_exercise_completed = False
        st.success("Progresul a fost resetat!")
        st.rerun()



