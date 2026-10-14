from __future__ import annotations

import time
from pathlib import Path

import pandas as pd
import streamlit as st

from src.exercise_bank import load_practice_bank
from src.model_utils import load_assets, predict_domain_from_text, predict_structured_difficulty
from src.pedagogical_engine import (
    METACOGNITIVE_QUESTIONS,
    answer_result_is_verified,
    assess_diagnostic_results,
    automatically_gradable_bank,
    choose_hint,
    diagnose_learning_state,
    evaluate_answer,
    generate_diagnostic_bank,
    load_hint_display_corrections,
    load_semantic_grading_exclusion_hashes,
    next_review_date,
    recommend_next_exercise,
    target_difficulty_from_mastery,
    tutor_answer_can_be_recorded,
    update_mastery,
)

ROOT = Path(__file__).resolve().parent
FLAGS_PATH = ROOT / "data" / "processed" / "exercise_review_flags.csv"
FEATURE_COLUMNS = [
    "Itemul", "Sursa_year", "problem_chars", "problem_words", "steps_chars", "answer_chars",
    "n_digits", "n_math_symbols", "has_percent", "has_geometry_word", "has_equation_word",
    "has_radical", "has_function_word", "has_real_life_context", "Tema_norm", "Domeniu", "Sursa_type",
]

st.set_page_config(page_title="Didact AI - Tutor adaptiv de matematică", page_icon="🧠", layout="wide")
st.markdown(
    """
    <style>
    .main-card {padding: 1rem 1.2rem; border: 1px solid #CBD5E1; border-radius: 12px;
      background: #FFFFFF; color: #0F172A;}
    .hero-card {padding: 1rem; border-radius: 12px; background: #EEF2FF;
      border: 1px solid #C7D2FE; color: #172554;}
    </style>
    """,
    unsafe_allow_html=True,
)


def display_answer_feedback(result: dict) -> None:
    if result["status"] == "correct":
        st.success(result["feedback"])
    elif result["status"] == "incorrect":
        st.error(result["feedback"])
    elif result["status"] in {"no_answer", "invalid_key"}:
        st.warning(result["feedback"])
    else:
        st.info(result["feedback"])


try:
    structured_model, domain_model, raw_data, report = load_assets()
    data = load_practice_bank(raw_data, FLAGS_PATH)
    load_hint_display_corrections()
    semantic_grading_exclusion_hashes = load_semantic_grading_exclusion_hashes()
    missing_features = set(FEATURE_COLUMNS).difference(data.columns)
    if missing_features:
        raise ValueError(f"Exercise data is missing model features: {', '.join(sorted(missing_features))}")
    automatically_gradable_data = automatically_gradable_bank(data)
except Exception as exc:
    st.error(f"Nu s-au putut încărca resursele necesare: {exc}")
    st.info("Verifică fișierele de date, modelele salvate și data/processed/exercise_review_flags.csv.")
    st.stop()

data = data.copy()
for column in ("Dificultate", "Itemul", "Sursa_year"):
    data[column] = pd.to_numeric(data[column], errors="coerce")

try:
    from src import neural_student_state_model as neural_module

    neural_available = (
        neural_module.tf is not None
        and all(
            (ROOT / "models" / name).is_file()
            for name in (
                "neural_student_state_model.keras",
                "neural_student_state_scaler.joblib",
                "neural_student_state_label_encoder.joblib",
            )
        )
    )
    predict_neural_student_state = neural_module.predict_neural_student_state if neural_available else None
except ModuleNotFoundError:
    neural_available = False
    predict_neural_student_state = None


SESSION_DEFAULTS = {
    "mastery": 0.55,
    "interaction_log": [],
    "current_exercise_start_time": None,
    "current_exercise_attempt_count": 0,
    "current_exercise_hint_count": 0,
    "current_exercise_mistake_count": 0,
    "current_exercise_consecutive_errors": 0,
    "current_exercise_completed": False,
    "selected_exercise_idx": None,
    "tutor_run_id": 0,
    "tutor_selection_generation": 0,
    "tutor_latest_hint": None,
    "tutor_feedback": None,
    "tutor_analysis": None,
    "tutor_recommendation": None,
    "tutor_error": None,
    "diagnostic_started": False,
    "diagnostic_seed": 42,
    "diagnostic_results": None,
    "diagnostic_results_snapshot": None,
    "diagnostic_generation": 0,
}
for state_key, default in SESSION_DEFAULTS.items():
    if state_key not in st.session_state:
        st.session_state[state_key] = default
st.session_state.neural_available = neural_available


def formatted_exercise(index: int) -> str:
    row = data.loc[index]
    return f"#{index} · {row['Tema_norm']} · {row['Dificultate_group']} · {str(row['Problema'])[:85]}"


def reset_exercise_state() -> None:
    st.session_state.current_exercise_start_time = time.time()
    st.session_state.current_exercise_attempt_count = 0
    st.session_state.current_exercise_hint_count = 0
    st.session_state.current_exercise_mistake_count = 0
    st.session_state.current_exercise_consecutive_errors = 0
    st.session_state.current_exercise_completed = False
    st.session_state.tutor_latest_hint = None
    st.session_state.tutor_feedback = None
    st.session_state.tutor_analysis = None
    st.session_state.tutor_recommendation = None
    st.session_state.tutor_error = None
    st.session_state.tutor_run_id += 1


if (
    st.session_state.selected_exercise_idx is not None
    and st.session_state.selected_exercise_idx not in data.index
):
    st.session_state.selected_exercise_idx = None
    reset_exercise_state()


def process_verified_tutor_answer(result: dict, current_row: pd.Series, exercise_idx: int) -> bool:
    if not tutor_answer_can_be_recorded(result, st.session_state.current_exercise_completed):
        return False

    is_correct = bool(result["correct"])
    next_attempt_count = st.session_state.current_exercise_attempt_count + 1
    next_mistake_count = st.session_state.current_exercise_mistake_count + (not is_correct)
    next_consecutive_errors = 0 if is_correct else st.session_state.current_exercise_consecutive_errors + 1

    elapsed = max(0.0, time.time() - (st.session_state.current_exercise_start_time or time.time()))
    try:
        domain_prediction = predict_domain_from_text(domain_model, str(current_row["Problema"]))["prediction"]
        feature_row = data.loc[[exercise_idx], FEATURE_COLUMNS]
        difficulty_prediction = predict_structured_difficulty(structured_model, feature_row)["prediction"]
    except (KeyError, ValueError, TypeError) as exc:
        st.session_state.tutor_error = f"Nu s-a putut analiza exercițiul selectat: {exc}"
        return False

    next_mastery = update_mastery(
        st.session_state.mastery,
        is_correct,
        st.session_state.current_exercise_hint_count,
        next_attempt_count,
    )
    prediction = None
    neural_error = None
    if neural_available and predict_neural_student_state:
        try:
            prediction = predict_neural_student_state({
                "time_spent_seconds": elapsed,
                "hint_count": st.session_state.current_exercise_hint_count,
                "attempt_count": next_attempt_count,
                "is_correct": float(is_correct),
                "mistake_count": next_mistake_count,
                "exercise_difficulty_encoded": {
                    "1 - bază": 1, "2 - mediu": 2, "3 - consolidare": 3, "4 - avansat": 4,
                }.get(str(current_row["Dificultate_group"]), 2),
                "previous_mastery": st.session_state.mastery,
                "consecutive_errors": next_consecutive_errors,
                "help_level_requested": min(st.session_state.current_exercise_hint_count, 3),
            })
        except (OSError, ValueError, RuntimeError, ModuleNotFoundError) as exc:
            neural_error = f"Predicția neurală opțională nu este disponibilă: {exc}"

    entry = {
        "bank_index": int(exercise_idx),
        "exercise_id": int(exercise_idx),
        "problem_text": str(current_row["Problema"])[:200],
        "actual_domain": str(current_row["Domeniu"]),
        "predicted_domain": str(domain_prediction),
        "predicted_difficulty": str(difficulty_prediction),
        "time_spent_seconds": elapsed,
        "hint_count": int(st.session_state.current_exercise_hint_count),
        "attempt_count": int(next_attempt_count),
        "mistake_count": int(next_mistake_count),
        "consecutive_errors": int(next_consecutive_errors),
        "is_correct": is_correct,
        "predicted_learning_state": prediction["predicted_state"] if prediction else None,
        "timestamp": time.time(),
    }
    st.session_state.interaction_log.append(entry)
    st.session_state.mastery = next_mastery

    learning_state = diagnose_learning_state(
        is_correct,
        st.session_state.current_exercise_hint_count,
        next_attempt_count,
        int(elapsed),
    )
    st.session_state.tutor_feedback = result
    st.session_state.tutor_analysis = {
        "elapsed": int(elapsed),
        "domain": domain_prediction,
        "difficulty": difficulty_prediction,
        "mastery": next_mastery,
        "learning_state": learning_state,
        "neural": prediction,
        "neural_error": neural_error,
        "review_date": next_review_date(next_mastery, is_correct),
    }
    st.session_state.current_exercise_attempt_count = next_attempt_count
    st.session_state.current_exercise_mistake_count = next_mistake_count
    st.session_state.current_exercise_consecutive_errors = next_consecutive_errors
    if is_correct:
        st.session_state.current_exercise_completed = True
    target = target_difficulty_from_mastery(next_mastery, is_correct)
    recommended = recommend_next_exercise(
        data, str(current_row["Domeniu"]), target,
        exclude_problem=str(current_row["Problema"]), random_state=int(exercise_idx) + len(st.session_state.interaction_log),
    )
    st.session_state.tutor_recommendation = (
        {"index": int(recommended.index[0]), "target": target} if not recommended.empty else None
    )
    return True


def clear_preview_prefix(prefix: str) -> None:
    for suffix in (
        "answer", "hint_count", "latest_hint", "feedback", "estimate", "next",
        "hint_button", "evaluate",
    ):
        st.session_state.pop(f"{prefix}_{suffix}", None)


def clear_preview_context(context: str) -> None:
    active_key = f"{context}_preview_active_prefix"
    active_prefix = st.session_state.pop(active_key, None)
    if active_prefix:
        clear_preview_prefix(active_prefix)


def render_preview(row: pd.Series, prefix: str, context: str) -> None:
    """Show an isolated demonstration interaction; never write tracked Tutor state."""
    active_key = f"{context}_preview_active_prefix"
    previous_prefix = st.session_state.get(active_key)
    if previous_prefix and previous_prefix != prefix:
        clear_preview_prefix(previous_prefix)
    st.session_state[active_key] = prefix

    exercise_idx = int(row.name)
    st.caption("Simulare: această previzualizare nu modifică progresul Tutorului.")
    hint_key = f"{prefix}_hint_count"
    if hint_key not in st.session_state:
        st.session_state[hint_key] = 0
    if st.button("💡 Cere un indiciu", key=f"{prefix}_hint_button"):
        st.session_state[hint_key] += 1
        st.session_state[f"{prefix}_latest_hint"] = choose_hint(
            str(row["Problema"]), row.get("Pasii de rezolvare", ""), st.session_state.mastery,
            st.session_state[hint_key], exercise_idx, str(row.get("problem_hash", "")),
        )
    shown_hint = st.session_state.get(f"{prefix}_latest_hint")
    if shown_hint:
        st.info(shown_hint["hint"])
    st.info(METACOGNITIVE_QUESTIONS[exercise_idx % len(METACOGNITIVE_QUESTIONS)])
    answer = st.text_input("Răspuns (simulare)", key=f"{prefix}_answer")
    if st.button("Evaluează simularea", key=f"{prefix}_evaluate"):
        result = evaluate_answer(answer, row.get("Raspunsul"))
        st.session_state[f"{prefix}_feedback"] = result
        if answer_result_is_verified(result):
            hypothetical = update_mastery(
                st.session_state.mastery, result["correct"], st.session_state[hint_key],
                max(1, st.session_state[hint_key] + 1),
            )
            st.session_state[f"{prefix}_estimate"] = hypothetical
            target = target_difficulty_from_mastery(hypothetical, result["correct"])
            candidate = recommend_next_exercise(
                data, str(row["Domeniu"]), target, exclude_problem=str(row["Problema"]),
                random_state=exercise_idx + 1,
            )
            st.session_state[f"{prefix}_next"] = (
                candidate.iloc[0].to_dict() if not candidate.empty else None
            )
        else:
            st.session_state.pop(f"{prefix}_estimate", None)
            st.session_state.pop(f"{prefix}_next", None)
    feedback = st.session_state.get(f"{prefix}_feedback")
    if feedback:
        display_answer_feedback(feedback)
    estimate = st.session_state.get(f"{prefix}_estimate")
    if estimate is not None:
        st.write(f"Estimare simulată a indicatorului de stăpânire: **{estimate:.2f}**")
    preview_next = st.session_state.get(f"{prefix}_next")
    if preview_next:
        st.markdown("Recomandare simulată:")
        st.write(preview_next["Problema"])


st.title("🧠 Didact AI")
st.subheader("Un tutor de matematică clar și practic.")
st.caption(
    f"{len(data):,} exerciții pentru explorare; {len(automatically_gradable_data):,} "
    "pot fi evaluate automat. Întrebările de reflecție sunt prestabilite."
)
metrics = st.columns(4)
metrics[0].metric("Pentru explorare", f"{len(data):,}")
metrics[1].metric("Gradabile automat", f"{len(automatically_gradable_data):,}")
metrics[2].metric("Macro-F1 dificultate", f"{report['structured_model']['model']['macro_f1']:.3f}")
metrics[3].metric("Macro-F1 domeniu", f"{report['unstructured_model']['model']['macro_f1']:.3f}")

tabs = st.tabs(["🏠 Acasă", "🤖 Tutor AI", "📈 Progresul meu"])

with tabs[0]:
    st.header("Acasă · explorează o simulare")
    st.markdown(
        "<div class='hero-card'>Alege un domeniu și un exercițiu pentru a vedea "
        "clasificările modelelor și o interacțiune demonstrativă.</div>",
        unsafe_allow_html=True,
    )
    domains = ["Toate"] + sorted(data["Domeniu"].dropna().unique().tolist())
    home_domain = st.selectbox("Domeniu (simulare)", domains, key="home_domain")
    home_rows = data if home_domain == "Toate" else data[data["Domeniu"] == home_domain]
    topics = ["Toate"] + sorted(home_rows["Tema_norm"].dropna().unique().tolist())
    home_topic = st.selectbox("Temă (simulare)", topics, key=f"home_topic_{home_domain}")
    if home_topic != "Toate":
        home_rows = home_rows[home_rows["Tema_norm"] == home_topic]
    if home_rows.empty:
        st.warning("Nu există exerciții disponibile pentru filtrele alese.")
    else:
        home_idx = st.selectbox(
            "Alege exercițiul (simulare)", home_rows.index.tolist(),
            index=home_rows.index.tolist().index(460) if 460 in home_rows.index else 0,
            format_func=formatted_exercise,
            key=f"home_exercise_{home_domain}_{home_topic}",
        )
        home_row = data.loc[home_idx]
        st.markdown("### Problemă")
        st.write(home_row["Problema"])
        st.caption(f"{home_row['Domeniu']} · {home_row['Tema_norm']} · {home_row['Dificultate_group']}")
        if home_idx not in automatically_gradable_data.index:
            if str(home_row.get("problem_hash", "")) in semantic_grading_exclusion_hashes:
                st.warning(
                    "Acest exercițiu cu răspuns deschis este disponibil pentru explorare și "
                    "verificare manuală; răspunsurile numerice valide pot varia."
                )
            else:
                st.warning(
                    "Acest exercițiu este disponibil pentru explorare, dar formatul cheii nu "
                    "permite verificarea automată. Nu se poate înregistra un progres automat."
                )
        left, right = st.columns(2)
        with left:
            render_preview(home_row, f"home_preview_{home_idx}", "home")
        with right:
            domain_prediction = predict_domain_from_text(domain_model, str(home_row["Problema"]))
            feature_row = data.loc[[home_idx], FEATURE_COLUMNS]
            difficulty_prediction = predict_structured_difficulty(structured_model, feature_row)
            st.subheader("Clasificări pentru acest exercițiu")
            st.write(f"Domeniu clasificat din text: **{domain_prediction['prediction']}**")
            st.write(f"Dificultate estimată din metadate: **{difficulty_prediction['prediction']}**")
            st.caption("Clasificările descriu exercițiul; nu măsoară rezultatele învățării.")

    st.divider()
    st.subheader("Diagnostic scurt (separat de progresul Tutorului)")
    if not st.session_state.diagnostic_started:
        if st.button("Începe diagnosticul", key="diagnostic_start"):
            st.session_state.diagnostic_started = True
            st.session_state.diagnostic_seed = int(time.time()) % 100000
            st.session_state.diagnostic_results = None
            st.session_state.diagnostic_results_snapshot = None
            clear_preview_context("diagnostic")
            st.session_state.diagnostic_generation += 1
            st.rerun()
    else:
        quiz = generate_diagnostic_bank(automatically_gradable_data, 5, st.session_state.diagnostic_seed)
        generation = st.session_state.diagnostic_generation
        if not quiz:
            st.info("Nu există suficiente exerciții eligibile pentru diagnostic.")
        else:
            responses = []
            submitted_answers = []
            for number, item in enumerate(quiz):
                st.write(f"{number + 1}. {item['Problema']}")
                answer = st.text_input(
                    f"Răspunsul {number + 1}",
                    key=f"diagnostic_answer_{generation}_{number}",
                )
                submitted_answers.append(answer)
                if answer:
                    verdict = evaluate_answer(answer, item["Raspunsul"])
                    if answer_result_is_verified(verdict):
                        responses.append({
                            "problem": item["Problema"], "domain": item["Domeniu"],
                            "correct": verdict["correct"], "verified": True,
                            "status": verdict["status"],
                        })
                    elif verdict["status"] not in {"no_answer"}:
                        st.caption(f"Răspunsul {number + 1}: neverificat.")
            submitted_snapshot = tuple(submitted_answers)
            if (
                st.session_state.diagnostic_results is not None
                and st.session_state.diagnostic_results_snapshot != submitted_snapshot
            ):
                st.session_state.diagnostic_results = None
                st.session_state.diagnostic_results_snapshot = None
                clear_preview_context("diagnostic")
            if st.button("Finalizează diagnosticul", key="diagnostic_finish"):
                st.session_state.diagnostic_results = assess_diagnostic_results(responses, data)
                st.session_state.diagnostic_results_snapshot = submitted_snapshot
                st.rerun()
            if st.button("Generează un diagnostic nou", key="diagnostic_new"):
                for number in range(len(quiz)):
                    st.session_state.pop(f"diagnostic_answer_{generation}_{number}", None)
                st.session_state.diagnostic_results = None
                st.session_state.diagnostic_results_snapshot = None
                clear_preview_context("diagnostic")
                st.session_state.diagnostic_generation += 1
                st.session_state.diagnostic_seed += 1
                st.rerun()

    if st.session_state.diagnostic_results:
        profile = st.session_state.diagnostic_results
        st.subheader("Rezultate observate")
        for domain, score in profile["domain_scores"].items():
            if score["status"] == "untested":
                st.write(f"{domain}: netestat")
            else:
                st.write(f"{domain}: {score['correct']}/{score['total']} corecte")
        weak_domains = profile["weak_domains"]
        tested_answers = sum(score["total"] for score in profile["domain_scores"].values())
        if tested_answers == 0:
            st.info("Nu există răspunsuri verificate; toate domeniile disponibile rămân netestate.")
        elif weak_domains:
            st.write("Domenii cu scor observat mai mic: " + ", ".join(weak_domains[:3]))
            recommended = recommend_next_exercise(
                automatically_gradable_data, weak_domains[0], "2 - mediu", random_state=11,
            )
            if not recommended.empty:
                st.write("Exercițiu sugerat pentru explorare:")
                st.write(recommended.iloc[0]["Problema"])
                render_preview(
                    recommended.iloc[0],
                    f"diagnostic_preview_{generation}_{int(recommended.index[0])}",
                    "diagnostic",
                )
            else:
                st.info("Nu există un exercițiu disponibil în acest domeniu.")
        else:
            st.info("Răspunsurile verificate nu indică încă un domeniu cu scor redus.")

with tabs[1]:
    st.header("Tutor AI · practică urmărită")
    st.caption("Răspunsurile verificate actualizează progresul în sesiunea curentă; cele neverificate nu îl modifică.")
    if not neural_available:
        st.caption("Predicția neurală opțională nu este disponibilă; practica rulează fără TensorFlow.")
    tutor_domains = sorted(data["Domeniu"].dropna().unique().tolist())
    gradable_domains = set(automatically_gradable_data["Domeniu"].dropna().unique().tolist())
    if automatically_gradable_data.empty or not tutor_domains or not gradable_domains:
        st.info(
            "Nu există exerciții cu chei compatibile cu verificarea automată. "
            "Poți explora banca în fila Acasă; controalele Tutorului nu sunt disponibile."
        )
    else:
        preferred_domain = (
            "Funcții" if "Funcții" in gradable_domains
            else next(domain for domain in tutor_domains if domain in gradable_domains)
        )
        if st.session_state.get("tutor_domain") not in tutor_domains:
            st.session_state.tutor_domain = preferred_domain
        st.selectbox("Alege domeniul", tutor_domains, key="tutor_domain")
        domain_rows = automatically_gradable_data[
            automatically_gradable_data["Domeniu"] == st.session_state.tutor_domain
        ]
        active_index = st.session_state.selected_exercise_idx
        if active_index is not None and active_index not in domain_rows.index:
            st.session_state.selected_exercise_idx = None
            reset_exercise_state()
        if domain_rows.empty:
            st.info(
                "Nu există exerciții care pot fi evaluate automat în acest domeniu. "
                "Alege un alt domeniu sau explorează exercițiile în fila Acasă."
            )
        else:
            active_index = st.session_state.selected_exercise_idx
            if active_index is None:
                active_index = 460 if 460 in domain_rows.index else int(domain_rows.index[0])
            selector_key = (
                f"tutor_exercise_choice_{st.session_state.tutor_domain}_"
                f"{st.session_state.tutor_selection_generation}"
            )
            selected_index = st.selectbox(
                "Alege un exercițiu", domain_rows.index.tolist(),
                index=domain_rows.index.tolist().index(active_index),
                format_func=formatted_exercise,
                key=selector_key,
            )
            action_cols = st.columns(2)
            if action_cols[0].button("▶ Începe exercițiul selectat", type="primary", key="start_selected"):
                st.session_state.selected_exercise_idx = int(selected_index)
                reset_exercise_state()
                st.rerun()
            if action_cols[1].button("Alege automat", key="start_automatic"):
                candidates = domain_rows.drop(index=selected_index, errors="ignore")
                if candidates.empty:
                    candidates = domain_rows
                next_index = int(
                    candidates.sample(1, random_state=len(st.session_state.interaction_log) + 42).index[0]
                )
                st.session_state.selected_exercise_idx = next_index
                st.session_state.tutor_selection_generation += 1
                reset_exercise_state()
                st.rerun()

            exercise_index = st.session_state.selected_exercise_idx
            if exercise_index is None or exercise_index not in automatically_gradable_data.index:
                st.info("Alege un exercițiu pentru a începe practica.")
            else:
                current = data.loc[exercise_index]
                st.divider()
                st.subheader("Problemă")
                st.write(current["Problema"])
                st.caption(f"{current['Domeniu']} · {current['Tema_norm']} · {current['Dificultate_group']}")
                st.caption(
                    f"Încercări verificate: {st.session_state.current_exercise_attempt_count} · "
                    f"indicii: {st.session_state.current_exercise_hint_count} · "
                    f"greșeli: {st.session_state.current_exercise_mistake_count} · "
                    f"erori consecutive: {st.session_state.current_exercise_consecutive_errors}"
                )
                run_id = st.session_state.tutor_run_id
                answer = st.text_area("Scrie răspunsul tău", key=f"tutor_answer_{run_id}_{exercise_index}")
                st.info(METACOGNITIVE_QUESTIONS[(int(exercise_index) + st.session_state.current_exercise_hint_count) % len(METACOGNITIVE_QUESTIONS)])
                hint_column, submit_column = st.columns(2)
                with hint_column:
                    if st.button("💡 Cere un indiciu", key=f"tutor_hint_{run_id}", disabled=st.session_state.current_exercise_completed):
                        st.session_state.current_exercise_hint_count += 1
                        st.session_state.tutor_latest_hint = choose_hint(
                            str(current["Problema"]), current.get("Pasii de rezolvare", ""),
                            st.session_state.mastery, st.session_state.current_exercise_hint_count,
                            int(exercise_index), str(current.get("problem_hash", "")),
                        )
                        st.rerun()
                if st.session_state.tutor_latest_hint:
                    st.info(st.session_state.tutor_latest_hint["hint"])
                    if st.session_state.tutor_latest_hint["shown_solution_steps"]:
                        st.caption(f"Pași ai soluției afișați: {st.session_state.tutor_latest_hint['shown_solution_steps']}")
                with submit_column:
                    if st.button(
                        "✓ Verifică răspunsul", type="primary", key=f"tutor_submit_{run_id}",
                        disabled=st.session_state.current_exercise_completed,
                    ):
                        verdict = evaluate_answer(answer, current.get("Raspunsul"))
                        st.session_state.tutor_feedback = verdict
                        st.session_state.tutor_error = None
                        if answer_result_is_verified(verdict):
                            process_verified_tutor_answer(verdict, current, int(exercise_index))
                        st.rerun()

                if st.session_state.tutor_feedback:
                    display_answer_feedback(st.session_state.tutor_feedback)
                if st.session_state.tutor_error:
                    st.error(st.session_state.tutor_error)
                analysis = st.session_state.tutor_analysis
                if analysis:
                    if (
                        st.session_state.tutor_feedback
                        and not answer_result_is_verified(st.session_state.tutor_feedback)
                    ):
                        st.caption("Analiza de mai jos aparține ultimei încercări verificate, nu răspunsului neverificat curent.")
                    with st.expander("Analiza răspunsului", expanded=True):
                        st.write(f"Timp estimat în această problemă: {analysis['elapsed']} s")
                        st.write(f"Domeniu clasificat din text: **{analysis['domain']}**")
                        st.write(f"Dificultate estimată din metadate: **{analysis['difficulty']}**")
                        st.write(f"Indicator rule-based, estimare aproximativă: **{analysis['mastery']:.2f}**")
                        st.write(f"Feedback: {analysis['learning_state']['state']}. {analysis['learning_state']['intervention']}")
                        st.caption(f"Dată orientativă de revizuire: {analysis['review_date']} (nu este salvată și nu trimite mementouri).")
                        if analysis["neural"]:
                            st.caption(f"Predicție neurală experimentală: {analysis['neural']['predicted_state']}")
                        if analysis["neural_error"]:
                            st.warning(analysis["neural_error"])

                recommendation = st.session_state.tutor_recommendation
                if st.session_state.current_exercise_completed:
                    if recommendation and recommendation["index"] in automatically_gradable_data.index:
                        next_row = data.loc[recommendation["index"]]
                        st.subheader("Următorul exercițiu sugerat")
                        st.write(f"Țintă orientativă: {recommendation['target']} · domeniu: {next_row['Domeniu']}")
                        st.write(next_row["Problema"])
                        if st.button("Continuă cu exercițiul următor ➜", key=f"tutor_continue_{run_id}"):
                            st.session_state.selected_exercise_idx = recommendation["index"]
                            st.session_state.tutor_selection_generation += 1
                            reset_exercise_state()
                            st.rerun()
                    else:
                        st.info("Nu a rămas un alt exercițiu eligibil în acest domeniu.")

with tabs[2]:
    st.header("Progresul meu")
    history = st.session_state.interaction_log
    if not history:
        st.info("Nu există încă răspunsuri verificate. Începe un exercițiu în fila Tutor AI.")
    else:
        history_frame = pd.DataFrame(history)
        columns = st.columns(4)
        columns[0].metric("Încercări evaluate", len(history_frame))
        columns[1].metric("Răspunsuri corecte", int(history_frame["is_correct"].sum()))
        columns[2].metric("Exerciții completate", int(history_frame.loc[history_frame.is_correct, "bank_index"].nunique()))
        columns[3].metric("Indicator rule-based aproximativ", f"{st.session_state.mastery:.2f}")
        st.caption("Indicator global orientativ, bazat pe reguli; nu este o probabilitate calibrată.")
        st.subheader("Încercări pe domenii")
        by_domain = history_frame.groupby("actual_domain").agg(
            Corecte=("is_correct", "sum"),
            Încercări=("is_correct", "count"),
            Timp_mediu_s=("time_spent_seconds", "mean"),
        ).round(1)
        st.dataframe(by_domain, width="stretch")
        predictions = history_frame["predicted_learning_state"].dropna()
        if not predictions.empty:
            st.subheader("Predicții neurale opționale")
            st.bar_chart(predictions.value_counts())
        st.subheader("Istoric recent")
        display = history_frame.tail(10)[
            ["problem_text", "actual_domain", "predicted_difficulty", "hint_count", "attempt_count", "is_correct"]
        ].copy()
        display["Rezultat"] = display.pop("is_correct").map({True: "Corect", False: "Incorect"})
        st.dataframe(display, width="stretch", hide_index=True)

    if st.button("🔄 Resetează progresul Tutorului", key="reset_progress"):
        st.session_state.interaction_log = []
        st.session_state.mastery = 0.55
        st.session_state.selected_exercise_idx = None
        reset_exercise_state()
        st.rerun()
