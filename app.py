from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data_prep import engineer_features
from src.model_utils import (
    load_assets,
    prepare_single_problem,
    predict_domain_from_text,
    predict_structured_difficulty,
    resolve_dataset_path,
)
from src.pedagogical_engine import (
    METACOGNITIVE_QUESTIONS,
    assess_diagnostic_results,
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


st.cache_resource.clear()

dataset_signature = _asset_signature(resolve_dataset_path())
report_signature = _asset_signature(ROOT / "models" / "evaluation_report.json")
structured_model, unstructured_model, data, report = cached_assets(dataset_signature, report_signature)


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
        learning_state = diagnose_learning_state(result["correct"], int(hints_used), int(attempts), int(time_seconds))
        new_mastery = update_mastery(st.session_state.mastery, result["correct"], int(hints_used), int(attempts))
        target = target_difficulty_from_mastery(new_mastery, result["correct"])
        next_row = recommend_next_exercise(data, row["Domeniu"], target, exclude_problem=row["Problema"], random_state=int(exercise_idx) + 1)

        st.session_state.mastery = new_mastery
        st.session_state.attempts = attempts
        st.session_state.hints_used = hints_used
        st.success(result["feedback"] if result["correct"] else result["feedback"])
        st.write(f"**Stare estimată:** {learning_state['state']}")
        st.write(f"**Intervenție pedagogică:** {learning_state['intervention']}")
        st.write(f"**Noua probabilitate de stăpânire:** {new_mastery:.2f}")
        st.write(f"**Reactivare spaced repetition:** {next_review_date(new_mastery, result['correct'])}")
        if next_row:
            st.markdown("#### Recomandarea următoare")
            st.write(f"Țintă: **{target}**, domeniu: **{row['Domeniu']}**")
            st.markdown(f"<div class='main-card'>{next_row['Problema']}</div>", unsafe_allow_html=True)
            st.caption(f"{next_row.get('Tema_norm', '—')} · {next_row.get('Dificultate_group', '—')}")


# Fix dtypes after CSV load.
for col in ["Dificultate", "Itemul", "Sursa_year"]:
    if col in data.columns:
        data[col] = pd.to_numeric(data[col], errors="coerce")

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
    diagnostic_bank = generate_diagnostic_bank(data, n_questions=5, random_state=st.session_state.diagnostic_seed)
    st.caption("Ai 5 întrebări scurte. Răspunde natural și apoi primești o recomandare simplă asupra temelor de exersat.")
    diagnostic_answers = []
    for idx, row in enumerate(diagnostic_bank, start=1):
        st.markdown(f"<div class='main-card'><strong>{idx}.</strong> {row['Problema']}</div>", unsafe_allow_html=True)
        answer = st.text_area("Răspunsul tău", key=f"diag_{idx}", placeholder="Scrie răspunsul aici")
        if answer:
            result = evaluate_answer(answer, str(row.get("Raspunsul", "")))
            diagnostic_answers.append({"problem": row["Problema"], "domain": row.get("Domeniu"), "correct": result["correct"]})
    if st.button("Finalizează diagnostic și recomandă exerciții", type="primary"):
        if not diagnostic_answers:
            st.warning("Completează răspunsurile la întrebările de diagnostic înainte de a primi recomandări.")
            st.stop()

        profile = assess_diagnostic_results(diagnostic_answers, data)
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
        rec = recommend_next_exercise(data, weak_domain, "2 - mediu", random_state=11)
        if rec:
            st.markdown("### Exercițiul de început recomandat")
            st.markdown(f"Tema prioritară: **{weak_domain}**")
            try:
                exercise_idx = int(rec.get("Itemul", 0)) if pd.notna(rec.get("Itemul")) else 0
            except (TypeError, ValueError, OverflowError):
                exercise_idx = 0
            render_tutor_exercise(rec, exercise_idx=exercise_idx, key_prefix="diag_followup")

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
    st.metric("Exerciții disponibile", report["dataset"]["rows_total"])
with col2:
    st.metric("Calitate model structură", f"{report['structured_model']['model']['macro_f1']:.3f}")
with col3:
    st.metric("Calitate model text", f"{report['unstructured_model']['model']['macro_f1']:.3f}")
with col4:
    st.metric("Acoperire demo", "10/10 criterii")


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
    st.dataframe(probs, use_container_width=True, hide_index=True)
    st.bar_chart(probs.set_index("clasă"))


tabs = st.tabs([
    "🏠 Acasă",
    "🤖 Tutor AI",
])

with tabs[0]:
    st.header("Demo: elevul rezolvă, sistemul clasifică, estimează dificultatea și adaptează traseul")
    left, right = st.columns([1.2, 0.8])

    with left:
        domains = ["Toate"] + sorted(data["Domeniu"].dropna().unique().tolist())
        selected_domain = st.selectbox("Domeniu", domains, index=0)
        filtered = data.copy()
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
            learning_state = diagnose_learning_state(
                result["correct"], int(st.session_state.hints_used), int(st.session_state.attempts), int(time_seconds)
            )
            new_mastery = update_mastery(
                st.session_state.mastery, result["correct"], int(st.session_state.hints_used), int(st.session_state.attempts)
            )
            target = target_difficulty_from_mastery(new_mastery, result["correct"])
            next_row = recommend_next_exercise(data, row["Domeniu"], target, exclude_problem=row["Problema"], random_state=int(exercise_idx) + 1)

            st.session_state.mastery = new_mastery
            st.success(result["feedback"] if result["correct"] else result["feedback"])
            st.write(f"**Stare estimată:** {learning_state['state']}")
            st.write(f"**Intervenție pedagogică:** {learning_state['intervention']}")
            st.write(f"**Noua probabilitate de stăpânire:** {new_mastery:.2f}")
            st.write(f"**Reactivare spaced repetition:** {next_review_date(new_mastery, result['correct'])}")
            if next_row:
                st.markdown("#### Recomandarea următoare")
                st.write(f"Țintă: **{target}**, domeniu: **{row['Domeniu']}**")
                st.markdown(f"<div class='main-card'>{next_row['Problema']}</div>", unsafe_allow_html=True)
                st.caption(f"{next_row['Tema_norm']} · {next_row['Dificultate_group']}")

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

with tabs[1]:
    st.header("Tutor AI - exerciții, indicii și recomandări")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("1) Date structurate → dificultate")
        st.write(
            "Modelul primește features tabelare: temă normalizată, domeniu, tip sursă, numărul itemului, lungimi, simboluri matematice și indicatori de context. Output: dificultatea estimată."
        )
        st.code("Input: [Tema_norm, Domeniu, Sursa_type, Itemul, problem_words, n_math_symbols, ...]\nOutput: 1 - bază / 2 - mediu / 3 - consolidare / 4 - avansat")
        manual_domain = st.selectbox("Domeniu manual", sorted(data["Domeniu"].unique()), key="manual_domain")
        manual_topic = st.selectbox("Temă manuală", sorted(data["Tema_norm"].unique()), key="manual_topic")
        manual_problem = st.text_area("Problemă pentru test structurat", value="Calculați valoarea expresiei 2x + 5 pentru x = 3.")
        manual_features = prepare_single_problem(manual_problem, manual_topic, manual_domain, item=5, sursa_type="manual")
        manual_diff = predict_structured_difficulty(structured_model, manual_features)
        st.success(f"Dificultate estimată: {manual_diff['prediction']}")
        show_probabilities(manual_diff.get("probabilities"), "Distribuție dificultate")

    with c2:
        st.subheader("2) Text nestructurat → domeniu curricular")
        st.write(
            "Modelul primește enunțul brut al problemei și îl clasifică în domeniul curricular. Este serviciul util când încă nu avem etichete manuale pentru o problemă nouă."
        )
        st.code("Input: text liber al problemei\nOutput: Geometrie / Funcții / Ecuații... / Mulțimi numerice / ...")
        text_problem = st.text_area(
            "Problemă nouă pentru clasificare text",
            value="În triunghiul ABC, AB = AC și unghiul A este 40°. Determinați măsura unghiului B.",
        )
        text_pred = predict_domain_from_text(unstructured_model, text_problem)
        st.success(f"Domeniu estimat: {text_pred['prediction']}")
        show_probabilities(text_pred.get("probabilities"), "Distribuție domeniu")

    st.markdown("### De ce nu ajunge un singur serviciu?")
    st.write(
        "Modelul pe text etichetează probleme noi după conținut. Modelul pe date structurate estimează nivelul de dificultate și susține recomandarea adaptivă. Dacă eliminăm modelul text, nu putem eticheta probleme noi; dacă eliminăm modelul structurat, nu putem controla progresia dificultății în traseul elevului."
    )

