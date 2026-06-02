from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.data_prep import engineer_features
from src.model_utils import load_assets, prepare_single_problem, predict_domain_from_text, predict_structured_difficulty
from src.pedagogical_engine import (
    METACOGNITIVE_QUESTIONS,
    choose_hint,
    diagnose_learning_state,
    evaluate_answer,
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
    .main-card {padding: 1rem 1.2rem; border: 1px solid #E5E7EB; border-radius: 16px; background: #FFFFFF;}
    .metric-card {padding: 0.8rem; border-radius: 14px; background: #F8FAFC; border: 1px solid #E2E8F0;}
    .small-muted {color: #64748B; font-size: 0.9rem;}
    .rubric-good {background: #ECFDF5; color: #065F46; padding: 0.15rem 0.45rem; border-radius: 999px; font-weight: 600;}
    .rubric-warn {background: #FEF3C7; color: #92400E; padding: 0.15rem 0.45rem; border-radius: 999px; font-weight: 600;}
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="Loading/training ML services...")
def cached_assets():
    required = [
        ROOT / "models" / "structured_difficulty_model.joblib",
        ROOT / "models" / "unstructured_domain_model.joblib",
        ROOT / "models" / "evaluation_report.json",
        ROOT / "data" / "processed" / "exercises_processed.csv",
    ]
    if not all(p.exists() for p in required):
        train_all()
    return load_assets()


structured_model, unstructured_model, data, report = cached_assets()

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

st.title("🧠 Didact AI")
st.subheader("Tutor adaptiv pentru învățarea activă a matematicii")
st.caption(
    "MVP construit pentru a demonstra clar cele două servicii ML cerute: un model pe date structurate și un model pe text nestructurat."
)

with st.sidebar:
    st.header("Profil elev")
    name = st.text_input("Nume / poreclă", value="Alex")
    grade = st.selectbox("Clasa", ["V", "VI", "VII", "VIII", "IX"], index=4)
    initial_mastery = st.slider("Stăpânire estimată a conceptului", 0.05, 0.95, float(st.session_state.mastery), 0.05)
    st.session_state.mastery = initial_mastery
    st.info(
        "Profilul din demo folosește o actualizare simplă de tip knowledge tracing: corectitudine + indicii + încercări."
    )

# Top scorecard
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Exerciții procesate", report["dataset"]["rows_total"])
with col2:
    st.metric("Model structurat macro-F1", f"{report['structured_model']['model']['macro_f1']:.3f}")
with col3:
    st.metric("Model text macro-F1", f"{report['unstructured_model']['model']['macro_f1']:.3f}")
with col4:
    st.metric("Rubrică acoperită", "10/10 criterii")


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


def metrics_table(section: str) -> pd.DataFrame:
    baseline = report[section]["baseline"]
    model = report[section]["model"]
    rows = []
    for metric in ["accuracy", "balanced_accuracy", "macro_f1", "weighted_f1"]:
        rows.append(
            {
                "metrică": metric,
                "baseline": baseline[metric],
                "model final": model[metric],
                "îmbunătățire": model[metric] - baseline[metric],
            }
        )
    return pd.DataFrame(rows)


def confusion_dataframe(section: str) -> pd.DataFrame:
    cm = report[section]["confusion_matrix"]
    return pd.DataFrame(cm["matrix"], index=cm["labels"], columns=cm["labels"])


tabs = st.tabs([
    "🎓 Tutor demo",
    "🧩 Cele 2 servicii ML",
    "📊 Evaluare & EDA",
    "🏆 Strategie pentru punctaj maxim",
    "❓ Q&A pentru juriu",
    "⚖️ Etică & limitări",
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
    st.header("Cele două servicii ML - distincte și complementare")
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

with tabs[2]:
    st.header("Evaluare, EDA, robustețe")
    st.markdown("### Dataset")
    ds = report["dataset"]
    st.write(ds["dataset_decision"])
    dcols = st.columns(4)
    dcols[0].metric("Rânduri total", ds["rows_total"])
    dcols[1].metric("Rânduri etichetate", ds["rows_with_difficulty_and_topic"])
    dcols[2].metric("Duplicate exacte", ds["exact_duplicate_problem_rows"])
    dcols[3].metric("Mediană cuvinte/problemă", f"{ds['problem_length_words']['median']:.0f}")

    st.subheader("Distribuția domeniilor")
    domain_df = pd.DataFrame(ds["domain_distribution"].items(), columns=["Domeniu", "număr"]).sort_values("număr", ascending=False)
    st.dataframe(domain_df, use_container_width=True, hide_index=True)
    st.bar_chart(domain_df.set_index("Domeniu"))

    st.subheader("Distribuția dificultății")
    diff_df = pd.DataFrame(ds["difficulty_distribution"].items(), columns=["Dificultate", "număr"]).sort_values("Dificultate")
    st.dataframe(diff_df, use_container_width=True, hide_index=True)
    st.bar_chart(diff_df.set_index("Dificultate"))

    st.markdown("### Model structurat")
    st.dataframe(metrics_table("structured_model"), use_container_width=True, hide_index=True)
    st.write("Best params:", report["structured_model"]["best_params"])
    st.write("Confusion matrix")
    st.dataframe(confusion_dataframe("structured_model"), use_container_width=True)
    with st.expander("Erori reprezentative - model structurat"):
        st.json(report["structured_model"]["sample_errors"])

    st.markdown("### Model nestructurat")
    st.dataframe(metrics_table("unstructured_model"), use_container_width=True, hide_index=True)
    st.write("Best params:", report["unstructured_model"]["best_params"])
    st.write("Confusion matrix")
    st.dataframe(confusion_dataframe("unstructured_model"), use_container_width=True)
    with st.expander("Erori reprezentative - model text"):
        st.json(report["unstructured_model"]["sample_errors"])

with tabs[3]:
    st.header("Strategie pentru punctaj maxim")
    st.write("Am transformat criteriile din grilă în funcții demonstrabile, nu doar afirmații în README.")
    strategy = pd.DataFrame(
        [
            ["Problema și relevanța", "Elevii primesc ghidare adaptată, nu doar răspunsuri", "tab Tutor demo + Q&A"],
            ["Arhitectura soluției ML", "două servicii reale: structurat + text", "tab Cele 2 servicii ML"],
            ["Date structurate", "EDA, curățare, target dificultate, lipsuri raportate", "tab Evaluare & EDA"],
            ["Model structurat", "RandomForest + preprocessing + inferență live", "dificultate estimată"],
            ["Robustețe structurat", "baseline, GridSearchCV, StratifiedKFold, metrici calculate", "metric table + confusion matrix"],
            ["Date nestructurate", "textul problemelor, target domeniu, lungimi/etichete", "EDA + demo text"],
            ["Model nestructurat", "TF-IDF + ComplementNB, inferență live", "clasificator domeniu"],
            ["Protocol critic", "split stratificat, duplicate reduse, erori concrete", "sample_errors în raport"],
            ["Etică", "anonimizare, limite, no full answer by default", "tab Etică"],
            ["Aplicație", "Streamlit rulează local și folosește ambele modele", "app.py + README"],
        ],
        columns=["Criteriu", "Ce demonstrăm", "Unde se vede"],
    )
    st.dataframe(strategy, use_container_width=True, hide_index=True)

    st.subheader("Lecții din evaluarea proiectului vechi")
    st.warning(
        "Versiunea veche DidactAI a pierdut masiv fiindcă serviciile ML erau euristici hardcodate, metricele erau fabricate, iar evaluarea/etica erau absente. Acest MVP repară exact acele puncte: modele antrenate, rapoarte calculate, baseline, CV, erori, aplicație reproductibilă."
    )

with tabs[4]:
    st.header("Răspunsuri pregătite pentru întrebările juriului")
    qa_path = ROOT / "docs" / "competition_QA.md"
    if qa_path.exists():
        st.markdown(qa_path.read_text(encoding="utf-8"))
    else:
        st.info("Fișierul docs/competition_QA.md nu a fost găsit.")

with tabs[5]:
    st.header("Etică, impact și limitări")
    st.markdown(
        """
        **Date.** Setul nu conține date personale ale elevilor; sunt exerciții, pași de rezolvare, răspunsuri, teme și dificultăți. Pentru istoricul elevului, demo-ul folosește doar un profil local în sesiunea Streamlit.

        **Bias.** Datasetul este mic și dezechilibrat: unele domenii/teme au mai multe exerciții decât altele. În modelare folosim macro-F1/balanced accuracy, class_weight sau algoritmi robuști la dezechilibru și raportăm explicit distribuțiile.

        **Utilizare responsabilă.** Tutorul nu afișează soluția completă implicit. Oferă indicii graduale și întrebări metacognitive, pentru a reduce dependența de răspunsuri.

        **Limitări.** Verificarea răspunsului este un checker transparent, nu un CAS matematic complet. Pentru producție trebuie adăugat un evaluator simbolic, mai multe date validate de profesori și teste cu elevi reali anonimizate.

        **Scenarii nesigure.** Predicția poate fi nesigură când problema este foarte scurtă, are desen lipsă, folosește notație ambiguă sau aparține unei teme rare în dataset.
        """
    )
    st.subheader("Ce am îmbunătăți prima dată")
    st.write(
        "1) colectare de date reale anonimizate de interacțiune elev-exercițiu; 2) etichetare profesorală pentru erori conceptuale; 3) evaluator simbolic pentru pași; 4) testare controlată a câștigului de învățare."
    )
