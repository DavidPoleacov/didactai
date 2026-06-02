# Didact AI - tutor adaptiv de matematică

Didact AI este un MVP pentru Olimpiada Națională de Inteligență Artificială: o aplicație Streamlit care ajută elevul să exerseze matematică fără să primească automat soluția completă. Sistemul combină reguli pedagogice controlabile cu două servicii ML reale și demonstrabile.

## Decizia despre dataset

Datasetul furnizat este **utilizabil pentru o demonstrație competitivă**, dar nu este încă suficient pentru producție. Are 489 exerciții, 477 rânduri cu temă + dificultate, 51 duplicate exacte și 12 rânduri fără etichete. Nu recomand înlocuirea lui acum, pentru că este în limba română, este aliniat la matematica de examen și conține exact câmpurile necesare pentru cele două servicii ML: textul problemei, tema și dificultatea. Pentru versiunea următoare, cel mai bun dataset ar fi o colecție extinsă din arhive oficiale + etichetare profesorală pentru competențe și erori conceptuale.

## Cele două servicii ML

### 1. Serviciu pe date structurate

- **Input:** `Tema_norm`, `Domeniu`, `Sursa_type`, `Itemul`, anul sursei, lungimi, număr de simboluri matematice, indicatori de procente/geometrie/ecuații/radicali/funcții/context real.
- **Output:** `Dificultate_group`: `1 - bază`, `2 - mediu`, `3 - consolidare`, `4 - avansat`.
- **Model:** `RandomForestClassifier` în pipeline cu imputare, scalare și OneHotEncoder.
- **Evaluare:** macro-F1 0.686, baseline macro-F1 0.180.

### 2. Serviciu pe date nestructurate

- **Input:** enunț brut de problemă matematică.
- **Output:** domeniu curricular: Geometrie, Funcții, Ecuații/Inecuații/Sisteme, Mulțimi numerice etc.
- **Model:** TF-IDF + ComplementNB.
- **Evaluare:** macro-F1 0.876, baseline macro-F1 0.071.

Cele două servicii sunt complementare: modelul text etichetează probleme noi, iar modelul structurat controlează progresia dificultății și recomandarea adaptivă.

## Rulare rapidă

```bash
pip install -r requirements.txt
streamlit run app.py
```

Aplicația folosește modelele deja salvate în `/models`. Pentru reantrenare:

```bash
python -m src.train_models
```

## Structura proiectului

```text
app.py                              # UI Streamlit
src/data_prep.py                    # curățare, normalizare, feature engineering
src/train_models.py                 # antrenare + baseline + GridSearchCV + evaluare
src/model_utils.py                  # încărcare modele + inferență
src/pedagogical_engine.py           # indicii, mastery update, recomandări
data/raw/                           # fișierele .xlsx furnizate
data/processed/exercises_processed.csv
models/structured_difficulty_model.joblib
models/unstructured_domain_model.joblib
models/evaluation_report.json
docs/competition_QA.md              # răspunsuri pregătite pentru juriu
```

## Ce demonstrează aplicația

1. Elevul alege o problemă și introduce un răspuns.
2. Modelul text prezice domeniul curricular din enunț.
3. Modelul structurat prezice dificultatea.
4. Motorul pedagogic oferă indiciu gradual și întrebare de conștientizare.
5. Sistemul actualizează stăpânirea estimată și recomandă următorul exercițiu.
6. Taburile de evaluare arată baseline, metrici, tuning, confuzii și erori concrete.

## Lecții din proiectul vechi / greșeli evitate

Versiunea veche DidactAI a fost penalizată deoarece „serviciile ML” erau euristici hardcodate, metricele erau constante fabricate, nu exista split, baseline, tuning sau analiză de erori, iar etica era absentă. Acest proiect remediază direct acele probleme: modelele sunt antrenate, metricile sunt calculate în `evaluation_report.json`, iar aplicația folosește efectiv ambele servicii.

## Limitări oneste

- Nu avem încă istoric real de elevi, deci knowledge tracing-ul din demo este o actualizare transparentă, nu model secvențial LSTM.
- Verificarea răspunsului este un checker simplu, nu un evaluator simbolic complet.
- Datasetul este mic și dezechilibrat; de aceea raportăm macro-F1 și balanced accuracy, nu doar accuracy.
- Problemele cu imagini/diagrame lipsă pot fi clasificate incorect.

## Etică

Datasetul conține exerciții, nu date personale. Demo-ul nu persistă date personale. Sistemul nu oferă soluția completă implicit și comunică faptul că predicțiile sunt suport educațional, nu verdict final.
