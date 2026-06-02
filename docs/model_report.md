# Raport model Didact AI

## Dataset

{
  "rows_total": 489,
  "rows_with_difficulty_and_topic": 477,
  "exact_duplicate_problem_rows": 51,
  "missing_by_column": {
    "Sursa": 48,
    "Itemul": 48,
    "Problema": 0,
    "Pasii de rezolvare": 0,
    "Raspunsul": 0,
    "Dificultate": 12,
    "Tema": 12,
    "Tema_norm": 0,
    "Domeniu": 0,
    "Dificultate_group": 0,
    "Sursa_type": 0,
    "Sursa_year": 48,
    "Problema_clean": 0,
    "Pasi_clean": 0,
    "Raspuns_clean": 0,
    "problem_chars": 0,
    "problem_words": 0,
    "steps_chars": 0,
    "answer_chars": 0,
    "n_digits": 0,
    "n_math_symbols": 0,
    "has_percent": 0,
    "has_geometry_word": 0,
    "has_equation_word": 0,
    "has_radical": 0,
    "has_function_word": 0,
    "has_real_life_context": 0,
    "problem_hash": 0
  },
  "difficulty_distribution": {
    "2 - mediu": 256,
    "1 - bază": 91,
    "3 - consolidare": 88,
    "4 - avansat": 42
  },
  "domain_distribution": {
    "Geometrie": 124,
    "Ecuații, inecuații și sisteme": 113,
    "Mulțimi numerice": 82,
    "Funcții": 77,
    "Rapoarte și proporții": 44,
    "Calcul algebric": 32,
    "Altele": 17
  },
  "topic_top_20": {
    "Geometrie 3D": 43,
    "Mulțimi numerice": 41,
    "Ecuații de gradul II": 38,
    "Funcția de gradul II": 36,
    "Calcul algebric": 32,
    "Procente": 32,
    "Sisteme de ecuații": 31,
    "Geometrie - Triunghiuri": 28,
    "Radicali": 28,
    "Inecuații": 25,
    "Funcții": 22,
    "Ecuații": 19,
    "Funcții liniare": 19,
    "Geometrie - Cercul": 16,
    "Puteri": 13,
    "Necunoscut": 12,
    "Rapoarte și proporții": 12,
    "Geometrie - Trapeze": 9,
    "Geometrie": 8,
    "Geometrie - Paralelograme": 8
  },
  "problem_length_words": {
    "min": 2.0,
    "median": 13.0,
    "mean": 16.222903885480573,
    "max": 78.0
  },
  "dataset_decision": "Usable for a strong competition MVP because it contains Romanian math problem text, solutions, topic labels, and difficulty labels. Not production-grade: it is small, topic labels were noisy before normalization, and there are exact duplicates/missing labels; the code explicitly cleans these and reports the limitations."
}

## Model structurat

- Task: Structured difficulty classification
- Target: Dificultate_group
- Rows train/test: 332/94
- Baseline: {"accuracy": 0.5638297872340425, "balanced_accuracy": 0.25, "macro_f1": 0.18027210884353742, "weighted_f1": 0.40657113909393544}
- Model: {"accuracy": 0.7553191489361702, "balanced_accuracy": 0.7296258125891867, "macro_f1": 0.6860119047619048, "weighted_f1": 0.7580420466058764}
- Best params: {"clf__class_weight": "balanced", "clf__max_depth": null, "clf__min_samples_leaf": 3, "clf__n_estimators": 120}
- Best CV macro-F1: 0.716

## Model nestructurat

- Task: Unstructured text domain classification
- Target: Domeniu
- Rows train/test: 329/93
- Baseline: {"accuracy": 0.26881720430107525, "balanced_accuracy": 0.16666666666666666, "macro_f1": 0.07062146892655367, "weighted_f1": 0.1139055950428285}
- Model: {"accuracy": 0.8924731182795699, "balanced_accuracy": 0.8685858585858587, "macro_f1": 0.8764292031573139, "weighted_f1": 0.8934701985987848}
- Best params: {"clf__alpha": 0.2, "tfidf__max_features": 6000, "tfidf__min_df": 1, "tfidf__ngram_range": [1, 2]}
- Best CV macro-F1: 0.873

## Erori reprezentative

### Structurat
[
  {
    "problem": "Funcția f(x) = x² - 4x + 4. Câte puncte de intersecție cu axa Ox?",
    "true": "2 - mediu",
    "predicted": "1 - bază",
    "tema": "Funcția de gradul II",
    "domeniu": "Funcții",
    "difficulty": "2 - mediu"
  },
  {
    "problem": "Fie a = 0,5 : 1/4 si b = -11 + 5. Completati casetele: a =?, b =?, b/a=?",
    "true": "3 - consolidare",
    "predicted": "1 - bază",
    "tema": "Mulțimi numerice",
    "domeniu": "Mulțimi numerice",
    "difficulty": "3 - consolidare"
  },
  {
    "problem": "Fie E(x) = (x−3)² − x(x−4). Determinați E(√2).",
    "true": "2 - mediu",
    "predicted": "3 - consolidare",
    "tema": "Calcul algebric",
    "domeniu": "Calcul algebric",
    "difficulty": "2 - mediu"
  },
  {
    "problem": "Fie functia f:R->R, f(x) = 2x + m - 1, m apartine R. Graficul functiei f intersecteaza axa Oy intr-un punct cu ordonata egala cu -3. Determinati zeroul functiei f.",
    "true": "3 - consolidare",
    "predicted": "4 - avansat",
    "tema": "Funcții",
    "domeniu": "Funcții",
    "difficulty": "3 - consolidare"
  },
  {
    "problem": "Fie functia f:R->R, f(x) = 2ax - 9, a ≠ 0. Determinati valorile reale ale lui a, pentru care graficul functiei f trece prin punctul A(a; a^2) si functia f este monoton descrescatoare.",
    "true": "3 - consolidare",
    "predicted": "4 - avansat",
    "tema": "Funcții",
    "domeniu": "Funcții",
    "difficulty": "3 - consolidare"
  },
  {
    "problem": "Un tractorist are de arat un teren de 116 hectare. In primele 6 zile a arat 87 hectare. Determinati in cate zile tractoristul va ara suprafata ramasa.",
    "true": "3 - consolidare",
    "predicted": "4 - avansat",
    "tema": "Rapoarte și proporții",
    "domeniu": "Rapoarte și proporții",
    "difficulty": "3 - consolidare"
  },
  {
    "problem": "Baza triunghiului isoscel cu A=60cm² și h=12cm",
    "true": "2 - mediu",
    "predicted": "3 - consolidare",
    "tema": "Geometrie - Arii",
    "domeniu": "Geometrie",
    "difficulty": "2 - mediu"
  },
  {
    "problem": "Fie funcția f: R→R, f(x) = -3x + 2. Scrieți în casetă un număr real, astfel încât punctul (__, 5) să aparțină graficului funcției f.",
    "true": "2 - mediu",
    "predicted": "1 - bază",
    "tema": "Funcții liniare",
    "domeniu": "Funcții",
    "difficulty": "2 - mediu"
  }
]

### Text
[
  {
    "problem": "Maria: 12 bătăi ale inimii în 10 s. Bătăi pe minut.",
    "true": "Rapoarte și proporții",
    "predicted": "Ecuații, inecuații și sisteme",
    "tema": "Rapoarte și proporții",
    "domeniu": "Rapoarte și proporții",
    "difficulty": "2 - mediu"
  },
  {
    "problem": "Ion si Maria au primit de la parinti sume egale de bani pentru a cumpara martisoare. Ion a cumparat martisoare de cate 8 lei fiecare si i-au ramas 3 lei, iar Maria a cumparat martisoare de cate 6 lei fiecare si i-a ramas 1 leu. Determinati ",
    "true": "Ecuații, inecuații și sisteme",
    "predicted": "Rapoarte și proporții",
    "tema": "Sisteme de ecuații",
    "domeniu": "Ecuații, inecuații și sisteme",
    "difficulty": "2 - mediu"
  },
  {
    "problem": "Barcă cu saci de cartofi: 20 saci → 1200 kg, 25 saci → 1425 kg. Greutatea bărcii și a unui sac?",
    "true": "Ecuații, inecuații și sisteme",
    "predicted": "Rapoarte și proporții",
    "tema": "Sisteme de ecuații",
    "domeniu": "Ecuații, inecuații și sisteme",
    "difficulty": "3 - consolidare"
  },
  {
    "problem": "Monitor și imprimantă: 4200 lei. După reduceri (200 lei monitor, 50% imprimantă): 2750 lei. Prețurile inițiale.",
    "true": "Ecuații, inecuații și sisteme",
    "predicted": "Rapoarte și proporții",
    "tema": "Sisteme de ecuații",
    "domeniu": "Ecuații, inecuații și sisteme",
    "difficulty": "2 - mediu"
  },
  {
    "problem": "Cost transport tonă pe x km: auto f(x)=(3/500)x+7, tren g(x)=(7/500)x+3. De la ce distanță trenul e mai ieftin?",
    "true": "Funcții",
    "predicted": "Ecuații, inecuații și sisteme",
    "tema": "Funcții liniare",
    "domeniu": "Funcții",
    "difficulty": "2 - mediu"
  },
  {
    "problem": "Un agent cumpără un televizor la 7500 lei. Achită suplimentar 5% pentru transport și 200 lei pentru depozitare. Cu câți lei mai mult achită pentru transport decât pentru depozitare?",
    "true": "Rapoarte și proporții",
    "predicted": "Ecuații, inecuații și sisteme",
    "tema": "Procente",
    "domeniu": "Rapoarte și proporții",
    "difficulty": "2 - mediu"
  },
  {
    "problem": "Fie functia f:R->R, f(x) = -2x + 10. Determinati valorile lui x, care sunt patrate perfecte si pentru care f(x) + f(2) >= x + 2.",
    "true": "Ecuații, inecuații și sisteme",
    "predicted": "Funcții",
    "tema": "Inecuații",
    "domeniu": "Ecuații, inecuații și sisteme",
    "difficulty": "3 - consolidare"
  },
  {
    "problem": "Simplificați (X³−3X²−X+3)/(9−X²), X∈ℝ\\{-3,3}.",
    "true": "Calcul algebric",
    "predicted": "Ecuații, inecuații și sisteme",
    "tema": "Calcul algebric",
    "domeniu": "Calcul algebric",
    "difficulty": "3 - consolidare"
  }
]
