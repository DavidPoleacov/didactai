# Didact AI - răspunsuri pregătite pentru juriu

## 1. Problema și relevanța

**Ce problemă concretă rezolvăm?**  
Rezolvăm lipsa de ghidare personalizată la matematică. Elevul primește de obicei un răspuns corect/greșit, nu un traseu adaptat la ce concepte stăpânește, unde se blochează și cât ajutor cere.

**Cine este utilizatorul final?**  
Elevul de gimnaziu/liceu care exersează matematică, iar beneficiar indirect este profesorul, care poate vedea ce teme sunt fragile și ce tipuri de intervenții ajută.

**De ce este important în practică?**  
Matematica cere pași, justificare și transfer. Un sistem care dă imediat soluția poate crea dependență; Didact AI oferă indicii graduale și întrebări de conștientizare.

**Ce obiectiv măsurabil urmărim?**  
Obiectivul MVP: clasificarea automată a domeniului curricular din text și estimarea dificultății din date structurate, apoi folosirea lor într-o recomandare adaptivă. Măsurăm macro-F1, accuracy, balanced accuracy, comparație cu baseline, rata de erori și demonstrație live în Streamlit.

**Cum știm că proiectul a avut succes?**  
Succesul pentru olimpiadă este: aplicație funcțională, ambele modele integrate, metrici calculate reproductibil, baseline depășit, EDA și limitări afișate. În raportul curent, modelul structurat are macro-F1 0.686 față de baseline 0.180; modelul text are macro-F1 0.876 față de baseline 0.071.

## 2. Arhitectura soluției ML

**Care este serviciul ML pe date structurate?**  
Serviciul `structured_difficulty_model.joblib`: prezice dificultatea exercițiului pe baza metadatelor și feature-urilor tabelare.

**Care este serviciul ML pe date nestructurate?**  
Serviciul `unstructured_domain_model.joblib`: clasifică domeniul curricular din textul brut al problemei.

**Ce input și output are fiecare serviciu?**  
Structurat: input = tema normalizată, domeniu, tip sursă, item, lungimi, număr de simboluri, indicatori precum procente/geometrie/ecuație; output = `1 - bază`, `2 - mediu`, `3 - consolidare`, `4 - avansat`.  
Nestructurat: input = enunțul problemei ca text liber; output = domeniu curricular: Geometrie, Funcții, Ecuații/Inecuații/Sisteme, Mulțimi numerice etc.

**Sunt ambele implementate și demonstrabile?**  
Da. În Streamlit, tabul „Cele 2 servicii ML” permite inferență separată pentru fiecare, iar tabul „Tutor demo” le combină în același flux.

**Unde apar cele două servicii în proiect?**  
Modelele sunt în `/models`, antrenarea în `src/train_models.py`, preprocesarea în `src/data_prep.py`, inferența în `src/model_utils.py`, integrarea în `app.py`.

**De ce nu era suficient un singur serviciu?**  
Textul rezolvă etichetarea problemelor noi, dar nu controlează progresia dificultății. Modelul structurat estimează dificultatea, dar are nevoie de etichete/metadate. Împreună permit: problemă nouă -> domeniu -> dificultate -> traseu adaptiv.

**Ce pierdem dacă eliminăm unul?**  
Fără text, nu putem încadra automat exerciții noi. Fără structurat, recomandarea adaptivă nu mai știe cât de greu este exercițiul.

## 3. Date și preprocesare structurate

**De unde provin datele structurate?**  
Din workbook-ul furnizat `Exercises_CORRECTED (2).xlsx`, cu exerciții de matematică, pași de rezolvare, răspunsuri, dificultate și temă. Am folosit și `Domenii, categorii.xlsx` ca ghid curricular pentru maparea temelor în domenii.

**Ce reprezintă variabilele principale?**  
`Tema_norm`, `Domeniu`, `Sursa_type`, `Itemul`, `Sursa_year`, `problem_words`, `steps_chars`, `n_digits`, `n_math_symbols`, `has_percent`, `has_geometry_word`, `has_equation_word`, `has_radical`, `has_function_word`, `has_real_life_context`.

**Care este targetul?**  
`Dificultate_group`: dificultatea normalizată în patru clase: bază, mediu, consolidare, avansat.

**De ce sunt potrivite datele?**  
Pentru un tutor adaptiv avem nevoie exact de conținut matematic, etichete tematice și dificultate. Datasetul are 489 exerciții, dintre care 477 au temă și dificultate.

**Limitări?**  
Setul este mic pentru producție, are 51 duplicate exacte și 12 rânduri fără temă/dificultate. Etichetele tematice inițiale erau zgomotoase (`Functii` vs `Funcții`, `Ecuatii` vs `Ecuații`), deci le-am normalizat.

**Cum am tratat lipsurile/anomaliile?**  
Am filtrat targeturile necunoscute la antrenare, am imputat numeric cu mediană și categoric cu cea mai frecventă valoare în pipeline, am grupat dificultățile rare 4/5/6 în clasa `4 - avansat`.

**Cum am tratat categoricele și numericele?**  
Categorice: `SimpleImputer` + `OneHotEncoder(handle_unknown='ignore')`. Numerice: `SimpleImputer(strategy='median')` + `StandardScaler`. Model final: `RandomForestClassifier`.

**Ce am observat în EDA?**  
Distribuția este dezechilibrată: clasa `2 - mediu` este dominantă, iar unele domenii precum geometria sunt mai frecvente decât altele. De aceea folosim balanced accuracy și macro-F1, nu doar accuracy.

**Ce decizie de modelare a influențat EDA?**  
Am folosit macro-F1 la GridSearchCV, am grupat dificultățile rare, am folosit split stratificat și am raportat baseline-ul majoritar.

## 4. Model structurat

**Ce model am ales și de ce?**  
RandomForestClassifier într-un pipeline scikit-learn. Este potrivit pentru amestec de features numerice și categorice, poate modela interacțiuni neliniare și este robust pe dataseturi mici/medii.

**Alternative analizate?**  
Baseline `DummyClassifier`. O alternativă simplă ar fi LogisticRegression cu OneHotEncoder; am preferat RandomForest pentru interacțiuni între temă, simboluri și lungime.

**Putem demonstra inferență?**  
Da, în tabul „Cele 2 servicii ML” se introduce o problemă și metadate, iar modelul returnează dificultatea estimată și probabilitățile.

**Cum justificăm complexitatea?**  
Nu folosim deep learning pentru date structurate mici. RandomForest este suficient de puternic, rapid, explicabil la nivel de feature importance și ușor de rulat local.

## 5. Evaluare și robustețe structurate

**Metrici folosite?**  
Accuracy, balanced accuracy, macro-F1, weighted-F1, classification report și confusion matrix.

**Baseline și depășire?**  
Baseline macro-F1: 0.180. Model final macro-F1: 0.686. Baseline accuracy: 0.564. Model accuracy: 0.755.

**Tuning?**  
Da, `GridSearchCV` pe `n_estimators`, `max_depth`, `min_samples_leaf`, `class_weight`.

**Validare robustă?**  
Da, `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` pe train și test holdout stratificat.

**Anti-overfitting?**  
Limităm `max_depth`, testăm `min_samples_leaf`, folosim CV și holdout separat, iar pipeline-ul evită leakage deoarece preprocessing-ul este fit doar pe train în CV.

## 6. Date nestructurate

**Tip de date nestructurate?**  
Text: enunțuri matematice în limba română, cu notație matematică.

**De unde provin?**  
Din aceeași bancă de exerciții furnizată. Textul problemei este coloana `Problema`.

**Cum sunt etichetate?**  
Fiecare problemă are `Tema`, normalizată în `Tema_norm` și apoi mapată în `Domeniu`.

**Care este targetul?**  
`Domeniu` curricular.

**Preprocesare?**  
Normalizare diacritice/case în vectorizator, tokenizare care păstrează simboluri matematice, TF-IDF cu unigrams/bigrams, eliminare duplicate exacte pentru reducerea leakage-ului.

**Analiză exploratorie?**  
Am analizat distribuția domeniilor și lungimea problemelor. Lungimea mediană este 13 cuvinte; există probleme foarte scurte, ceea ce poate reduce siguranța modelului.

**Concluzie EDA care a influențat modelul?**  
Pentru text scurt și dataset mic, un model TF-IDF + ComplementNB este mai potrivit și mai robust decât un transformer greu de justificat/rulat local.

## 7. Model și robustețe nestructurate

**Model ales?**  
TF-IDF + ComplementNB. ComplementNB este potrivit pentru clasificare text și clase dezechilibrate.

**Am folosit transfer learning/model preantrenat?**  
Nu în MVP. Am ales un model clasic, reproductibil, rapid, care poate rula offline. Dacă avem timp, putem compara cu embeddings Romanian/RoBERT ca extensie, dar fără să riscăm demo-ul.

**Input/output exact?**  
Input: enunț brut. Output: domeniu curricular probabilistic.

**Inferență live?**  
Da, în tabul „Cele 2 servicii ML”.

**Metrici?**  
Accuracy, balanced accuracy, macro-F1, weighted-F1, confusion matrix, erori concrete.

**Baseline?**  
Baseline macro-F1: 0.071. Model final macro-F1: 0.876. Baseline accuracy: 0.269. Model accuracy: 0.892.

**Validare și anti-overfitting?**  
Split stratificat, GridSearchCV cu StratifiedKFold, limitare `max_features`, `min_df`, comparație cu baseline și eliminare duplicate exacte.

**Dovadă că generalizează?**  
Raportăm performanța pe test holdout care nu a fost folosit la tuning, plus exemple de erori.

## 8. Protocol de evaluare și analiză critică

**Cum am împărțit datele?**  
Train/test stratificat 78/22, apoi GridSearchCV cu 5-fold stratificat pe train.

**Cum prevenim leakage-ul?**  
Pipeline scikit-learn fit-uit în CV, nu preprocesăm pe tot datasetul înainte de split. Pentru text, eliminăm duplicate exacte înainte de split.

**De ce aceste metrici?**  
Accuracy singură poate ascunde dezechilibrul. Macro-F1 și balanced accuracy tratează mai corect clasele rare.

**Cum tratăm dezechilibrul?**  
Metrici macro, `class_weight='balanced'` la RandomForest și ComplementNB pentru text, plus analiză de distribuție.

**Erori frecvente?**  
Modelul text confundă teme apropiate: ecuații vs funcții când enunțul conține `f(x)` și rezolvare de ecuații. Modelul structurat confundă dificultăți vecine, mai ales `2 - mediu` vs `3 - consolidare`.

**Limitări principale?**  
Dataset mic, etichete tematice normalizate automat, lipsa istoricului real al elevilor, lipsa unui evaluator simbolic complet.

**Scenarii nesigure?**  
Probleme cu imagini lipsă, enunțuri foarte scurte, teme rare, formule ambigue sau probleme care cer diagramă.

**Ce am îmbunătăți prima dată?**  
Colectare de interacțiuni reale anonimizate, etichetare profesorală pentru erori conceptuale, evaluator simbolic al pașilor, comparație cu embeddings preantrenate.

## 9. Etică și impact

**Poate modelul introduce bias?**  
Da, prin dezechilibrul temelor și tipurilor de exerciții. Raportăm distribuțiile, folosim macro-F1 și prezentăm limitele.

**Cum protejăm datele sensibile?**  
Datasetul conține exerciții, nu date personale. Pentru elevi, MVP-ul nu persistă date personale; în producție am salva doar profil anonimizat.

**Riscuri de utilizare incorectă?**  
Elevul ar putea încerca să obțină soluția fără gândire. De aceea indiciile sunt graduale și soluția completă nu apare implicit.

**Măsuri de utilizare responsabilă?**  
Feedback descriptiv, întrebări metacognitive, transparență asupra limitărilor, fără metrici fabricate, fără promisiunea că modelul înlocuiește profesorul.

**Cum comunicăm limitele?**  
În tabul Etică & limitări și în README: modelul recomandă, nu decide definitiv; răspunsurile trebuie verificate de elev/profesor.

## 10. Aplicație, reproducibilitate și prezentare

**Ce aplicație am construit?**  
O aplicație Streamlit cu tutor demo, două servicii ML testabile separat, EDA, evaluare, Q&A și etică.

**Aplicația este funcțională la jurizare?**  
Da. Comanda: `pip install -r requirements.txt` și `streamlit run app.py`.

**Ce poate face utilizatorul?**  
Alege exerciții, răspunde, cere indicii, vede evaluarea, vede recomandarea următoarei probleme și testează separat modelele ML.

**Cum se observă valoarea practică?**  
Un exercițiu nou este încadrat curricular, dificultatea este estimată, iar elevul primește un traseu adaptiv în loc de soluție imediată.

**Poate juriul testa separat fiecare serviciu?**  
Da, tabul „Cele 2 servicii ML” are două formulare separate.

**Scenariu real cu ambele servicii?**  
Elevul introduce/alege o problemă. Modelul text prezice domeniul. Modelul structurat estimează dificultatea. Motorul pedagogic decide indiciu și următorul exercițiu.

**Organizarea proiectului?**  
`app.py` UI Streamlit; `src/data_prep.py` curățare/feature engineering; `src/train_models.py` antrenare/evaluare; `src/model_utils.py` inferență; `src/pedagogical_engine.py` reguli tutor; `/models` artifacts; `/data` date; `/docs` documentație.

**Dependințe?**  
Python, Streamlit, pandas, numpy, scikit-learn, joblib, openpyxl.

**Poate un evaluator reproduce?**  
Da. Modelele sunt salvate, dar pot fi reantrenate cu `python -m src.train_models`. Raportul JSON este regenerat din date.
