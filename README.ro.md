# DidactAI

DidactAI este un prototip de tutor la matematică în limba română, destinat elevilor de gimnaziu și liceu din Republica Moldova. Aplicația Streamlit folosește o bancă de exerciții în română și combină conținut pedagogic prestabilit, reguli explicite și clasificatoare antrenate pe datele exercițiilor.

Acest README descrie starea actuală a repository-ului. Corectările verificatorului de răspunsuri, ale pornirii modelului neural, testele de regresie și documentația de portofoliu din această revizie sunt îmbunătățiri ulterioare competiției; ele nu trebuie prezentate drept funcționalități ale versiunii inițiale fără dovezi din acea versiune.

## Funcționalități implementate

- Selectarea exercițiilor din bancă după domeniu și temă sau pornirea unui exercițiu de tutor dintr-un domeniu selectat.
- Oferirea unor indicii graduale bazate pe pașii de rezolvare salvați și apoi a unei întrebări reflexive selectate dintr-o listă prestabilită în română.
- Estimarea stăpânirii printr-o actualizare simplă, bazată pe reguli, și recomandarea unui exercițiu după domeniu și dificultatea-țintă. Selectarea candidatului folosește filtrare și eșantionare după reguli, nu un model ML de clasare a recomandărilor.
- Clasificarea domeniului și dificultății cu două modele scikit-learn salvate; demo-ul principal afișează predicțiile lor.
- Estimarea opțională a unei etichete privind starea elevului cu un model neural salvat, dacă TensorFlow este instalat și artefactul poate fi încărcat.
- Păstrarea istoricului interacțiunilor și a stăpânirii în starea sesiunii Streamlit; nu există cont de utilizator sau istoric de progres într-o bază de date.

Indiciile, întrebările reflexive, agregarea diagnosticului, actualizarea stăpânirii și majoritatea recomandărilor folosesc conținut prestabilit sau reguli pedagogice. Aplicația nu folosește un LLM, nu generează explicații originale și nu evaluează demonstrații matematice arbitrare.

## Modele de învățare automată

**Clasificatorul de dificultate pe date structurate.** Un `RandomForestClassifier` din scikit-learn prezice `Dificultate_group` (`1 - bază`, `2 - mediu`, `3 - consolidare`, `4 - avansat`). Intrările sunt 14 caracteristici numerice (`Itemul`, `Sursa_year`, lungimile enunțului/soluției/răspunsului, numărul de cifre și simboluri matematice și indicatori pentru procente, geometrie, ecuații, radicali, funcții și contexte reale) și câmpurile categorice `Tema_norm`, `Domeniu` și `Sursa_type`. Pipeline-ul impută valorile lipsă, scalează caracteristicile numerice și aplică one-hot encoding câmpurilor categorice.

**Clasificatorul domeniului din text.** Un vectorizator TF-IDF și `ComplementNB` prezic `Domeniu` pe baza enunțului problemei în limba română (`Problema`). Modelul clasifică textul; nu îl generează și nu explică problema.

**Clasificator neural opțional al stării elevului.** Un model TensorFlow/Keras de tip feed-forward prezice una dintre clasele `autonomie_buna`, `blocaj`, `progres` sau `supraincarcare`. Cele nouă intrări sunt timpul de lucru, numărul de indicii, numărul de încercări, corectitudinea, numărul de greșeli, dificultatea exercițiului, stăpânirea anterioară, greșelile consecutive și nivelul de ajutor solicitat. Fișierul `data/processed/student_interactions.csv` conține 2.200 de rânduri sintetice. Etichetele și caracteristicile sunt generate prin reguli în `src/neural_student_state_model.py`; nu provin de la elevi reali. Metricile salvate măsoară concordanța cu aceste etichete sintetice, nu progresul în învățare sau performanța pe elevi reali.

Componenta neurală poate selecta dificultatea-țintă folosind o mapare fixă a stărilor prezise. Exercițiul este selectat în continuare prin filtrare după domeniu/dificultate și eșantionare.

## Seturi de date și metrici salvate

Repository-ul conține două seturi distincte de exerciții:

- `data/processed/exercises_processed.csv`: setul procesat legacy, cu 489 de rânduri; 477 au etichete pentru temă și dificultate, iar 51 de rânduri repetă un enunț normalizat.
- `data/processed/exercises_augmented.csv`: setul augmentat, cu 1.344 de rânduri; toate cele 1.344 au etichete pentru temă și dificultate, iar raportul de evaluare salvat înregistrează zero repetări ale enunțului normalizat.

Clasificatoarele salvate și `models/evaluation_report.json` corespund setului augmentat, nu setului legacy de 489 de rânduri. Metricile următoare sunt preluate din acel raport salvat; sunt metrici de clasificare pe setul de test, nu dovezi ale îmbunătățirii învățării elevilor.

| Sarcină | Set de test | Macro-F1 baseline | Macro-F1 model | Acuratețe model |
| --- | ---: | ---: | ---: | ---: |
| Dificultate structurată, 1.344 rânduri sursă | 296 | 0,163 | 0,831 | 0,922 |
| Domeniu din text, 1.265 rânduri după filtrare | 279 | 0,078 | 0,974 | 0,982 |

`models/neural_student_state_report.json` raportează acuratețe 0,923 și macro-F1 0,914 pe un set de test de 440 de rânduri, extras din setul sintetic de 2.200 de rânduri. Aceste scoruri descriu predicția etichetelor generate prin reguli. Clasificatoarele pentru exerciții folosesc, de asemenea, o singură împărțire train/test și trebuie evaluate pe date externe înainte de a formula concluzii despre generalizare.

## Instalare și pornire

Se recomandă Python 3.11 sau o versiune mai nouă. Dockerfile-ul folosește Python 3.11.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements-light.txt
streamlit run app.py
```

Deschide adresa locală afișată de Streamlit (de regulă `http://localhost:8501`). Dependențele lightweight nu includ TensorFlow; tutorul bazat pe reguli și cele două clasificatoare scikit-learn rămân disponibile, iar predicțiile neuronale despre starea elevului sunt omise.

Pentru calea TensorFlow opțională, folosește `python -m pip install -r requirements.txt`. Dacă TensorFlow este disponibil, dar lipsește fișierul modelului neural, pornirea încearcă să îl antreneze din CSV-ul interacțiunilor, generând date sintetice dacă lipsește și CSV-ul. Această antrenare nu este necesară pentru varianta lightweight.

În mod normal, aplicația încarcă modelele și CSV-ul augmentat din repository. Dacă lipsesc artefactele modelelor clasice, raportul este invalid sau numărul de rânduri din raport nu corespunde setului, `app.py` apelează `train_all()`. Funcția reconstruiește ambele CSV-uri procesate din fișierele Excel brute și reantrenează modelele clasice, scriind artefacte și rapoarte. Această cale de rezervă modifică fișiere locale; păstrează copii ale artefactelor pe care dorești să le conservi înainte de activarea ei.

Pentru Docker, rulează `./run_docker.sh`; imaginea se construiește din `requirements.txt`, inclusiv TensorFlow. Pentru suita de regresie:

```bash
python -m unittest discover -s tests -v
```

## Limitele verificării răspunsurilor

Verificatorul acceptă un răspuns complet, egal după normalizare, sau echivalența rațională exactă pentru răspunsuri scalare acceptate. Formatele numerice acceptate includ numere întregi, valori cu semn, fracții, zecimale cu punct și zecimale cu virgulă românească. La compararea scalară, etichetele variabilelor și unitățile recunoscute trebuie să coincidă; conversia unităților nu este implementată. Spațiile dintre tokenuri distincte sunt păstrate, iar verificatorul nu execută inputul elevului.

Verificatorul returnează `needs_review` pentru echivalențe simbolice pe care nu le poate demonstra, răspunsuri cu mai multe componente, intervale/mulțimi, text liber, unități nesuportate și formate ambigue. O cheie numerică lipsă sau malformată returnează `invalid_key`; un răspuns gol returnează `no_answer`. Aceste rezultate neverificate nu modifică stăpânirea, numărul de greșeli, predicțiile neuronale sau jurnalul de sesiune. Echivalența simbolică generală și evaluarea demonstrațiilor nu sunt implementate.

## Echipa și starea proiectului

David Poleacov a lucrat la backend. Colegul său/colega sa a lucrat la frontend; numele și împărțirea exactă a contribuțiilor trebuie confirmate de autori înainte de trimitere. Interfața și exercițiile rămân în română pentru elevii vizați; explicațiile în engleză pentru evaluatori sunt în acest README și în `docs/maker_portfolio_notes.md`.

Limitările actuale includ o bancă de exerciții mică și dezechilibrată, etichete normalizate automat, lipsa datelor de interacțiune de la elevi reali, progres păstrat doar în sesiune, lipsa unui verificator simbolic general și posibile diagrame lipsă în unele enunțuri. Modelele joblib salvate se încarcă în mediul verificat, dar emit avertismente de compatibilitate scikit-learn 1.9.0–1.9.1. Versiunile dependențelor nu sunt fixate exact, deci reproducerea artefactelor serializate între medii nu este garantată.