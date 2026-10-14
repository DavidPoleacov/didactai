# DidactAI

DidactAI este un prototip open-source de tutorat la matematică, cu interfață în limba română, construit cu Streamlit. Combină un subset de exerciții disponibil pentru practică, indicii prestabilite, feedback bazat pe reguli și două clasificatoare salvate. Proiectul este destinat experimentării și învățării, nu înlocuiește un profesor.

## Funcționalități

- Explorează probleme după domeniu și temă în Acasă. Interacțiunile de aici sunt **simulări** și nu schimbă progresul din Tutor.
- Rezolvă un exercițiu selectat în Tutor AI, cere indicii etapizate din soluții stocate, primește feedback conservator și o sugestie din același domeniu după un răspuns corect verificat.
- Consultă încercările verificate, indicatorul aproximativ bazat pe reguli și istoricul disponibil în sesiunea curentă în Progres.
- Completează un diagnostic scurt. Numai răspunsurile clar corecte sau greșite intră în scorurile domeniilor; diagnosticul este separat de istoricul Tutorului.
- Clasifică domeniul după text și dificultatea după metadatele problemei.

Indiciile și întrebările de reflecție în română sunt prestabilite. Recomandările folosesc reguli simple de domeniu/dificultate și selecție aleatorie cu seed; nu identifică concepții greșite și nu folosesc un model de ranking. Aplicația nu generează explicații cu un LLM și nu analizează răspunsurile la reflecție.

## Instalare și pornire rapidă

Python 3.12 este runtime-ul testat. `requirements.txt` este varianta implicită, fără TensorFlow, și fișierul recunoscut de Streamlit Community Cloud. TensorFlow nu este necesar.

**Windows PowerShell sau Command Prompt** (folosește direct Python-ul din mediul virtual; activarea este opțională):

```powershell
py -3.12 -m venv .venv-demo
.\.venv-demo\Scripts\python.exe -m pip install -r requirements-light.txt
.\.venv-demo\Scripts\python.exe -m streamlit run app.py
```

**Linux/macOS:**

```bash
python3.12 -m venv .venv-demo
.venv-demo/bin/python -m pip install -r requirements-light.txt
.venv-demo/bin/python -m streamlit run app.py
```

Deschide URL-ul local afișat de Streamlit (de regulă `http://localhost:8501`). Verifică proiectul înainte de pornire:

```bash
python -m unittest discover -s tests -v
python diagnose.py
```

`requirements-demo.lock` păstrează setul exact de dependențe Linux/Python 3.12.3 folosit la verificare; este un snapshot specific platformei, în timp ce fișierele requirements declară intervalele de instalare acceptate.

Pentru inferență TensorFlow opțională, instalează `requirements-neural.txt`. Inferența neurală rulează numai dacă TensorFlow și modelul neural salvat sunt disponibile. Lipsa modelului nu declanșează antrenarea la pornire. Recrearea modelelor este o operație explicită de dezvoltare: `python src/train_models.py` (modelele clasice și rapoartele) sau `python src/neural_student_state_model.py` (modelul neural).

## Cum funcționează interfața

- **Acasă:** filtrează banca de practică și vezi predicțiile pentru domeniu/dificultate. Răspunsurile, indiciile și recomandările de aici sunt simulări etichetate și nu scriu în progresul Tutorului.
- **Tutor AI:** alege un domeniu și un exercițiu din bancă sau selecția automată. Răspunsurile clar greșite și corecte sunt încercări distincte. Comparațiile neverificate nu actualizează progresul. Un exercițiu completat corect nu poate fi creditat de două ori.
- **Progres:** numără încercările verificate și le grupează după domeniul real al exercițiului. Exercițiile distincte completate sunt numărate separat. Indicatorul de stăpânire este o estimare globală aproximativă bazată pe reguli, nu o probabilitate calibrată. Istoricul există numai în sesiunea Streamlit curentă; nu este salvat într-un cont. Data de revizuire este doar orientativă: nu se programează și nu trimite mementouri.

## Arhitectură și modele

- `app.py` conține interfața Streamlit și fluxul Tutor cu stare de sesiune.
- `src/exercise_bank.py` aplică `data/processed/exercise_review_flags.csv` după `problem_hash` normalizat, elimină enunțurile/răspunsurile goale și păstrează indicii CSV originali pentru features și istoric.
- `src/pedagogical_engine.py` implementează comparația conservatoare a numerelor raționale, indiciile etapizate, scorarea diagnosticului și recomandările simple.
- Un `RandomForestClassifier` scikit-learn estimează dificultatea folosind metadate: identificatori și sursă, lungimi de enunț/soluție/răspuns, indicatori pentru simboluri matematice și câmpuri categoriale pentru temă, domeniul cunoscut și tipul sursei.
- Un pipeline TF-IDF/Complement Naive Bayes clasifică domeniul din textul problemelor românești.
- Un model TensorFlow opțional clasifică interacțiuni dintr-un set sintetic de stări ale elevului. Nu este folosit în instalarea lightweight, iar etichetele sale sunt generate prin reguli, nu provin din observații reale ale elevilor.

## Date și proveniență

`data/processed/exercises_augmented.csv` conține 1.344 de rânduri de antrenare: 1.212 generate și 132 provenite din surse. Fișierul de revizuire exclude 59 de probleme după textul normalizat—42 derivate din surse și 17 probleme ambigue generate despre economii—lăsând 1.285 de exerciții dacă banca rămâne neschimbată. Aceste marcaje semnalează limitări cunoscute; nu înseamnă că toate celelalte exerciții au fost revizuite de profesori. `data/processed/exercises_processed.csv` este un set mai vechi de 489 de rânduri, păstrat pentru proveniență, nu banca activă. Setul sintetic de interacțiuni neuronale are 2.200 de rânduri.

Metricile salvate pentru holdout-urile clasice:

| Clasificator | Rânduri holdout | Acuratețe | Macro-F1 |
| --- | ---: | ---: | ---: |
| Dificultate structurată | 296 | 0.922297 | 0.831212 |
| Domeniu din text | 279 | 0.982079 | 0.974313 |

Raportul setului neural sintetic indică 440 de rânduri de test și acuratețe 0.922727 / macro-F1 0.914048. Aceste scoruri măsoară clasificarea, nu rezultatele învățării. Setul include familii de șabloane asemănătoare între rândurile împărțite aleatoriu; scorurile pot reflecta această suprapunere. Este necesară evaluare externă pe date independente, revizuite de profesori. `python tools/check_generated_math.py` auditează proveniența și limitările fără a executa expresii. `python tools/verify_saved_models.py` reproduce independent holdout-urile clasice salvate.

Graficele active din `assets/` se regenerează din banca CSV și raportul salvate cu `python tools/refresh_figures.py`. Graficele vechi, dacă există în `assets/legacy/`, sunt istorice.

## Testare și mentenanță

```bash
python -m unittest discover -s tests -v
python diagnose.py
python tools/check_generated_math.py
python tools/verify_saved_models.py
```

Utilitarele de antrenare sunt operații explicite de dezvoltare și pot suprascrie datele procesate, modelele și rapoartele. Nu sunt rulate la pornirea aplicației, în diagnostice, teste sau demonstrație.

## Limitări actuale

- Verificatorul acceptă text normalizat identic și echivalență exactă de numere raționale cu etichete de variabilă și unități identice. Nu convertește unități și nu demonstrează echivalență simbolică generală, demonstrații, mulțimi, intervale sau text ambiguu.
- Unele exerciții sunt incomplete, ambigue ori fără diagrame. Marcajele exclud limitări cunoscute, dar nu certifică întreaga bancă printr-un proces de revizie profesorală.
- Stăpânirea și recomandările folosesc reguli simple; modelul neural opțional prezice etichete sintetice de interacțiune.
- Nu există conturi salvate, progres persistent, mementouri automate, analiză a reflecțiilor sau rezultate măsurate cu elevi reali.
- Holdout-urile modelelor sunt împărțiri interne aleatorii, nu evaluări externe independente.

## Contribuitori

- **David Poleacov — backend**
- **Beatrice Batrinu — frontend**

Mentenanța asistată de AI este consemnată în istoricul ulterior al proiectului; nu este prezentată drept funcționalitate originală a concursului.
