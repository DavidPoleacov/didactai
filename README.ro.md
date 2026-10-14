# DidactAI

DidactAI este un prototip open-source de tutorat la matematică, cu interfață în limba română, construit cu Streamlit. Combină o bancă de exerciții filtrată pentru explorare, un subset compatibil cu verificatorul pentru practică evaluată automat, indicii prestabilite, feedback bazat pe reguli și două clasificatoare salvate. Proiectul este destinat experimentării și învățării, nu înlocuiește un profesor.

## Funcționalități

- Explorează probleme după domeniu și temă în Acasă. Interacțiunile de aici sunt **simulări** și nu schimbă progresul din Tutor.
- Rezolvă în Tutor AI un exercițiu care poate fi evaluat automat, cere indicii etapizate din soluții stocate, primește feedback conservator și o sugestie din același domeniu după un răspuns corect verificat.
- Consultă încercările verificate, indicatorul aproximativ bazat pe reguli și istoricul disponibil în sesiunea curentă în Progres.
- Completează un diagnostic scurt. Numai răspunsurile clar corecte sau greșite intră în scorurile domeniilor; diagnosticul este separat de istoricul Tutorului.
- Clasifică domeniul după text și dificultatea după metadatele problemei.

Indiciile și întrebările de reflecție în română sunt prestabilite. Recomandările folosesc reguli simple de domeniu/dificultate și selecție aleatorie cu seed; nu identifică concepții greșite și nu folosesc un model de ranking. Aplicația nu generează explicații cu un LLM și nu analizează răspunsurile la reflecție.

## Instalare și pornire rapidă

Python 3.12 este runtime-ul testat. `requirements.txt` este varianta implicită, fără TensorFlow, și fișierul recunoscut de Streamlit Community Cloud. TensorFlow nu este necesar.

**Windows PowerShell sau Command Prompt** (folosește direct Python-ul din mediul virtual; activarea este opțională):

```powershell
py -3.12 -m venv .venv-demo
.\.venv-demo\Scripts\python.exe -m pip install --upgrade pip
.\.venv-demo\Scripts\python.exe -m pip install -r requirements-light.txt
.\.venv-demo\Scripts\python.exe -m streamlit run app.py
```

**Linux/macOS:**

```bash
python3.12 -m venv .venv-demo
.venv-demo/bin/python -m pip install --upgrade pip
.venv-demo/bin/python -m pip install -r requirements-light.txt
.venv-demo/bin/python -m streamlit run app.py
```

Deschide URL-ul local afișat de Streamlit (de regulă `http://localhost:8501`). Pentru instrucțiunile oficiale de upgrade pip, vezi [pip.pypa.io/en/stable/installation](https://pip.pypa.io/en/stable/installation/). Verifică proiectul înainte de pornire:

```bash
python -m unittest discover -s tests -v
python diagnose.py
```

`requirements-demo.lock` este un snapshot specific Linux/Python 3.12.3 și fixează Streamlit 1.65.0. Mediul nou instalat din intervalele din requirements pentru această reparație a folosit Streamlit 1.66.0; snapshot-ul lock nu a fost instalat sau testat aici.

Pentru inferență TensorFlow opțională, instalează `requirements-neural.txt`. Inferența neurală rulează numai dacă TensorFlow și modelul neural salvat sunt disponibile. Lipsa modelului nu declanșează antrenarea la pornire. Recrearea modelelor este o operație explicită de dezvoltare: `python src/train_models.py` (modelele clasice și rapoartele) sau `python src/neural_student_state_model.py` (modelul neural).

## Cum funcționează interfața

- **Acasă:** explorează toate exercițiile eligibile pentru navigare și vezi predicțiile pentru domeniu/dificultate. Cele 15 formate incompatibile cu verificatorul și o problemă cu răspuns deschis rămân vizibile pentru revizuire manuală. Răspunsurile, indiciile și recomandările de aici sunt simulări și nu scriu în progresul Tutorului. Schimbarea exercițiului din previzualizare șterge răspunsul, indiciile, feedback-ul, estimarea și recomandarea; la revenire simularea este goală.
- **Tutor AI:** alege un domeniu și un exercițiu evaluabil automat sau folosește selecția automată. Răspunsurile clar greșite și corecte sunt încercări distincte. Comparațiile neverificate nu actualizează progresul. Dublul credit este blocat în aceeași rulare a exercițiului; reluările deliberate într-o rulare nouă rămân permise.
- **Progres:** numără încercările verificate și le grupează după domeniul real al exercițiului. Exercițiile distincte completate sunt numărate separat. Indicatorul de stăpânire este o estimare globală aproximativă bazată pe reguli, nu o probabilitate calibrată. Istoricul există numai în sesiunea Streamlit curentă; nu este salvat într-un cont. Data de revizuire este doar orientativă: nu se programează și nu trimite mementouri.

## Arhitectură și modele

- `app.py` conține interfața Streamlit și fluxul Tutor cu stare de sesiune.
- `src/exercise_bank.py` aplică `data/processed/exercise_review_flags.csv` după `problem_hash` normalizat, elimină enunțurile/răspunsurile goale și păstrează indicii CSV originali pentru features și istoric.
- `src/pedagogical_engine.py` implementează comparația conservatoare a numerelor raționale, eligibilitatea separată după format și semantică, corecțiile de afișare ale indiciilor validate prin hash, indiciile etapizate, scorarea diagnosticului și recomandările simple. Scorarea acceptă verdicte verificate explicit; formatul vechi prevalidat este acceptat numai cu un câmp boolean `correct` și fără marcaje `verified` sau `status`.
- Un `RandomForestClassifier` scikit-learn estimează dificultatea folosind metadate: identificatori și sursă, lungimi de enunț/soluție/răspuns, indicatori pentru simboluri matematice și câmpuri categoriale pentru temă, domeniul cunoscut și tipul sursei.
- Un pipeline TF-IDF/Complement Naive Bayes clasifică domeniul din textul problemelor românești.
- Un model TensorFlow opțional clasifică interacțiuni dintr-un set sintetic de stări ale elevului. Nu este folosit în instalarea lightweight, iar etichetele sale sunt generate prin reguli, nu provin din observații reale ale elevilor.

## Date și proveniență

`data/processed/exercises_augmented.csv` conține 1.344 de rânduri originale de antrenare: 1.212 generate și 132 provenite din surse. Marcajele exclud 59 de probleme după textul normalizat—42 derivate din surse și 17 probleme ambigue generate despre economii—lăsând 1.285 de rânduri eligibile pentru explorare. Compararea fiecărei chei cu ea însăși indică 1.270 de rânduri compatibile ca format. O excludere semantică revizuită separat elimină problema deschisă #37 din subsetul evaluabil automat, rezultând 1.269 de rânduri evaluabile și 16 disponibile pentru navigare/verificare manuală (15 cu formate incompatibile și #37). Compatibilitatea formatului nu dovedește corectitudinea matematică sau aprobarea profesorilor. CSV-ul original, marcajele, features, modelele și rapoartele salvate rămân neschimbate. `data/processed/exercises_processed.csv` este un set mai vechi de 489 de rânduri, păstrat pentru proveniență, nu banca activă. Setul sintetic de interacțiuni neuronale are 2.200 de rânduri.

Corecțiile de afișare ale discriminantului sunt versionate în `data/processed/hint_display_corrections.csv`, cu indici stabili, hash-uri ale problemelor, pașii vechi/noi, justificare și SHA-256 pentru pasul-sursă. Verificarea aritmetică acoperă substituțiile revizuite, nu întreaga bancă.

Metricile salvate pentru holdout-urile clasice:

| Clasificator | Rânduri holdout | Acuratețe | Macro-F1 |
| --- | ---: | ---: | ---: |
| Dificultate structurată | 296 | 0.922297 | 0.831212 |
| Domeniu din text | 279 | 0.982079 | 0.974313 |

Raportul setului neural sintetic indică 440 de rânduri de test și acuratețe 0.922727 / macro-F1 0.914048. Aceste scoruri măsoară clasificarea, nu rezultatele învățării. Setul include familii de șabloane asemănătoare între rândurile împărțite aleatoriu; scorurile pot reflecta această suprapunere. Este necesară evaluare externă pe date independente, revizuite de profesori. `python tools/check_generated_math.py` auditează proveniența și limitările fără a executa expresii. `python tools/verify_saved_models.py` reproduce independent holdout-urile clasice salvate.

Graficele active despre populația de exerciții din `assets/` reprezintă cele 1.285 de rânduri eligibile pentru explorare, nu subsetul de 1.269 evaluabil automat în Tutor. Graficul de holdout arată metrici de clasificare, nu populația curentă a băncii. Regenerează doar graficele afectate cu `python tools/refresh_figures.py --population-only`. Graficele vechi, dacă există în `assets/legacy/`, sunt istorice.

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
- Unele exerciții sunt incomplete, ambigue ori fără diagrame. Cele 59 de excluderi de conținut, cele 15 formate incompatibile și excluderea semantică pentru #37 sunt filtre distincte; nu certifică restul băncii printr-un proces de revizie profesorală.
- Stăpânirea și recomandările folosesc reguli simple; modelul neural opțional prezice etichete sintetice de interacțiune.
- Nu există conturi salvate, progres persistent, mementouri automate, analiză a reflecțiilor sau rezultate măsurate cu elevi reali.
- Holdout-urile modelelor sunt împărțiri interne aleatorii, nu evaluări externe independente.

## Docker

Imaginea lightweight folosește Python 3.12 fără TensorFlow. Din rădăcina repository-ului, rulează `bash run_docker.sh`; nu presupune că arhiva ZIP păstrează permisiunea executabilă.

## Contribuitori

- **David Poleacov — backend**
- **Beatrice Batrinu — frontend**


