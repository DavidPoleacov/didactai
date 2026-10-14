# Two-minute DidactAI demonstration

## Verified recording setup

The lightweight environment was checked on Linux with Python 3.12.3, Streamlit 1.65.0, pandas 3.0.6, NumPy 2.5.4, scikit-learn 1.9.0, joblib 1.6.0, and matplotlib 3.11.2. TensorFlow was absent; the walkthrough does not claim to use neural inference. Start Streamlit locally at `http://localhost:8501`. Windows commands are provided below, but Windows execution was not tested.

**Windows PowerShell or Command Prompt** (no activation is needed):

```powershell
py -3.12 -m venv .venv-demo
.\.venv-demo\Scripts\python.exe -m pip install -r requirements-light.txt
.\.venv-demo\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-demo\Scripts\python.exe diagnose.py
.\.venv-demo\Scripts\python.exe -m streamlit run app.py
```

**Linux/macOS:**

```bash
python3.12 -m venv .venv-demo
.venv-demo/bin/python -m pip install -r requirements-light.txt
.venv-demo/bin/python -m unittest discover -s tests -v
.venv-demo/bin/python diagnose.py
.venv-demo/bin/python -m streamlit run app.py
```

Use a fresh browser session for recording. The tracked demo sequence starts from the Tutor domain **Funcții**, exercise **#460** (`Fie f: R → R, f(x) = 4x + 12. Calculati f(0).`). The chosen exercise remains available by its stable bank index even though its `Itemul` metadata is empty. Tutor history is session-only.

## Timed walkthrough

Elapsed time totals **110 seconds**. The spoken narration below is **167 words**; pauses during Streamlit reruns are included in the timeline. Do not narrate continuously through slow clicks—let the result render.

| Elapsed | Screen action | Narration |
| --- | --- | --- |
| 00:00–00:11 | Show DidactAI, then open **Tutor AI**. | “DidactAI is a Romanian-language mathematics tutoring prototype. It offers practice, staged hints, and feedback; it is not a teacher replacement.” |
| 00:11–00:23 | Confirm **Funcții** and select exercise **#460**; click **Începe exercițiul selectat**. | “I’m choosing Funcții and exercise 460, asking for f of zero. Its bank index connects this problem to model features and session history.” |
| 00:23–00:37 | Read the problem and click **Cere un indiciu** once. Pause on the hint and reflection prompt. | “The first hint suggests substitution and operation order without giving twelve. This reflection prompt is predefined, not analyzed.” |
| 00:37–00:49 | Enter **1** and click **Verifică răspunsul**. | “Submitting one is incorrect: one verified attempt is logged and the consecutive-error count increases. Unverified answers change no progress.” |
| 00:49–01:03 | Keep **Analiza răspunsului** visible; point to the displayed domain and difficulty predictions. | “For this item, the classifiers display Funcții and 1 - bază. Difficulty uses exercise metadata; neither classification measures learning gains.” |
| 01:03–01:15 | Replace the answer with **12** and click **Verifică răspunsul**. | “Twelve matches the stored key. The second attempt resets consecutive errors and updates approximate rule-based mastery. TensorFlow is absent.” |
| 01:15–01:26 | Show the suggested same-domain exercise; click **Continuă cu exercițiul următor**. | “Continue selects another eligible function exercise and resets its counters, preserving both attempts in session history.” |
| 01:26–01:50 | Open **Progresul meu** and point to attempts, completed exercises, and history; hold the final view. | “Progress separates evaluated attempts from completed exercises and groups by actual domain. History is session-only; mastery is not a calibrated probability. Hints are stored, recommendations use rules, and neural labels are synthetic.” |

**Narration count:** 167 whitespace-separated words (table narration only). Hold on the Progress view for the closing shot; use the remaining recording time for a silent pause rather than adding unverified claims.

## Manual visual check still required

Automated Streamlit AppTest verified widget state, reruns, feedback persistence, hint state, answer attempts, Continue, reset, and preview isolation. It does not verify desktop typography or physical clipping. Before publishing a recording, inspect the running page at a typical desktop capture size and confirm:

- Romanian labels, problem text, feedback, and hint have readable contrast and are not clipped.
- Tutor controls and **Continue** are visible after each rerun without awkward scrolling.
- Progress metrics and history fit the capture frame and retain their labels.
- No learner-facing traceback, deprecation message, or TensorFlow claim appears.

No browser-based visual inspection was available during this repair; visual verification remains pending.
