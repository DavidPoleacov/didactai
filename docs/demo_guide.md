# Two-minute DidactAI demonstration

## Verified recording setup

The fresh lightweight environment for this repair was checked on Linux with Python 3.12.3, pip 26.2.1, Streamlit 1.66.0, pandas 3.0.6, NumPy 2.5.4, scikit-learn 1.9.0, joblib 1.6.0, and matplotlib 3.11.2. TensorFlow was absent; the walkthrough does not claim to use neural inference. The checked-in `requirements-demo.lock` is a separate Linux snapshot with Streamlit 1.65.0 and was not installed or tested for this repair. `pip-audit` was unavailable, so no package vulnerability scan is claimed. Start Streamlit locally at `http://localhost:8501`. Windows commands are provided below, but Windows execution was not tested.

The unchanged bank contains 1,344 training rows, 59 content exclusions, and 1,285 content-eligible browsing rows. There are 1,270 key-format-compatible rows; after the separate semantic exclusion for open-choice exercise #37, 1,269 are eligible for automatic grading and 16 remain available for manual review. This is not teacher approval of the remaining bank.

**Windows PowerShell or Command Prompt** (no activation is needed):

```powershell
py -3.12 -m venv .venv-demo
.\.venv-demo\Scripts\python.exe -m pip install --upgrade pip
.\.venv-demo\Scripts\python.exe -m pip install -r requirements-light.txt
.\.venv-demo\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-demo\Scripts\python.exe diagnose.py
.\.venv-demo\Scripts\python.exe -m streamlit run app.py
```

**Linux/macOS:**

```bash
python3.12 -m venv .venv-demo
.venv-demo/bin/python -m pip install --upgrade pip
.venv-demo/bin/python -m pip install -r requirements-light.txt
.venv-demo/bin/python -m unittest discover -s tests -v
.venv-demo/bin/python diagnose.py
.venv-demo/bin/python -m streamlit run app.py
```

See [pip's official installation and upgrade guidance](https://pip.pypa.io/en/stable/installation/).

Use a fresh browser session for recording. The tracked demo sequence starts from the Tutor domain **Funcții**, exercise **#460** (`Fie f: R → R, f(x) = 4x + 12. Calculati f(0).`). The chosen exercise remains available by its stable bank index even though its `Itemul` metadata is empty. Tutor history is session-only.

## Timed walkthrough

Elapsed time totals **115 seconds**. The spoken narration below contains **191 whitespace-separated words**, counted across the nine quoted narration cells only (split on whitespace; table labels and screen actions are excluded). Pauses during Streamlit reruns are included in the timeline. Do not narrate continuously through slow clicks—let the result render.

| Elapsed | Screen action | Narration |
| --- | --- | --- |
| 00:00–00:11 | Show DidactAI, then open **Tutor AI**. | “DidactAI is a Romanian-language mathematics tutoring prototype. It offers practice, staged hints, and feedback; it is not a teacher replacement.” |
| 00:11–00:23 | Confirm **Funcții** and select exercise **#460**; click **Începe exercițiul selectat**. | “I’m choosing Funcții and exercise 460, asking for f of zero. Its bank index connects this problem to model features and session history.” |
| 00:23–00:37 | Read the problem and click **Cere un indiciu** once. Pause on the hint and reflection prompt. | “The first hint suggests substitution and operation order without giving twelve. This reflection prompt is predefined, not analyzed.” |
| 00:37–00:49 | Enter **1** and click **Verifică răspunsul**. | “Submitting one is incorrect: one verified attempt is logged and the consecutive-error count increases. Unverified answers change no progress.” |
| 00:49–00:58 | Enter **12 or 99** and submit; pause on the unverified feedback. | “This answer is ambiguous to the conservative checker, so it is not counted and changes no progress.” |
| 00:58–01:08 | Keep **Analiza răspunsului** visible; point to the displayed domain and difficulty predictions. | “The retained analysis is clearly identified as belonging to the last verified attempt. Difficulty uses metadata; neither classification measures learning gains.” |
| 01:08–01:20 | Replace the answer with **12** and click **Verifică răspunsul**. | “Twelve matches the stored key. This is the second verified attempt; consecutive errors reset and approximate rule-based mastery updates. TensorFlow is absent.” |
| 01:20–01:31 | Show the suggested same-domain exercise; click **Continuă cu exercițiul următor**. | “Continue selects the displayed eligible function exercise, synchronizes the dropdown, and resets its counters while preserving both verified attempts.” |
| 01:31–01:55 | Open **Progresul meu** and point to attempts, completed exercises, and history; hold the final view. | “Progress separates evaluated attempts from completed exercises and groups by actual domain. History is session-only; mastery is not a calibrated probability. Hints are stored, recommendations use rules, and neural labels are synthetic.” |

Hold on the Progress view for the closing shot; the 115-second timeline includes a silent pause rather than unverified claims.

## Manual visual check still required

Automated Streamlit AppTest verified widget state and dropdown synchronization, reruns, feedback persistence, verified/unverified attempts, Continue, reset, corrected #337 hints across Home/Tutor/diagnostic previews, diagnostic result invalidation, untested browsing-only domains, leave-and-return preview cleanup, preview isolation, and empty-bank rendering. It does not verify desktop typography or physical clipping. Before publishing a recording, inspect the running page at a typical desktop capture size and confirm:

- Romanian labels, problem text, feedback, and hint have readable contrast and are not clipped.
- Tutor controls and **Continue** are visible after each rerun without awkward scrolling.
- Progress metrics and history fit the capture frame and retain their labels.
- No learner-facing traceback, deprecation message, or TensorFlow claim appears.

No browser-based visual inspection was available during this repair; visual verification remains pending.
