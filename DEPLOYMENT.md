# Local run and optional hosting

## Local lightweight run

Use Python 3.12 and the declared lightweight dependencies:

```bash
python3.12 -m venv .venv-demo
.venv-demo/bin/python -m pip install -r requirements-light.txt
.venv-demo/bin/python -m unittest discover -s tests -v
.venv-demo/bin/python diagnose.py
.venv-demo/bin/python -m streamlit run app.py
```

On Windows, use the venv interpreter directly to avoid PowerShell activation-policy issues:

```powershell
py -3.12 -m venv .venv-demo
.\.venv-demo\Scripts\python.exe -m pip install -r requirements-light.txt
.\.venv-demo\Scripts\python.exe -m unittest discover -s tests -v
.\.venv-demo\Scripts\python.exe diagnose.py
.\.venv-demo\Scripts\python.exe -m streamlit run app.py
```

The local Streamlit URL is normally `http://localhost:8501`. TensorFlow is not required.

## Streamlit Community Cloud

`requirements.txt` includes the lightweight default dependency set and is the repository's recognized requirements file. Select `app.py` as the entrypoint. No secrets file or credentials are needed by this application. Do not configure a nonexistent Cloud setting to choose `requirements-light.txt`.

The optional TensorFlow path is a separate local development installation via `requirements-neural.txt`; Cloud deployment with TensorFlow is not part of the verified demo and is not recommended without separate resource/runtime testing. The app never trains models at startup.

## Docker

`Dockerfile` installs the default lightweight `requirements.txt`. `./run_docker.sh` builds and starts the local container. No deployment is performed by this repository's checks.

## Explicit model development

Normal app startup, tests, and diagnostics load the saved classifiers without retraining. `python src/train_models.py` explicitly rebuilds the classical model/data artifacts. `python src/neural_student_state_model.py` explicitly trains the optional neural model. Both can overwrite files; back up artifacts before running these development commands.
