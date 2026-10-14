# Deployment notes

The lightweight local build was tested with Python 3.12.14 and scikit-learn 1.9.0. This audit did not deploy to Community Cloud or execute Docker. A local pass does not guarantee a cloud deployment will fit the platform's resource limits.

## Local recording

Follow `docs/demo_guide.md`. Install `requirements-light.txt` in a fresh environment to keep the optional TensorFlow path disabled. Run the behavioral suite and `python diagnose.py`, then `streamlit run app.py`.

## Streamlit Community Cloud

Community Cloud discovers a dependency file with a recognized filename. `requirements-light.txt` is an alternative for local pip installation; its presence does not select it for a cloud deployment.

For a lightweight deployment, use a deployment branch or copy whose **requirements.txt contains the contents of requirements-light.txt**. Keep `app.py`, `.streamlit/config.toml`, the `src` modules, the classical model artifacts, `data/processed/exercises_augmented.csv` and `data/processed/exercise_review_flags.csv`. Select `app.py` as the entrypoint and the same Python version you tested locally.

For the full optional build, retain the ordinary `requirements.txt`. TensorFlow installation and inference need their own verification. The app can use its two classical models and pedagogical rules without neural inference.

After deployment, verify the exact build from a signed-out browser: launch, a hint, incorrect/correct scalar responses, next-exercise navigation and session history. Use the actual build logs to investigate a failure.

Official documentation: [dependency discovery](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies) and [file organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization).

## Artifact compatibility and startup

Both requirement files pin scikit-learn 1.9.0, matching the saved joblib artifacts. Other dependencies use ranges. The artifacts loaded and reproduced their scores in the tested environment; this is not a full dependency lock or a compatibility promise for every environment. See [scikit-learn's persistence guidance](https://scikit-learn.org/stable/model_persistence.html).

Normally the app loads the supplied models. If required classical assets are missing, the report is invalid, or its row count differs from the dataset, startup calls `train_all()` and writes regenerated CSVs/models. If TensorFlow is present but the neural model is absent, the optional startup helper attempts training. Keep the supplied artifacts for the recorded build and document intentional retraining.

The practice filter excludes flagged normalized problem IDs without altering training CSVs or saved models. Retain the flag file when deploying. It does not establish that all unflagged items have undergone teacher review.

## Docker

`bash run_docker.sh` builds a Python 3.11 image from full `requirements.txt`. Docker was not tested here and differs from the verified Python 3.12 lightweight environment. Verify it separately if offering it as your executable demonstration.
