# AI-Powered Fake News Detector backend

This backend follows the workflow in the project document: WELFake CSV → cleaning → stratified 80/20 split → TF-IDF (unigrams and bigrams) → Logistic Regression → evaluation → saved artifact → FastAPI prediction endpoint.

The classifier predicts whether text resembles the WELFake dataset's Fake or Real class. It does not verify claims or sources; its confidence is a model estimate, not the probability that a story is factually true.

## Setup and train

From this directory, create an environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
```

Training reads `Data/WELFake_Dataset.csv`, prints held-out accuracy, precision, recall, F1-score and confusion matrix, then writes `artifacts/fake_news_model.joblib`. To use a different CSV or output path, pass `--dataset` and `--output`.

## Run the API

```bash
uvicorn app:app --reload
```

The API is available at `http://127.0.0.1:8000`, with interactive docs at `/docs`.

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"A headline or article goes here"}'
```

The response contains `prediction`, `confidence` (0–1), per-class `probabilities`, and a disclaimer. Empty text returns HTTP 422; text over 100,000 characters returns HTTP 413. `GET /health` reports process liveness. The model artifact must be built before starting the API. Set `FAKE_NEWS_MODEL_PATH` to load an artifact from another location.
