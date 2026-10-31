"""FastAPI service for the saved WELFake classifier."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


BACKEND_DIR = Path(__file__).resolve().parent
ARTIFACT_PATH = Path(os.getenv(
    "FAKE_NEWS_MODEL_PATH", str(BACKEND_DIR / "artifacts" / "fake_news_model.joblib")
)).expanduser()
MAX_TEXT_LENGTH = 100_000


class PredictionRequest(BaseModel):
    text: str = Field(..., description="A news headline or article to classify")


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    probabilities: dict[str, float]
    disclaimer: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not ARTIFACT_PATH.is_file():
        raise RuntimeError(
            f"Model artifact not found at {ARTIFACT_PATH}. "
            "Run `python train_model.py` from the backend directory first."
        )
    try:
        bundle: Any = joblib.load(ARTIFACT_PATH)
        vectorizer = bundle["vectorizer"]
        model = bundle["model"]
        labels = bundle["labels"]
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"Could not load model artifact {ARTIFACT_PATH}: {exc}") from exc

    if not hasattr(vectorizer, "transform") or not hasattr(model, "predict_proba"):
        raise RuntimeError("Model artifact must contain a vectorizer and a probabilistic classifier.")
    app.state.vectorizer = vectorizer
    app.state.model = model
    app.state.labels = labels
    yield


app = FastAPI(
    title="AI-Powered Fake News Detector",
    description="Classifies text using the WELFake-trained model. A prediction is not fact verification.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Text must not be empty.")
    if len(text) > MAX_TEXT_LENGTH:
        raise HTTPException(
            status_code=413,
            detail=f"Text exceeds the maximum length of {MAX_TEXT_LENGTH} characters.",
        )

    vectorizer = request.app.state.vectorizer
    model = request.app.state.model
    labels = request.app.state.labels
    vector = vectorizer.transform([text])
    probabilities = model.predict_proba(vector)[0]
    classes = model.classes_
    class_probabilities = {
        str(labels[int(label)]): min(1.0, max(0.0, float(probability)))
        for label, probability in zip(classes, probabilities)
    }
    predicted_label = int(classes[probabilities.argmax()])
    confidence = min(1.0, max(0.0, float(probabilities.max())))

    return PredictionResponse(
        prediction=str(labels[predicted_label]),
        confidence=confidence,
        probabilities=class_probabilities,
        disclaimer="This is a model prediction, not verified proof that the story is true or false.",
    )
