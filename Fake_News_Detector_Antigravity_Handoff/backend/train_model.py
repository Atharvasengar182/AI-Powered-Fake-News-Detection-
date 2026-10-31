"""Train and save the WELFake TF-IDF + Logistic Regression artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


BACKEND_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = BACKEND_DIR / "Data" / "WELFake_Dataset.csv"
DEFAULT_ARTIFACT = BACKEND_DIR / "artifacts" / "fake_news_model.joblib"


def prepare_dataset(csv_path: Path) -> tuple[pd.Series, pd.Series]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Dataset not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    required = {"title", "text", "label"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    content = (
        frame["title"].fillna("").astype(str).str.strip()
        + " "
        + frame["text"].fillna("").astype(str).str.strip()
    ).str.strip()
    cleaned = pd.DataFrame({"content": content, "label": frame["label"]})
    cleaned = cleaned[cleaned["content"].ne("") & cleaned["label"].notna()]
    cleaned = cleaned.drop_duplicates(subset="content").reset_index(drop=True)

    labels = set(cleaned["label"].unique())
    if labels != {0, 1}:
        raise ValueError(f"Expected WELFake labels 0 and 1; found {sorted(labels, key=str)}")
    if cleaned["label"].value_counts().min() < 2:
        raise ValueError("Each label needs at least two examples for a stratified split.")

    return cleaned["content"], cleaned["label"].astype(int)


def train(dataset_path: Path, artifact_path: Path) -> dict[str, object]:
    texts, labels = prepare_dataset(dataset_path)
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.20, random_state=42, stratify=labels
    )

    vectorizer = TfidfVectorizer(
        max_features=50_000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2,
    )
    x_train_tfidf = vectorizer.fit_transform(x_train)
    x_test_tfidf = vectorizer.transform(x_test)
    model = LogisticRegression(max_iter=1000)
    model.fit(x_train_tfidf, y_train)

    predictions = model.predict(x_test_tfidf)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Test accuracy: {accuracy:.4%}")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Real News", "Fake News"],
        zero_division=0,
    ))
    print("Confusion matrix (actual rows / predicted columns; real, fake):")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "vectorizer": vectorizer,
            "model": model,
            "labels": {0: "REAL NEWS", 1: "FAKE NEWS"},
            "test_accuracy": float(accuracy),
        },
        artifact_path,
    )
    print(f"Saved model artifacts to {artifact_path}")
    return {"accuracy": float(accuracy), "artifact_path": str(artifact_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT)
    args = parser.parse_args()
    train(args.dataset, args.output)


if __name__ == "__main__":
    main()
