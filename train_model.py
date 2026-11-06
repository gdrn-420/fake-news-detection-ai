"""
Train a TF-IDF + Logistic Regression text classifier.
Run from the project folder: python train_model.py
"""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "news_samples.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

def main():
    df = pd.read_csv(DATA)
    df = df.dropna(subset=["text", "label"])
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.upper().str.strip()
    df = df[df["label"].isin(["REAL", "FAKE"])]
    if len(df) < 10 or df["label"].nunique() != 2:
        raise ValueError("Dataset must contain at least 10 rows and both REAL and FAKE labels.")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2), max_features=5000)),
        ("classifier", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42))
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("TEST SET RESULTS (for this small synthetic dataset only)")
    print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")
    print(f"Accuracy : {accuracy_score(y_test, predictions):.3f}")
    print(f"Precision: {precision_score(y_test, predictions, pos_label='FAKE', zero_division=0):.3f}")
    print(f"Recall   : {recall_score(y_test, predictions, pos_label='FAKE', zero_division=0):.3f}")
    print(f"F1-score : {f1_score(y_test, predictions, pos_label='FAKE', zero_division=0):.3f}")
    print("Confusion matrix (labels order: FAKE, REAL):")
    print(confusion_matrix(y_test, predictions, labels=["FAKE", "REAL"]))
    print("\nClassification report:")
    print(classification_report(y_test, predictions, zero_division=0))
    joblib.dump(model, MODEL_DIR / "fake_news_pipeline.joblib")
    print(f"Saved model to {MODEL_DIR / 'fake_news_pipeline.joblib'}")
    print("Reminder: these scores measure performance on invented examples, not real-world fact-checking.")

if __name__ == "__main__":
    main()
