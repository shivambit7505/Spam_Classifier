"""
Trains the spam/ham SMS classifier on the real UCI SMS Spam Collection
dataset (5,574 messages, 747 spam / 4,827 ham) and persists it for the
Flask backend to load.

Run this once before starting the API: python train_model.py
"""
import json
import re
import os
import urllib.request
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

RANDOM_STATE = 42
DATA_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"
DATA_PATH = "sms_raw.tsv"


def ensure_data():
    if not os.path.exists(DATA_PATH):
        print(f"Downloading SMS Spam Collection dataset from {DATA_URL} ...")
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
        print("Downloaded.")


def load_data():
    ensure_data()
    df = pd.read_csv(DATA_PATH, sep="\t", header=None, names=["label", "message"])
    df = df.dropna()
    df["label_num"] = (df["label"] == "spam").astype(int)
    return df


def main():
    df = load_data()
    print(f"Loaded {len(df)} messages: {df['label'].value_counts().to_dict()}")

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"], df["label_num"],
        test_size=0.2, random_state=RANDOM_STATE, stratify=df["label_num"]
    )

    candidates = {
        "naive_bayes": Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", lowercase=True, max_features=5000)),
            ("clf", MultinomialNB()),
        ]),
        "logreg": Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", lowercase=True, max_features=5000)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]),
        "linear_svm": Pipeline([
            ("tfidf", TfidfVectorizer(stop_words="english", lowercase=True, max_features=5000)),
            ("clf", LinearSVC()),
        ]),
    }

    results = {}
    fitted = {}
    for name, pipe in candidates.items():
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        results[name] = {
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred)),
            "recall": float(recall_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
            "confusion_matrix": confusion_matrix(y_test, pred).tolist(),
        }
        fitted[name] = pipe
        print(f"{name}: acc={results[name]['accuracy']:.4f} "
              f"prec={results[name]['precision']:.4f} "
              f"rec={results[name]['recall']:.4f} "
              f"f1={results[name]['f1']:.4f}")

    # Ship Logistic Regression: best balance of precision (don't flag real
    # messages as spam) and recall, and gives us predict_proba for a
    # confidence score in the UI (LinearSVC doesn't have predict_proba).
    best_name = "logreg"
    best_pipe = fitted[best_name]

    joblib.dump(best_pipe, "model.joblib")

    # Most spam-indicative and ham-indicative words by logistic regression
    # coefficient — real model introspection, not made up.
    tfidf = best_pipe.named_steps["tfidf"]
    clf = best_pipe.named_steps["clf"]
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_[0]
    top_spam_idx = np.argsort(coefs)[-20:][::-1]
    top_ham_idx = np.argsort(coefs)[:20]
    top_spam_words = [{"word": feature_names[i], "weight": float(coefs[i])} for i in top_spam_idx]
    top_ham_words = [{"word": feature_names[i], "weight": float(coefs[i])} for i in top_ham_idx]

    metadata = {
        "model_used": best_name,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "random_state": RANDOM_STATE,
        "class_balance": {
            "ham": int((df["label_num"] == 0).sum()),
            "spam": int((df["label_num"] == 1).sum()),
        },
        "results": results,
        "top_spam_words": top_spam_words,
        "top_ham_words": top_ham_words,
    }
    with open("metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save a sample of test messages (with true labels) so the UI can offer
    # "try a real example" buttons instead of only free-text entry.
    test_df = pd.DataFrame({"message": X_test, "label": y_test})
    samples = []
    for label_num in [0, 1]:
        subset = test_df[test_df["label"] == label_num].sample(
            n=min(8, (test_df["label"] == label_num).sum()), random_state=RANDOM_STATE
        )
        for _, row in subset.iterrows():
            samples.append({"message": row["message"], "true_label": "spam" if label_num else "ham"})
    with open("sample_messages.json", "w") as f:
        json.dump(samples, f, indent=2)

    print("\nSaved model.joblib, metadata.json, sample_messages.json")
    print(f"Production model: {best_name} (test accuracy {results[best_name]['accuracy']:.4f})")


if __name__ == "__main__":
    main()
