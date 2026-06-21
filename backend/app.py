"""
Flask API for the SMS spam/ham classifier.

Endpoints:
  GET  /api/health         -> service status + model info
  GET  /api/metadata        -> accuracy/precision/recall, confusion matrix, top words
  GET  /api/samples         -> real example messages from the test set (try-it buttons)
  POST /api/predict         -> { message: "..." } -> spam/ham prediction + confidence + flagged words

Run:
  python train_model.py   # once, to produce model.joblib etc.
  python app.py            # starts the API on :5001
"""
import html
import json
import re
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

MODEL = joblib.load("model.joblib")
with open("metadata.json") as f:
    METADATA = json.load(f)
with open("sample_messages.json") as f:
    SAMPLE_MESSAGES = json.load(f)

TFIDF = MODEL.named_steps["tfidf"]
CLF = MODEL.named_steps["clf"]
VOCAB = TFIDF.vocabulary_
FEATURE_NAMES = TFIDF.get_feature_names_out()
COEFS = CLF.coef_[0]


def clean_text(s):
    return html.unescape(s).strip()


def validate_payload(data):
    errors = []
    if not isinstance(data, dict):
        return ["Request body must be a JSON object."]
    if "message" not in data:
        errors.append("Missing field: message")
    elif not isinstance(data["message"], str):
        errors.append("Field 'message' must be a string.")
    elif len(data["message"].strip()) == 0:
        errors.append("Field 'message' cannot be empty.")
    elif len(data["message"]) > 2000:
        errors.append("Field 'message' is too long (max 2000 characters).")
    return errors


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model": METADATA["model_used"],
        "test_accuracy": METADATA["results"][METADATA["model_used"]]["accuracy"],
    })


@app.route("/api/metadata", methods=["GET"])
def metadata():
    return jsonify(METADATA)


@app.route("/api/samples", methods=["GET"])
def samples():
    cleaned = [{"message": clean_text(s["message"]), "true_label": s["true_label"]} for s in SAMPLE_MESSAGES]
    return jsonify({"count": len(cleaned), "samples": cleaned})


@app.route("/api/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    errors = validate_payload(data)
    if errors:
        return jsonify({"error": "Validation failed", "details": errors}), 400

    message = data["message"]

    pred = int(MODEL.predict([message])[0])
    proba = MODEL.predict_proba([message])[0].tolist()

    # Real per-word contribution: tokenize with the model's own TF-IDF
    # vectorizer, and report each present vocabulary word's learned
    # logistic-regression coefficient (positive = pushes toward spam).
    tfidf_vec = TFIDF.transform([message])
    nonzero_idx = tfidf_vec.nonzero()[1]
    word_contributions = []
    for idx in nonzero_idx:
        word_contributions.append({
            "word": FEATURE_NAMES[idx],
            "weight": float(COEFS[idx]),
            "tfidf": float(tfidf_vec[0, idx]),
        })
    word_contributions.sort(key=lambda w: abs(w["weight"]), reverse=True)

    response = {
        "prediction": "spam" if pred == 1 else "ham",
        "confidence": round(max(proba), 4),
        "probabilities": {"ham": round(proba[0], 4), "spam": round(proba[1], 4)},
        "flagged_words": word_contributions[:8],
        "message": message,
    }
    return jsonify(response)


@app.errorhandler(404)
def not_found(e):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
