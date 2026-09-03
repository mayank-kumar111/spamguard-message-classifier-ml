"""Flask app for the spam/ham classifier.

Routes:
    /            landing page
    /predict     single + batch prediction
    /analytics   charts built from chart_data.json
    /about       model metrics and notes

    POST /api/predict         {"text": "..."}  -> one prediction
    POST /api/predict_batch    CSV upload       -> a prediction per row

Run with `python app.py` and open http://127.0.0.1:5000.
"""

import io
import json
import os
import sys
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from preprocessing import PLACEHOLDER_LABELS, clean_text, simple_tokens

MODELS_DIR = os.path.join(BASE_DIR, "models")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB upload cap
MAX_BATCH_ROWS = 2000


def _load_artifacts():
    required = ["model.pkl", "vectorizer.pkl", "metadata.joblib", "metrics.json"]
    missing = [f for f in required if not os.path.exists(os.path.join(MODELS_DIR, f))]
    if missing:
        raise FileNotFoundError(
            "Missing model files: %s. Run `python train_model.py` first."
            % ", ".join(missing)
        )
    model = joblib.load(os.path.join(MODELS_DIR, "model.pkl"))
    vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl"))
    metadata = joblib.load(os.path.join(MODELS_DIR, "metadata.joblib"))
    with open(os.path.join(MODELS_DIR, "metrics.json")) as f:
        metrics = json.load(f)

    chart_path = os.path.join(MODELS_DIR, "chart_data.json")
    chart = {}
    if os.path.exists(chart_path):
        with open(chart_path) as f:
            chart = json.load(f)
    return model, vectorizer, metadata, metrics, chart


MODEL, VECTORIZER, METADATA, METRICS, CHART = _load_artifacts()
WEIGHTS = np.asarray(METADATA["feature_weights"])
FEATURE_NAMES = METADATA["feature_names"]
BEST_MODEL_NAME = METADATA["best_model_name"]


def _friendly(token):
    return PLACEHOLDER_LABELS.get(token, token)


def _top_influential(x_row, k=8):
    # Weight each word present in the message by tfidf * model weight.
    # Positive pushes the message toward spam, negative toward ham.
    idx = x_row.nonzero()[1]
    items = []
    for i in idx:
        contrib = float(x_row[0, i]) * float(WEIGHTS[i])
        if contrib != 0:
            items.append((FEATURE_NAMES[i], contrib))
    items.sort(key=lambda t: abs(t[1]), reverse=True)
    out = []
    for token, contrib in items[:k]:
        out.append({
            "token": _friendly(token),
            "direction": "spam" if contrib > 0 else "ham",
            "weight": round(abs(contrib), 4),
        })
    return out


def predict_message(text):
    cleaned = clean_text(text)
    x = VECTORIZER.transform([cleaned])

    if x.nnz == 0:
        return {
            "label": "ham",
            "prediction": "Ham",
            "prob_spam": 0.0,
            "prob_ham": 1.0,
            "confidence": 0.0,
            "top_words": [],
            "words": [],
            "note": "No usable words left after cleaning, treated as ham by default.",
        }

    proba = MODEL.predict_proba(x)[0]
    prob_spam = float(proba[1])
    prob_ham = float(proba[0])
    is_spam = prob_spam >= 0.5
    return {
        "label": "spam" if is_spam else "ham",
        "prediction": "Spam" if is_spam else "Ham",
        "prob_spam": round(prob_spam, 4),
        "prob_ham": round(prob_ham, 4),
        "confidence": round(100 * max(prob_spam, prob_ham), 2),
        "top_words": _top_influential(x),
        "words": simple_tokens(text)[:50],   # readable words, for the page's word charts
        "note": None,
    }


def _find_text_column(df):
    # Figure out which column holds the message text.
    lower = {str(c).lower(): c for c in df.columns}
    for key in ("text", "message", "email", "body", "v2", "content"):
        if key in lower:
            return lower[key]
    obj_cols = [c for c in df.columns if df[c].dtype == object]
    if not obj_cols:
        return df.columns[0]
    return max(obj_cols, key=lambda c: df[c].astype(str).str.len().mean())


@app.route("/")
def index():
    return render_template("index.html", active="home",
                           metrics=METRICS, best_model=BEST_MODEL_NAME)


@app.route("/predict")
def predict_page():
    return render_template("predict.html", active="predict",
                           metrics=METRICS, best_model=BEST_MODEL_NAME)


@app.route("/analytics")
def analytics_page():
    return render_template("analytics.html", active="analytics",
                           metrics=METRICS, best_model=BEST_MODEL_NAME, chart=CHART)


@app.route("/about")
def about_page():
    return render_template("about.html", active="about",
                           metrics=METRICS, best_model=BEST_MODEL_NAME)


@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.get_json(silent=True) or {}
    text = (data.get("text") or "").strip()
    if not text:
        return jsonify({"error": "Please provide some text to classify."}), 400
    return jsonify(predict_message(text))


@app.route("/api/predict_batch", methods=["POST"])
def api_predict_batch():
    if "file" not in request.files or request.files["file"].filename == "":
        return jsonify({"error": "No CSV file uploaded."}), 400

    file = request.files["file"]
    try:
        raw = file.read()
        df = pd.read_csv(io.BytesIO(raw), encoding="latin-1")
    except Exception as exc:
        return jsonify({"error": f"Could not read CSV: {exc}"}), 400

    if df.empty:
        return jsonify({"error": "The uploaded CSV is empty."}), 400

    text_col = _find_text_column(df)
    truncated = len(df) > MAX_BATCH_ROWS
    df = df.head(MAX_BATCH_ROWS)

    results = []
    spam_count = 0
    # Count the trigger words across the whole file so the page can show a
    # summary without us shipping every row's tokens to the browser.
    token_counts = Counter()
    token_dir = {}
    # Readable words per class, for the top spam/ham word charts.
    word_counts = {"spam": Counter(), "ham": Counter()}

    for value in df[text_col].astype(str).tolist():
        res = predict_message(value)
        if res["label"] == "spam":
            spam_count += 1
        for word in res["top_words"][:5]:
            token_counts[word["token"]] += 1
            token_dir[word["token"]] = word["direction"]
        for w in res["words"]:
            word_counts[res["label"]][w] += 1
        preview = value if len(value) <= 160 else value[:157] + "..."
        results.append({
            "text": preview,
            "chars": len(value),          # real length, not the shortened preview
            "label": res["label"],
            "prediction": res["prediction"],
            "prob_spam": res["prob_spam"],
            "confidence": res["confidence"],
        })

    token_summary = [
        {"token": tok, "count": int(cnt), "direction": token_dir.get(tok, "spam")}
        for tok, cnt in token_counts.most_common(20)
    ]
    word_freq = {
        cls: [{"word": w, "count": int(c)} for w, c in counter.most_common(30)]
        for cls, counter in word_counts.items()
    }

    return jsonify({
        "text_column": str(text_col),
        "total": len(results),
        "spam": spam_count,
        "ham": len(results) - spam_count,
        "truncated": truncated,
        "max_rows": MAX_BATCH_ROWS,
        "results": results,
        "token_summary": token_summary,
        "word_freq": word_freq,
    })


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "File too large (max 5 MB)."}), 413


if __name__ == "__main__":
    print(f"Loaded best model: {BEST_MODEL_NAME}")
    print("Open http://127.0.0.1:5001 in your browser.")
    app.run(host="127.0.0.1", port=5001, debug=True)
