"""Train and compare the spam/ham models, then save the best one.

Loads the dataset, builds TF-IDF features (with a CountVectorizer baseline for
comparison), trains Naive Bayes, Logistic Regression, Linear SVM and Random
Forest, picks the best by F1, and writes the model, vectorizer, metrics and the
chart data used by the web pages.

    python train_model.py                      # uses spam.csv next to this file
    python train_model.py --data my.csv        # a different dataset
    python train_model.py --vectorizer count   # save the CountVectorizer model
"""

import argparse
import json
import os
import time
from collections import Counter

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from preprocessing import clean_text, simple_tokens

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
IMG_DIR = os.path.join(BASE_DIR, "static", "images")
INTERACTIVE_DIR = os.path.join(BASE_DIR, "static", "interactive")

for d in (MODELS_DIR, IMG_DIR, INTERACTIVE_DIR):
    os.makedirs(d, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
np.random.seed(RANDOM_STATE)

sns.set_theme(style="whitegrid")
SPAM_COLOR = "#e74c3c"
HAM_COLOR = "#2ecc71"
PALETTE = ["#4e79a7", "#f28e2b", "#59a14f", "#e15759"]


def load_data(path):
    """Read the CSV and return a tidy frame with label + text columns."""
    df = pd.read_csv(path, encoding="latin-1")
    cols_lower = [str(c).lower() for c in df.columns]

    if "text" in cols_lower and "label" in cols_lower:
        label_col = df.columns[cols_lower.index("label")]
        text_col = df.columns[cols_lower.index("text")]
        df = df.rename(columns={label_col: "label", text_col: "text"})[["label", "text"]]
    else:
        # Kaggle spam.csv layout: label, text, then a few stray overflow columns
        # where commas in the message spilled into extra fields. Stitch them back.
        label_col, text_col = df.columns[0], df.columns[1]
        extra_cols = list(df.columns[2:])

        def _merge(row):
            parts = []
            if pd.notna(row[text_col]):
                parts.append(str(row[text_col]))
            for c in extra_cols:
                if pd.notna(row[c]):
                    parts.append(str(row[c]))
            return " ".join(parts).strip()

        df["text"] = df.apply(_merge, axis=1)
        df = df.rename(columns={label_col: "label"})[["label", "text"]]

    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df = df[df["label"].isin(["ham", "spam"])]
    df = df.dropna(subset=["text"])
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.strip() != ""]

    n_raw = len(df)
    df = df.drop_duplicates(subset=["label", "text"]).reset_index(drop=True)
    n_dupes = n_raw - len(df)

    df["target"] = (df["label"] == "spam").astype(int)
    df["clean"] = df["text"].apply(clean_text)
    df["char_len"] = df["text"].str.len()
    df["word_len"] = df["text"].str.split().apply(len)

    print(f"Loaded {n_raw} rows -> {len(df)} unique ({n_dupes} duplicates removed)")
    print(df["label"].value_counts().to_string())
    return df, n_raw, n_dupes


def build_vectorizer(kind="tfidf"):
    common = dict(ngram_range=(1, 2), min_df=2, max_features=5000)
    if kind == "count":
        return CountVectorizer(**common)
    return TfidfVectorizer(sublinear_tf=True, **common)


def get_models():
    return {
        "Multinomial Naive Bayes": MultinomialNB(alpha=0.1),
        "Logistic Regression": LogisticRegression(
            C=3.0, max_iter=2000, random_state=RANDOM_STATE
        ),
        "Linear SVM": LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }


def get_scores(model, X):
    # LinearSVC has no predict_proba, so fall back to its decision function.
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    scores = get_scores(model, X_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, pos_label=1, zero_division=0),
        "recall": recall_score(y_test, y_pred, pos_label=1, zero_division=0),
        "f1": f1_score(y_test, y_pred, pos_label=1, zero_division=0),
        "roc_auc": roc_auc_score(y_test, scores),
        "_y_pred": y_pred,
        "_scores": scores,
    }


def feature_weights_from(model, X_train, y_train):
    # One signed weight per feature: positive leans spam, negative leans ham.
    # This lets the app show which words pushed a prediction each way.
    if hasattr(model, "coef_"):  # logistic regression, linear svm
        return np.asarray(model.coef_).ravel()
    if hasattr(model, "feature_log_prob_"):  # naive bayes
        return model.feature_log_prob_[1] - model.feature_log_prob_[0]
    if hasattr(model, "feature_importances_"):  # random forest: give it a sign
        importances = model.feature_importances_
        X = X_train.tocsr()
        spam_mean = np.asarray(X[y_train == 1].mean(axis=0)).ravel()
        ham_mean = np.asarray(X[y_train == 0].mean(axis=0)).ravel()
        sign = np.sign(spam_mean - ham_mean)
        return importances * sign
    return np.zeros(X_train.shape[1])


def make_wordclouds(df):
    from wordcloud import WordCloud

    for label, colormap, fname in [
        ("spam", "Reds", "wordcloud_spam.png"),
        ("ham", "Greens", "wordcloud_ham.png"),
    ]:
        tokens = []
        for t in df[df["label"] == label]["text"]:
            tokens.extend(simple_tokens(t))
        freq = Counter(tokens)
        wc = WordCloud(
            width=800, height=400, background_color=None, mode="RGBA",
            colormap=colormap, max_words=150, prefer_horizontal=0.9,
        ).generate_from_frequencies(freq)
        wc.to_file(os.path.join(IMG_DIR, fname))
    print("  - word clouds saved")


def make_class_distribution(df):
    counts = df["label"].value_counts()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    order = ["ham", "spam"]
    vals = [int(counts.get(k, 0)) for k in order]
    colors = [HAM_COLOR, SPAM_COLOR]

    axes[0].bar(order, vals, color=colors)
    for i, v in enumerate(vals):
        axes[0].text(i, v + max(vals) * 0.01, f"{v:,}", ha="center", fontweight="bold")
    axes[0].set_title("Message count by class")
    axes[0].set_ylabel("Number of messages")

    axes[1].pie(vals, labels=[o.upper() for o in order], colors=colors,
                autopct="%1.1f%%", startangle=90,
                wedgeprops=dict(edgecolor="white", linewidth=2))
    axes[1].set_title("Class proportion")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "class_distribution.png"), dpi=130)
    plt.close(fig)
    print("  - class distribution saved")


def make_frequent_words(df, top_n=20):
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    for ax, label, color in [(axes[0], "spam", SPAM_COLOR), (axes[1], "ham", HAM_COLOR)]:
        tokens = []
        for t in df[df["label"] == label]["text"]:
            tokens.extend(simple_tokens(t))
        common = Counter(tokens).most_common(top_n)
        words = [w for w, _ in common][::-1]
        freqs = [c for _, c in common][::-1]
        ax.barh(words, freqs, color=color)
        ax.set_title(f"Top {top_n} words in {label.upper()}")
        ax.set_xlabel("Frequency")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "frequent_words.png"), dpi=130)
    plt.close(fig)
    print("  - frequent words saved")


def make_length_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 5))
    bins = np.linspace(0, df["char_len"].quantile(0.99), 50)
    ax.hist(df[df["label"] == "ham"]["char_len"], bins=bins, alpha=0.6,
            label="Ham", color=HAM_COLOR)
    ax.hist(df[df["label"] == "spam"]["char_len"], bins=bins, alpha=0.6,
            label="Spam", color=SPAM_COLOR)
    ax.set_title("Message length distribution (characters)")
    ax.set_xlabel("Message length (characters)")
    ax.set_ylabel("Number of messages")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "length_distribution.png"), dpi=130)
    plt.close(fig)
    print("  - length distribution saved")


def make_model_comparison(results):
    metrics = ["accuracy", "precision", "recall", "f1"]
    names = list(results.keys())
    x = np.arange(len(metrics))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 6))
    for i, name in enumerate(names):
        vals = [results[name][m] for m in metrics]
        ax.bar(x + (i - 1.5) * width, vals, width, label=name, color=PALETTE[i])
    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0.8, 1.005)
    ax.set_ylabel("Score")
    ax.set_title("Model performance comparison")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "model_comparison.png"), dpi=130)
    plt.close(fig)

    import plotly.graph_objects as go

    fig2 = go.Figure()
    for i, name in enumerate(names):
        fig2.add_trace(go.Bar(
            name=name, x=[m.capitalize() for m in metrics],
            y=[round(results[name][m], 4) for m in metrics],
            marker_color=PALETTE[i],
        ))
    fig2.update_layout(
        barmode="group", title="Model performance comparison (interactive)",
        yaxis=dict(range=[0.8, 1.01], title="Score"),
        template="plotly_white", legend=dict(orientation="h", y=-0.2),
        margin=dict(l=40, r=20, t=60, b=40),
    )
    fig2.write_html(os.path.join(INTERACTIVE_DIR, "model_comparison.html"),
                    include_plotlyjs="cdn", full_html=True)
    print("  - model comparison saved")


def make_confusion_matrix(y_test, y_pred, best_name):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Ham", "Spam"], yticklabels=["Ham", "Spam"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix ({best_name})")
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "confusion_matrix.png"), dpi=130)
    plt.close(fig)
    print("  - confusion matrix saved")


def make_roc_curves(results, y_test):
    fig, ax = plt.subplots(figsize=(8, 6))
    roc_data = {}
    for i, (name, res) in enumerate(results.items()):
        fpr, tpr, _ = roc_curve(y_test, res["_scores"])
        ax.plot(fpr, tpr, color=PALETTE[i], lw=2,
                label=f"{name} (AUC = {res['roc_auc']:.3f})")
        roc_data[name] = (fpr.tolist(), tpr.tolist(), res["roc_auc"])
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curves")
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "roc_curve.png"), dpi=130)
    plt.close(fig)

    import plotly.graph_objects as go

    fig2 = go.Figure()
    for i, (name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
        fig2.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc:.3f})",
                                  line=dict(color=PALETTE[i], width=2)))
    fig2.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines",
                              line=dict(dash="dash", color="grey"), showlegend=False))
    fig2.update_layout(title="ROC curves (interactive)", template="plotly_white",
                       xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                       legend=dict(orientation="h", y=-0.2),
                       margin=dict(l=40, r=20, t=60, b=40))
    fig2.write_html(os.path.join(INTERACTIVE_DIR, "roc_curve.html"),
                    include_plotlyjs="cdn", full_html=True)
    print("  - ROC curves saved")


def export_chart_data(df, results, y_test, best_name):
    # Save the raw numbers so the pages can draw their own charts client-side.
    cm = confusion_matrix(y_test, results[best_name]["_y_pred"])
    tn, fp, fn, tp = (int(v) for v in cm.ravel())

    grid = np.linspace(0, 1, 100)
    roc = {}
    for name, res in results.items():
        fpr, tpr, _ = roc_curve(y_test, res["_scores"])
        roc[name] = {
            "fpr": [round(float(x), 4) for x in grid],
            "tpr": [round(float(np.interp(g, fpr, tpr)), 4) for g in grid],
            "auc": round(float(res["roc_auc"]), 4),
        }

    edges = list(range(0, 320, 20))
    ham_len = df[df.label == "ham"]["char_len"].clip(upper=315)
    spam_len = df[df.label == "spam"]["char_len"].clip(upper=315)
    ham_counts, _ = np.histogram(ham_len, bins=edges)
    spam_counts, _ = np.histogram(spam_len, bins=edges)
    length_labels = [f"{edges[i]}-{edges[i + 1]}" for i in range(len(edges) - 1)]

    def top_words(label, n=12):
        toks = []
        for t in df[df["label"] == label]["text"]:
            toks.extend(simple_tokens(t))
        return [{"word": w, "count": int(c)} for w, c in Counter(toks).most_common(n)]

    chart_data = {
        "confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp, "model": best_name},
        "roc": roc,
        "length": {
            "labels": length_labels,
            "ham": [int(x) for x in ham_counts],
            "spam": [int(x) for x in spam_counts],
        },
        "frequent_words": {"spam": top_words("spam"), "ham": top_words("ham")},
        "models": {
            name: {
                "accuracy": round(r["accuracy"], 4),
                "precision": round(r["precision"], 4),
                "recall": round(r["recall"], 4),
                "f1": round(r["f1"], 4),
                "roc_auc": round(r["roc_auc"], 4),
            }
            for name, r in results.items()
        },
    }
    with open(os.path.join(MODELS_DIR, "chart_data.json"), "w") as f:
        json.dump(chart_data, f, indent=2)
    print("  - chart_data.json saved")


def main():
    parser = argparse.ArgumentParser(description="Train the spam/ham classifier.")
    parser.add_argument("--data", default=os.path.join(BASE_DIR, "spam.csv"),
                        help="Path to a CSV with text + label columns.")
    parser.add_argument("--vectorizer", choices=["tfidf", "count"], default="tfidf",
                        help="Which vectorizer to save for the app.")
    args = parser.parse_args()

    print("=" * 64)
    print("Spam/ham classifier - training")
    print("=" * 64)

    df, n_raw, n_dupes = load_data(args.data)

    X_train_txt, X_test_txt, y_train, y_test = train_test_split(
        df["clean"], df["target"], test_size=TEST_SIZE,
        stratify=df["target"], random_state=RANDOM_STATE,
    )

    # Fit both vectorizers so we can report how TF-IDF stacks up against counts.
    print("\nBuilding features and comparing vectorizers...")
    vectorizer_comparison = {}
    fitted = {}
    for kind in ("tfidf", "count"):
        vec = build_vectorizer(kind)
        Xtr = vec.fit_transform(X_train_txt)
        Xte = vec.transform(X_test_txt)
        fitted[kind] = (vec, Xtr, Xte)
        per_model = {}
        for name, model in get_models().items():
            model.fit(Xtr, y_train)
            per_model[name] = round(f1_score(y_test, model.predict(Xte), zero_division=0), 4)
        vectorizer_comparison[kind] = per_model
        print(f"  {kind:6s} vocab size: {len(vec.get_feature_names_out())}")

    vectorizer, X_train, X_test = fitted[args.vectorizer]
    feature_names = list(vectorizer.get_feature_names_out())

    # Train each model on the chosen vectorizer and score it on the test set.
    print(f"\nTraining models on {args.vectorizer.upper()} features...")
    results = {}
    trained = {}
    for name, model in get_models().items():
        t0 = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - t0
        res = evaluate(model, X_test, y_test)
        res["train_time"] = round(train_time, 3)
        results[name] = res
        trained[name] = model
        print(f"  {name:26s} acc={res['accuracy']:.4f}  f1={res['f1']:.4f}  "
              f"auc={res['roc_auc']:.4f}  ({train_time:.2f}s)")

    best_name = max(results, key=lambda n: (results[n]["f1"], results[n]["accuracy"]))
    best_model = trained[best_name]
    print(f"\nBest model: {best_name} (F1={results[best_name]['f1']:.4f})")

    weights = feature_weights_from(best_model, X_train, y_train)

    # LinearSVC can't give probabilities, so calibrate it before saving.
    if hasattr(best_model, "predict_proba"):
        app_model = best_model
    else:
        print("Calibrating the best model for probability estimates...")
        app_model = CalibratedClassifierCV(
            LinearSVC(C=1.0, max_iter=5000, random_state=RANDOM_STATE), cv=5
        )
        app_model.fit(X_train, y_train)

    order = np.argsort(weights)
    top_ham_tokens = [feature_names[i] for i in order[:15]]
    top_spam_tokens = [feature_names[i] for i in order[::-1][:15]]

    joblib.dump(app_model, os.path.join(MODELS_DIR, "model.pkl"))
    joblib.dump(vectorizer, os.path.join(MODELS_DIR, "vectorizer.pkl"))
    joblib.dump(
        {
            "feature_weights": weights,
            "feature_names": feature_names,
            "best_model_name": best_name,
            "vectorizer_kind": args.vectorizer,
            "classes": ["ham", "spam"],
        },
        os.path.join(MODELS_DIR, "metadata.joblib"),
    )

    metrics = {
        "dataset": {
            "total_raw": int(n_raw),
            "duplicates_removed": int(n_dupes),
            "total_unique": int(len(df)),
            "ham": int((df["label"] == "ham").sum()),
            "spam": int((df["label"] == "spam").sum()),
            "spam_pct": round(100 * (df["label"] == "spam").mean(), 2),
            "train_size": int(len(y_train)),
            "test_size": int(len(y_test)),
            "avg_len_ham": round(df[df.label == "ham"]["char_len"].mean(), 1),
            "avg_len_spam": round(df[df.label == "spam"]["char_len"].mean(), 1),
        },
        "vectorizer": {
            "used_for_app": args.vectorizer,
            "ngram_range": "(1, 2)",
            "min_df": 2,
            "max_features": 5000,
            "vocab_size": len(feature_names),
        },
        "best_model": best_name,
        "models": {
            name: {
                "accuracy": round(r["accuracy"], 4),
                "precision": round(r["precision"], 4),
                "recall": round(r["recall"], 4),
                "f1": round(r["f1"], 4),
                "roc_auc": round(r["roc_auc"], 4),
                "train_time": r["train_time"],
            }
            for name, r in results.items()
        },
        "vectorizer_comparison": vectorizer_comparison,
        "top_spam_tokens": top_spam_tokens,
        "top_ham_tokens": top_ham_tokens,
        "preprocessing_steps": [
            "Lowercase the message",
            "Swap URLs, emails, currency, phone numbers and digits for placeholder tokens",
            "Strip the leftover punctuation",
            "Split into word tokens",
            "Drop English stopwords and single letters",
            "Apply Porter stemming",
        ],
    }
    with open(os.path.join(MODELS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print("Saved model.pkl, vectorizer.pkl, metadata.joblib, metrics.json")

    print("\nGenerating figures...")
    make_wordclouds(df)
    make_class_distribution(df)
    make_frequent_words(df)
    make_length_distribution(df)
    make_model_comparison(results)
    make_confusion_matrix(y_test, results[best_name]["_y_pred"], best_name)
    make_roc_curves(results, y_test)
    export_chart_data(df, results, y_test, best_name)

    print("\nDone. Best model:", best_name)
    print("=" * 64)


if __name__ == "__main__":
    main()
