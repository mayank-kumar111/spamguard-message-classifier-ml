# SpamGuard — Spam/Ham Email Classifier

> An end-to-end machine learning web application for classifying email or SMS messages as **Spam** or **Ham**, with model comparison, TF-IDF text features, explainable predictions, batch CSV inference, analytics, and an interactive Flask interface.

![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.12-blue.svg) ![Flask](https://img.shields.io/badge/Flask-3.x-black.svg) ![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)

## Project Overview

SpamGuard demonstrates a complete classical NLP workflow: dataset cleaning, shared preprocessing, feature extraction, model comparison, evaluation, saved artifacts, and Flask inference.

The included SMS Spam Collection contains 5,572 raw messages; after duplicate removal, 5,158 unique records are used. The current recorded best model is **Linear SVM**.

### Recorded benchmark

| Metric | Result |
|---|---:|
| Accuracy | **98.93%** |
| Precision | **98.35%** |
| Recall | **92.97%** |
| F1 | **0.9558** |
| ROC-AUC | **0.9979** |
| Test messages | **1,032** |

> These are recorded benchmark results on the included dataset, not guaranteed real-world performance.

## Key Features

- Spam/Ham binary classification.
- Multinomial Naive Bayes, Logistic Regression, Linear SVM, and Random Forest comparison.
- TF-IDF unigrams/bigrams with `min_df=2`, `max_features=5000`, and `sublinear_tf=True`.
- CountVectorizer comparison.
- Shared preprocessing for training and inference.
- Feature-level prediction explanations.
- Single-message and CSV batch prediction.
- Interactive analytics, ROC curves, confusion matrix, frequent words, and word clouds.
- Dark/light theme.

## Machine Learning Pipeline

```text
Raw message
  → preprocessing
  → TF-IDF / CountVectorizer
  → four candidate classifiers
  → Accuracy / Precision / Recall / F1 / ROC-AUC
  → select best model by F1
  → calibrate Linear SVM when selected
  → save model/vectorizer/metadata/metrics
  → Flask web app + JSON API
```

### Text preprocessing

`preprocessing.py` lowercases text, replaces URLs, email addresses, currency, phone numbers and numeric values with placeholders, removes punctuation/stopwords/single-character tokens, and applies Porter stemming. NLTK stopwords have a built-in fallback.

| Placeholder | Meaning |
|---|---|
| `httpaddr` | URL |
| `emailaddr` | email address |
| `moneysymb` | currency |
| `phonenumbr` | phone number |
| `numbr` | number |

### Explainability

For active TF-IDF features, the application uses `TF-IDF value × model weight` as a lightweight local contribution. Positive values push toward Spam and negative values toward Ham.

## Project Structure

```text
spamguard-email-classifier-ml/
├── README.md
├── .gitignore
├── 21.Spam Ham Project.ipynb
├── app.py
├── preprocessing.py
├── train_model.py
├── requirements.txt
├── spam.csv
├── models/
│   ├── model.pkl
│   ├── vectorizer.pkl
│   ├── metadata.joblib
│   ├── metrics.json
│   └── chart_data.json
├── static/
│   ├── css/
│   ├── js/
│   ├── images/
│   └── interactive/
└── templates/
    ├── base.html
    ├── index.html
    ├── predict.html
    ├── analytics.html
    └── about.html
```

The tracked runnable workflow is centered on `train_model.py` and `app.py`. Local helper files excluded by `.gitignore` are not part of the documented repository structure.

## Application

| Page | Route | Purpose |
|---|---|---|
| Home | `/` | Overview and ML pipeline |
| Predict | `/predict` | Single and batch classification |
| Analytics | `/analytics` | Dataset/model visualizations |
| About | `/about` | Methodology and metrics |

### JSON API

Single prediction:

```http
POST /api/predict
Content-Type: application/json
```

```json
{"text":"You won a FREE prize, call now to claim!"}
```

The response includes `label`, `prediction`, `prob_spam`, `prob_ham`, `confidence`, and influential features.

Batch prediction:

```http
POST /api/predict_batch
Content-Type: multipart/form-data
```

Upload the CSV using the `file` field. Common text columns include `text`, `message`, `email`, `body`, `v2`, and `content`. The endpoint processes at most 2,000 rows and 5 MB per request.

## Quick Start

```bash
python -m venv venv

# Windows PowerShell
.\venv\Scripts\Activate.ps1

# macOS / Linux
# source venv/bin/activate

pip install -r requirements.txt
python train_model.py
python app.py
```

Open `http://127.0.0.1:5001`.

## Training

```bash
python train_model.py
python train_model.py --data path/to/your.csv
python train_model.py --vectorizer count
```

Training generates or refreshes model artifacts and visualization outputs.

## Requirements

Python 3.10–3.12 is the documented/tested range. Install dependencies with:

```bash
pip install -r requirements.txt
```

Main packages: Flask, scikit-learn, pandas, NumPy, Matplotlib, Seaborn, WordCloud, NLTK, Joblib, and Plotly.

## Saved Artifacts

- `models/model.pkl` — application classifier
- `models/vectorizer.pkl` — saved feature vectorizer
- `models/metadata.joblib` — feature/model metadata
- `models/metrics.json` — dataset and evaluation metrics
- `models/chart_data.json` — analytics data

## Reproducibility and Limitations

The recorded experiment uses a stratified 80/20 train/test split with random state `42`. Results can change when source data or dependency versions change. This is a college-level local ML application, not a hardened production email-security gateway, and its benchmark is based on the included SMS dataset.

## Future Enhancements

Broader datasets, cross-dataset evaluation, threshold tuning, deeper error analysis, experiment tracking, automated tests/CI, and containerized deployment are possible future work and are intentionally outside the current scope.

## Tech Stack

**Python · Flask · scikit-learn · NLTK · pandas · NumPy · Matplotlib · Seaborn · Plotly · Chart.js · Bootstrap · JavaScript**
