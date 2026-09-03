# SpamGuard — Spam/Ham Email Classifier

> End-to-end NLP and machine learning application for classifying email or SMS-style messages as **Spam** or **Ham**, with model comparison, TF-IDF features, explainable predictions, batch CSV inference, analytics, and a Flask web interface.

![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.12-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.x-black.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-orange.svg)
![Project Status](https://img.shields.io/badge/Status-Active-success.svg)

## Table of Contents

- [Project Overview](#project-overview)
- [What It Does](#what-it-does)
- [Key Features](#key-features)
- [Machine Learning Pipeline](#machine-learning-pipeline)
- [Dataset](#dataset)
- [Text Preprocessing](#text-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Models](#models)
- [Model Selection and Evaluation](#model-selection-and-evaluation)
- [Explainable Predictions](#explainable-predictions)
- [Application Architecture](#application-architecture)
- [Project Structure](#project-structure)
- [Application Pages](#application-pages)
- [REST API](#rest-api)
- [Batch Prediction](#batch-prediction)
- [Session Analytics](#session-analytics)
- [Quick Start (Windows)](#quick-start-windows)
- [Manual Installation](#manual-installation)
- [Training the Model](#training-the-model)
- [Running the Application](#running-the-application)
- [Requirements](#requirements)
- [Model Artifacts](#model-artifacts)
- [Visualization and Reporting](#visualization-and-reporting)
- [Tech Stack](#tech-stack)
- [Reproducibility Notes](#reproducibility-notes)
- [Limitations and Notes](#limitations-and-notes)
- [Future Enhancements](#future-enhancements)
- [Project Status](#project-status)

---

## Project Overview

**SpamGuard** is an end-to-end Spam/Ham text classification project built with Python, scikit-learn, NLTK, and Flask.

The project takes raw email/SMS-style text, cleans and normalizes it, converts the text into numerical TF-IDF features, evaluates multiple machine learning algorithms, selects the strongest model based on F1 score, stores the trained artifacts, and serves predictions through a web interface and JSON API.

The application is designed not only to demonstrate classification accuracy, but also to make the model more understandable through feature-level explanations and interactive analytics.

### Problem Statement

Spam messages are unwanted or potentially harmful messages that may contain promotional content, fraudulent offers, suspicious links, or attempts to manipulate users into taking an action. The objective of SpamGuard is to automatically classify a message into one of two classes:

- **Ham** — legitimate/non-spam message
- **Spam** — unwanted or suspicious message

The project demonstrates a practical Natural Language Processing (NLP) workflow from raw text through model inference.

---

## What It Does

The project trains and compares four models on the **SMS Spam Collection dataset** (`spam.csv`, 5,572 messages) and serves predictions through a dark/light themed web interface with a live prediction gauge, interactive charts, explanations, and CSV batch processing.

The current recorded experiment reports **Linear SVM** as the best model on the held-out test set:

| Metric | Result |
|---|---:|
| Accuracy | **98.9%** |
| F1 Score | **0.956** |
| ROC-AUC | **0.998** |
| False Positives | **2 / 1,032 test messages** |

> These are the project's recorded benchmark results on the available dataset and should not be interpreted as guaranteed real-world production performance.

---

## Key Features

- Binary Spam/Ham classification for email or SMS-style messages.
- Model comparison across **Multinomial Naive Bayes, Logistic Regression, Linear SVM, and Random Forest**.
- TF-IDF features using **unigrams and bigrams**, with a maximum vocabulary of 5,000 terms.
- CountVectorizer comparison against TF-IDF.
- Shared preprocessing between training and inference.
- Feature-level explanations showing terms that pushed predictions toward Spam or Ham.
- Single-message prediction through the web UI and REST API.
- Live prediction that updates as the user types.
- CSV batch upload and inference.
- Session analytics on submitted predictions.
- Interactive Chart.js analytics plus standalone Plotly charts.
- Static figures for reports, including confusion matrix, ROC curve, class distribution, message-length distribution, frequent words, and word clouds.
- Dark/light theme with `localStorage` persistence.
- Windows setup and training shortcuts.
- Offline-friendly NLTK stopword fallback.

---

## Machine Learning Pipeline

```text
Raw message
    ↓
Text preprocessing
    ├── lowercase
    ├── URL/email/currency/phone/number placeholders
    ├── punctuation cleanup
    ├── stopword removal
    └── Porter stemming
    ↓
TF-IDF / CountVectorizer
    ↓
Train candidate models
    ├── Multinomial Naive Bayes
    ├── Logistic Regression
    ├── Linear SVM
    └── Random Forest
    ↓
Evaluate
    ├── Accuracy
    ├── Precision
    ├── Recall
    ├── F1
    └── ROC-AUC
    ↓
Select best model by F1 (accuracy tie-break)
    ↓
Calibrate Linear SVM probabilities when selected
    ↓
Save model + vectorizer + metadata + metrics
    ↓
Flask web application + JSON API
```

The training and inference workflow is implemented so the same `clean_text()` logic is used in both places, reducing the risk of training/serving preprocessing mismatch.

---

## Dataset

The project uses the **SMS Spam Collection** dataset stored in `spam.csv`.

The training script also supports a different CSV through:

```bash
python train_model.py --data path/to/your.csv
```

### Dataset handling

The training code:

1. Reads the CSV using `latin-1` encoding.
2. Supports a standard `text` + `label` format.
3. Supports the common Kaggle `spam.csv` layout and reconstructs overflow columns caused by commas inside messages.
4. Normalizes labels to lowercase.
5. Keeps only `ham` and `spam` records.
6. Removes missing and empty messages.
7. Removes duplicate `(label, text)` records.
8. Creates a binary target where Spam = `1` and Ham = `0`.

---

## Text Preprocessing

`preprocessing.py` contains the shared `clean_text()` function used during both training and prediction.

### Preprocessing steps

1. Lowercase the text.
2. Replace URLs with `httpaddr`.
3. Replace email addresses with `emailaddr`.
4. Replace currency values/symbols with `moneysymb`.
5. Replace phone numbers with `phonenumbr`.
6. Replace other numeric values with `numbr`.
7. Strip remaining non-alphabetic characters.
8. Tokenize the message.
9. Remove English stopwords and one-character tokens.
10. Apply Porter stemming.

### Why placeholders are preserved

URLs, email addresses, currency values, phone numbers, and numbers are swapped to placeholders instead of being deleted because their presence can itself be a useful spam signal.

The UI converts these internal placeholders to readable labels:

| Placeholder | UI label |
|---|---|
| `httpaddr` | a URL |
| `emailaddr` | an email address |
| `moneysymb` | a currency symbol |
| `phonenumbr` | a phone number |
| `numbr` | a number |

NLTK stopwords are used when available. A built-in fallback list is used when the NLTK data is unavailable, helping the project continue to run offline.

---

## Feature Engineering

The main application uses **TF-IDF** with:

```text
ngram_range = (1, 2)
min_df      = 2
max_features= 5000
sublinear_tf= True
```

A **CountVectorizer** configuration is also trained for comparison.

The chosen vectorizer is saved as `vectorizer.pkl` so inference uses the same feature representation that was used during training.

---

## Models

SpamGuard compares four classical machine learning models:

| Model | Role |
|---|---|
| **Multinomial Naive Bayes** | Strong probabilistic text baseline |
| **Logistic Regression** | Linear probabilistic baseline with interpretable coefficients |
| **Linear SVM** | Strong sparse-text classifier; current selected model |
| **Random Forest** | Non-linear ensemble baseline |

Model selection uses **F1 score** as the primary criterion, with accuracy as the tie-breaker.

---

## Model Selection and Evaluation

The current training implementation uses:

```text
Train/test split : 80/20
Split strategy    : stratified
Random state      : 42
```

Evaluation includes:

- **Accuracy** — overall percentage of correct predictions.
- **Precision** — fraction of predicted spam messages that are actually spam.
- **Recall** — fraction of actual spam messages detected.
- **F1** — harmonic mean of precision and recall.
- **ROC-AUC** — ranking quality across thresholds.
- **Confusion Matrix** — true/false positive and negative counts.

The training process also compares TF-IDF with CountVectorizer and exports ROC curves, class distribution, message-length analysis, frequent-word charts, model comparison, and word clouds.

### Probability handling

`LinearSVC` does not natively implement `predict_proba`. When Linear SVM is selected as the best model, the project applies `CalibratedClassifierCV` before saving the application model so the Flask interface can expose Spam/Ham probabilities.

---

## Explainable Predictions

SpamGuard provides a lightweight local explanation for each prediction.

For a new message, active TF-IDF features are combined with learned signed feature weights:

```text
feature contribution = TF-IDF value × model weight
```

- Positive contribution → pushes the prediction toward **Spam**.
- Negative contribution → pushes the prediction toward **Ham**.

The API returns the most influential terms/features with their direction and contribution weight. Placeholder features are displayed using readable labels such as **a URL** or **a phone number**.

---

## Application Architecture

```text
                     ┌─────────────────────┐
                     │      Web UI         │
                     │ HTML/CSS/JavaScript  │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │     Flask App       │
                     │ pages + JSON APIs   │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Prediction Pipeline │
                     │ preprocess → vector │
                     │ → model → explain   │
                     └──────────┬──────────┘
                                │
                                ▼
                     ┌─────────────────────┐
                     │ Saved ML Artifacts  │
                     │ model + vectorizer  │
                     │ metadata + metrics  │
                     └─────────────────────┘

Training side:

Dataset → preprocessing → feature engineering → model comparison
       → evaluation → best model → probability calibration
       → artifact export → Flask inference
```

---

## Project Structure

```text
spamguard-email-classifier-ml/
│
├── README.md
├── .gitignore
├── 21.Spam Ham Project.ipynb
├── 22.TextProcessing.ipynb
├── app.py
├── train_model.py
├── preprocessing.py
├── requirements.txt
├── run_app.bat
├── train_model.bat
├── spam.csv
│
├── models/
│   ├── model.pkl
│   ├── vectorizer.pkl
│   ├── metadata.joblib
│   ├── metrics.json
│   └── chart_data.json
│
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   ├── js/vendor/chart.umd.js
│   ├── images/
│   └── interactive/
│
└── templates/
    ├── base.html
    ├── index.html
    ├── predict.html
    ├── analytics.html
    └── about.html
```

### Main components

| File / Directory | Purpose |
|---|---|
| `app.py` | Flask routes, prediction endpoints, artifact loading, batch inference and session analytics |
| `train_model.py` | Dataset loading, preprocessing, model training, evaluation, model selection and artifact generation |
| `preprocessing.py` | Shared text cleaning, placeholder replacement, stopwords and stemming |
| `spam.csv` | SMS Spam Collection dataset |
| `models/` | Saved model, vectorizer, metadata, metrics and chart data |
| `templates/` | Flask HTML templates |
| `static/` | CSS, JavaScript, charts and visual assets |
| `21.Spam Ham Project.ipynb` | Existing exploratory/model-development notebook |
| `22.TextProcessing.ipynb` | Existing text-processing notebook |
| `run_app.bat` | Windows setup/run launcher |
| `train_model.bat` | Windows retraining shortcut |

---

## Application Pages

| Page | Route | Description |
|---|---|---|
| Landing | `/` | Hero area, animated statistics, ticker and pipeline presentation |
| Predict | `/predict` | Single prediction, live typing, CSV batch mode and session analytics |
| Analytics | `/analytics` | Interactive charts and Spam/Ham word clouds |
| About | `/about` | Metrics, preprocessing, TF-IDF and algorithm details |

The interface supports dark/light themes and remembers the selected theme through `localStorage`.

---

## REST API

### Single-message prediction

```http
POST /api/predict
Content-Type: application/json
```

Request:

```json
{
  "text": "You won a FREE prize, call now to claim!"
}
```

Example response:

```json
{
  "label": "spam",
  "prediction": "Spam",
  "prob_spam": 0.9999,
  "prob_ham": 0.0001,
  "confidence": 99.99,
  "top_words": [
    {
      "token": "a phone number",
      "direction": "spam",
      "weight": 0.92
    }
  ]
}
```

### cURL

```bash
curl -X POST http://127.0.0.1:5001/api/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "You won a FREE prize, call now to claim!"}'
```

> When `app.py` is executed directly, the current Flask entry point uses `127.0.0.1:5001`.

### Batch prediction

```http
POST /api/predict_batch
Content-Type: multipart/form-data
```

Upload the CSV using the `file` field.

The endpoint returns:

- detected text column
- total rows processed
- Spam count and Ham count
- truncation status
- per-row predictions
- token summary
- word-frequency summary

Current limits:

- **5 MB** maximum upload size
- **2,000 rows** maximum processed rows per request

---

## Batch Prediction

The batch endpoint tries to identify the message column using common names such as:

```text
text
message
email
body
v2
content
```

If none is found, it selects the object/text column with the greatest average text length as a fallback.

For each row, the application records:

- message preview
- original character length
- predicted label
- prediction text
- spam probability
- reported confidence

It also aggregates the most influential tokens and readable word frequencies across the uploaded file.

---

## Session Analytics

The Predict page keeps an in-memory record of submitted scans. Live typing previews are not counted.

The session analytics view includes:

1. Verdict split — Spam vs Ham.
2. Risk distribution across five probability bands.
3. Spam probability per submitted scan with the 50% reference line.
4. Most common trigger/influential words.

The session history:

- clears on reload
- has a Clear button
- supports CSV export

---

## Quick Start (Windows)

The project includes `run_app.bat`.

Double-clicking it is intended to:

1. Find Python.
2. Create/reuse the virtual environment.
3. Install packages from `requirements.txt`.
4. Train the model when the required artifacts are missing.
5. Start the Flask application.
6. Open the local browser page.

Later runs can skip directly to the application when artifacts already exist.

Press `Ctrl+C` in the console to stop the server.

---

## Manual Installation

Create a virtual environment:

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Training the Model

### Default dataset

```bash
python train_model.py
```

### Custom CSV

```bash
python train_model.py --data path/to/your.csv
```

The input CSV should contain compatible label/text information.

### CountVectorizer application model

```bash
python train_model.py --vectorizer count
```

The default is TF-IDF.

### Windows shortcut

```text
train_model.bat
```

Arguments can be passed through, for example:

```text
train_model.bat --vectorizer count
```

Training generates the saved classifier, vectorizer, metadata, metrics, chart data, static figures, and interactive HTML charts.

---

## Running the Application

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5001
```

The trained `models/` directory is included, so the application can run without retraining when the expected artifacts are present.

Retrain whenever you change the:

- dataset
- preprocessing logic
- vectorizer configuration
- model configuration

The application reads metrics from `metrics.json` and chart values from `chart_data.json`, so retrained results can flow into the web pages without manually editing the frontend.

---

## Requirements

The project currently documents compatibility with **Python 3.10–3.12**.

```text
Flask>=3.0
scikit-learn>=1.4
pandas>=2.0
numpy>=1.26
matplotlib>=3.8
seaborn>=0.13
wordcloud>=1.9
nltk>=3.8
joblib>=1.3
plotly>=5.20
```

The pages also use Bootstrap, Chart.js and Google Fonts through CDNs, so browser internet access is needed for those external resources. Training and prediction code is designed to run offline.

---

## Model Artifacts

Training creates or updates the following files:

| Artifact | Purpose |
|---|---|
| `models/model.pkl` | Best classifier used by the Flask application |
| `models/vectorizer.pkl` | Fitted TF-IDF or CountVectorizer |
| `models/metadata.joblib` | Feature weights/names, selected model and vectorizer metadata |
| `models/metrics.json` | Dataset statistics, model metrics and preprocessing information |
| `models/chart_data.json` | Numerical data used by analytics charts |

`metadata.joblib` is also used to support local feature-level explanations in the application.

---

## Visualization and Reporting

Training generates static and interactive reporting artifacts.

### Static images

Stored under `static/images/`:

- `class_distribution.png`
- `confusion_matrix.png`
- `frequent_words.png`
- `length_distribution.png`
- `model_comparison.png`
- `roc_curve.png`
- `wordcloud_ham.png`
- `wordcloud_spam.png`

### Interactive reports

Stored under `static/interactive/`:

- `model_comparison.html`
- `roc_curve.html`

The application primarily draws analytics from `chart_data.json` using Chart.js. The static and standalone Plotly outputs are useful for reports and presentations. Only the two word clouds are required directly by the application according to the current project design; the other generated reports can be removed from a deployment if not needed.

---

## Tech Stack

### Backend / Machine Learning

- Python
- Flask
- scikit-learn
- pandas
- NumPy
- NLTK
- joblib

### NLP / Visualization

- TF-IDF
- CountVectorizer
- Porter Stemmer
- Matplotlib
- Seaborn
- WordCloud
- Plotly
- Chart.js

### Frontend

- HTML
- CSS
- JavaScript
- Bootstrap
- Google Fonts

The frontend is plain JavaScript and has no separate frontend build step.

---

## Reproducibility Notes

The current training pipeline uses:

```text
Random state : 42
Test size    : 20%
Split        : stratified train/test split
```

NumPy is also seeded with `42`.

Because serialized scikit-learn/joblib artifacts can depend on package versions, exact reproducibility is improved when the Python and dependency versions used for training are kept consistent with the environment used for inference.

---

## Limitations and Notes

- The model is trained on the SMS Spam Collection dataset, so real-world email traffic can differ from the training distribution.
- Reported metrics are benchmark results on the held-out test split and are not guarantees of production performance.
- The project is currently optimized as a local/demo Flask application.
- The current direct-entry Flask server is a development server and should not be treated as a production WSGI deployment.
- The browser depends on CDN-hosted Bootstrap, Chart.js and Google Fonts.
- NLTK stopwords use a built-in fallback when NLTK resources are unavailable.
- When preprocessing produces no usable vectorized features, the current application defaults to Ham and returns a note explaining that no usable words remained.
- The repository intentionally keeps trained model and visualization artifacts alongside source code for convenience.

---

## Future Enhancements

Possible future improvements include:

- Cross-validation and a separate validation/test protocol for stronger model selection.
- Automated unit and integration tests.
- API schema/input validation.
- Structured application logging and monitoring.
- Docker-based deployment.
- GitHub Actions CI/CD.
- Model and dataset versioning.
- Experiment tracking.
- More robust probability calibration and threshold tuning.
- Additional metrics such as PR-AUC.
- Larger and more diverse email datasets.
- Production WSGI deployment.

These are optional next steps; the current project remains fully usable as a local ML + Flask application.

---

## Project Status

**Status: Functional ML + Flask portfolio application.**

Current capabilities:

```text
Dataset handling             ✓
Text preprocessing           ✓
Feature engineering          ✓
Multiple-model comparison   ✓
Model evaluation             ✓
Model persistence             ✓
Explainable inference        ✓
Single prediction API        ✓
Batch CSV prediction         ✓
Interactive web UI           ✓
Analytics dashboard          ✓
Visualization exports        ✓
```

---

## Existing Project Information Retained

This README upgrade preserves the original project information and workflow, including:

- the original model choices and recorded results
- TF-IDF and CountVectorizer comparison
- preprocessing rules and placeholder tokens
- existing page routes and page descriptions
- JSON API request/response examples
- batch prediction behavior and limits
- session analytics behavior
- model artifact filenames
- visualization outputs
- Windows quick-start workflow
- manual installation and training commands
- dependency list
- technology stack
- NLTK fallback behavior
- offline/online notes

The primary change is the **organization and presentation of the existing information**, with additional professional documentation around the ML workflow, architecture, API, evaluation, artifacts, limitations, and project status.
