# SpamGuard — Complete Project Documentation

> End-to-end machine learning application for classifying SMS/email-style messages as **Spam** or **Ham**, with classical NLP, model comparison, saved artifacts, explainable predictions, batch CSV inference, analytics, and a Flask web interface.

---

## 1. Project Overview

SpamGuard demonstrates a complete classical Natural Language Processing (NLP) and Machine Learning workflow:

```text
Raw message
    ↓
Data cleaning
    ↓
Shared text preprocessing
    ↓
TF-IDF / CountVectorizer
    ↓
Multiple candidate classifiers
    ↓
Evaluation
    ↓
Best-model selection by F1
    ↓
Probability calibration when Linear SVM is selected
    ↓
Saved model + vectorizer + metadata
    ↓
Flask web application + JSON API
    ↓
Single-message prediction / CSV batch prediction
    ↓
Prediction explanations + analytics
```

The project is intentionally implemented as a **college-level local ML application** rather than a hardened production email-security gateway. Its benchmark is based on the included SMS dataset.

---

## 2. Goals

SpamGuard is designed to:

- classify messages into **Spam** or **Ham**
- demonstrate a reusable NLP preprocessing pipeline
- compare multiple supervised learning algorithms
- compare TF-IDF and count-based features
- evaluate models using more than accuracy
- persist trained artifacts for inference
- provide a usable Flask interface
- expose a JSON prediction API
- support bulk CSV inference
- provide lightweight feature-level explanations
- visualize dataset and model behavior

---

## 3. Key Features

### Machine Learning

- Binary Spam/Ham classification
- Multinomial Naive Bayes
- Logistic Regression
- Linear SVM
- Random Forest
- TF-IDF unigrams and bigrams
- CountVectorizer comparison
- Stratified 80/20 train/test split
- F1-based model selection
- ROC-AUC, precision, recall and accuracy evaluation
- Probability calibration for Linear SVM

### NLP preprocessing

The shared preprocessing pipeline:

1. lowercases text
2. replaces URLs with `httpaddr`
3. replaces email addresses with `emailaddr`
4. replaces currency with `moneysymb`
5. replaces phone numbers with `phonenumbr`
6. replaces other numbers with `numbr`
7. removes remaining non-alphabetic characters
8. tokenizes by whitespace
9. removes English stopwords
10. removes single-character tokens
11. applies Porter stemming

The placeholders deliberately retain useful structural spam signals rather than deleting them.

### Application

- Home page
- Single-message prediction
- Live prediction mode
- Batch CSV prediction
- Downloadable batch results
- Session analytics
- Model/data analytics
- ROC curves
- Confusion matrix
- Frequent-word charts
- Word clouds
- Dark/light theme
- Responsive Flask + Bootstrap interface

---

## 4. Dataset

The repository contains the included SMS spam dataset in `spam.csv`.

### Recorded dataset statistics

| Metric | Value |
|---|---:|
| Raw messages | 5,572 |
| Duplicate records removed | 414 |
| Unique messages | 5,158 |
| Ham | 4,516 |
| Spam | 642 |
| Spam share | 12.45% |
| Training set | 4,126 |
| Test set | 1,032 |

The dataset is therefore **imbalanced**, with Ham representing the large majority.

### Data loading behavior

The training script supports:

- the project's `v1/v2` style CSV layout
- a simpler `label/text` layout
- reconstruction of message text when commas have spilled into additional CSV columns
- filtering to valid `ham` / `spam` labels
- removal of empty messages
- duplicate removal

---

## 5. Preprocessing Pipeline

Preprocessing is centralized in `preprocessing.py` so that training and inference use the **same transformation logic**.

### 5.1 Normalization

Input text is converted to lowercase.

Example:

```text
"Congratulations! You WON a FREE prize."
```

becomes conceptually:

```text
"congratulations! you won a free prize."
```

### 5.2 Pattern placeholders

High-signal entities are replaced before punctuation removal:

| Original pattern | Placeholder |
|---|---|
| URL | `httpaddr` |
| Email address | `emailaddr` |
| Currency | `moneysymb` |
| Phone number | `phonenumbr` |
| Numeric value | `numbr` |

Example:

```text
WIN £1000! Call 9876543210 http://example.com
```

becomes approximately:

```text
win moneysymb call phonenumbr httpaddr
```

This preserves information such as "the message contains a URL" instead of treating the URL as meaningless noise.

### 5.3 Stopwords

Common English stopwords are removed.

The implementation first attempts to use NLTK's English stopword list and contains a built-in fallback list for environments where NLTK data is unavailable.

### 5.4 Stemming

Porter stemming is applied to reduce related word forms to a common representation.

Example:

```text
claim / claiming / claimed
```

may map toward a shared stem such as:

```text
claim
```

### 5.5 Inference consistency

The Flask application imports the same `clean_text()` function used by training. This avoids a common deployment problem where training and inference accidentally create different feature representations.

---

## 6. Feature Engineering

SpamGuard evaluates two feature representations.

### 6.1 TF-IDF

The default vectorizer is:

```python
TfidfVectorizer(
    ngram_range=(1, 2),
    min_df=2,
    max_features=5000,
    sublinear_tf=True
)
```

This creates:

- unigram features
- bigram features
- a minimum document-frequency threshold of 2
- a maximum vocabulary of 5,000 features
- sublinear term-frequency scaling

For example, the message:

```text
free prize
```

can contribute:

```text
free
prize
free prize
```

### 6.2 CountVectorizer

CountVectorizer is also evaluated as a comparison baseline.

The recorded experiment is especially useful because it shows that the "best representation" cannot simply be assumed from convention.

### Vectorizer experiment

For Linear SVM:

| Vectorizer | Recorded F1 |
|---|---:|
| TF-IDF | 0.9558 |
| CountVectorizer | 0.9600 |

The application currently uses **TF-IDF by default**, while `train_model.py --vectorizer count` can generate a count-based saved model.

---

## 7. Model Training

The training script compares four classifiers:

1. Multinomial Naive Bayes
2. Logistic Regression
3. Linear SVM
4. Random Forest

### Model configuration

#### Multinomial Naive Bayes

```python
MultinomialNB(alpha=0.1)
```

#### Logistic Regression

```python
LogisticRegression(
    C=3.0,
    max_iter=2000,
    random_state=42
)
```

#### Linear SVM

```python
LinearSVC(
    C=1.0,
    max_iter=5000,
    random_state=42
)
```

#### Random Forest

```python
RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)
```

---

## 8. Train/Test Split

SpamGuard uses:

```python
train_test_split(
    ...,
    test_size=0.20,
    stratify=target,
    random_state=42
)
```

This creates an 80/20 split while maintaining a similar class distribution in both partitions.

### Why stratification matters

Only about 12.45% of the cleaned dataset is spam. Without stratification, random variation in a small minority class could make evaluation less stable or representative.

---

## 9. Model Selection Strategy

The current training script selects the final model using:

```python
best_name = max(
    results,
    key=lambda n: (results[n]["f1"], results[n]["accuracy"])
)
```

Therefore:

1. F1 is the primary selection metric
2. Accuracy is the tie-breaker

This is more appropriate than selecting only by accuracy when the dataset is imbalanced.

---

## 10. Recorded Benchmark Results

The repository's recorded TF-IDF experiment produced:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Multinomial Naive Bayes | 98.16% | 95.80% | 89.06% | 0.9231 | 0.9930 |
| Logistic Regression | 98.55% | 100.00% | 88.28% | 0.9378 | 0.9987 |
| Linear SVM | **98.93%** | 98.35% | **92.97%** | **0.9558** | 0.9979 |
| Random Forest | 98.64% | 99.14% | 89.84% | 0.9426 | 0.9978 |

The recorded application model is **Linear SVM**.

> These numbers are benchmark results on the included dataset and fixed recorded split. They are not a guarantee of real-world performance.

---

## 11. Confusion Matrix

The saved test-set confusion matrix for Linear SVM is:

|  | Predicted Ham | Predicted Spam |
|---|---:|---:|
| Actual Ham | 902 | 2 |
| Actual Spam | 9 | 119 |

Therefore:

- True Negatives (TN): 902
- False Positives (FP): 2
- False Negatives (FN): 9
- True Positives (TP): 119

### Precision

```text
Precision = TP / (TP + FP)
          = 119 / 121
          ≈ 98.35%
```

### Recall

```text
Recall = TP / (TP + FN)
       = 119 / 128
       ≈ 92.97%
```

The model made very few false-positive spam predictions in this recorded test set, while nine actual spam messages were missed.

---

## 12. Evaluation Metrics

SpamGuard reports:

### Accuracy

```text
(TP + TN) / (TP + TN + FP + FN)
```

Useful as an overall metric, but potentially misleading when one class dominates.

### Precision

```text
TP / (TP + FP)
```

Answers:

> Of the messages predicted as spam, how many were actually spam?

### Recall

```text
TP / (TP + FN)
```

Answers:

> Of the actual spam messages, how many did the model catch?

### F1

```text
2 × Precision × Recall / (Precision + Recall)
```

Balances precision and recall.

### ROC-AUC

Measures ranking quality across decision thresholds.

---

## 13. Probability Calibration

LinearSVC does not natively expose `predict_proba()`.

Because the Flask application provides:

- spam probability
- ham probability
- confidence

the training script wraps Linear SVM with:

```python
CalibratedClassifierCV(
    LinearSVC(...),
    cv=5
)
```

This creates probability estimates from the SVM decision scores using five-fold calibration.

### Important implementation detail

The recorded benchmark metrics are calculated from the original selected model on the held-out test set. The saved application classifier is calibrated when the selected model is a Linear SVM.

---

## 14. Explainability

SpamGuard provides lightweight local explanations for individual predictions.

For active features, the application computes a contribution using the idea:

```text
feature contribution = TF-IDF value × model weight
```

Positive contribution:

```text
pushes toward Spam
```

Negative contribution:

```text
pushes toward Ham
```

The application displays the strongest contributing tokens/features.

Examples of recorded spam-oriented features include:

```text
phonenumbr
httpaddr
moneysymb
call phonenumbr
tone
mobil
rington
txt
servic
claim
emailaddr
```

This is a **lightweight model-weight explanation**, not a SHAP or LIME implementation.

---

## 15. Application Architecture

### Backend

- Python
- Flask
- pandas
- NumPy
- scikit-learn
- joblib

### NLP

- NLTK
- Porter stemmer
- stopword filtering

### Visualization

- Matplotlib
- Seaborn
- Plotly
- Chart.js

### Frontend

- HTML/Jinja templates
- Bootstrap
- Bootstrap Icons
- custom CSS
- JavaScript

### High-level structure

```text
                ┌───────────────────┐
                │     spam.csv      │
                └─────────┬─────────┘
                          │
                          ▼
                ┌───────────────────┐
                │ train_model.py    │
                └─────────┬─────────┘
                          │
          ┌───────────────┼────────────────┐
          ▼               ▼                ▼
    model.pkl       vectorizer.pkl   metadata/metrics
          │               │                │
          └───────────────┼────────────────┘
                          ▼
                     app.py
                          │
              ┌───────────┴───────────┐
              ▼                       ▼
           Web UI                 JSON API
              │                       │
              └───────────┬───────────┘
                          ▼
                   Spam / Ham result
```

---

## 16. Project Structure

```text
spamguard-message-classifier-ml/
├── README.md
├── DOCUMENTATION.md
├── .gitignore
├── 21.Spam Ham Project.ipynb
├── app.py
├── preprocessing.py
├── train_model.py
├── requirements.txt
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
│   ├── css/
│   ├── js/
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

Local helper files excluded by `.gitignore` are not part of the documented runnable workflow.

---

## 17. File-by-File Responsibility

### `spam.csv`

Training dataset containing message labels and message text.

### `preprocessing.py`

Shared text preprocessing and tokenization utilities.

Key functions include:

- `clean_text()`
- `simple_tokens()`

### `train_model.py`

Main training pipeline:

- reads dataset
- cleans data
- removes duplicates
- creates train/test partitions
- builds vectorizers
- compares vectorizers
- trains four models
- evaluates metrics
- selects best model
- calibrates Linear SVM when required
- saves model artifacts
- generates charts
- exports chart data

### `app.py`

Flask inference server:

- loads saved artifacts
- serves web pages
- performs single prediction
- performs batch CSV prediction
- produces probability/confidence values
- returns lightweight explanations
- applies upload/row limits

### `models/model.pkl`

Serialized application classifier.

### `models/vectorizer.pkl`

Serialized fitted vectorizer. It must match the model used during training.

### `models/metadata.joblib`

Contains metadata such as:

- feature weights
- feature names
- best model name
- vectorizer kind
- class order

### `models/metrics.json`

Stores dataset information, model metrics, preprocessing description, and vectorizer-comparison results.

### `models/chart_data.json`

Stores chart-ready analytics data used by the Flask front end.

---

## 18. Web Routes

| Page | Route | Purpose |
|---|---|---|
| Home | `/` | Project overview and key metrics |
| Predict | `/predict` | Single + batch inference |
| Analytics | `/analytics` | Dataset/model visualizations |
| About | `/about` | Methodology, metrics and project notes |

The prediction page supports:

- single-message input
- live mode
- spam/ham sample messages
- probability gauge
- trigger-token explanation
- CSV upload
- batch table
- CSV export
- session analytics

---

## 19. JSON API

### 19.1 Single prediction

**Endpoint**

```http
POST /api/predict
Content-Type: application/json
```

**Request**

```json
{
  "text": "You won a FREE prize, call now to claim!"
}
```

**Response fields**

```json
{
  "label": "spam",
  "prediction": "Spam",
  "prob_spam": 0.99,
  "prob_ham": 0.01,
  "confidence": 99.0,
  "top_words": [],
  "words": [],
  "note": null
}
```

Exact numeric values depend on the saved model and input.

### 19.2 Batch prediction

**Endpoint**

```http
POST /api/predict_batch
Content-Type: multipart/form-data
```

Upload a CSV using the `file` field.

Recognized text-column names include:

- `text`
- `message`
- `email`
- `body`
- `v2`
- `content`

If no known text-column name is found, the application attempts to choose the most suitable object-type column based on average text length.

### Batch limits

- Maximum upload size: **5 MB**
- Maximum processed rows: **2,000**

The endpoint returns:

- selected text column
- total rows processed
- spam count
- ham count
- whether truncation occurred
- per-row prediction data
- token summary
- class-specific word frequencies

---

## 20. Quick Start

### Requirements

The documented/tested Python range is:

```text
Python 3.10–3.12
```

Main packages:

- Flask
- scikit-learn
- pandas
- NumPy
- Matplotlib
- Seaborn
- WordCloud
- NLTK
- Joblib
- Plotly

### Windows PowerShell

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
python train_model.py
python app.py
```

Open:

```text
http://127.0.0.1:5001
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python train_model.py
python app.py
```

Open:

```text
http://127.0.0.1:5001
```

---

## 21. Training Commands

### Default training

```bash
python train_model.py
```

Uses the repository's `spam.csv` and saves a TF-IDF based application pipeline.

### Custom dataset

```bash
python train_model.py --data path/to/your.csv
```

### CountVectorizer application model

```bash
python train_model.py --vectorizer count
```

Training regenerates model artifacts and visualization outputs.

---

## 22. Saved Artifacts

| Artifact | Purpose |
|---|---|
| `models/model.pkl` | Application classifier |
| `models/vectorizer.pkl` | Fitted text feature extractor |
| `models/metadata.joblib` | Feature/model metadata |
| `models/metrics.json` | Evaluation + dataset metrics |
| `models/chart_data.json` | Front-end analytics data |

These artifacts allow the Flask app to perform inference without retraining when it starts.

---

## 23. Reproducibility

The recorded experiment uses:

```text
random_state = 42
test_size = 0.20
stratified split
```

Results can change when:

- the source dataset changes
- dependency versions change
- preprocessing logic changes
- hyperparameters change
- the train/test partition changes

The README and metric files should therefore be treated as **recorded benchmark results for the repository state**, not as universal model performance claims.

---

## 24. Data Leakage Considerations

The vectorizer is fitted on training text and then applied to test text:

```text
training text
    ↓
fit vectorizer
    ↓
transform training/test
```

The test data is not used to fit the vectorizer.

This is important because fitting TF-IDF on the complete dataset before splitting could leak test-set information into the training process.

---

## 25. Current Evaluation Limitation

The current implementation uses the held-out test set to compare candidate models and select the best model.

A stricter experimental design would be:

```text
Full dataset
     ↓
Train / Test
     ↓
Training subset
     ↓
Cross-validation
     ↓
Model / hyperparameter selection
     ↓
Final model
     ↓
Untouched test set
     ↓
Final reported benchmark
```

This would reduce the risk of indirectly tuning the model toward the held-out test set.

For a production-oriented experiment, cross-validation should be added.

---

## 26. Real-World Generalization Limitations

The repository is based on an SMS dataset.

That means the recorded result should **not** be interpreted as proof that the same model will perform equally well on unrestricted corporate email.

Real email can contain:

- HTML
- headers
- longer documents
- attachments
- multilingual text
- URLs with different patterns
- phishing language
- domain-specific terminology

A broader production evaluation should include:

- additional SMS datasets
- real email datasets
- cross-dataset testing
- temporal testing
- false-positive analysis
- threshold tuning

---

## 27. Known Edge Case: No Usable Features

If cleaning produces a message for which the vectorizer has no active known features, the current application follows a default Ham path:

```text
No usable features
      ↓
Ham
```

This is simple and deterministic, but it can be unsafe for production use.

Potential improvements:

- return an "unknown / low confidence" state
- use character n-grams
- use subword features
- add a fallback model
- expose a minimum-confidence policy

---

## 28. Security and Production Considerations

The application includes basic operational protections:

- 5 MB request-size limit
- 2,000-row batch limit
- input validation for missing/empty CSV uploads
- JSON validation for missing text

However, the repository is not a hardened production security service.

Before Internet-facing deployment, consider:

- disabling Flask `debug=True`
- using a production WSGI server such as Gunicorn
- rate limiting
- authentication if required
- request logging
- structured error logging
- health/readiness endpoints
- dependency vulnerability scanning
- automated tests
- CI/CD
- containerization
- secret/config management
- monitoring and alerting

---

## 29. Testing and CI Status

The current repository does not include a dedicated automated test suite or a visible GitHub Actions workflow.

Recommended tests include:

### Unit tests

- placeholder replacement
- stopword removal
- stemming
- empty input behavior
- known-label predictions

### API tests

- valid `/api/predict`
- missing text
- invalid JSON
- empty CSV
- invalid CSV
- oversized upload
- row-limit truncation

### ML tests

- model artifact loading
- vectorizer/model compatibility
- prediction schema
- reproducibility with fixed seed

### CI

A GitHub Actions workflow could automatically:

1. install dependencies
2. run unit/API tests
3. train a small smoke-test model
4. verify artifact creation
5. optionally run linting

---

## 30. Production Model Improvement Roadmap

### Phase 1 — Evaluation quality

- stratified cross-validation
- hyperparameter search
- confidence/threshold analysis
- precision-recall curves
- error buckets
- statistical confidence intervals where appropriate

### Phase 2 — Better NLP

- character n-grams
- word + character feature fusion
- stronger text normalization
- HTML-aware email preprocessing
- language detection
- multilingual handling

### Phase 3 — Better datasets

- multiple public spam datasets
- real-world email datasets
- time-based evaluation
- cross-domain validation

### Phase 4 — Production engineering

- automated tests
- Docker
- Gunicorn
- GitHub Actions
- API monitoring
- model versioning
- data/model drift monitoring
- deployment health checks

### Phase 5 — Advanced ML

Potential experiments:

- class-weighted models
- threshold optimization
- ensemble methods
- calibrated linear models
- transformer-based text classification
- hybrid rules + ML
- retrieval or LLM-based explanation layers

These are future directions, not claims about the current implementation.

---

## 31. Troubleshooting

### Error: missing model files

The Flask application expects:

```text
models/model.pkl
models/vectorizer.pkl
models/metadata.joblib
models/metrics.json
```

Run:

```bash
python train_model.py
```

before starting the application.

### NLTK stopword issue

The preprocessing module attempts to download the English stopword data and contains a fallback list when that is unavailable.

If working in an offline environment, confirm that dependencies and NLTK data are available or allow the fallback to be used.

### CSV upload failure

Check:

- file is valid CSV
- file is not empty
- text column exists or is inferable
- file is below 5 MB

### Large CSV

Only the first 2,000 rows are processed by the batch endpoint.

---

## 32. Reusable Prediction Flow

The inference path is:

```text
Input string
   ↓
clean_text()
   ↓
saved vectorizer.transform()
   ↓
saved classifier.predict_proba()
   ↓
Spam/Ham label
   ↓
Probability + confidence
   ↓
Feature contribution explanation
```

For a calibrated Linear SVM model, the probability layer is provided by `CalibratedClassifierCV`.

---

## 33. Why the Project Is Technically Useful

SpamGuard demonstrates several interview-relevant concepts in one system:

- supervised learning
- NLP preprocessing
- feature engineering
- sparse feature representations
- model comparison
- class imbalance
- precision vs recall
- F1 optimization
- ROC-AUC
- probability calibration
- model serialization
- REST APIs
- Flask inference
- batch processing
- visualization
- lightweight explainability
- reproducibility
- deployment considerations

---

## 34. Suggested Interview Explanation

A concise technical explanation of the project is:

> SpamGuard is an end-to-end NLP-based spam classification system built with Python, scikit-learn and Flask. I preprocess text by lowercasing it and replacing URLs, email addresses, phone numbers, currencies and numeric values with dedicated placeholder tokens, followed by stopword removal and Porter stemming. I compare TF-IDF and count-based features and train Multinomial Naive Bayes, Logistic Regression, Linear SVM and Random Forest models. The application selects a model primarily using F1 because the dataset is imbalanced. The recorded TF-IDF experiment achieved 98.93% accuracy, 98.35% precision, 92.97% recall and 0.9558 F1 with Linear SVM on the held-out test set. Because Linear SVM does not directly provide probabilities, I calibrate it before using it in the Flask application. The final system supports single prediction, CSV batch prediction, lightweight feature explanations and interactive analytics.

---

## 35. Important Interview Follow-ups

You should be able to explain:

1. Why F1 instead of accuracy?
2. Why stratified splitting?
3. Why TF-IDF?
4. Why did you use bigrams?
5. Why replace URLs instead of deleting them?
6. Why Linear SVM?
7. Why did you compare multiple algorithms?
8. How does Linear SVM differ from Logistic Regression?
9. Why does LinearSVC not have `predict_proba()`?
10. What does probability calibration do?
11. Explain the confusion matrix.
12. What is false positive vs false negative here?
13. Why can accuracy be misleading?
14. How do you prevent data leakage?
15. Why do training and inference need the same preprocessing?
16. How does the model explanation work?
17. What happens with unknown words?
18. Why is an SMS benchmark not enough for production email?
19. How would you improve the experiment with cross-validation?
20. How would you deploy the Flask service safely?

---

## 36. Future Enhancements

The project README already identifies these possible improvements:

- broader datasets
- cross-dataset evaluation
- threshold tuning
- deeper error analysis
- experiment tracking
- automated tests / CI
- containerized deployment

Additional extensions can include:

- character-level features
- multilingual classification
- phishing-specific classification
- model drift monitoring
- model registry/versioning
- A/B evaluation of preprocessing strategies
- production API observability

---

## 37. License and Usage Note

The current repository documentation does not define a dedicated open-source license. Before distributing or presenting the project as an open-source package, add an appropriate `LICENSE` file and clarify dataset licensing/usage requirements.

---

## 38. Final Repository Summary

**Project:** SpamGuard — Spam/Ham Message Classifier

**Primary stack:**

```text
Python
Flask
scikit-learn
NLTK
pandas
NumPy
Matplotlib
Seaborn
Plotly
Chart.js
Bootstrap
Joblib
```

**Core ML approach:**

```text
NLP preprocessing
      +
TF-IDF / CountVectorizer
      +
classical supervised ML
      +
probability calibration
      +
Flask inference
```

**Recorded best application model:** Linear SVM

**Recorded TF-IDF benchmark:**

```text
Accuracy   98.93%
Precision  98.35%
Recall     92.97%
F1         0.9558
ROC-AUC    0.9979
```

**Dataset:** 5,158 unique messages after duplicate removal from 5,572 raw records.

**Primary limitation:** the benchmark is based on the included SMS dataset and should not be treated as evidence of production-grade email-security performance.

---

## 39. Related Files

- `README.md` — project overview and quick-start documentation
- `DOCUMENTATION.md` — detailed technical documentation
- `preprocessing.py` — shared NLP preprocessing
- `train_model.py` — training, evaluation and artifact generation
- `app.py` — Flask application and API
- `spam.csv` — included training dataset
- `models/metrics.json` — recorded evaluation metrics
- `models/chart_data.json` — chart-ready analytics data


---

## 39. UI / UX Refresh

The application UI was refreshed without changing the machine-learning or API functionality.

### Theme behavior

- **Light theme is now the default.**
- Dark theme remains available through the existing theme toggle.
- The selected theme is stored in browser `localStorage` under `sg-theme`.
- Theme is applied before the page paints to reduce theme flashing.
- Charts listen for the existing `sg-theme` event and redraw using the current theme colors.

### Prediction interface improvements

The prediction page retains both existing workflows:

- Single message classification
- Batch CSV classification

The following UI issues were addressed:

1. The **Single message** active tab could become white/invisible in light mode.
   - The active state now has an explicit high-contrast cyan/blue/purple gradient.
   - Icon and text remain white and visible.

2. The **Live mode** switch could disappear against the light background.
   - The switch now has an explicit track and thumb style.
   - ON/OFF states have separate high-contrast colors.
   - Keyboard focus and hover states are visible.
   - The same control remains functional in dark mode.

3. Bootstrap alert components were given explicit light-theme colors for:
   - warnings
   - information messages
   - errors

4. CSV file inputs were styled for the light theme so the filename and file-selector button remain readable.

5. Analytics **Radar/Bars** toggle buttons now have a clear selected state in light mode.

### Responsive behavior

The navigation and prediction interface were hardened for smaller screens:

- Collapsed navigation receives a readable card-style container.
- Navigation links remain touch-friendly.
- Prediction tabs expand to balanced full-width controls on small screens.
- The Live mode control wraps cleanly instead of being squeezed beside buttons.
- Batch-result tables can scroll horizontally on narrow screens.
- Long navigation/footer labels are prevented from breaking awkwardly.

### Asset cache versioning

The main stylesheet and JavaScript assets use version query parameters so browsers are less likely to retain stale UI assets after deployment.

---

## 40. UI Maintenance Guidelines

When adding new UI components:

1. Define the component using the existing CSS variables such as `--text`, `--muted`, `--surface`, `--border`, `--cyan`, `--ham`, and `--spam`.
2. Add explicit light-theme rules when a component depends on Bootstrap's default colors.
3. Ensure active/selected states have sufficient contrast in both themes.
4. Do not modify prediction/API IDs such as `liveToggle`, `classifyBtn`, `messageInput`, `batchBtn`, or API routes without updating the corresponding JavaScript.
5. Test both `data-theme="light"` and `data-theme="dark"`.
6. Test the prediction page at desktop and mobile widths.
7. Bump the static asset version when a browser-cache-sensitive UI change is deployed.

---

## 41. Recent UI Fix Log

| Area | Issue | Resolution |
|---|---|---|
| Theme | Light mode was not the HTML fallback | Light is now the default fallback |
| Prediction tabs | Active Single message tab could disappear in light mode | Added explicit high-contrast active state |
| Live mode | Switch track/thumb was difficult to see in light mode | Added explicit accessible switch styling |
| Alerts | Bootstrap alert colors could have poor light-theme contrast | Added light-theme warning/info/error colors |
| CSV upload | File input styling could be difficult to read | Added light-theme file-selector styling |
| Analytics toggle | Radar/Bars selected state could be ambiguous | Added explicit selected-state styling |
| Mobile navigation | Collapsed menu could blend into page | Added bordered/raised mobile menu |
| Mobile prediction | Controls could become cramped | Added responsive tab and switch layout |
| Batch table | Narrow screens could clip results | Added horizontal overflow handling |
| Browser cache | Updated CSS/JS could remain stale | Refreshed asset version parameters |

> These changes are presentation-layer changes. The existing model artifacts, preprocessing pipeline, prediction endpoints, batch processing, session analytics, and chart logic remain part of the application.

---

## 42. Current UI Validation Checklist

Before considering a UI change complete, verify:

- [x] Light theme loads by default.
- [x] Dark theme remains available.
- [x] Theme preference persists after reload.
- [x] Home page remains readable in both themes.
- [x] Predict page remains readable in both themes.
- [x] Single message tab is visible when selected.
- [x] Batch CSV tab remains visible and selectable.
- [x] Live mode switch is visible in both ON and OFF states.
- [x] Classify and Clear controls remain visible.
- [x] Probability cards retain Ham/Spam contrast.
- [x] Trigger-token area remains readable.
- [x] Batch CSV input remains readable.
- [x] Batch result table remains readable.
- [x] Session analytics controls remain usable.
- [x] Analytics Radar/Bars toggle has a visible selected state.
- [x] Analytics charts can redraw after theme changes.
- [x] Navigation remains usable on smaller screens.
- [x] Long tables can scroll on narrow screens.
- [x] Existing routes and API IDs remain unchanged.

