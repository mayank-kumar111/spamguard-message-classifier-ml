# SpamGuard - Spam/Ham Email Classifier

A Flask web app that classifies email or SMS messages as spam or ham. It trains
and compares four models on the SMS Spam Collection dataset (`spam.csv`, 5,572
messages) and serves predictions through a dark/light themed interface with a
live prediction gauge and interactive charts.

## What it does

- Trains and compares Multinomial Naive Bayes, Logistic Regression, Linear SVM
  and Random Forest, then keeps the best by F1. On this dataset Linear SVM wins:
  98.9% accuracy, F1 0.956, ROC-AUC 0.998 on the held-out test set, with only 2
  false positives out of 1,032 messages.
- TF-IDF features (unigrams and bigrams, 5,000 terms), with a CountVectorizer
  comparison reported on the About page.
- Shows the words that pushed each prediction toward spam or ham.
- Single prediction that updates as you type, plus CSV batch upload.
- A session analytics panel on the predict page and six charts on the analytics
  page (built with Chart.js).
- Dark/light theme, remembered with `localStorage`.

## Project layout

```
Email Classifier/
├── run_app.bat            Windows launcher (setup, train if needed, serve)
├── train_model.bat        Windows retraining shortcut
├── app.py                 Flask app: page routes and the prediction API
├── train_model.py         Trains the models, saves the best one, exports charts
├── preprocessing.py       Text cleaning used by both training and the app
├── requirements.txt
├── spam.csv               dataset (label + text)
│
├── models/                created by train_model.py
│   ├── model.pkl          best classifier (probability-calibrated)
│   ├── vectorizer.pkl     fitted TF-IDF vectorizer
│   ├── metadata.joblib    feature weights and names, used for explanations
│   ├── metrics.json       metrics and dataset stats, read by every page
│   └── chart_data.json    numbers behind the analytics charts
│
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   ├── images/            word clouds and static chart PNGs
│   └── interactive/       standalone Plotly HTML charts
│
└── templates/
    ├── base.html          shared layout (navbar, theme toggle, footer)
    ├── index.html         landing page
    ├── predict.html       prediction and batch page
    ├── analytics.html     charts
    └── about.html         model details
```

## Quick start (Windows)

Double-click `run_app.bat`. The first run finds Python, sets up a virtual
environment, installs the packages, trains the model if it hasn't been trained
yet, then starts the server and opens http://127.0.0.1:5000 in your browser.
Later runs skip straight to starting the server. Press Ctrl+C in the console
window to stop it.

`train_model.bat` retrains on demand and passes any arguments through, for
example `train_model.bat --vectorizer count`.

## Running it manually

Install the dependencies:

```bash
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # macOS / Linux
pip install -r requirements.txt
```

Train the models (reads `spam.csv`, saves the best model and the chart data):

```bash
python train_model.py
python train_model.py --data path/to/your.csv    # a different CSV (needs text + label)
python train_model.py --vectorizer count          # save the CountVectorizer model instead
```

Start the app:

```bash
python app.py
```

Then open http://127.0.0.1:5000.

The trained `models/` folder is included, so you can run `python app.py`
straight away without training first. Retrain whenever you change the dataset or
the preprocessing; every page reads its numbers from `metrics.json` and
`chart_data.json`, so the pages update on their own.

The pages load Bootstrap, Chart.js and Google Fonts from a CDN, so the first
load needs an internet connection. The training and prediction code runs
offline.

## The pages

| Page | Route | What it shows |
|------|-------|---------------|
| Landing | `/` | Hero with a looping demo, animated stats, ticker, pipeline diagram |
| Predict | `/predict` | Prediction that updates as you type, CSV batch mode, session analytics |
| Analytics | `/analytics` | Six charts plus spam and ham word clouds |
| About | `/about` | Metrics table, TF-IDF and algorithm notes, preprocessing steps |

The six analytics charts are the class balance (doughnut), model comparison
(radar or bar, toggleable), ROC curves, message-length distribution, and the top
words for spam and ham.

### JSON API

```bash
curl -X POST http://127.0.0.1:5000/api/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "You won a FREE prize, call now to claim!"}'
```

```json
{
  "label": "spam",
  "prediction": "Spam",
  "prob_spam": 0.9999,
  "prob_ham": 0.0001,
  "confidence": 99.99,
  "top_words": [{"token": "a phone number", "direction": "spam", "weight": 0.92}]
}
```

`POST /api/predict_batch` takes a CSV upload and returns a prediction per row
plus a `token_summary` with the trigger words counted across the whole file.

### Session analytics

The predict page keeps an in-memory record of what you classify this session
(single scans and batch rows) and draws four charts from it: verdict split, risk
distribution across five probability bands, spam probability per scan with the
50% line marked, and the most common trigger words. Only submitted messages are
counted, not live-typing previews, so one message can't fill the history. It
clears on reload, and there is a Clear button and a CSV export.

## How it works

```
text -> preprocessing -> TF-IDF -> Linear SVM -> spam/ham + confidence
```

`preprocessing.py` lowercases the text, swaps URLs, emails, currency, phone
numbers and digits for placeholder tokens (their presence is a useful spam
signal), strips punctuation, drops stopwords and applies Porter stemming. The
same function runs during training and at prediction time so the features match.

For the explanation, one signed weight per feature is saved at train time. For a
new message its TF-IDF values are multiplied by those weights, which gives the
words pushing it toward spam or ham. Placeholder tokens are shown in plain words
(for example `phonenumbr` becomes "a phone number").

## Tech stack

Python, Flask, scikit-learn, pandas, NumPy, NLTK, matplotlib, seaborn,
WordCloud, Plotly, Chart.js and Bootstrap. The front-end is plain JavaScript
with no build step.

## Notes

- NLTK stopwords are used when available, with a built-in list as a fallback so
  the pipeline still runs offline.
- The pages draw their charts live from `chart_data.json`. `train_model.py` also
  writes static PNG charts and standalone Plotly HTML into `static/images/` and
  `static/interactive/`, which are handy for reports. Only the two word clouds
  are used by the app itself; the rest can be deleted if you don't need them.
