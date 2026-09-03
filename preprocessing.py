"""Text cleaning shared by the training script and the web app.

Both import clean_text so a message is processed the same way when the model
is trained and when it is used to make a prediction.
"""

import re
import string


# Built-in stopword list, used as a fallback when the nltk data can't be
# downloaded (e.g. no internet on the machine running this).
_FALLBACK_STOPWORDS = {
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while", "of",
    "at", "by", "for", "with", "about", "against", "between", "into", "through",
    "during", "before", "after", "above", "below", "to", "from", "up", "down",
    "in", "out", "on", "off", "over", "under", "again", "further", "then",
    "once", "here", "there", "when", "where", "why", "how", "all", "any",
    "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor",
    "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can",
    "will", "just", "don", "should", "now", "d", "ll", "m", "o", "re", "ve",
    "y", "ain", "aren", "couldn", "didn", "doesn", "hadn", "hasn", "haven",
    "isn", "ma", "mightn", "mustn", "needn", "shan", "shouldn", "wasn", "weren",
    "won", "wouldn",
}

try:
    import nltk
    from nltk.corpus import stopwords as _nltk_stopwords
    try:
        STOPWORDS = set(_nltk_stopwords.words("english"))
    except LookupError:
        nltk.download("stopwords", quiet=True)
        STOPWORDS = set(_nltk_stopwords.words("english"))
except Exception:
    STOPWORDS = set(_FALLBACK_STOPWORDS)

try:
    from nltk.stem import PorterStemmer
    _STEMMER = PorterStemmer()

    def _stem(token):
        return _STEMMER.stem(token)
except Exception:
    def _stem(token):
        return token


_URL_RE = re.compile(r"(http[s]?://\S+|www\.\S+)")
_EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_MONEY_RE = re.compile(r"[£$€]\d+(?:\.\d+)?|\b\d+(?:\.\d+)?\s?(?:usd|gbp|eur|dollars?|pounds?)\b")
_PHONE_RE = re.compile(r"\b\d{5,}\b")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_NON_ALPHA_RE = re.compile(r"[^a-z\s]")

# Placeholders kept intact through stemming.
_PLACEHOLDERS = {"httpaddr", "emailaddr", "moneysymb", "phonenumbr", "numbr"}


def clean_text(text, stem=True):
    """Return the cleaned, space-joined tokens for a message.

    URLs, emails, currency, phone numbers and plain digits are swapped for
    placeholder tokens instead of being deleted, since their presence is a
    useful spam signal on its own.
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)

    text = text.lower()

    # Swap the high-signal patterns for placeholders before we strip punctuation.
    text = _URL_RE.sub(" httpaddr ", text)
    text = _EMAIL_RE.sub(" emailaddr ", text)
    text = _MONEY_RE.sub(" moneysymb ", text)
    text = re.sub(r"[£$€]", " moneysymb ", text)
    text = _PHONE_RE.sub(" phonenumbr ", text)
    text = _NUMBER_RE.sub(" numbr ", text)

    text = _NON_ALPHA_RE.sub(" ", text)

    tokens = text.split()
    tokens = [tok for tok in tokens if tok not in STOPWORDS and len(tok) > 1]

    if stem:
        tokens = [tok if tok in _PLACEHOLDERS else _stem(tok) for tok in tokens]

    return " ".join(tokens)


def simple_tokens(text):
    """Readable tokens for the word clouds (no placeholders, no stemming)."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    return [t for t in tokens if t not in STOPWORDS and len(t) > 2]


# Friendly labels for the placeholder tokens, used when the app shows which
# words drove a prediction.
PLACEHOLDER_LABELS = {
    "httpaddr": "a URL",
    "emailaddr": "an email address",
    "moneysymb": "a currency symbol",
    "phonenumbr": "a phone number",
    "numbr": "a number",
}


if __name__ == "__main__":
    for s in [
        "WINNER!! You have won a £1000 cash prize! Call 09061701461 now http://bit.ly/x",
        "Hey, are we still meeting for lunch tomorrow at 12?",
    ]:
        print("raw  :", s)
        print("clean:", clean_text(s))
        print("-" * 60)
