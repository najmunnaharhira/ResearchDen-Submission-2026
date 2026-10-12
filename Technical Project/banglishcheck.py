"""BanglishCheck: detect AI-generated Banglish text and explain the decision.

One module shared by the Colab notebook and the Gradio demo:
    - Banglish stylometric features
    - dataset building (human reviews + AI texts from several generators)
    - the classical detector (word + character TF-IDF + stylometric features,
      logistic regression)
    - word-level explanations (exact SHAP values for a linear model)
    - shared metrics
"""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
CHATGPT_RELEASE = "2022-11-30"  # human texts must be older than this

# ---------------------------------------------------------------- features
BANGLISH_WORDS = {
    "ami", "amra", "amar", "amader", "tumi", "tomar", "tomra", "apni", "apnar",
    "tar", "tader", "ki", "keno", "kivabe", "kemon", "kothay",
    "kobe", "nai", "nei", "ha", "hya", "haan", "ache", "achi", "acho",
    "chilo", "chhilo", "hobe", "hoy", "hoye", "hoyeche", "korbo", "kori", "kore",
    "korche", "korchi", "korechi", "korte", "jabo", "jai", "jacchi", "jete",
    "gelam", "gechi", "khabo", "khai", "khete", "bolo", "bolchi", "bolte",
    "dekho", "dekhi", "dekhte", "valo", "bhalo", "kharap", "onek", "ektu",
    "shob", "sob", "sobai", "eta", "ota", "oita", "eita", "ei", "oi", "ekhon",
    "ekhane", "okhane", "tokhon", "kal", "aaj", "aj", "porshu", "din", "rat",
    "raat", "bhai", "vai", "apu", "dost", "mama", "khub", "ar", "aro", "kintu",
    "tai", "jodi", "tahole", "karon", "jonno", "theke", "diye", "shathe",
    "sathe", "moto", "mone", "lagche", "lage", "parbo", "pari", "parina",
    "janina", "jani", "bujhi", "bujhlam", "thik", "accha", "achha", "acha",
    "er", "re", "ke", "te", "tho", "ta", "ti", "gula", "gulo", "dam", "taka",
    "pore", "age", "dise", "disi", "paisi", "peyechi", "nilam", "dilam",
}
ENGLISH_WORDS = {
    "the", "an", "is", "are", "was", "were", "be", "been", "am", "and",
    "but", "if", "because", "of", "in", "on", "at", "for", "with",
    "from", "by", "about", "into", "this", "that", "these", "those", "it",
    "its", "i'm", "you", "your", "he", "she", "they", "we", "our", "my",
    "him", "her", "them", "what", "which", "who", "when", "where", "why",
    "how", "not", "no", "yes", "do", "does", "did", "have", "has", "had",
    "will", "would", "can", "could", "should", "just", "very", "really",
    "too", "also", "then", "than", "there", "here", "all", "some", "more",
    "most", "good", "bad", "great", "nice", "love", "like", "time", "today",
    "product", "quality", "price", "delivery", "seller", "order", "thanks",
    "thank", "packaging", "original", "fast", "service", "size", "color",
}
AMBIGUOUS = {"to", "i", "e", "o", "se", "or", "na", "a", "so", "me"}
BANGLISH_WORDS -= AMBIGUOUS
ENGLISH_WORDS -= AMBIGUOUS | BANGLISH_WORDS

SPELLING_VARIANTS = [
    {"valo", "bhalo", "bhaalo", "vhalo", "vlo", "balo"},
    {"kemon", "kamon", "kmn"},
    {"achi", "asi", "aci"},
    {"ache", "ase", "ace", "achhe", "ase"},
    {"korbo", "krbo", "korbu"},
    {"kichu", "kisu", "kicu", "kichhu"},
    {"accha", "achha", "acha"},
    {"bhai", "vai", "bhaiya", "vaiya"},
    {"shob", "sob", "sb"},
    {"ekhon", "akhon", "ekhn"},
    {"onek", "onk", "oneek"},
    {"chilo", "cilo", "silo", "chhilo"},
]
VARIANT_WORDS = set().union(*SPELLING_VARIANTS)

WORD_RE = re.compile(r"[A-Za-z']+|[ঀ-৿]+")
SENT_RE = re.compile(r"[.!?।]+")
EMOJI_RE = re.compile("[\U0001F300-\U0001FAFF☀-➿]")
ELONGATED_RE = re.compile(r"([A-Za-z])\1{2,}")
BENGALI_RE = re.compile("[ঀ-৿]")

FEATURE_NAMES = [
    "n_words", "avg_word_len", "type_token_ratio", "banglish_word_ratio",
    "english_word_ratio", "code_switch_rate", "spelling_variant_ratio",
    "avg_sentence_len", "punctuation_rate", "repeated_punct_rate", "emoji_rate",
    "elongation_rate", "uppercase_ratio", "digit_rate",
]
FEATURE_LABELS = {
    "n_words": "Number of words",
    "avg_word_len": "Average word length",
    "type_token_ratio": "Vocabulary variety",
    "banglish_word_ratio": "Share of Bangla words",
    "english_word_ratio": "Share of English words",
    "code_switch_rate": "Bangla/English switching",
    "spelling_variant_ratio": "Informal spelling variants",
    "avg_sentence_len": "Average sentence length",
    "punctuation_rate": "Punctuation",
    "repeated_punct_rate": "Repeated punctuation (..., !!)",
    "emoji_rate": "Emoji",
    "elongation_rate": "Stretched words (sooo)",
    "uppercase_ratio": "Capital letters",
    "digit_rate": "Digits",
}


def _lang_tag(word):
    w = word.lower()
    if w in BANGLISH_WORDS:
        return "bn"
    if w in ENGLISH_WORDS:
        return "en"
    return None


def text_features(text):
    """Return a dict of FEATURE_NAMES -> float for one text."""
    text = str(text)
    words = WORD_RE.findall(text)
    lower = [w.lower() for w in words]
    n = max(len(words), 1)
    tags = [t for t in (_lang_tag(w) for w in words) if t is not None]
    switches = sum(1 for a, b in zip(tags, tags[1:]) if a != b)
    sentences = [s for s in SENT_RE.split(text) if s.strip()]
    sent_lens = [len(WORD_RE.findall(s)) for s in sentences] or [0]
    letters = [c for c in text if c.isalpha()]
    n_letters = max(len(letters), 1)
    return {
        "n_words": float(len(words)),
        "avg_word_len": float(np.mean([len(w) for w in words])) if words else 0.0,
        "type_token_ratio": len(set(lower)) / n,
        "banglish_word_ratio": sum(w in BANGLISH_WORDS for w in lower) / n,
        "english_word_ratio": sum(w in ENGLISH_WORDS for w in lower) / n,
        "code_switch_rate": switches / max(len(tags) - 1, 1),
        "spelling_variant_ratio": sum(w in VARIANT_WORDS for w in lower) / n,
        "avg_sentence_len": float(np.mean(sent_lens)),
        "punctuation_rate": sum(c in ".,!?;:-" for c in text) / max(len(text), 1),
        "repeated_punct_rate": len(re.findall(r"[.!?,]{2,}", text)) / n,
        "emoji_rate": len(EMOJI_RE.findall(text)) / n,
        "elongation_rate": len(ELONGATED_RE.findall(text)) / n,
        "uppercase_ratio": sum(c.isupper() for c in letters) / n_letters,
        "digit_rate": sum(c.isdigit() for c in text) / max(len(text), 1),
    }


class StyloFeatures(BaseEstimator, TransformerMixin):
    """scikit-learn transformer: list of texts -> (n_texts, n_features)."""

    def __init__(self, names=None):
        self.names = names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        names = self.names or FEATURE_NAMES
        return np.array([[text_features(t)[k] for k in names] for t in X])

    def get_feature_names_out(self, input_features=None):
        return np.array(self.names or FEATURE_NAMES, dtype=object)


# ---------------------------------------------------------------- dataset
def n_words(text):
    return len(str(text).split())


def clean_text(text):
    text = re.sub(r"\s+", " ", str(text)).strip()
    return text.strip('"').strip()


def load_human(path, max_rows=None):
    """Human Banglish reviews written before ChatGPT's release, Latin script only."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, nrows=max_rows)
    df["text"] = df["text"].map(clean_text)
    df = df[df["language"] == "Banglish"]
    df = df[(df["date"] != "") & (df["date"] < CHATGPT_RELEASE)]
    df = df[~df["text"].str.contains(BENGALI_RE)]
    df = df[df["text"].map(n_words).between(5, 60)]
    df = df.drop_duplicates("text")
    out = pd.DataFrame({"text": df["text"], "label": "human", "generator": "human",
                        "domain": "ecommerce", "source": "BanglishRev reviews",
                        "date": df["date"]})
    return out.reset_index(drop=True)


def load_ai(paths, domains=("ecommerce",)):
    """paths: {generator_name: csv_path}. Keeps only the given domains."""
    parts = []
    for gen, path in paths.items():
        df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
        df["text"] = df["text"].map(clean_text)
        if domains:
            df = df[df["domain"].isin(domains)]
        df = df[~df["text"].str.contains(BENGALI_RE)]
        df = df[df["text"].map(n_words) >= 4]
        parts.append(pd.DataFrame({"text": df["text"], "label": "ai", "generator": gen,
                                   "domain": df["domain"], "source": gen, "date": ""}))
    out = pd.concat(parts).drop_duplicates("text")
    return out.reset_index(drop=True)


LENGTH_BINS = [0, 6, 8, 10, 12, 14, 16, 18, 20, 23, 26, 30, 40, 1000]


def length_matched_sample(human, ai, ratio=1.0, seed=SEED):
    """Sample human texts so their word-count histogram matches the AI texts.

    Without this, a model can separate the classes by length alone.
    """
    rng = np.random.default_rng(seed)
    h_bins = pd.cut(human["text"].map(n_words), LENGTH_BINS, right=False)
    a_bins = pd.cut(ai["text"].map(n_words), LENGTH_BINS, right=False)
    picks = []
    for b, want in a_bins.value_counts().items():
        pool = human.index[h_bins == b].to_numpy()
        k = min(len(pool), int(round(want * ratio)))
        if k:
            picks.extend(rng.choice(pool, size=k, replace=False))
    return human.loc[picks]


def build_dataset(human, ai, seed=SEED):
    """Length-matched, deduplicated dataset with a 70/15/15 split."""
    human = length_matched_sample(human, ai, seed=seed)
    df = pd.concat([human, ai]).drop_duplicates("text").sample(frac=1, random_state=seed)
    df = df.reset_index(drop=True)
    df.insert(0, "id", [f"{'h' if l == 'human' else 'a'}{i:05d}" for i, l in enumerate(df["label"])])
    df["y"] = (df["label"] == "ai").astype(int)
    strata = df["generator"]
    train, rest = train_test_split(df, test_size=0.30, stratify=strata, random_state=seed)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest["generator"], random_state=seed)
    df["split"] = "train"
    df.loc[val.index, "split"] = "val"
    df.loc[test.index, "split"] = "test"
    return df


# ---------------------------------------------------------------- models
def make_model(kind="full", C=4.0):
    """kind: 'length', 'stylo', 'chars', 'words' or 'full'."""
    stylo = lambda names=None: Pipeline([("extract", StyloFeatures(names)),
                                         ("scale", StandardScaler())])
    words = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2,
                            sublinear_tf=True, token_pattern=r"(?u)\b\w+\b", lowercase=True)
    chars = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=3,
                            sublinear_tf=True, max_features=60000, lowercase=True)
    parts = {
        "length": [("stylo", stylo(["n_words", "avg_word_len"]))],
        "stylo": [("stylo", stylo())],
        "chars": [("chars", chars)],
        "words": [("words", words)],
        "full": [("words", words), ("chars", chars), ("stylo", stylo())],
    }[kind]
    clf = LogisticRegression(C=C, max_iter=3000, class_weight="balanced")
    return Pipeline([("features", FeatureUnion(parts)), ("clf", clf)])


# ---------------------------------------------------------------- metrics
def threshold_at_fpr(y_true, p_ai, max_fpr=0.05):
    """Lowest threshold whose false-positive rate is at most max_fpr."""
    fpr, _, thr = roc_curve(y_true, p_ai)
    ok = np.where(fpr <= max_fpr)[0]
    return float(thr[ok[-1]])


def evaluate(y_true, p_ai, threshold=0.5):
    y_true, p_ai = np.asarray(y_true), np.asarray(p_ai)
    y_pred = (p_ai >= threshold).astype(int)
    humans, ais = y_true == 0, y_true == 1
    res = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "human_flagged_as_ai": float(y_pred[humans].mean()) if humans.any() else None,
        "ai_caught": float(y_pred[ais].mean()) if ais.any() else None,
    }
    res["auroc"] = float(roc_auc_score(y_true, p_ai)) if len(set(y_true.tolist())) == 2 else None
    return res


# ---------------------------------------------------------------- explanations
class LinearExplainer:
    """Exact SHAP values for the 'full' logistic-regression pipeline.

    For a linear model with independent features, the SHAP value of feature j
    is coef_j * (x_j - mean_j). Word and character n-gram values are mapped
    back onto the words of the text; stylometric values are reported by name.
    """

    def __init__(self, model, background_texts):
        self.model = model
        self.union = model.named_steps["features"]
        self.coef = model.named_steps["clf"].coef_.ravel()
        X = self.union.transform(list(background_texts))
        self.mean = np.asarray(X.mean(axis=0)).ravel()
        self.names = list(self.union.get_feature_names_out())
        self.base = float(model.named_steps["clf"].intercept_[0] + self.coef @ self.mean)

    def explain(self, text):
        x = self.union.transform([text])
        x = x.tocsr() if hasattr(x, "tocsr") else x
        dense = np.asarray(x.todense()).ravel() if hasattr(x, "todense") else np.asarray(x).ravel()
        shap = self.coef * (dense - self.mean)
        words = WORD_RE.findall(text)
        lw = [w.lower() for w in words]
        padded = [f" {w} " for w in lw]
        word_scores = np.zeros(len(words))
        stylo = {}
        for j in np.flatnonzero(dense):
            name, val = self.names[j], shap[j]
            block, feat = name.split("__", 1)
            if block == "words":
                toks = feat.split()
                hits = [i for i, w in enumerate(lw) if w in toks]
                for i in hits:
                    word_scores[i] += val / len(hits)
            elif block == "chars":
                hits = [i for i, p in enumerate(padded) if feat in p]
                for i in hits:
                    word_scores[i] += val / len(hits)
        for j, name in enumerate(self.names):
            if name.startswith("stylo__"):
                stylo[name.split("__", 1)[1]] = float(shap[j])
        logit = self.base + float(shap.sum())
        return {
            "p_ai": float(1 / (1 + np.exp(-logit))),
            "words": list(zip(words, word_scores.tolist())),
            "stylo": dict(sorted(stylo.items(), key=lambda kv: -abs(kv[1]))),
        }

    def global_importance(self, top=15):
        """Largest coefficients towards AI (+) and human (-), per feature block."""
        s = pd.Series(self.coef, index=self.names)
        return s.sort_values()

    def stylo_importance(self):
        s = pd.Series({n.split("__", 1)[1]: c for n, c in zip(self.names, self.coef)
                       if n.startswith("stylo__")})
        return s.reindex(s.abs().sort_values().index)
