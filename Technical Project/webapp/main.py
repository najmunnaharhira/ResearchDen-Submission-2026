"""BanglishCheck web app: FastAPI backend + static frontend.

Standalone, dependency-light alternative to the Gradio/Hugging Face Space demo,
meant to be deployed behind your own domain (see README.md).

Needs, next to this file:
    banglishcheck.py   the shared module (already copied here)
    explainer.joblib    saved by the Colab notebook — place it in this folder
"""

import os

import joblib
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from banglishcheck import FEATURE_LABELS, WORD_RE, n_words

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
EXPLAINER_PATH = os.path.join(BASE_DIR, "explainer.joblib")

app = FastAPI(title="BanglishCheck API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

explainer = None
if os.path.exists(EXPLAINER_PATH):
    explainer = joblib.load(EXPLAINER_PATH)

NOTICE = (
    "Please read: this score is an estimate from a student project. It can be wrong, "
    "especially on short texts and on writing styles it hasn't seen. Never use it to accuse "
    "anyone of using AI."
)

EXAMPLES = [
    "onek valo product. packaging o onek valo chilo. seller vai ke dhonnobad 🥰",
    "Daraz theke earbuds order dilam 3 din age, tracking e dekhay 'out for delivery' kintu rider er kono khobor nai.",
    "Product er quality dekhe mone hocche price ta worth it, packaging o neat and clean chilo.",
    "size ta aktu choto hoise.. but overall kharap na,, nite paren",
]


class AnalyzeRequest(BaseModel):
    text: str


class Segment(BaseModel):
    text: str
    score: float | None = None


class StyloSignal(BaseModel):
    key: str
    label: str
    score: float


class AnalyzeResponse(BaseModel):
    p_ai: float
    p_human: float
    segments: list[Segment]
    stylo: list[StyloSignal]
    note: str
    short_text: bool


@app.get("/api/examples")
def get_examples():
    return {"examples": EXAMPLES}


@app.get("/api/health")
def health():
    return {"status": "ok", "model_loaded": explainer is not None}


@app.post("/api/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    if explainer is None:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded: put explainer.joblib next to main.py and restart.",
        )

    text = (req.text or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="Paste some Banglish text first.")

    res = explainer.explain(text)
    p = res["p_ai"]
    biggest = max((abs(s) for _, s in res["words"]), default=1.0) or 1.0

    segments, pos = [], 0
    for m, (_, score) in zip(WORD_RE.finditer(text), res["words"]):
        if m.start() > pos:
            segments.append(Segment(text=text[pos:m.start()], score=None))
        segments.append(Segment(text=m.group(), score=round(score / biggest, 3)))
        pos = m.end()
    if pos < len(text):
        segments.append(Segment(text=text[pos:], score=None))

    stylo = [
        StyloSignal(key=k, label=FEATURE_LABELS.get(k, k), score=v)
        for k, v in res["stylo"].items()
    ]

    short = n_words(text) < 8
    note = ("This text is very short, so the score is unreliable. " + NOTICE) if short else NOTICE

    return AnalyzeResponse(
        p_ai=p, p_human=1 - p, segments=segments, stylo=stylo, note=note, short_text=short,
    )


app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "static"), html=True), name="static")
