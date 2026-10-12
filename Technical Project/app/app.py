"""BanglishCheck demo: paste Banglish text, get an AI score and the reasons.

Runs on a free Hugging Face Space (CPU). Needs, next to this file:
    banglishcheck.py   the shared module
    explainer.joblib   saved by the Colab notebook
"""

import gradio as gr
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from banglishcheck import FEATURE_LABELS, WORD_RE, n_words

explainer = joblib.load("explainer.joblib")

EXAMPLES = [
    "onek valo product. packaging o onek valo chilo. seller vai ke dhonnobad 🥰",
    "Daraz theke earbuds order dilam 3 din age, tracking e dekhay 'out for delivery' kintu rider er kono khobor nai.",
    "Product er quality dekhe mone hocche price ta worth it, packaging o neat and clean chilo.",
    "size ta aktu choto hoise.. but overall kharap na,, nite paren",
]

NOTICE = (
    "**Please read:** this score is an estimate from a student project. It can be wrong, "
    "especially on short texts and on writing styles it hasn't seen. Never use it to accuse "
    "anyone of using AI."
)


def stylo_plot(stylo, top=8):
    items = list(stylo.items())[:top][::-1]
    labels = [FEATURE_LABELS.get(k, k) for k, _ in items]
    vals = [v for _, v in items]
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.barh(labels, vals, color=["#C2410C" if v > 0 else "#0F766E" for v in vals])
    ax.axvline(0, color="#444", lw=0.8)
    ax.set_xlabel("pushes towards Human (−)   |   towards AI (+)")
    ax.set_title("Writing-style signals")
    fig.tight_layout()
    return fig


def analyse(text):
    text = (text or "").strip()
    if not text:
        return {}, [], None, "Paste some Banglish text first."
    res = explainer.explain(text)
    p = res["p_ai"]
    biggest = max((abs(s) for _, s in res["words"]), default=1.0) or 1.0
    # Rebuild the full text: scored words, with the spaces, punctuation and
    # emoji between them kept unscored.
    highlighted, pos = [], 0
    for m, (_, score) in zip(WORD_RE.finditer(text), res["words"]):
        if m.start() > pos:
            highlighted.append((text[pos:m.start()], None))
        highlighted.append((m.group(), round(score / biggest, 3)))
        pos = m.end()
    if pos < len(text):
        highlighted.append((text[pos:], None))
    note = NOTICE
    if n_words(text) < 8:
        note = "⚠️ This text is very short, so the score is unreliable.\n\n" + NOTICE
    return {"AI-generated": p, "Human-written": 1 - p}, highlighted, stylo_plot(res["stylo"]), note


with gr.Blocks(title="BanglishCheck") as demo:
    gr.Markdown(
        "# BanglishCheck\n"
        "Paste a Banglish text (Bangla written in English letters). The detector estimates whether "
        "it was written by a person or by an AI model, and shows which words and writing-style "
        "signals drove the decision."
    )
    with gr.Row():
        with gr.Column():
            inp = gr.Textbox(label="Banglish text", lines=5, placeholder="ami kal class e jabo na, too tired...")
            btn = gr.Button("Check", variant="primary")
            gr.Examples(EXAMPLES, inputs=inp)
        with gr.Column():
            label = gr.Label(label="Prediction", num_top_classes=2)
            words = gr.HighlightedText(label="Word evidence: red = AI-like, blue = human-like")
            plot = gr.Plot(label="Writing-style signals")
            note = gr.Markdown(NOTICE)
    btn.click(analyse, inputs=inp, outputs=[label, words, plot, note])
    inp.submit(analyse, inputs=inp, outputs=[label, words, plot, note])

if __name__ == "__main__":
    demo.launch()
