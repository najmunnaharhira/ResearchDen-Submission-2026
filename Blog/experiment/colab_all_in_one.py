# Why AI Text Detectors Struggle with Banglish: experiment
# Paste into ONE Colab cell. Runtime > Change runtime type > T4 GPU first.
# A file picker opens partway through: choose samples.csv.

# ===== 1. Install and import =====
!pip install -q transformers pandas scikit-learn matplotlib

import math
import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

DEVICE = 0 if torch.cuda.is_available() else -1
print("GPU" if DEVICE == 0 else "CPU (slower, still fine)")

# ===== 2. Upload samples.csv =====
from google.colab import files
uploaded = files.upload()
CSV_NAME = list(uploaded)[0]  # Colab renames repeat uploads, e.g. "samples (1).csv"
print("Using", CSV_NAME)

# ===== 3. Load and check the data =====
df = pd.read_csv(CSV_NAME, dtype=str, keep_default_na=False)
df = df[~df["id"].str.startswith("example")]
df["text"] = df["text"].str.strip()
df["label"] = df["label"].str.strip().str.lower()
df = df[df["text"] != ""].drop_duplicates("text").reset_index(drop=True)

assert len(df) > 0, ("No samples found. samples.csv only has the example rows. "
                     "Add your own human and AI texts, then upload it again.")
for lab in ["human", "ai"]:
    assert (df["label"] == lab).any(), f"No rows labelled {lab!r}. You need both human and ai rows."

bad = df[~df["label"].isin(["human", "ai"])]
assert bad.empty, f"These rows need label human or ai:\n{bad[['id', 'label']]}"

print(df["label"].value_counts())
print(df.groupby("label")["generator"].value_counts())
display(df.sample(min(5, len(df)), random_state=0))

# ===== 4. Figure 1: how tokenizers split Banglish =====
SENTENCES = {
    "English": "I won't go to class tomorrow, I'm too tired.",
    "Bangla script": "আমি কাল ক্লাসে যাব না, খুব ক্লান্ত।",
    "Banglish": "ami kal class e jabo na, too tired.",
}
rows = []
for name in ["gpt2", "xlm-roberta-base"]:
    tok = AutoTokenizer.from_pretrained(name)
    for form, text in SENTENCES.items():
        pieces = tok.tokenize(text)
        rows.append({"tokenizer": name, "form": form, "tokens": len(pieces)})
        print(f"{name:18} {form:14} {len(pieces):3}  {pieces[:12]}")

tokens = pd.DataFrame(rows).pivot(index="form", columns="tokenizer", values="tokens")
ax = tokens.loc[list(SENTENCES)].plot.bar(rot=0, figsize=(7, 4))
ax.set_ylabel("Number of tokens")
ax.set_title("Same sentence, different tokenization")
plt.tight_layout(); plt.savefig("fig1_tokens.png", dpi=200); plt.show()
display(tokens)

# ===== 5. Run the trained classifiers =====
CLASSIFIERS = {
    "ChatGPT-RoBERTa (HC3)": "Hello-SimpleAI/chatgpt-detector-roberta",
    "OpenAI GPT-2 detector": "openai-community/roberta-base-openai-detector",
}
# Label names each model uses for "AI". Check the printed labels below.
AI_LABELS = {"chatgpt", "fake", "label_1", "ai"}

texts = df["text"].tolist()
for name, model_id in CLASSIFIERS.items():
    clf = pipeline("text-classification", model=model_id, device=DEVICE, truncation=True)
    out = clf(texts, batch_size=16)
    print(name, "returns labels:", sorted({o["label"] for o in out}))
    df[name] = ["ai" if o["label"].lower() in AI_LABELS else "human" for o in out]
    # Confidence that the text is AI, used for Figure 3
    df[name + " p(ai)"] = [o["score"] if o["label"].lower() in AI_LABELS else 1 - o["score"]
                           for o in out]

# ===== 6. Run the perplexity detector (zero-shot) =====
PPL_MODEL = "bigscience/bloom-560m"
tok = AutoTokenizer.from_pretrained(PPL_MODEL)
lm = AutoModelForCausalLM.from_pretrained(PPL_MODEL).eval()
if DEVICE == 0:
    lm = lm.cuda()

ppl = []
for text in texts:
    enc = tok(text, return_tensors="pt", truncation=True, max_length=256)
    enc = {k: v.to(lm.device) for k, v in enc.items()}
    with torch.no_grad():
        ppl.append(math.exp(lm(**enc, labels=enc["input_ids"]).loss.item()))
df["perplexity"] = ppl

threshold = df["perplexity"].median()
df["Perplexity (BLOOM-560m)"] = ["ai" if p < threshold else "human" for p in ppl]
print(df.groupby("label")["perplexity"].describe().round(1))

# ===== 7. Figure 2: results for each detector =====
DETECTORS = list(CLASSIFIERS) + ["Perplexity (BLOOM-560m)"]
y = df["label"]

def rates(pred):
    human, ai = pred[y == "human"], pred[y == "ai"]
    return {
        "Accuracy": accuracy_score(y, pred),
        "False positive rate\n(human flagged as AI)": (human == "ai").mean(),
        "False negative rate\n(AI passed as human)": (ai == "human").mean(),
    }

results = pd.DataFrame({d: rates(df[d]) for d in DETECTORS}).T
ax = results.plot.bar(rot=10, figsize=(9, 4.5), ylim=(0, 1))
ax.set_title(f"Detectors on Banglish text (n = {len(df)})")
ax.legend(fontsize=8)
plt.tight_layout(); plt.savefig("fig2_detectors.png", dpi=200); plt.show()
display(results.round(3))

# How often each generator's texts passed as human (false negatives per model)
ai_rows = df[df["label"] == "ai"]
by_gen = pd.DataFrame({d: ai_rows.groupby("generator")[d].apply(lambda s: (s == "human").mean())
                       for d in DETECTORS})
print("Share of AI texts that passed as human, by generator:")
display(by_gen.round(3))

# ===== 8. Figure 3: where the detectors went wrong =====
main = DETECTORS[0]
wrong = df[df[main] != df["label"]].copy()
wrong["confidence"] = (wrong[main + " p(ai)"] - 0.5).abs() * 2
cols = ["text", "label", main, main + " p(ai)", "generator"]
display(wrong.sort_values("confidence", ascending=False)[cols].head(10))

# ===== 9. Save and download everything =====
df.to_csv("predictions.csv", index=False)
results.to_csv("results.csv")
by_gen.to_csv("results_by_generator.csv")
for f in ["fig1_tokens.png", "fig2_detectors.png", "predictions.csv", "results.csv", "results_by_generator.csv"]:
    files.download(f)
