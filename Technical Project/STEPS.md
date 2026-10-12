# BanglishCheck: what to do, step by step

## 1. Run the notebook (about 30–45 minutes, mostly waiting)
1. In Google Drive, make the folder `MyDrive/banglishcheck/data/` and upload these four files with these exact names:
   - `banglish_filtered_quality.csv` (human reviews)
   - `banglish_ai_samples.csv` (Claude)
   - `chatgpt_banglish_samples.csv`
   - `gemini_banglish_samples.csv`
2. Open `BanglishCheck_Colab.ipynb` in Colab (File > Upload notebook).
3. Runtime > Change runtime type > **T4 GPU**.
4. Runtime > **Run all**. Allow Drive access when asked.
5. When it finishes, everything is in `MyDrive/banglishcheck/outputs/`:
   - `results/` tables for your report
   - `figures/` images for your report and slides
   - `space.zip` the demo files
   - The last cell opens the demo with a public link you can try straight away.

If XLM-R runs out of memory, find the line `MODEL_NAME, MAX_LEN = "xlm-roberta-base", 128` in the XLM-R cell, change 128 to 96, and run from that cell again.

## 2. Put the demo online (10 minutes)
1. Make a free account on huggingface.co.
2. New > Space. Name it `BanglishCheck`, SDK **Gradio**, hardware **CPU basic (free)**.
3. Unzip `space.zip` and upload all five files (Files > Add file > Upload files).
4. Wait for "Running". Copy the link: this is your **live demo**.

## 3. GitHub
Upload `README.md`, `.gitignore`, `banglishcheck.py`, `app/` and `notebooks/` to your repo.
Do **not** upload `.env`, the CSV files, or anything in `outputs/models`.
Fill in the results table and links in the README.

## 4. Demo video (bonus, 2–3 minutes)
Screen-record the Space: paste a human review, then an AI one, and explain the highlighted words and the style chart.

## 5. Technical report
Sections: Introduction, Related work, Data, Methods, Results, Explanations, Robustness, Limitations, Ethics, References.
Use the tables in `results/` and the figures in `figures/`. Write it in your own words.

## 6. Presentation (10 minutes max)
| Slide | Content | Time |
|---|---|---|
| 1 | Title, your name | 0:15 |
| 2 | Introduction: what Banglish is, why AI text is a problem | 1:00 |
| 3 | Motivation: detectors are built for English; false accusations hurt people | 1:00 |
| 4 | Background: perplexity, classifiers, XLM-R, SHAP, IG | 1:00 |
| 5 | System design and tech stack: data → models → explainer → Gradio app (`fig_lengths`) | 1:30 |
| 6 | Results: model comparison (`fig_models`, `fig_roc`) | 1:30 |
| 7 | Explanations: `fig_style_features`, one highlighted example, faithfulness test | 1:00 |
| 8 | Robustness: unseen generator (`fig_unseen_generator`) | 0:45 |
| 9 | Live walkthrough of the Space | 1:00 |
| 10 | Limitations and future scope | 0:30 |
| 11 | References | 0:15 |
