# Blog: Why AI Text Detectors Struggle with Banglish

The article is the PDF in the main folder: `Why_AI_Text_Detectors_Struggle_with_Banglish.docx.pdf`.

This folder has the experiment behind it:

| Path | What it is |
|---|---|
| `experiment/banglish_detectors_experiment.ipynb` | Colab notebook that runs three existing detectors on 100 human and 100 AI Banglish texts |
| [Google Colab](https://colab.research.google.com/drive/1A_5cKP-HulXIOz3C2_SC3QrxM52M-dKa) | Shared notebook with the completed T4 GPU run and saved outputs |
| `experiment/colab_all_in_one.py` | The same experiment as one script |
| `results/` | Accuracy of each detector, overall and per AI model |
| `figures/` | The figures used in the article |

The text samples themselves are not included.

## Saved results

The shared Colab notebook was run on 200 samples: 100 human texts and 100 AI texts (50 ChatGPT and 50 Gemini). Its aggregate result tables are included in `results/`.
