# BanglishCheck to-do list

**Deadline: Tuesday 20 October 2026.** Submit a live demo (a video is a bonus), a technical report and a GitHub link.
The blog is due the same day. Tick each box as you go (change `[ ]` to `[x]`).

## Mon 12 Oct: demo online
- [ ] Rotate the Google API key that was in `.env`, if not done yet
- [ ] Fix the Hugging Face Space: delete `index.html`, upload the 5 files from `space.zip`, wait for "Running"
- [ ] Test the Space with one human review and one AI review
- [ ] Send Claude the Space link (it goes in the README)

## Tue 13 Oct: run the notebook
- [ ] Upload the 4 CSV files to `MyDrive/banglishcheck/data/`
- [ ] Open `notebooks/BanglishCheck_Colab.ipynb` in Colab, pick the T4 GPU, choose Run all
- [ ] Send Claude `model_comparison.csv`, `leave_one_generator_out.csv` and `faithfulness.csv` from `outputs/results/`
- [ ] If Colab fails or runs out of time, skip it: the classical results already in the README are enough to submit

## Wed 14 – Fri 16 Oct: technical report
- [x] Finalize the report (`report/BanglishCheck_Technical_Report_FINAL.docx`) and remove draft placeholders/highlights
- [x] Keep XLM-R explicitly as future work until a completed Colab run is available
- [x] Add the BanglishRev citation
- [x] Export it as PDF (`report/BanglishCheck_Technical_Report_FINAL.pdf`)

## Sat 17 Oct: demo video (bonus, 3–5 minutes)
- [ ] Record your screen: what the problem is, then the Space with a human text and an AI text, then the highlighted words and style chart
- [ ] Upload it (YouTube unlisted or Google Drive, set to "anyone with the link")

## Sun 18 Oct: GitHub
- [ ] Check the README links: demo, video, report
- [ ] Never upload `.env`, the CSV files or model files

## Mon 19 Oct: final check
- [ ] Open every link in a private browser window to make sure it works for others
- [ ] Finish the blog (also due 20 Oct)

## Tue 20 Oct: submit
- [ ] Live demo link
- [ ] Video link (bonus)
- [ ] Technical report (PDF)
- [ ] GitHub link
