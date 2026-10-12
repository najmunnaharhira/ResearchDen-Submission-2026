# Literature Review Plan: Human-Centered Explainable AI

## Proposed topic

**Working title:** Human-Centered Explainable AI Beyond English: Trust, Reliance and Explanation Quality for Low-Resource and Code-Mixed Language Users, with Bangla and Banglish as a Case.

The 20 collected papers fall into two groups that rarely cite each other. One group studies how people understand and rely on AI explanations, almost always in English. The other builds Bangla and code-mixed datasets and models, but does not ask how users understand those models. The review sits where the two meet, and it connects to the BanglishCheck technical project.

**Research questions:**

1. How do current studies measure whether an explanation works for people (trust, understanding, reliance)?
2. What changes when the user, the model or the explanation is not in English?
3. What would human-centered XAI for Bangla and Banglish need that existing resources do not yet provide?

## What the papers cover

Nine papers are about people and explanations; eleven are about Bangla or code-mixed language technology. Per-paper details are in [reading_matrix.csv](reading_matrix.csv) and links in [papers.md](papers.md).

| Group | Papers | What the group gives the review |
| --- | --- | --- |
| A. Measuring human-centered XAI | Hoffman et al. 2018; Rong et al. 2024 (survey of 97 XAI user studies) | The vocabulary: trust, understanding, satisfaction, reliance, human-AI performance, and how each is measured |
| B. Reliance on LLM explanations | Si et al. 2024; Steyvers et al. 2025; Bo et al. 2025 | Evidence that people over-rely on fluent, confident explanations, and what partly helps |
| C. XAI beyond English | Rathi et al. 2025; Wang et al. 2025; Mehrazar et al. 2025 | Language changes how explanations are read and how faithful they are |
| D. Why LLM outputs mislead | Huang et al. 2024 | Background on why explanations can be fluent but wrong |
| E. Code-mixed NLP | Banerjee et al. 2018; Vavre et al. 2022; Winata et al. 2023; Yang and Chai 2025 | Code-mixed text is hard for models, so explanations about it are hard too |
| F. Bangla resources | Ekram et al. 2022; Fahim et al. 2024; Alam et al. 2025; Nahin et al. 2025; Mahjabin et al. 2025; Jawad et al. 2025; Hasan and Dipta 2025 | Many datasets and benchmarks, none with a human-centered explanation study |

## Planned structure

About 3,500 words, to be adjusted once the expected length is announced.

| # | Section | Question answered | Papers | Words |
| --- | --- | --- | --- | --- |
| 1 | Introduction | Why should explanations be judged by the people who use them, and why does language matter? | Rong; Rathi; Winata | 400 |
| 2 | Method of the review | How were the papers selected? | All | 250 |
| 3 | Evaluating explanations with people | What do trust, understanding and appropriate reliance mean, and how are they measured? | Hoffman; Rong | 600 |
| 4 | Over-reliance on LLM explanations | When do explanations help, and when do they make people trust wrong answers? | Si; Steyvers; Bo; Huang | 650 |
| 5 | Explanations across languages | Does an explanation work the same in Japanese, Persian or Telugu as in English? | Rathi; Wang; Mehrazar | 600 |
| 6 | The Bangla and code-mixed landscape | What exists for Bangla and Banglish, and what is missing for explainability? | Groups E and F | 600 |
| 7 | Gaps, research agenda and conclusion | What should a human-centered XAI study for Banglish users look like? | Synthesis; BanglishCheck as an example | 400 |

## Gaps to argue

- **User studies are English-centric.** Si et al. list their English-speaking crowdworker sample as a limitation, and Rathi et al. show reliance results from English do not transfer to Japanese.
- **Faithfulness in low-resource languages is barely tested.** Mehrazar et al. find LLM self-explanations in Persian agree more with each other than with humans; nothing similar exists for Bangla.
- **Bangla papers stop at accuracy.** BanglaRQA, BnSentMix, TituLLMs and the dialect benchmarks report EM, F1 or BLEU, with no explanation or user study.
- **Romanized spelling varies a lot.** BanglaTLit shows one word written many ways ("bhalo", "valo", "vlo"), so word-level explanations may be unstable for Banglish (an open question, not yet tested).
- **Hedging differs by language.** Rathi et al. show people read uncertainty phrases differently across languages; this has not been checked for Bangla.

## Foundational papers to add (to verify before citing)

- Doshi-Velez and Kim (2017), "Towards a Rigorous Science of Interpretable Machine Learning"
- Miller (2019), "Explanation in Artificial Intelligence: Insights from the Social Sciences"
- Ehsan and Riedl (2020), "Human-centered Explainable AI: Towards a Reflective Sociotechnical Approach"
- Liao, Gruen and Miller (2020), "Questioning the AI: Informing Design Practices for Explainable AI User Experiences"
- Jacovi and Goldberg (2020), "Towards Faithfully Interpretable NLP Systems"
- Bansal et al. (2021), "Does the Whole Exceed its Parts? The Effect of AI Explanations on Complementary Team Performance"
- Vasconcelos et al. (2023), "Explanations Can Reduce Overreliance on AI Systems During Decision-Making"

## Steps

- [ ] Topic approved by the leads
- [ ] Notes written for every paper in the reading matrix
- [ ] Sections 3, 4 and 5 drafted
- [ ] Section 6 drafted
- [ ] Section 7 and introduction drafted
- [ ] Reference list checked against every source
