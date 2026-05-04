# Taglish Sentiment Analysis — Multilingual Transformers for Low-Resource Code-Switched Philippine Social Media

> **An Analysis of Multilingual Transformers for Low-Resource Taglish Sentiment Analysis: Effects of Code-Switch Density**  
> Beldad, J. P. · Burac, A. A. · Cea, G. I. M. S. · Guevara, K. — Partido State University, BSCS 2026  
> Adviser: Shane Catolico-Briones

---

## Overview

Taglish — the fluid code-switched blend of Tagalog and English ubiquitous on Philippine social media — presents a unique and underexplored challenge for automated sentiment analysis. Standard multilingual NLP systems fail to capture its cultural nuance, informal syntax, and varying degrees of language mixing.

This repository contains the fine-tuned transformer models, density-stratified dataset, and supporting research artifacts from a systematic study investigating how **code-switch density** affects the sentiment classification performance of three multilingual transformer architectures across three distinct training conditions.

---

## 🤗 Models & Dataset on Hugging Face

All models and the annotated dataset are publicly available at **[huggingface.co/mggy](https://huggingface.co/mggy)**.

| Resource | Link | Size | Description |
|---|---|---|---|
| **XLM-RoBERTa** (fine-tuned) | [`mggy/taglish-xlm-roberta-sentiment`](https://huggingface.co/mggy/taglish-xlm-roberta-sentiment) | 0.3B | Best overall — macro-F1: **0.7504** |
| **mBERT** (fine-tuned) | [`mggy/taglish-mbert-sentiment`](https://huggingface.co/mggy/taglish-mbert-sentiment) | 0.2B | Strongest curriculum response |
| **TLUnified-RoBERTa** (fine-tuned) | [`mggy/taglish-tlunified-sentiment`](https://huggingface.co/mggy/taglish-tlunified-sentiment) | 0.1B | Best Tagalog-dominant stability |
| **Dataset** | [`mggy/taglish-socialmedia-dataset`](https://huggingface.co/datasets/mggy/taglish-socialmedia-dataset) | 7,707 posts | Density-stratified, near-balanced |

---

## Task: Taglish Sentiment Classification

Given a social media post written in Taglish, Tagalog, or English, the models classify it into one of three sentiment categories:

- **Positive**
- **Negative**
- **Neutral**

Input posts are drawn from Facebook, Reddit, and YouTube, and vary across five code-switch density bands — from predominantly Tagalog (0–20% English tokens) to predominantly English (81–100% English tokens).

---

## Dataset

The corpus was constructed entirely in-house using a custom data collection pipeline called **TAGCOL**, without reliance on any pre-existing annotated datasets.

**7,707 annotated social media posts** sourced from:

| Platform | N | Dominant Sentiment | Neg. Rate |
|---|---|---|---|
| Reddit | 4,047 | Negative | 47.9% |
| YouTube | 2,198 | Positive | 6.6% |
| Facebook | 977 | Negative | 48.1% |
| Augmented | 485 | Neutral (only) | — |
| **Total** | **7,707** | **Near-balanced** | — |

**Class balance:** Positive 33.4% · Negative 33.0% · Neutral 33.6%

### Code-Switch Density Bands

Each post is labeled with its **code-switch density** — the proportion of English tokens to total tokens — and stratified into five bands for systematic evaluation:

| Band | English Token Range | Dominant Language | Notes |
|---|---|---|---|
| Band 1 | 0–20% | Tagalog | Monolingual Filipino |
| Band 2 | 21–40% | Tagalog-dominant | Light switching |
| Band 3 | 41–60% | Mixed | **Best model performance** |
| Band 4 | 61–80% | English-dominant | Heavy switching |
| Band 5 | 81–100% | Near-English | **Hardest for all models** |

---

## Models

### Architecture Comparison

All three models were fine-tuned on the same 7,707-post corpus under three training conditions: **Standard Fine-Tuning** (full FP16), **QLoRA Baseline** (parameter-efficient, no curriculum), and **QLoRA + SPCL** (density-ordered Self-Paced Curriculum Learning).

### Overall Macro-F1 Results

*Benchmark threshold: 0.70 (state-of-the-art on comparable Taglish tasks)*

| Model | Standard (FP16) | QLoRA Baseline | QLoRA + SPCL |
|---|---|---|---|
| **XLM-RoBERTa** | **0.7504 ✓** | **0.7278 ✓** | 0.6942 |
| **mBERT** | 0.7214 ✓ | 0.6599 | 0.6745 |
| **TLUnified-RoBERTa** | 0.7181 ✓ | 0.6822 | 0.6570 |

✓ = meets or exceeds prior benchmark (0.70)

### Per-Density-Band Macro-F1

| Model / Condition | Band 1 (0–20%) | Band 2 (21–40%) | Band 3 (41–60%) ★ | Band 4 (61–80%) | Band 5 (81–100%) ▼ |
|---|---|---|---|---|---|
| XLM-RoBERTa (Standard) | 0.739 | 0.713 | **0.864** | 0.703 | 0.691 |
| mBERT (Standard) | 0.720 | 0.664 | **0.824** | 0.742 | 0.624 |
| TLUnified (Standard) | 0.764 | 0.654 | **0.818** | 0.650 | 0.681 |

★ Band 3 consistently highest across all models · ▼ Band 5 consistently lowest

---

## Key Findings

### 1. XLM-RoBERTa is the Recommended Model for Deployment
XLM-RoBERTa under standard fine-tuning achieves the highest overall macro-F1 (0.7504), the lowest misclassification rate (24.9% of the test set), and the most failure-resistant behavior across all linguistic categories. Under QLoRA, it still exceeds the 0.70 benchmark (0.7278), making it the best choice for resource-constrained deployments.

### 2. Mixed Code-Switching (Band 3) Is Actually the Easiest to Classify
Contrary to intuition, posts with 41–60% English tokens yield the *best* classification performance — peaking at macro-F1 = 0.864 for XLM-RoBERTa. Balanced bilingual mixing provides cross-lingual redundancy within an utterance, helping models disambiguate sentiment more reliably. This is the study's most theoretically significant structural finding.

### 3. High-English Taglish (Band 5) Is the Hardest
Posts with 81–100% English tokens consistently yield the lowest macro-F1 (0.532–0.691). This occurs because high-density Taglish is *not* standard English — it applies Filipino syntactic patterns, Filipino pragmatic conventions (indirect disagreement, face-saving), and Filipino internet slang to English vocabulary. Models' English representations activate without the required Filipino cultural "overlay."

### 4. QLoRA Is Viable for Low-Resource Deployment
QLoRA reduces full fine-tuning performance by 2–6 points but enables 4-bit quantization (~8× memory reduction). XLM-RoBERTa under QLoRA still exceeds the benchmark, making it feasible to deploy Taglish sentiment classifiers on consumer-grade hardware without GPU clusters.

### 5. Density-Ordered Curriculum Learning Did Not Help
The QLoRA + SPCL curriculum consistently underperformed both baselines across all models. Three contributing factors: (1) the 7,707-sample corpus is below the scale at which curriculum learning reliably improves performance; (2) code-switch density is a noisy difficulty proxy — sample-level difficulty varies widely within each band; (3) QLoRA's low-rank adapters are sensitive to distribution shifts at stage transitions. The sole exception: mBERT showed a marginal +0.015 improvement over its QLoRA baseline, suggesting weaker base models may benefit more from staged training.

### 6. Emojis Generally Hurt — But 4+ Emojis Paradoxically Help
Emoji presence is associated with 2–9 point macro-F1 drops (caused by tokenization failures on non-linguistic symbols). However, posts with 4+ emojis achieve anomalously *high* performance (up to 0.872), consistent with an "emoji consensus signal" — redundant affective marking that eliminates classification ambiguity. TLUnified-RoBERTa shows near-zero emoji sensitivity (Δ = +0.001) under curriculum training, suggesting native-language pretraining confers pragmatic advantages for Filipino emoji conventions.

---

## Qualitative Failure Analysis

The most frequent failure modes across all models (as % of the 770-post test set):

| Category | Rate Range | Description |
|---|---|---|
| **Other (untagged)** | 13.0–18.4% | Label noise, deep pragmatic inference, cultural background knowledge |
| **Context-dependent** | 5.1–8.7% | Polarity determined by external context unavailable to the model |
| **Mixed sentiment** | 5.7–8.8% | Contrast markers — positive and negative co-occur in one utterance |
| **Netspeak / slang** | 2.3–4.8% | Filipino internet vernacular (e.g., "marites", "sana all", "charot") |
| **Sarcasm / irony** | 1.3–2.7% | Affirmative words paired with negative emojis or context |
| **Emoji-heavy** | 0.3–0.7% | Sentiment encoded primarily through paralinguistic emoji signals |
| **Baliktad na salita** | 0.1–0.4% | Syllable-reversed Filipino slang (e.g., "lodi" = idol, "erpat" = father) |

---

## Training Setup

| Parameter | Value |
|---|---|
| Hardware | Google Colab (free-tier T4 GPU) |
| Max sequence length | 128 tokens (covers 92.8% of posts) |
| QLoRA rank | r = 16 |
| Training precision | FP16 (standard) / 4-bit (QLoRA) |
| Loss function | Weighted cross-entropy (balanced class weights) |
| Curriculum stages | 4 stages: Band 1 → Band 2 → Band 5 → Band 3+4 |
| Stage learning rates | 3×10⁻⁵ → 2×10⁻⁵ → 1×10⁻⁵ → 8×10⁻⁶ |
| Early stopping | Patience = 3 (per stage), warmup = 2 evaluations |

---

## Quick Start

### Installation

```bash
pip install transformers torch
```

### Inference with XLM-RoBERTa (Recommended)

```python
from transformers import pipeline

classifier = pipeline(
    "text-classification",
    model="mggy/taglish-xlm-roberta-sentiment"
)

# Taglish examples
posts = [
    "Grabe ang ganda niya, pero charot lang.",          # Sarcasm
    "I'm so proud of you! Laking achievement nito!",   # Positive
    "Hindi ko maintindihan bakit ganito ang nangyari.", # Neutral
]

for post in posts:
    result = classifier(post)
    print(f"{post}\n→ {result[0]['label']} ({result[0]['score']:.3f})\n")
```

### Loading the Dataset

```python
from datasets import load_dataset

dataset = load_dataset("mggy/taglish-socialmedia-dataset")
print(dataset)

# Filter by density band
band3 = dataset['train'].filter(lambda x: x['density_band'] == 3)
```

---

## Repository Structure

```
.
.
│   app.py
│   enhancer.py
│   logger.py
│   predict.py
│   utils.py
│   requirements.txt
│   README.md
│   .gitignore
│   history.db
│
├── models/
│   ├── taglish-mbert-sentiment/
│   ├── taglish-tlunified-sentiment/
│   └── taglish-xlm-roberta-sentiment/
│
├── static/
│   ├── css/style.css
│   ├── js/main.js
│   ├── js/chart.umd.min.js
│   └── images/
│       ├── architecture.png
│       └── shocked.png
│
├── templates/
│   ├── base.html
│   ├── landing.html
│   ├── index.html
│   ├── about.html
│   ├── batch.html
│   ├── compare.html
│   └── history.html
---

## Societal Impact & Intended Use

These models are intended for research and public-benefit applications in Philippine NLP contexts, including:

- **Social listening and brand monitoring** in Philippine markets
- **Public discourse monitoring** for government agencies and civic tech developers
- **Disaster response and community feedback** tracking in Filipino
- **Academic baseline** for future Filipino NLP studies — including ABSA, ensemble modeling, and hate speech detection

The density-stratified evaluation framework is designed to be replicable. Future researchers can directly benchmark against this corpus across the five defined density bands rather than treating Taglish as a uniform monolithic category.

### Limitations

- Models were trained on social media posts from Facebook, Reddit, and YouTube only; performance on other domains (news, formal writing, customer service) is untested.
- Band 5 (81–100% English token density) performance remains the weakest across all models due to Filipino pragmatic conventions being surface-camouflaged as standard English.
- Sarcasm, irony, *baliktad na salita*, and deep context-dependent posts remain reliably misclassified across all architectures.
- AI-generated augmentation was used for neutral posts in Bands 3–4; these samples are an imperfect proxy for organic code-switching.
- Models were evaluated under Google Colab free-tier GPU constraints; results under different hardware configurations may vary slightly.

---

## Citation

If you use these models, the dataset, or the density-stratified evaluation framework in your research, please cite:

```bibtex
@thesis{beldad2026taglish,
  title     = {An Analysis of Multilingual Transformers for Low-Resource Taglish Sentiment Analysis: Effects of Code-Switch Density},
  author    = {Beldad, Jerald P. and Burac, Alzyn A. and Cea, Gil IV Miguel Salvador I. and Guevara, Kimberly},
  year      = {2026},
  school    = {Partido State University},
  degree    = {Bachelor of Science in Computer Science},
  adviser   = {Shane Catolico-Briones}
}
```

---

## Authors

**Jerald P. Beldad** · **Alzyn A. Burac** · **Gil IV Miguel Salvador I. Cea** · **Kimberly Guevara**  
Partido State University — Bachelor of Science in Computer Science, April 2026  
Adviser: Shane Catolico-Briones  

🤗 Hugging Face: [huggingface.co/mggy](https://huggingface.co/mggy)  
GitHub: [github.com/mggyslz](https://github.com/mggyslz)

---

## License

This project is released for academic and research use. See individual model and dataset cards on Hugging Face for specific licensing terms.