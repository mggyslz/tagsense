# An Analysis of Multilingual Transformers for Low-Resource Taglish Sentiment Analysis: Effects of Code-Switch Density

**Authors:** Beldad, Jerald P. | Burac, Alzyn A. | Cea, Gil IV Miguel Salvador I. | Guevara, Kimberly  
**Institution:** Partido State University — Bachelor of Science in Computer Science  
**Adviser:** Shane Catolico-Briones  
**Date:** April 2026

---

# Chapter I: Introduction

## Background of the Study

On Philippine social media, millions of Filipinos communicate their ideas, emotions, and responses using **Taglish** — a fluid blend of Tagalog and English that represents the lived bilingualism of Filipino digital discourse. Despite how common code-switching is in Philippine digital communication, automated sentiment analysis systems remain poorly equipped to handle it, particularly in the low-resource environments typical of Filipino NLP research. Traditional algorithms trained on monolingual or formally organized corpora often struggle to reliably assess sentiment because Taglish is so context-dependent, culturally ingrained, and structurally changeable.

In Natural Language Processing (NLP), code-switching is widely acknowledged as a major barrier. Due to the lack of annotated training data and the structural difficulty of intra-sentential language mixing, recent developments in multilingual modeling have shown inconsistent performance on code-switched text, particularly in low-resource settings (Sheth et al., 2025; Winata et al., 2021). According to Winata et al. (2021), sentiment analysis in this area has only recently attracted systematic scholarly interest, despite social media being the main source of code-switched data. The ongoing need for lightweight, domain-adapted transformer solutions is highlighted by the fact that, despite large-scale models like XLM-R showing strong performance on conventional benchmarks, their computing needs make them impractical for deployment in resource-constrained contexts (Aliyu et al., 2024).

Although low-resource sentiment analysis frequently uses transfer learning with transformers, applying pre-trained models to noisy multilingual data runs the risk of **catastrophic forgetting** — which reduces sensitivity to localized linguistic phenomena like humor, irony, and slang (Aliyu et al., 2024). Token-level language identification is required because local research shows that code-switching is common on Philippine social media (Boquiren et al., 2022; Cosme & De Leon, 2024). Emojis, backward slang (*baliktad na salita*), and netspeak expressions (like "marites," "sana all") are examples of cultural markers that significantly impact sentiment but are consistently underrepresented in current modeling approaches.

---

## Purpose and Description

This study develops and evaluates transformer-based sentiment analysis models fine-tuned on low-resource Taglish social media text. Three pretrained multilingual architectures — **mBERT**, **XLM-RoBERTa**, and **TLUnified-RoBERTa** — are compared across systematically varied levels of code-switch density, with the goal of identifying which architecture is most robust to increasing language mixing in Philippine social media contexts.

The study constructs a curated dataset of approximately **6,000–7,000 annotated social media posts** collected entirely in-house from **Facebook**, **Reddit**, and **YouTube**, without reliance on pre-existing online datasets. Three training conditions are evaluated:

1. Standard fine-tuning
2. QLoRA fine-tuning
3. QLoRA combined with Self-Paced Curriculum Learning (SPCL)

### Code-Switch Density Definition

Code-switch density is operationally redefined as:

> **Code-Switch Density = Number of English Tokens ÷ Total Number of Tokens in a Post**

Posts are stratified into five quantifiable density bands:

| Band | English Token Proportion |
|------|--------------------------|
| Band 1 | 0–20% |
| Band 2 | 21–40% |
| Band 3 | 41–60% |
| Band 4 | 61–80% |
| Band 5 | 81–100% |

Evaluation metrics include **accuracy**, **macro-F1 score**, and qualitative error analysis of misclassifications.

---

## Objectives of the Study

The general objective is to investigate how code-switch density may influence the performance of multilingual transformer models in sentiment classification of low-resource Taglish, Tagalog, and English social media text.

Specifically, this study aims to:

1. **Construct** a curated dataset of approximately 6,000–7,000 social media posts from Facebook, Reddit, and YouTube, manually labeled as positive (33%), negative (33%), and neutral (34%), annotated with token-level code-switch density and emoji usage, stratified into five density bands.

2. **Compare** mBERT, XLM-RoBERTa, and TLUnified-RoBERTa using accuracy, macro-F1, and confusion matrix analysis across three training conditions: conventional fine-tuning, QLoRA fine-tuning, and QLoRA + SPCL.

3. **Analyze** the impact of Density-Primary SPCL and emoji presence on macro-F1 performance across code-switch density bands (0–20% to 81–100%).

4. **Investigate** model failure modes — including sarcasm, slang, netspeak, reversed Filipino expressions (*baliktad na salita*), and emoji-heavy posts — through qualitative error analysis and per-band confusion matrices.

5. **Release** the curated density-stratified dataset and fine-tuned models as publicly available research resources and develop a demonstration application for practical Taglish sentiment analysis.

---

## Scope and Delimitation of the Study

This research focuses on **sentiment classification** of social media text in Taglish, Tagalog, and English into three classes: **positive**, **negative**, and **neutral**, based on approximately 6,000–7,000 manually annotated posts from Facebook, Reddit, and YouTube.

**Training employs a staged learning rate schedule:**

| Stage | Content | Learning Rate |
|-------|---------|---------------|
| Stage 1 | Pure monolingual posts | 3×10⁻⁵ |
| Stage 2 | Light code-switching (21–40%) | 2×10⁻⁵ |
| Stage 3 | Heavy code-switching (61–80%) | 1×10⁻⁵ |
| Stage 4 | Balanced Taglish (41–60%) | 8×10⁻⁶ |

**Limitations:**
- Restricted to transformer-based models (mBERT, XLM-RoBERTa, TLUnified-RoBERTa) and QLoRA fine-tuning due to computational constraints
- Experiments conducted on Google Colab's free-tier T4 GPU
- Maximum sequence length of 128 tokens
- Does **not** involve LLMs such as GPT-4 or multimodal input like images or audio (except emoji processing)
- Mobile app development, real-time system integration, and cross-domain analysis are excluded

---

## Significance of the Study

The findings of this study may benefit the following groups:

- **Researchers** — Access to a well-structured, density-stratified Taglish dataset benchmarked for low-resource fine-tuning. The five-band density framework allows future researchers to replicate across quantified levels of code-switching. Released datasets and trained models provide a reproducible baseline for Filipino-focused NLP studies.

- **Industry practitioners and data analysts** — A density-aware evaluation framework that pinpoints code-switching levels where sentiment models perform poorly, enabling targeted enhancements to analytics pipelines used for social listening, brand monitoring, and customer feedback analysis in Philippine markets.

- **Government agencies, platform moderators, and civic technology developers** — Density-stratified models and evaluation frameworks applicable to public discourse monitoring, hate speech detection, and Filipino-language tools (e.g., community feedback systems, public health dashboards, disaster response trackers) without requiring large annotated corpora from scratch.

- **Students and emerging researchers in IT and Computer Science** — A practical case study of the full NLP pipeline — from dataset curation and annotation to transformer fine-tuning and error analysis — applied to a locally relevant, code-switched language. Lowers the entry barrier for student-led Filipino NLP projects.

**Overall contributions at three levels:**
- **Empirical** — Explores how code-switch density quantifiably affects transformer sentiment performance across five density bands
- **Methodological** — Proposes a density-stratified evaluation approach and a purpose-built fine-tuning pipeline
- **Practical** — Releases a curated Taglish dataset and fine-tuned models to lower the entry barrier for resource-constrained NLP development in the Philippines

---

*This README covers Chapter I only. For the full research proposal including the Review of Related Literature (Chapter II), Technical Background (Chapter III), and Methodology (Chapter IV), refer to the complete document.*