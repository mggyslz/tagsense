# An Analysis of Multilingual Transformers for Low-Resource Taglish Sentiment Analysis: Effects of Code-Switch Density

**Authors:** Beldad, Jerald P. | Burac, Alzyn A. | Cea, Gil IV Miguel Salvador I. | Guevara, Kimberly  
**Institution:** Partido State University — Bachelor of Science in Computer Science  
**Adviser:** Shane Catolico-Briones  
**Date:** April 2026

---

# Chapter II: Review of Related Literature, Studies, and Systems

This chapter reviews the literature, studies, and systems relevant to the study. It examines the theoretical foundations of transformer-based architectures, the linguistic challenges of code-switching, and prior research on multilingual sentiment analysis in low-resource settings. It critically examines methodological limitations that existing research has left unresolved and identifies research gaps that justify the present study.

Existing NLP studies generally draw on 5,000 to 20,000 data samples (Cosme & De Leon, 2024; Maceda et al., 2023), employ fine-tuning approaches such as gradual unfreezing and curriculum learning over 3 to 10 training epochs (Aliyu et al., 2024; Gupta et al., 2021), and evaluate model performance using macro-F1 score to account for class imbalance (Khan et al., 2022; Nazir et al., 2025).

---

## Review of Related Literature

This section establishes the technical and conceptual foundation for developing multilingual sentiment analysis systems, focusing on Taglish — a code-switched combination of English and Tagalog. It reviews research related to code-switching, transformer architectures, and low-resource languages, and explains how these concepts apply to multilingual sentiment analysis using distributional semantics, transformer attention mechanisms, and pre-trained models such as mBERT, XLM-RoBERTa, and TLUnified-RoBERTa.

---

### A. Code-Switching and Sentiment Analysis in Natural Language Processing

**Code-switching (CS)** is broadly defined in sociolinguistics as the use of two or more languages interchangeably within a single conversational setting. Traditional linguistic definitions distinguish three types:

| Type | Description |
|------|-------------|
| Inter-sentential switching | Switching between sentences |
| Intra-sentential switching | Switching within a single sentence |
| Tag-switching | Insertion of short phrases from one language into another |

These distinctions are difficult to implement reliably in a computational pipeline, particularly in social media text — where the boundary between borrowed words, transliterations, and deliberate code-switching cannot always be clearly drawn. Code-mixing of multiple linguistic structures affects tokenization, embedding representations, and context modeling — consequently lowering the performance of LLMs, especially in zero-shot and resource-scarce scenarios (Winata et al., 2021; Aliyu et al., 2024).

**Taglish** — a mix of Tagalog and English — has become prevalent throughout social media platforms including Facebook, Reddit, and YouTube in the Philippines (Herrera et al., 2022). Researchers found that approximately:

- **55%** of Taglish-related post content consists of Tagalog
- **19%** from English
- **26%** from informal borrowed words, slang, and typographical alternatives

This complexity proves challenging for most established sentiment analysis models, which have primarily been constructed to analyze monolingual or formally structured text. For sentiment analysis of Taglish content to be accurate, a model must learn how sentence structure, lexical choices, and contextual usage collectively construct meaning in code-switched text — including sensitivity to sarcasm, humor, and culturally specific expressions.

---

### B. Transformer Architectures in Multilingual Sentiment Analysis

The **self-attention mechanism** has transformed how sentiment analysis is implemented using transformer-based models, allowing them to learn contextual relationships between consecutive tokens bidirectionally — in contrast to recurrent neural networks (RNNs), which stream information sequentially one word at a time.

This is particularly relevant in code-mixed utterances, where meaning depends on cross-lingual contextual relationships between tokens that are distant in surface position but semantically linked across languages.

Key findings from the literature:

- **XLM-RoBERTa** has consistently ranked among the top pre-trained transformers for multilingual prediction tasks (Manias et al., 2023); however, performance declines as code-switching increases, underscoring the need for domain adaptation.
- **XLM-T**, a multilingual transformer pre-trained on Twitter data, showed measurable improvements on social media corpora (Barbieri et al., 2022), demonstrating that pre-training on social media-specific datasets improves capacity to handle noisy and non-standard syntax.
- Winata et al. (2021) found that even state-of-the-art multilingual transformers produced inconsistent results on code-switched data, reinforcing the importance of domain-specific adaptation for mixed-language text like Taglish.

These findings motivate the comparative analysis of **mBERT**, **XLM-RoBERTa**, and **TLUnified-RoBERTa** conducted in the present study.

---

### C. Low-Resource Language Processing and Transfer Learning

**Low-resource NLP** refers to settings where models cannot be trained effectively due to data and computational constraints. **Transfer learning** addresses this by reusing models trained on multilingual corpora, capturing cross-lingual patterns while reducing *catastrophic forgetting* — the degradation of a model's prior knowledge when exposed to new task-specific data (Aliyu et al., 2024). This effect is particularly pronounced in Taglish, where models pre-trained on corpora far removed from Philippine social media exhibit a domain gap that fine-tuning alone cannot fully bridge.

Two key training approaches have been proposed:

- **Gupta et al. (2021)** demonstrated that semi-supervised learning can achieve results comparable to fully supervised training while requiring significantly less annotated data.
- **Aliyu et al. (2024)** proposed an incremental fine-tuning schedule that preserves pre-trained multilingual representations through carefully controlled exposure to task-specific data.

**Parameter-efficient methods** used in this study:

- **QLoRA (Quantized Low-Rank Adaptation)** — enables adaptation of large transformer models using a fraction of the trainable parameters, reducing GPU memory requirements without substantial performance loss.
- **Self-Paced Curriculum Learning (SPCL)** — dynamically orders training samples based on model confidence, allowing the model to progress from easier to harder examples within each data partition.

In this study, these two strategies are combined into a **four-stage cumulative training design**, where QLoRA constrains memory use across all stages while SPCL orders samples within each density band from easiest to hardest — beginning with purely monolingual text and progressing to balanced Taglish.

---

### D. Cultural Linguistic Features in Filipino Social Media

Filipino social media presents a dense concentration of culturally specific linguistic features that standard sentiment models are not equipped to process:

- ***Baliktad na salita*** — the practice of reversing syllables or words (e.g., *'erpat'* for *'father'*, *'lodi'* for *'idol'*), which encode sentiment indirectly (Boquiren et al., 2022)
- **Sarcasm and irony** — often expressed through combinations of affirmative words and negative emojis, making polarity detection unreliable without context-aware modeling
- **Informal borrowed words and slang** — approximately **26%** of Taglish social media content consists of such expressions (Herrera et al., 2022), large enough to systematically distort sentiment predictions when left unaddressed

Aquino et al. (2025) demonstrated that incorporating multimodal signals such as emojis and hashtags into graph-aware transformer models significantly improves classification on politically charged Filipino content, achieving a macro-F1 of **68.72%** on imbalanced election data. These findings establish that culturally informed feature engineering is **not supplementary but essential** for reliable Taglish sentiment analysis.

---

## Review of Related Studies

Research on multilingual sentiment analysis in Taglish draws from both local and international studies, each contributing insights into model performance and methodological gaps in low-resource settings.

---

### A. Local Studies

| Study | Model | Result | Limitation |
|-------|-------|--------|------------|
| Cosme & De Leon (2024) | XLM-RoBERTa | 84% accuracy on product/service reviews | Neutral sentiment detection remained challenging |
| Maceda et al. (2023) | mBERT | 80.21% accuracy on code-switched feedback | Did not examine varying levels of code-switch density |
| Cruz & Cheng (2022) | RoBERTa-Tagalog (TLUnified) | Significant accuracy improvements over standard multilingual models | Struggled with irony, sarcasm, and nuanced sentiment |
| Macrohon et al. (2022) | Various transformer models | Applied to 2022 Philippine presidential election tweets | Poor handling of politically charged sarcastic content |

**Key takeaways from local studies:**

- Transformer-based models outperform traditional methods in code-switched contexts.
- Domain-specific pretraining improves sensitivity to localized vocabulary.
- A persistent limitation is reliance on datasets not purpose-built to capture the full range of code-switching in Philippine social media.
- Inconsistent annotation standards and underrepresentation of the neutral class constrain comparability across studies.

The present study addresses these gaps by using **social media posts from Facebook, Reddit, and YouTube** — where linguistic behavior is more varied and code-switch density fluctuates considerably across posts — and by evaluating models across systematically varied density levels.

---

### B. Foreign Studies

| Study | Model | Context | Key Finding | Limitation |
|-------|-------|---------|-------------|------------|
| Khan et al. (2022) | mBERT | Urdu sentiment (low-resource) | F1 score of 81.49% | Lower performance on informal, contextually ambiguous posts |
| Aryal et al. (2022) | XLM-RoBERTa | Spanish-English code-switching | Beat RNN baselines in accuracy and F1 | Spanish-English has more grammatical/lexical overlap than Tagalog-English |
| Nazir et al. (2025) | CMSA-mBERT | Roman Urdu, Punjabi, English code-mixed | Accuracy gains of up to 22.55% over LSTM/CNN/SVM | Treats code-switching as a static feature, not a dynamic variable |

**Key takeaways from foreign studies:**

- Transformer architectures are consistently effective for code-switched sentiment analysis in low-resource settings.
- Performance limitations in informal grammar, cultural nuance, and irony are universal challenges.
- The present study builds on Nazir et al. (2025) by treating **code-switch density as a dynamic, continuous variable** rather than a static feature, enabling finer-grained analysis of how language mixing affects model performance.

---

## Review of Related Systems

The tools and platforms through which data are collected, processed, and analyzed impose constraints on what is measurable. This section examines three categories of existing systems.

---

### Commercial Cloud-Based NLP Systems

| System | Strengths | Limitations for Taglish |
|--------|-----------|--------------------------|
| **Google Cloud Natural Language API** | Scalable, fast, real-time analytics | Trained predominantly on formal English corpora; tokenizer does not accommodate mixed-language syntax |
| **IBM Watson Natural Language Understanding** | Sentiment detection, emotion classification, keyword extraction | English-centric tokenization fragments code-switched phrases; fails on *baliktad na salita* and netspeak |

Both systems are **proprietary and closed to fine-tuning** on localized datasets, making domain adaptation to Taglish structurally infeasible without external preprocessing pipelines.

> **Example:** A Taglish post such as *"Grabe ang ganda niya pero charot lang"* combines informal reversal, sarcasm, and code-switching — a combination that default English tokenizers fragment incorrectly, misassigning sentiment polarity as a result.

---

### Open-Source Transformer Frameworks

**Hugging Face Transformers** (Wolf et al., 2020) offers pre-trained models such as BERT, mBERT, XLM-RoBERTa, and TLUnified-RoBERTa that can be fine-tuned using locally available datasets. These models have proven effective in capturing bidirectional contextual relationships across language boundaries.

**Principal disadvantages:**
- High computational cost — fine-tuning requires GPU resources not consistently available in Philippine academic settings
- Performance is tightly coupled to dataset quality — without preprocessing steps such as emoji normalization, models produce unreliable sentiment predictions on raw Taglish input

---

### Social Media Sentiment Dashboards

Social media sentiment dashboards automate the analysis of public opinion across platforms. While useful for real-time visualization and trend tracking, they rely on **English-centric tokenization pipelines and lexicon-based sentiment models** that systematically fail on code-switched content.

Taglish posts containing mixed grammar, sarcasm, hashtags, and emoji combinations are routinely misclassified because none of these linguistic signals are handled by default English sentiment engines.

---

### Comparative Summary of Systems

*Table 1. Comparison of Transformer Models and Sentiment Analysis Systems Used in Multilingual Text Processing*

| System Name | Main Features | Strengths | Limitations | Code-Switch Handling |
|-------------|---------------|-----------|-------------|----------------------|
| **mBERT** | Multilingual transformer pre-trained on 104 languages | Strong baseline for low-resource code-switched tasks; widely validated on Philippine data | Performance drops with high code-switch density; struggles with sarcasm and irony | Moderate – supports Tagalog and English but not optimized for Taglish |
| **TLUnified-RoBERTa** | RoBERTa pre-trained on the TLUnified Filipino corpus | Strong sensitivity to localized Tagalog vocabulary, idioms, and syntactic structures | Limited exposure to English-dominant or high-density code-switched content | Moderate – better Tagalog coverage but not evaluated on density-stratified Taglish |
| **XLM-RoBERTa** | Multilingual transformer supporting 100+ languages | Strong cross-lingual transfer capability | Computationally expensive | Moderate – stronger cross-lingual transfer than mBERT but not adapted to Taglish-specific patterns |
| **Google Cloud Natural Language API** | Cloud-based sentiment analysis and entity recognition | Scalable and fast processing | Limited flexibility and customization | Low – optimized mainly for English |
| **Social Media Sentiment Dashboards** | Real-time sentiment monitoring and visualization | Automated trend tracking | Relies on default English models | Low – misclassifies mixed-language posts |

---

## Synthesis of the State of the Art

Based on the reviewed literature, studies, and existing systems, transformer-based models — particularly **mBERT**, **XLM-RoBERTa**, and **TLUnified-RoBERTa** — represent the state of the art in multilingual sentiment analysis. Across both local and international studies, these models consistently outperform traditional approaches such as Naive Bayes and LSTMs, achieving higher accuracy and F1 score in code-switched and low-resource settings (Cosme & De Leon, 2024; Nazir et al., 2025; Winata et al., 2021).

**Emerging trends in the literature:**
- Fine-tuning pre-trained multilingual transformer models on smaller domain-specific datasets to compensate for limited low-resource data
- Use of social media corpora rather than formal datasets, improving performance on informal and noisy text

**Persistent limitations across all reviewed areas:**
- Dataset scope and quality challenges — most studies rely on small or generic corpora not purpose-built for Filipino online conversations
- Limited studies examine classification reliability with respect to **code-switching density**, leaving this important factor unexplored
- Commercial systems prioritize scalability over linguistic flexibility
- Cultural factors such as sarcasm, slang, *baliktad na salita*, emojis, and hashtags remain largely unaddressed (Boquiren et al., 2022; Herrera et al., 2022)

These findings support the development of a **density-aware evaluation framework** for analyzing transformer performance in Taglish sentiment classification.

---

## Gap Bridged by the Study

The reviewed literature reveals **three converging gaps** that the current study seeks to address:

1. **No systematic density-stratified evaluation** — Existing research does not systematically evaluate transformer-based sentiment models across quantified levels of code-switch density. Prior studies either treat code-switching as a binary/static feature or fail to stratify evaluation datasets by measurable density levels (Maceda et al., 2023; Khan et al., 2022; Nazir et al., 2025).

2. **Limited comparative evaluation in Philippine social media contexts** — Only a limited number of studies directly compare mBERT, XLM-RoBERTa, and TLUnified-RoBERTa using Philippine social media sources such as Facebook, Reddit, and YouTube, leaving their relative effectiveness in local, code-switched contexts insufficiently established.

3. **Underrepresentation of cultural linguistic phenomena** — Features such as sarcasm, netspeak, and *baliktad na salita* remain largely underrepresented in existing evaluation frameworks, resulting in error analyses that do not fully capture real-world failure modes of sentiment classification in Taglish.

**To address these gaps, the present study:**
- Introduces a structured, density-aware evaluation framework where code-switch density is operationalized as the proportion of English tokens relative to total tokens per post — enabling reproducible computation and consistent stratification into defined density bands
- Conducts a direct comparative evaluation of mBERT, XLM-RoBERTa, and TLUnified-RoBERTa on Philippine-sourced social media data
- Utilizes a controlled, purpose-built dataset rather than repurposed corpora, ensuring that code-switch density distribution, sentiment labels, and cultural linguistic features more accurately reflect authentic Philippine social media discourse
- Incorporates **qualitative failure analysis** focused on culturally salient phenomena such as sarcasm, netspeak, and *baliktad na salita*

---

## Definition of Terms

| Term | Definition |
|------|------------|
| **Code-Switch Density** | The proportion of English tokens relative to the total number of tokens in a social media post. Formula: *Number of English Tokens ÷ Total Number of Tokens in a Post*. Posts are stratified into five bands (0–20%, 21–40%, 41–60%, 61–80%, 81–100%). |
| **Code-switching (CS)** | The alternation between Tagalog and English within a single social media post, characterized by inter-sentential, intra-sentential, or tag-switching patterns. Distinct from cultural features like slang, netspeak, and *baliktad na salita*. |
| **Curriculum Learning** | A training strategy ordering samples from easiest to hardest based on code-switch density and model confidence scores, implemented here as Density-Primary Self-Paced Curriculum Learning (SPCL). |
| **Dataset Annotation** | The process of labeling Taglish social media posts for sentiment (positive, negative, neutral) and other features such as emoji use and token-level code-switching. |
| **Fine-Tuning** | Adapting pre-trained multilingual transformer models (mBERT, XLM-RoBERTa, TLUnified-RoBERTa) on annotated Taglish datasets for improved performance on sentiment analysis. |
| **Low-Resource Language** | A language or variety such as Taglish with limited annotated resources and computational tools, making model training and testing challenging. |
| **Macro-F1 Score** | An evaluation metric calculating the overall F1 score across all three sentiment classes (positive, negative, neutral), giving equal weight to each class regardless of distribution. |
| **Multilingual Transformer Models** | Pre-trained language models (mBERT, XLM-RoBERTa, TLUnified-RoBERTa) that process and understand multiple languages simultaneously and establish contextual relationships in Taglish social media posts. |
| **QLoRA (Quantized Low-Rank Adaptation)** | A parameter-efficient fine-tuning method enabling adaptation of large transformer models using a fraction of trainable parameters through quantization and low-rank decomposition, reducing GPU memory requirements without substantial performance loss. |
| **Sentiment Analysis (SA)** | A subfield of NLP concerned with computationally identifying and classifying the emotional polarity of text — typically as positive, negative, or neutral — applied here to Taglish social media posts. |
| **Taglish** | Code-switching between Tagalog and English, characterized by mixed syntax and informal use of loanwords and cultural linguistic symbols, commonly used in Philippine social media. |
| **Tokenization** | The process of segmenting Taglish social media posts into individual words, subwords, or characters as input units for transformer models. |
| **Transfer Learning** | Adapting pre-trained multilingual transformer models to Taglish sentiment analysis tasks, allowing models to apply cross-lingual knowledge to a low-resource code-switched setting. |

---

*This README covers Chapter II only. For Chapter I (Introduction), refer to `README.md`. For the full research proposal including Technical Background (Chapter III) and Methodology (Chapter IV), refer to the complete document.*