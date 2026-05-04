# An Analysis of Multilingual Transformers for Low-Resource Taglish Sentiment Analysis: Effects of Code-Switch Density

**Authors:** Beldad, Jerald P. | Burac, Alzyn A. | Cea, Gil IV Miguel Salvador I. | Guevara, Kimberly  
**Institution:** Partido State University — Bachelor of Science in Computer Science  
**Adviser:** Shane Catolico-Briones  
**Date:** April 2026

---

# Chapter V: Results and Discussion

---

## 4.1 Dataset Overview

This study constructed a new, density-stratified Taglish sentiment corpus comprising **7,707 annotated social media posts** sourced from the following platforms:

| Source | N | Notes |
|--------|---|-------|
| Reddit | 4,047 | Negative-dominant; high CS density |
| YouTube | 2,198 | Positive-dominant; low CS density |
| Facebook | 977 | Negative-dominant; moderate density |
| AI-Augmented | 485 | Neutral augmentation; mid-density (41–80%) |
| **Total** | **7,707** | Near-balanced; all classes within ±0.6% |

### Sentiment Distribution

The dataset achieves near-perfect class balance — a deliberate departure from naturally occurring Taglish datasets, which tend to be positivity-skewed (Yap et al., 2021). This was enforced to prevent classifier bias and ensure macro-F1 results reflect per-class performance uniformly.

| Sentiment Class | Count | Proportion |
|----------------|-------|------------|
| Positive | 2,572 | 33.4% |
| Negative | 2,543 | 33.0% |
| Neutral | 2,592 | 33.6% |

### Platform-Level Sentiment Distributions

*Table 4.1. Distribution of Dataset Entries by Source, Sentiment Class, and Negativity Rate (N = 7,707)*

| Source | N | Positive | Negative | Neutral | Neg. Rate | Notes |
|--------|---|----------|----------|---------|-----------|-------|
| Reddit | 4,047 | 657 (16.2%) | 1,937 (47.9%) | 1,453 (35.9%) | 0.479 | Negative-dominant; high CS density |
| YouTube | 2,198 | 1,456 (66.2%) | 136 (6.2%) | 606 (27.6%) | 0.066 | Positive-dominant; low CS density |
| Facebook | 977 | 379 (38.8%) | 470 (48.1%) | 128 (13.1%) | 0.481 | Negative-dominant; moderate density |
| AI-Augmented | 485 | 80 (16.5%) | 0 (0.0%) | 405 (83.5%) | — | Neutral augmentation; mid-density (41–80%) |
| **Total** | **7,707** | **2,572 (33.4%)** | **2,543 (33.0%)** | **2,592 (33.6%)** | — | Near-balanced; all classes within ±0.6% |

> **Note.** CS = code-switching. Neg. Rate = proportion of negative posts within each source. Platform-level distributions were intentionally preserved rather than neutralized during sampling, as they reflect real-world deployment conditions.

### Platform Discourse Patterns

- **Reddit's negative skew (47.9%)** is consistent with its affordance for anonymous critical commentary — debate, complaint, and counter-opinion with relatively low social cost.
- **YouTube's positive dominance (66.2%)** reflects its fan-driven audience culture, where like-minded viewers congregate around creators and suppress critical speech through downvoting norms.
- **Facebook** occupies a moderate middle ground, hosting a wider spectrum of interpersonal and civic discourse.

### AI-Augmented Component Justification

Natural Taglish usage in the collected corpus concentrated heavily in Band 1 (predominantly Tagalog) and Band 5 (predominantly English), leaving **Bands 3 and 4** — the "genuine mixed" zone — underrepresented. Generative augmentation was employed exclusively to fill these structural gaps, producing **Neutral-only samples** to avoid introducing annotation noise on subjective sentiment.

### Code-Switch Density Distribution

| Band | English Token Proportion | Share of Dataset |
|------|--------------------------|-----------------|
| Band 1 | 0–20% | 24.1% (largest stratum) |
| Band 2 | 21–40% | ~17–20% |
| Band 3 | 41–60% | ~17–20% |
| Band 4 | 61–80% | ~17–20% |
| Band 5 | 81–100% | ~17–20% |

Reddit contributes disproportionately to Band 5 (31.2% of high-density samples), while YouTube is dominated by Band 1 posts (45.8%). This structural heterogeneity across platforms ensures per-band evaluation is not confounded by a single platform's idiosyncrasies.

---

## 4.2 Overall Macro-F1 Performance (Objective 2)

*Table 4.2. Overall Macro-F1 Scores by Model and Training Condition (Benchmark = 0.70)*

| Model | Standard (FP16) | QLoRA Baseline | Curriculum (QLoRA+SPCL) | Δ vs Standard | Δ vs QLoRA |
|-------|----------------|----------------|--------------------------|---------------|------------|
| **XLM-RoBERTa** | 0.7504 ✓ | 0.7278 ✓ | 0.6942 | −0.0562 | −0.0336 |
| **mBERT** | 0.7214 ✓ | 0.6599 | 0.6745 | −0.0469 | +0.0146 |
| **TLUnified-RoBERTa** | 0.7181 ✓ | 0.6822 | 0.6570 | −0.0611 | −0.0252 |

> **Note.** ✓ = meets or exceeds the 0.70 prior benchmark. Negative delta values indicate performance degradation relative to the respective baseline. The methodologically sound comparison for assessing curriculum learning is Δ vs QLoRA, as both conditions share identical parameter budgets.

---

### 4.2.1 Standard Fine-Tuning Results

All three models under standard FP16 fine-tuning **met or exceeded the 0.70 benchmark**:
- **XLM-RoBERTa**: 0.7504 (highest)
- **mBERT**: 0.7214
- **TLUnified-RoBERTa**: 0.7181

**Key implications:**
1. Validates the quality and adequacy of the constructed corpus — 7,707 samples, carefully stratified by density and class, is sufficient to support robust fine-tuning of large multilingual transformers.
2. Confirms the cross-lingual transferability of XLM-RoBERTa to code-switched Philippine social media text (consistent with Conneau et al., 2020).

The modest advantage of XLM-RoBERTa over TLUnified-RoBERTa (0.7504 vs. 0.7181) is notable given TLUnified's specialization in Filipino-dominant corpora. Taglish is not simply Filipino with English insertions — it is a hybrid register with its own pragmatic conventions and internet-specific vocabulary. XLM-RoBERTa's broader multilingual pretraining may better span the English-Tagalog continuum.

---

### 4.2.2 QLoRA Baseline Results

Under QLoRA fine-tuning without curriculum ordering, all three models experienced reduced macro-F1:

| Model | Standard | QLoRA | Drop |
|-------|----------|-------|------|
| XLM-RoBERTa | 0.7504 | 0.7278 | −0.023 |
| TLUnified | 0.7181 | 0.6822 | −0.036 |
| mBERT | 0.7214 | 0.6599 | −0.062 |

These reductions are consistent with the known **efficiency-accuracy trade-off** of parameter-efficient fine-tuning. QLoRA constrains the model to update only a low-rank decomposition of weight matrices (rank r = 16), reducing effective learning capacity.

**Practical significance:** XLM-RoBERTa under QLoRA still exceeds the prior benchmark at 0.7278. The 2.3-point performance cost relative to full fine-tuning should be weighed against the **4-bit quantization** it enables — reducing memory requirements by approximately **8×** compared to full-precision FP32 training. This makes deployment feasible for resource-constrained settings such as community health workers, local government units, and disaster response monitoring.

---

### 4.2.3 Curriculum Learning (QLoRA + SPCL) Results

Contrary to the initial hypothesis, the **Density-Primary SPCL curriculum consistently underperformed both baselines** across all three models.

| Model | QLoRA Baseline | Curriculum | Δ |
|-------|---------------|------------|---|
| XLM-RoBERTa | 0.7278 | 0.6942 | −0.0336 |
| TLUnified | 0.6822 | 0.6570 | −0.0252 |
| mBERT | 0.6599 | 0.6745 | **+0.0146** |

The sole partial exception is **mBERT**, which showed a marginal positive delta of +0.0146 over its QLoRA baseline — the only model where density-ordered curriculum learning provided any measurable benefit.

This negative result constitutes evidence against a specific instantiation of curriculum learning — density-ordered self-paced learning in a low-resource QLoRA regime. It does not invalidate curriculum learning as a general approach; rather, it constrains the assumptions underlying its application to Taglish NLP.

---

### 4.2.4 Visual Comparison

*(Figure 4.1 — Bar chart: Baseline vs. Curriculum Fine-Tuning — Overall Macro-F1 by Model and Condition. Dashed orange line = prior benchmark (0.70).)*

The consistent pattern across all three models: **standard fine-tuning leads → QLoRA baseline follows → curriculum training falls below both** in nearly all cases.

*(Figures 4.2a–4.4c — Confusion matrices for mBERT, TLUnified-RoBERTa, and XLM-RoBERTa across all three training conditions.)*

---

## 4.3 Per-Density-Band Macro-F1 Analysis (Objective 3)

*Table 4.3. Per-Density-Band Macro-F1 by Model and Training Condition (★ = consistently best band; ▼ = consistently worst band)*

| Model / Condition | Band 1 (0–20%) | Band 2 (21–40%) | Band 3 (41–60%) ★ | Band 4 (61–80%) | Band 5 (81–100%) ▼ |
|-------------------|----------------|-----------------|-------------------|-----------------|-------------------|
| XLM-RoBERTa (Standard) | 0.739 | 0.713 | **0.864** | 0.703 | 0.691 |
| XLM-RoBERTa (QLoRA Base) | 0.712 | 0.646 | **0.782** | 0.733 | 0.667 |
| XLM-RoBERTa (Curriculum) | 0.682 | 0.639 | **0.716** | 0.645 | 0.614 |
| mBERT (Standard) | 0.720 | 0.664 | **0.824** | 0.742 | 0.624 |
| mBERT (QLoRA Base) | 0.670 | 0.592 | **0.759** | 0.673 | 0.544 |
| mBERT (Curriculum) | 0.644 | 0.537 | **0.737** | 0.608 | 0.544 |
| TLUnified (Standard) | 0.764 | 0.654 | **0.818** | 0.650 | 0.681 |
| TLUnified (QLoRA Base) | 0.724 | 0.662 | **0.731** | 0.634 | 0.580 |
| TLUnified (Curriculum) | 0.718 | 0.619 | **0.688** | 0.654 | 0.532 |

> **Note.** Band 3 (41–60% density) consistently yields the highest macro-F1 across all model-condition pairs. Band 5 (81–100%) under curriculum conditions yields the steepest degradation.

---

### 4.3.1 The Band 3 Performance Peak: Structural Interpretation

**Band 3 (41–60% code-switch density) consistently yields the highest macro-F1** across all models and training conditions — peaking at **0.864** for XLM-RoBERTa under standard training.

Taglish utterances in Band 3 occupy a **"mixed equilibrium"** — both Tagalog and English tokens are proportionally present, providing the model with redundant contextual signals from both language families. When a sentiment-bearing expression occurs in one language, it is often contextualized or reinforced by sentiment-adjacent vocabulary in the other language within the same utterance. This within-utterance bilingual redundancy likely improves the model's ability to disambiguate borderline sentiment cases, particularly irony and mild positivity.

**Band 1 (0–20% English, predominantly Tagalog)** also performs well under standard training — especially for TLUnified-RoBERTa (0.764) and XLM-RoBERTa (0.739) — reflecting the strong Tagalog pretraining of both models.

**Band 2 (21–40%)** moderate performance may reflect syntactic ambiguity introduced by minimal but structurally active code-switching: a single English term inserted into a Tagalog sentence often occupies a discourse-salient position that can shift sentiment polarity in ways requiring fine-grained sensitivity to both languages.

---

### 4.3.2 The Band 5 Challenge: Why High-Density Taglish Is Hardest

**Band 5 (81–100% English tokens) consistently yields the lowest macro-F1** (range: 0.532–0.691 across all conditions).

High-density Taglish is **not simply English with minor Filipino influence**. It is a distinct register characterized by:

| Feature | Description |
|---------|-------------|
| Filipino syntactic patterns | Applied to English vocabulary (e.g., verb-initial constructions, particle insertion) |
| Filipino pragmatic conventions | Indirect disagreement, face-saving hedging |
| Filipino internet slang/netspeak | Does not appear in standard English pretraining corpora |
| Culturally specific sentiment anchors | Humor rooted in Philippine political or entertainment contexts, opaque without cultural knowledge |

This **pragmatic distribution mismatch** — where surface lexis is English but pragmatic and syntactic context is Filipino — causes systematic misclassification of nuanced sentiment, particularly *ironic positivity* and *softened negativity*.

---

### 4.3.3 Curriculum Learning at the Band Level

The curriculum condition's damage is **concentrated at Band 5**, where the largest divergence from standard baseline performance is observed. For XLM-RoBERTa, the curriculum condition falls 7.7 points below standard at Band 5 (0.614 vs. 0.691).

This suggests a **"catastrophic interference"** dynamic: weight updates during Stages 1–3 optimize for low-to-moderate density patterns, reducing the plasticity available to learn Band 5's qualitatively different vocabulary and syntax during Stage 4.

**Notable exception:** TLUnified-RoBERTa at Band 1 (0.718 under curriculum) nearly matches the QLoRA baseline and approaches the standard condition (0.764). This implies a "language-first" curriculum variant — organizing stages by dominant language (Tagalog → mixed → English) rather than by density percentage — might better leverage TLUnified's pretraining distribution.

---

## 4.4 Effect of Emoji Presence on Macro-F1 (Objective 3, continued)

*Table 4.4. Macro-F1 by Emoji Presence and Count Group (Δ = macro-F1 with emoji minus macro-F1 without emoji)*

| Model / Condition | No Emoji | Has Emoji | 1–3 Emoji | 4+ Emoji | Δ (Has−No) | Pattern |
|-------------------|----------|-----------|-----------|----------|------------|---------|
| XLM-R (Curriculum) | 0.643 | 0.586 | 0.596 | **0.697** | −0.057 | Emoji hurts; 4+ recovers |
| XLM-R (Standard) | 0.729 | 0.656 | 0.652 | **0.793** | −0.074 | Emoji hurts; 4+ recovers |
| XLM-R (QLoRA Base) | 0.693 | 0.600 | 0.582 | **0.872** | −0.093 | 4+ anomaly strongest |
| mBERT (Curriculum) | 0.592 | 0.502 | 0.469 | **0.793** | −0.090 | Emoji hurts most |
| mBERT (Standard) | 0.698 | 0.631 | 0.617 | **0.590** | −0.068 | Mixed; 4+ dips |
| mBERT (QLoRA Base) | 0.622 | 0.599 | 0.579 | **0.872** | −0.023 | 4+ anomaly |
| TLUnified (Curr.) | 0.622 | 0.623 | 0.642 | **0.579** | +0.001 | Stable; native advantage |
| TLUnified (Standard) | 0.694 | 0.689 | 0.677 | **0.872** | −0.005 | 4+ anomaly |
| TLUnified (QLoRA) | 0.647 | 0.615 | 0.633 | **0.579** | −0.032 | Mixed |

> **Note.** Red Δ values indicate emoji presence is associated with performance degradation. The 4+ emoji column highlights the anomalous performance boost at high emoji density. TLUnified (Curriculum) is the sole model-condition pair with a near-neutral emoji effect (Δ = +0.001).

*(Figures 4.5a–4.7c — Emoji effect on Macro-F1 for mBERT, TLUnified-RoBERTa, and XLM-RoBERTa across all three training conditions.)*

---

### 4.4.1 The General Emoji Penalty: Tokenization as a Root Cause

The predominant pattern is that **emoji presence is associated with reduced macro-F1** (Δ = −0.023 to −0.093 across most model-condition pairs). This is consistent with the structural limitations of subword tokenization in handling non-linguistic symbols. Emojis are typically represented as rare or OOV tokens that fail to decompose into semantically meaningful subword units.

**Three distinct pragmatic functions of emojis in Taglish social media:**

| Emoji Function | Example | Learnability |
|---------------|---------|-------------|
| Sentiment amplifiers | 😭 intensifying negative text | Relatively tractable with sufficient annotated data |
| Irony markers | 😊 following clearly negative text | Requires recognition of semantic incongruity |
| Pragmatic softeners | 😅 mitigating the force of a complaint | Requires metalinguistic competence not in pretraining |

---

### 4.4.2 The 4+ Emoji Anomaly: Homogeneity as Signal

A striking exception is the consistently high macro-F1 for posts with **four or more emojis**:
- XLM-RoBERTa QLoRA Baseline: **0.872**
- mBERT QLoRA Baseline: **0.872**
- TLUnified Standard: **0.872**

Posts with 4+ emojis constitute a **"maximally expressive"** subset where the writer's affective state is signaled so redundantly — through both text and multiple emojis — that classification becomes trivially easy. This "emoji consensus signal" overrides the model's uncertainty about the Taglish text itself.

**Practical implication:** Emoji count functions as a proxy for sentiment intensity and writer expressiveness. High-emoji posts could be deprioritized as annotation "easy cases" in active learning and data annotation pipelines.

**TLUnified-RoBERTa's near-neutral emoji effect** (Δ = +0.001 under curriculum training) suggests that native-language pretraining may confer pragmatic advantages for emoji processing in Filipino social media contexts — not just lexical and syntactic advantages.

---

## 4.5 Discussion: Why Curriculum Learning Underperformed

The consistent underperformance of QLoRA + SPCL is the study's **most theoretically consequential finding**. Three explanatory frameworks are advanced — no single explanation is sufficient; the negative result reflects a confluence of constraints that collectively undermine the conditions necessary for curriculum learning to be beneficial.

---

### 4.5.1 Dataset Scale Constraints and the Curriculum Learning Threshold

Curriculum learning is most effective at scale. Key references:
- Bengio et al. (2009) demonstrated benefits on datasets exceeding **100,000 examples**
- Xu et al. (2020) found marginal or negative effects below **10,000 samples**

The 7,707-sample corpus falls at the lower boundary of the range where curriculum effects are theoretically expected. Within a four-stage curriculum, each stage receives approximately **1,500–2,000 samples** — far below the threshold at which staged training is expected to confer meaningful representational improvements.

> **Recommendation:** Future implementations should consider minimum per-stage sample thresholds of at least **3,000–4,000 examples** before applying density-ordered curriculum learning.

---

### 4.5.2 Density as an Imperfect Difficulty Proxy

The per-band results validate the density-difficulty assumption at the **aggregate level** — Band 5 does yield the lowest macro-F1. However, difficulty at the **sample level** is poorly approximated by density alone:

- A monolingual Tagalog post in Band 1 may be grammatically complex and semantically ambiguous
- Many Band 5 posts consist of simple positive/negative English exclamations with Filipino orthographic conventions (e.g., *"grabe ang ganda niya omg 😭"*) — trivially easy to classify despite high English token density

**Proposed composite difficulty measure:**
1. Code-switch density
2. Model-assigned confidence on a held-out subset
3. Syntactic complexity (e.g., dependency tree depth)
4. Presence of known sentiment modifiers (negation, intensifiers, irony markers)

---

### 4.5.3 Stage-Transition Instability Under QLoRA Constraints

TLUnified-RoBERTa exhibited **non-monotonic convergence in Stage 4** (flagged as unstable: 0.650–0.657 oscillation). This is mechanistically consistent with QLoRA's sensitivity to distributional shifts: low-rank adapters (rank r = 16) optimized on the previous stage's distribution must rapidly adjust to the new distribution with fewer degrees of freedom — increasing the risk of over-correction and destabilization of previously learned representations.

**Recommended future stabilization mechanisms:**

| Mechanism | Description |
|-----------|-------------|
| Experience replay | Intersperse a fraction of samples from earlier stages during later stages to prevent forgetting |
| Elastic Weight Consolidation (EWC) | Penalize updates to parameters important for earlier stages (Kirkpatrick et al., 2017) |

---

### 4.5.4 Implications for the mBERT Exception

mBERT's marginal positive curriculum effect (Δ = +0.0146) is notable because mBERT is the **lowest-performing** of the three models under standard training. One interpretation: mBERT, having weaker baseline representations of Taglish, benefits more from the curriculum's gradual complexity exposure than models with stronger priors.

> **Hypothesis for future work:** Curriculum learning's utility in low-resource settings may be *inversely proportional to baseline model capability* — testable by evaluating curriculum effects across a range of model capacities and pretraining data sizes.

---

## 4.6 Qualitative Failure Analysis (Objective 4)

A qualitative failure analysis was conducted across all nine model-condition combinations. Misclassified samples from the **770-post test set** were extracted and tagged with seven linguistically motivated failure categories.

---

### 4.6.1 Overall Error Rates by Model and Training Method

| Model / Condition | Errors (N) | Error Rate |
|-------------------|-----------|------------|
| XLM-RoBERTa (Standard) | 192 | **24.9%** (lowest) |
| XLM-RoBERTa (QLoRA) | 214 | 27.8% |
| XLM-RoBERTa (Curriculum) | 251 | 32.6% |
| TLUnified (QLoRA) | 212 | 27.5% |
| TLUnified (Curriculum) | 265 | 34.4% |
| mBERT (QLoRA) | 262 | 34.0% |
| mBERT (Curriculum) | 286 | **37.1%** (highest) |

Error rate rankings are broadly consistent with the macro-F1 rankings in Table 4.2, confirming failure patterns are not artifacts of class imbalance.

---

### 4.6.2 Failure Category Distribution

**Failure categories by frequency across all model-condition pairs:**

| Failure Category | Rate (% of test set) | Description |
|-----------------|---------------------|-------------|
| **Other** | 13.0%–18.4% | Misclassifications not attributable to tagged phenomena (label noise, deep pragmatic inference, culturally specific anchors) |
| **Context-dependent** | 5.1%–8.7% | Polarity determined by contextual knowledge unavailable to the model |
| **Mixed sentiment** | 5.7%–8.8% | Positive and negative sentiments co-occur; overall polarity is ambiguous |
| **Sarcasm / Irony** | 1.3%–2.7% | Expressed sentiment contradicts literal meaning |
| **Netspeak / Slang** | 2.3%–4.8% | Internet expressions and informal text speak |
| ***Baliktad na salita*** | 0.1%–0.4% | Reversed Filipino slang |
| **Emoji-heavy** | 0.3%–0.7% | Posts where emojis are primary sentiment carriers |

The persistent **"other" residual** (13–18%) indicates a substantial proportion of Taglish misclassifications arise from phenomena not captured by the present taxonomy, including label noise, deep pragmatic inference, and culturally specific sentiment anchors.

---

### 4.6.3 Curriculum Learning Amplifies Known Failure Modes

The curriculum training **increases failure rates in nearly every category** relative to the QLoRA baseline — it does not introduce new failure modes but **exacerbates the severity of pre-existing ones**:

| Model | Failure Category | QLoRA Baseline | Curriculum |
|-------|-----------------|----------------|------------|
| mBERT | Context-dependent | 5.6% | 8.3% |
| mBERT | Mixed sentiment | 7.4% | 8.3% |
| TLUnified | Context-dependent | 5.1% | 8.3% |

This amplification is consistent with the stage-transition instability hypothesis: as curriculum stages shift the training distribution, previously stabilized representations for pragmatically complex inputs are partially destabilized.

**XLM-RoBERTa** is the most failure-resistant model across all categories and conditions, achieving the lowest "other" residual at 13.0% — attributable to its broader multilingual pretraining scope (100 languages, including Tagalog and English) and higher parameter count.

> **Recommendation:** XLM-RoBERTa is the recommended base model for Taglish sentiment classification in resource-constrained deployments, balancing competitive overall performance with the lowest susceptibility to known failure modes.

*(Figure 4.8 — Failure Analysis across all models and training methods. Left: overall error rate as % of test set (N = 770). Center: stacked failure category distribution. Right: per-category heatmap.)*

---

## 4.7 Summary of Findings and Contributions

| # | Finding | Contribution |
|---|---------|-------------|
| **1** | All three models under standard fine-tuning met or exceeded the 0.70 benchmark (XLM-RoBERTa: 0.7504, mBERT: 0.7214, TLUnified: 0.7181) | Validates dataset quality; establishes a reproducible performance reference for future work on low-resource Philippine NLP |
| **2** | QLoRA reduced performance by 2–6 points, but XLM-RoBERTa under QLoRA (0.7278) still exceeds the benchmark | Confirms QLoRA's viability for resource-constrained deployment of Taglish sentiment classifiers |
| **3** | Density-Primary SPCL consistently underperformed both baselines (sole exception: mBERT +0.0146 over QLoRA) | Constrains assumptions underlying curriculum learning in low-resource code-switched NLP; density alone is an insufficient difficulty proxy |
| **4** | Band 3 (41–60%) consistently yields the highest per-band macro-F1, peaking at 0.864; Band 5 (81–100%) consistently yields the lowest | Identifies "mixed equilibrium" as a meaningful evaluation dimension; documents the pragmatic distribution mismatch in high-English-token Taglish |
| **5** | Emoji presence generally reduced macro-F1 (Δ ≈ −0.02 to −0.09); 4+ emojis showed anomalously high performance; TLUnified under curriculum showed near-neutral effect (Δ = +0.001) | Documents tokenization limitations for emojis; identifies "emoji consensus signal" effect; suggests native-language pretraining may confer pragmatic advantages |

---

*This README covers Chapter V only. For earlier chapters, refer to:*
- *Chapter I (Introduction) → `README.md`*
- *Chapter II (Review of Related Literature, Studies, and Systems) → `README_Chapter2.md`*
- *Chapter III (Technical Background) → `README_Chapter3.md`*
- *Chapter IV (Methodology) → `README_Chapter4.md`*

*End of Chapter V · Beldad, Burac, Cea, Guevara · Partido State University, BSCS 2026*