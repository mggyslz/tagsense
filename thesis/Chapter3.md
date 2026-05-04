# An Analysis of Multilingual Transformers for Low-Resource Taglish Sentiment Analysis: Effects of Code-Switch Density

**Authors:** Beldad, Jerald P. | Burac, Alzyn A. | Cea, Gil IV Miguel Salvador I. | Guevara, Kimberly  
**Institution:** Partido State University — Bachelor of Science in Computer Science  
**Adviser:** Shane Catolico-Briones  
**Date:** April 2026

---

# Chapter III: Technical Background

This chapter deals with the software and hardware tools to be used by the research team during the development of this study. Each tool will serve a particular function during one or more phases of development. Overall, the tools form the foundation of **TAGCOL** — a custom data collection and processing pipeline developed specifically for this study. They also help the team establish a more organized workflow, ensuring consistency during data processing and improving the overall quality of the data used for training and evaluation.

---

## Software Requirements

*Table 2. Software Requirements*

| Software | Description / Use |
|----------|-------------------|
| Windows 11 Home Single Language 64-bit (Build 26200) | Operating System |
| Python 3.13 | Programming Language |
| Visual Studio Code (VS Code) – latest stable version | IDE / Code Editor |
| Google Colaboratory (Google Colab) | Cloud-Based Notebook Environment |
| mBERT, XLM-RoBERTa, and TLUnified-RoBERTa | AI Models |
| TAGCOL | Custom Python-Developed Application |
| TAGCOL – Facebook Scraper (custom-built) | Data Collection Tool |
| TAGCOL – Reddit Scraper (custom-built) | Data Collection Tool |
| TAGCOL – YouTube Scraper (custom-built) | Data Collection Tool |
| Dataset Annotator (TAGCOL) – custom-built | Dataset Annotation Tool |
| Sentiment Annotator (TAGCOL) – custom-built | Sentiment Annotation Tool |
| Data Cleaner – custom-built | Data Preprocessing Tool |
| Name Anonymizer – custom-built | Data Anonymization Tool |
| Claude / ChatGPT | AI Model / LLM (assistive) |

---

### Windows 11 Home Single Language 64-bit

The local development work carried out by the team will depend on the **Windows 11 Home Single Language 64-bit** operating system (Build 26200). This version has zero compatibility issues with Python 3.13, Visual Studio Code, and all third-party libraries necessary for this project, making it workable for coding, testing, and executing all scripts and custom tools developed in this research.

---

### Python 3.13

**Python 3.13** will be selected for this project due to its extensive support for natural language processing and data science applications. All custom tools developed under TAGCOL — including the scrapers, Dataset Annotator, and Sentiment Annotator — will be implemented using Python. Its readable syntax and extensive library ecosystem facilitate efficient collaboration and streamlined development among the research team.

---

### Visual Studio Code (VS Code)

**Visual Studio Code** will be the primary code editor used throughout the entire process. It will be used to create and test all Python scripts. The team will also make use of the built-in Git feature for version control of all scripts, making VS Code the primary environment for all development tasks.

---

### Google Colaboratory (Google Colab)

Certain tasks cannot be executed locally due to their memory and processing demands. For such tasks, the team will use **Google Colab**. For instance, calculating code-switch density across thousands of posts requires more memory than local machines can provide. The team will take advantage of the free GPU and TPU access provided by Google Colab and will use it for executing tasks and running scripts collaboratively.

---

### mBERT, XLM-RoBERTa, and TLUnified-RoBERTa

These three models will be chosen based on their strong multilingual capabilities, which make them suitable for handling mixed English and Filipino language data. Their roles include:

- Performing sentiment analysis on Taglish text
- Understanding linguistic patterns within the dataset
- Acting as the core intelligent layer during annotation, experimentation, and evaluation

---

### TAGCOL (Custom Pipeline)

**TAGCOL** is a custom-developed data collection and processing pipeline designed specifically for this study. It consists of multiple modules — **Collector**, **Reviewer**, and **Reader** — each corresponding to a specific phase of the data lifecycle. TAGCOL ensures consistency, automation, and control over collection, annotation, and preprocessing of Taglish social media data, eliminating the need for external tools and minimizing manual intervention.

#### TAGCOL – Facebook Scraper

The **Facebook Scraper** is a Python application utilizing the **Selenium** library for automating web browsers. Since the official Facebook API does not offer access to community posts and comments, the scraper behaves like a real user — scrolling down the page and clicking to expand comments — to access information that is not otherwise displayed.

#### TAGCOL – Reddit Scraper

The **Reddit Scraper** takes a different approach. Rather than using a browser automation tool or third-party API wrapper, it makes use of the **JSON API offered by Reddit**. For public content, no OAuth authentication is required. The scraper will collect content from various Philippine-related subreddits such as:

- `r/philippines`
- `r/phcareers`
- `r/phinvest`

#### TAGCOL – YouTube Scraper

The **YouTube Scraper** will be built using the `comments` library together with the **Flask** library for the entire workflow. YouTube comments from Filipino users — especially from reaction and opinion videos — serve as a good source of informal and emotionally charged language, making them a valuable addition to the corpus alongside Facebook and Reddit data.

#### Dataset Annotator

The **Dataset Annotator** serves two functions:

1. Determines the **code-switch density per post** using a seven-layer token-level language identification pipeline (explained further in the methodology).
2. Creates two different output files from the annotated data:
   - A **full JSON file** with all analysis fields, including `code_switch_density` (float: 0.0–1.0) and `density_range` (string)
   - A **fine-tune JSONL file** where each line is a post with:
     - `text` — the post content
     - `sentiment` — integer classification (0 = negative, 1 = neutral, 2 = positive)
     - `code_switch_density` — float value
     - `code_switch_density_band` — string (e.g., `'band1_0_20'`)

#### Sentiment Annotator

The **Sentiment Annotator** performs initial annotation using a combination of **lexicon-based scoring** and **rule-based heuristics** to assign posts a positive, negative, or neutral tag. The system will not be solely relied upon — particularly for Taglish texts containing sarcasm, irony, and culturally nuanced expressions. Low-confidence samples will be flagged for manual review, ensuring higher annotation accuracy and reducing noise in the training dataset.

#### Data Cleaner

Social media content is usually accompanied by unnecessary noise, including URLs, @mentions, hashtags, repetitive content, and posts below a minimum length threshold. The **Data Cleaner** script removes this unnecessary content to prevent noise from entering the training process. It will be run for every batch of scraped content to ensure that only useful content passes through.

#### Name Anonymizer

Since scrapers will gather publicly available text — some of which may contain names of actual people — the **Name Anonymizer** script utilizes **spaCy's named entity recognition** to scan each post and replace recognized person names with the placeholder `[PERSON]`. This ensures that the data does not include personally identifiable information, in accordance with ethical expectations for research using public data.

#### Claude / ChatGPT (Assistive AI Tools)

**Claude** and **ChatGPT** will be utilized as assistive tools to support annotation refinement, ambiguity resolution, and quality checking. These tools will **not** be used as primary annotators but rather as supplementary aids within a human-in-the-loop validation process to improve annotation consistency. Claude is chosen not only for its advanced reasoning abilities but also for its effectiveness in supporting human-in-the-loop validation within the TAGCOL pipeline.

---

## Hardware Requirements

*Table 3. Hardware Requirements*

| Hardware | Specification |
|----------|--------------|
| **HP Victus Gaming Laptop 15-fb0xxx (MIGGY)** | Processor: AMD Ryzen 5 5600H with Radeon Graphics (12 CPUs, ~3.3GHz) |
| | RAM: 8,192MB (8GB) |
| | GPU: NVIDIA GeForce RTX 3050 Laptop GPU / AMD Radeon Graphics (integrated) |
| | OS: Windows 11 Home Single Language 64-bit (Build 26200) |
| **LAPTOP-PDGG1B8B** | Processor: 11th Gen Intel Core i5-1155G7 @ 2.50GHz |
| | RAM: 8.00GB (7.78GB usable) |
| | OS: 64-bit, x64-based processor |
| **HP Victus Gaming Laptop 15 (Second Unit)** | Processor: AMD Ryzen 5 8645HS (4.3GHz base, up to 5.0GHz max, 16MB L3 cache, 6 cores, 12 threads) |
| | RAM: 16GB DDR5 4800MHz (1 × 16GB, removable) |
| | Storage: 512GB PCIe Gen4 NVMe M.2 SSD |
| | GPU: NVIDIA GeForce RTX 3050 6GB GDDR6 (Dedicated) / AMD Radeon 760M (Integrated) |
| | OS: Windows 11 Home 64-bit, Single Language |
| **Realme C75x (pactset)** | Model: Realme C75x (RMX5020) |
| | Processor: MediaTek Helio G81 Ultra |
| | RAM: 8.00GB + 16.0GB (extended virtual RAM) |
| | Storage: 128GB internal (96.4GB used) / SD card: 7.90GB (5.28GB used) |
| | Battery: 5600 mAh (TYP) |
| | OS: realme UI 6.0 |

---

### Hardware Usage Overview

The project will be carried out on **three laptops and one mobile device** connected via a local network to share scripts and data.

**Laptops** handle:
- Local development tasks — scripting, testing, and running TAGCOL on individual batches
- Dataset Annotator and Sentiment Annotator tasks for smaller inputs
- Text normalization and token-level language identification (leveraging multi-core architectures of the Ryzen 5 5600H and i5-1155G7)

**Google Colab** handles:
- Computationally intensive tasks such as corpus-wide density calculations across 6,000–7,000 posts
- Eliminates memory constraints without requiring additional hardware
- Scales as the corpus grows in future work

**Mobile device** is used primarily for:
- Monitoring
- Lightweight data review

---

Overall, the combination of custom-built tools and established machine learning frameworks enables the development of a **controlled and reproducible pipeline** for analyzing multilingual sentiment in Taglish social media.

---

*This README covers Chapter III only. For Chapter I (Introduction), refer to `README.md`. For Chapter II (Review of Related Literature, Studies, and Systems), refer to `README_Chapter2.md`. For the full research proposal including Methodology (Chapter IV), refer to the complete document.*