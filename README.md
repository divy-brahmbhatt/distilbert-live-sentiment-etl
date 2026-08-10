# Real-Time Headline Sentiment Analysis Pipeline

An end-to-end Python data pipeline that fetches live news headlines via REST API, performs batched GPU sentiment inference using a fine-tuned DistilBERT transformer, and visualizes sentiment distributions.

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=flat-square&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat-square&logo=pytorch)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-FFD21E?style=flat-square&logo=huggingface)
![Plotly](https://img.shields.io/badge/Plotly-Express-3F4F75?style=flat-square&logo=plotly)

---

## Overview

This project provides a lightweight, real-time NLP pipeline for monitoring topic sentiment across tech news feeds. It ingests live trending stories from the Hacker News REST API, pre-processes unstructured text payloads into PyTorch tensors, runs high-throughput batched inference through a Hugging Face `distilbert-base-uncased-finetuned-sst-2-english` model, and renders interactive distribution metrics.

### Live Sentiment Output

Below is an execution output showing real-time sentiment extraction across live tech news headlines:

![Real-Time Sentiment Distribution](assets/sentiment_distribution.png)

---

## Key Features

* **Automated Hardware Ingestion**: Detects CUDA availability dynamically to route tensor calculations directly to GPU memory.
* **REST API Pipeline with Safeguards**: Ingests live payloads with timeout handling and fallback dataset safety nets.
* **High-Throughput Batched Inference**: Implements vectorized token padding (`padding=True`) and disables gradient calculations (`torch.no_grad()`) for rapid inference pass execution.
* **Interactive Data Visualization**: Maps prediction confidence scores into customized Plotly Express distribution histograms.

---

## Pipeline Architecture
[ HackerNews API ] ──(REST/JSON)──> [ Data Ingestion & Formatting ] ──> [ Batched DistilBERT (GPU) ] ──> [ Plotly Dashboard ]

1. **Ingestion**: Requests top story IDs and fetches title payloads asynchronously with timeout checks.
2. **ETL & Tokenization**: Cleans unstructured text and converts titles into input IDs and attention masks.
3. **Inference**: Computes class probabilities using softmax over output logits in evaluation mode (`eval()`).
4. **Analytics**: Joins classification outputs back into Pandas DataFrames for visualization and statistical inspection.

---

## Getting Started

### Prerequisites

* Python 3.8 or higher
* CUDA-compatible GPU *(Optional, falls back to CPU automatically)*

### Installation
Clone the repository:
   ```bash
   git clone [https://github.com/](https://github.com/)<your-username>/distilbert-live-sentiment-etl.git
   cd distilbert-live-sentiment-etl
