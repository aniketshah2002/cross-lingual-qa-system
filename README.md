---
title: Cross-Lingual QA System
emoji: 🌍
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# Cross-Lingual Question Answering System

This project is a web-based application that demonstrates a cross-lingual semantic search engine. A user can ask a question in English, and the system will retrieve the most relevant answers from a knowledge base of German documents.

This project was built to showcase the power of modern NLP models in bridging language barriers, a key skill for a Master's program in AI/ML.

## Core Technologies & Concepts

-   **Backend:** Flask
-   **Frontend:** HTML, Tailwind CSS
-   **NLP Model:** `paraphrase-multilingual-MiniLM-L12-v2` from the Sentence-Transformers library.
-   **Vector Search:** Facebook AI Similarity Search (FAISS) for efficient nearest-neighbor search.
-   **Dataset:** A subset of the [Tatoeba](https://huggingface.co/datasets/tatoeba) dataset (German-English).

## How to Set Up and Run the Project Locally

**1. Clone the repository:**
```bash
git clone [https://github.com/YOUR_USERNAME/cross-lingual-qa-system.git](https://github.com/YOUR_USERNAME/cross-lingual-qa-system.git)
cd cross-lingual-qa-system