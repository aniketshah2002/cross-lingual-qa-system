# Cross-Lingual Customer Insight Tool 🌍📊

A Semantic Search Engine designed to uncover business insights from multilingual customer feedback.

## 🚀 The Problem
Global companies receive customer feedback in many languages (English, German, French, Spanish). Product Managers often miss critical insights—like bugs or quality issues—because they cannot search through foreign-language reviews effectively. Standard keyword search fails because it doesn't understand context across languages.

## 💡 The Solution
This tool allows users to query a database of international reviews using **Natural Language** in English. 

* **Semantic Search:** Uses `paraphrase-multilingual-MiniLM-L12-v2` to match meaning, not just keywords.
* **Vector Database:** Powered by **FAISS** for high-speed similarity search.
* **Metadata Filtering:** Combines vector search with "Star Rating" filters to isolate specific sentiments (e.g., "Find 1-star reviews about Battery Life").

## 🛠️ Tech Stack
* **Python 3.10+**
* **Sentence-Transformers** (Hugging Face)
* **FAISS** (Facebook AI Similarity Search)
* **Flask** (Web Interface)
* **Pandas & NumPy** (Data Processing)

## 📸 How It Works
1.  **Ingestion:** The system loads and balances real-world Amazon reviews from the `mteb/amazon_reviews_multi` dataset.
2.  **Embedding:** It converts reviews into 384-dimensional vectors.
3.  **Indexing:** Vectors are stored in a FAISS index for sub-millisecond retrieval.
4.  **Search:** The user's query is vectorized and compared against the database to find the nearest semantic neighbors.

## 💻 How to Run Locally

1. **Clone the repo**
   git clone https://github.com/aniketshah2002/cross-lingual-qa-system.git
   cd cross-lingual-insight

2. **Install dependencies**
    pip install -r requirements.txt

3. **Build the Knowledge Base (First time only)**
    python create_embeddings.py  # Downloads data and creates vectors
    python build_index.py        # Creates the FAISS index

4. **Run the Dashboard**
    python app.py
    Open http://127.0.0.1:5000 in your browser.

**📊 Example Scenarios**
**Query:** "The battery life is terrible" + Filter: 1 Star

***Result:*** Retrieves complaints in German ("Akku ist schwach") and French ("Batterie défaillante").

**Query:** "Fast delivery" + Filter: 5 Stars

***Result:*** Retrieves praise in Spanish ("Envío rápido") and English.