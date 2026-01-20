<<<<<<< HEAD
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
   git clone [your-repo-link]
   cd cross-lingual-insight

2. **Install dependencies**
    pip install -r requirements.txt

3. **Build the Knowledge Base (First Time Only)**
    python create_embeddings.py   #Downloads data and creates vectors
    python build_index.py         # Creates the FAISS index

4 **Run the Dashboard**
    python app.py

**Open** http://127.0.0.1:5000 in your browser.

**📊 Example Scenarios**
**Query:** "The battery life is terrible" + Filter: 1 Star

**Result:** Retrieves complaints in German ("Akku ist schwach") and French ("Batterie défaillante").

**Query:** "Fast delivery" + Filter: 5 Stars

**Result:** Retrieves praise in Spanish ("Envío rápido") and English.
=======
Cross-Lingual Question Answering System
This project is a web-based application that demonstrates a cross-lingual semantic search engine. A user can ask a question in English, and the system will retrieve the most relevant answers from a knowledge base of German documents.

This project was built to showcase the power of modern NLP models in bridging language barriers, a key skill for a Master's program in AI/ML.

Live Demo GIF:
(Recommendation: Record a short GIF of you using the web app and upload it to your repository. Then, you can embed it here like this: ![Demo GIF](demo.gif))

Core Technologies & Concepts
Backend: Flask

Frontend: HTML, Tailwind CSS

NLP Model: paraphrase-multilingual-MiniLM-L12-v2 from the Sentence-Transformers library.

Vector Search: Facebook AI Similarity Search (FAISS) for efficient nearest-neighbor search.

Dataset: A subset of the Tatoeba dataset (German-English).

Key Concepts: Semantic Search, Sentence Embeddings, Cross-Lingual Information Retrieval, Vector Databases.

How It Works
The system leverages a multilingual sentence embedding model to map text from different languages into a shared vector space.

Indexing Pipeline:

A corpus of 10,000 German sentences is loaded from the Tatoeba dataset.

The SentenceTransformer model encodes each German sentence into a 384-dimensional vector (embedding).

These embeddings are stored in a FAISS index, creating a highly efficient and searchable knowledge base.

Search Pipeline:

A user submits a query in English via the Flask web interface.

The same model encodes the English query into a vector.

FAISS performs a similarity search to find the vectors in the index that are closest to the query vector.

The German sentences corresponding to the top results are returned to the user, along with their original English translations from the parallel corpus.

How to Set Up and Run the Project Locally
Prerequisites:

Python 3.8+

pip

1. Clone the repository:

git clone https://github.com/YOUR_USERNAME/cross-lingual-qa-system.git
cd cross-lingual-qa-system

2. Create a virtual environment and install dependencies:

# Create and activate the virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows, use: .venv\Scripts\activate

# Install the required libraries
pip install -r requirements.txt

(You will need to create a requirements.txt file. See instructions below.)

3. Run the Pre-processing Scripts:
The application requires a pre-built knowledge base. Run the following scripts in order:

# This will download the dataset
python step1_load_data.py

# This will generate the embeddings (this may take a few minutes)
python step2_create_embeddings.py

# This will build the FAISS index
python step3_build_index.py

4. Run the Flask Application:

python app.py

The application will be available at http://127.0.0.1:5000.

Creating the requirements.txt file
In your local project directory, run the following command to generate the requirements.txt file. This is a standard practice that lists all the project's dependencies.

pip freeze > requirements.txt

After running this, make sure to add and commit the new requirements.txt file to your repository:

git add requirements.txt
git commit -m "Add requirements.txt for dependencies"
git push

>>>>>>> 12a97993d1425d5103c71a93c9ac1423827b0633
