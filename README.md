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

