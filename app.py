# Step 4: The Complete Web Application (Backend + Frontend)
# ---------------------------------------------------------
# This script creates a simple web server using Flask to provide a
# user interface for our cross-lingual search engine.
#
# Before running, you need to install Flask and Datasets:
#
# pip install Flask datasets
# ---------------------------------------------------------

from flask import Flask, request, jsonify, render_template_string
from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import numpy as np
import faiss
import os

# --- Global variables to hold our model and data ---
# We load these once when the server starts to avoid reloading on every request.
print("Loading model and knowledge base...")
MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
FAISS_INDEX = faiss.read_index("knowledge_base/faiss.index")

# --- MODIFICATION: Load parallel sentences ---
# Instead of just loading the German sentences from the text file,
# we reload the original dataset to get the parallel English sentences.
print("Loading dataset to get parallel sentences...")
dataset = load_dataset("tatoeba", lang1="de", lang2="en", split="train")
# This number MUST match the number used in step2_create_embeddings.py
# This ensures our indices align perfectly with the FAISS index.
num_documents = 10000 
SENTENCES = [ex['translation']['de'] for ex in dataset.select(range(num_documents))]
ENGLISH_SENTENCES = [ex['translation']['en'] for ex in dataset.select(range(num_documents))]
# --- END MODIFICATION ---

print("Model and knowledge base loaded successfully.")

# Initialize the Flask application
app = Flask(__name__)


# --- Frontend and Backend Routes ---

@app.route('/')
def home():
    """Renders the main HTML page."""
    # We use render_template_string to keep everything in one file.
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Cross-Lingual Question Answering</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Inter', sans-serif; }
        .loader {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 40px;
            height: 40px;
            animation: spin 1s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
    </style>
</head>
<body class="bg-gray-100 text-gray-800">
    <div class="container mx-auto p-4 md:p-8 max-w-3xl">
        <div class="bg-white rounded-lg shadow-lg p-6">
            <h1 class="text-2xl md:text-3xl font-bold text-center mb-2 text-gray-700">Cross-Lingual QA System</h1>
            <p class="text-center text-gray-500 mb-6">Ask a question in English, get answers from German documents.</p>
            
            <div class="relative">
                <input type="text" id="query-input" class="w-full p-3 pr-20 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:outline-none" placeholder="e.g., Who is the chancellor of Germany?">
                <button onclick="handleSearch()" class="absolute right-2 top-1/2 -translate-y-1/2 bg-blue-500 text-white px-4 py-2 rounded-lg hover:bg-blue-600 font-semibold">Search</button>
            </div>
            
            <div id="loading" class="hidden flex justify-center items-center my-8">
                <div class="loader"></div>
            </div>

            <div id="results-container" class="mt-8">
                <!-- Search results will be displayed here -->
            </div>
        </div>
        <footer class="text-center text-sm text-gray-500 mt-6">
            <p>A project built with Flask, FAISS, and Sentence-Transformers.</p>
        </footer>
    </div>

    <script>
        const queryInput = document.getElementById('query-input');
        const resultsContainer = document.getElementById('results-container');
        const loadingIndicator = document.getElementById('loading');

        async function handleSearch() {
            const query = queryInput.value.trim();
            if (!query) {
                resultsContainer.innerHTML = '<p class="text-center text-red-500">Please enter a question.</p>';
                return;
            }

            loadingIndicator.classList.remove('hidden');
            resultsContainer.innerHTML = '';

            try {
                const response = await fetch('/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query })
                });

                if (!response.ok) {
                    throw new Error('Network response was not ok');
                }

                const results = await response.json();
                displayResults(results);

            } catch (error) {
                console.error('Search error:', error);
                resultsContainer.innerHTML = '<p class="text-center text-red-500">An error occurred during the search.</p>';
            } finally {
                loadingIndicator.classList.add('hidden');
            }
        }
        
        function displayResults(results) {
            if (results.length === 0) {
                resultsContainer.innerHTML = '<p class="text-center text-gray-500">No results found.</p>';
                return;
            }

            let html = '<h2 class="text-xl font-semibold mb-4 text-gray-600">Top Results:</h2><div class="space-y-4">';
            results.forEach(result => {
                html += `
                    <div class="bg-gray-50 p-4 rounded-lg border border-gray-200">
                        <p class="text-lg text-gray-800">"${result.sentence}"</p>
                        <!-- MODIFICATION: Display the English translation -->
                        <p class="text-md text-gray-600 mt-2 pl-4 border-l-4 border-gray-300"><em>Translation:</em> ${result.translation}</p>
                        <!-- END MODIFICATION -->
                        <p class="text-right text-sm text-blue-500 font-medium mt-2">Similarity Score: ${result.score}</p>
                    </div>
                `;
            });
            html += '</div>';
            resultsContainer.innerHTML = html;
        }

        queryInput.addEventListener('keyup', (event) => {
            if (event.key === 'Enter') {
                handleSearch();
            }
        });
    </script>
</body>
</html>
    """)

@app.route('/search', methods=['POST'])
def search_endpoint():
    """Receives a query, performs the search, and returns JSON results."""
    data = request.get_json()
    query_text = data.get('query', '')
    top_k = 5

    if not query_text:
        return jsonify([])
        
    query_embedding = MODEL.encode([query_text])
    distances, indices = FAISS_INDEX.search(query_embedding, top_k)
    
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "sentence": SENTENCES[idx],
            # --- MODIFICATION: Add the English translation to the result ---
            "translation": ENGLISH_SENTENCES[idx],
            # --- END MODIFICATION ---
            "score": f"{distances[0][i]:.4f}"
        })
    
    return jsonify(results)


if __name__ == '__main__':
    # Starts the Flask web server.
    # Open your web browser and go to http://127.0.0.1:5000
    print("\n--- Starting Flask Server ---")
    print("Your application will be available at: http://127.0.0.1:5000")
    app.run(debug=True, use_reloader=False)

