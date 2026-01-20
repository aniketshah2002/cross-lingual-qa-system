from flask import Flask, request, jsonify, render_template_string
from sentence_transformers import SentenceTransformer
import faiss
import json
import numpy as np

# --- Global Load ---
print("Loading Knowledge Base...")
MODEL = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
FAISS_INDEX = faiss.read_index("knowledge_base/faiss.index")

# Load Sentences & Metadata
with open("knowledge_base/sentences.txt", "r", encoding="utf-8") as f:
    SENTENCES = [line.strip() for line in f.readlines()]

with open("knowledge_base/metadata.json", "r", encoding="utf-8") as f:
    METADATA = json.load(f)

print("System Ready!")

app = Flask(__name__)

@app.route('/')
def home():
    return render_template_string("""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Global Customer Insights</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <style>
        body { font-family: 'Segoe UI', sans-serif; background-color: #f8fafc; }
    </style>
</head>
<body class="text-slate-800">
    <div class="container mx-auto p-6 max-w-4xl">
        <div class="text-center mb-10">
            <h1 class="text-4xl font-bold text-slate-700 mb-2">Global Customer Insights</h1>
            <p class="text-slate-500">Analyze product feedback with <b>Semantic Search + Filters</b>.</p>
        </div>

        <div class="bg-white p-6 rounded-xl shadow-md mb-8">
            <div class="grid grid-cols-1 md:grid-cols-4 gap-4">
                <div class="col-span-3">
                    <label class="block text-xs font-bold text-slate-500 uppercase mb-1">Search Topic</label>
                    <input type="text" id="query-input" 
                        class="w-full p-3 border border-slate-300 rounded-lg focus:ring-2 focus:ring-indigo-500 focus:outline-none" 
                        placeholder="e.g., Battery life, Delivery speed, Quality">
                </div>
                
                <div class="col-span-1">
                    <label class="block text-xs font-bold text-slate-500 uppercase mb-1">Filter by Stars</label>
                    <select id="star-filter" class="w-full p-3 border border-slate-300 rounded-lg bg-white focus:ring-2 focus:ring-indigo-500 outline-none">
                        <option value="all">All Stars</option>
                        <option value="1">⭐ 1 Star Only (Issues)</option>
                        <option value="2">⭐⭐ 2 Stars</option>
                        <option value="3">⭐⭐⭐ 3 Stars</option>
                        <option value="4">⭐⭐⭐⭐ 4 Stars</option>
                        <option value="5">⭐⭐⭐⭐⭐ 5 Stars (Praise)</option>
                    </select>
                </div>
            </div>
            
            <button onclick="handleSearch()" 
                class="w-full mt-4 bg-indigo-600 text-white px-6 py-3 rounded-lg hover:bg-indigo-700 font-semibold transition">
                Analyze Feedback
            </button>
        </div>

        <div id="loading" class="hidden flex justify-center my-12">
            <div class="animate-spin rounded-full h-10 w-10 border-b-2 border-indigo-600"></div>
        </div>

        <div id="results-container" class="space-y-4"></div>
    </div>

    <script>
        async function handleSearch() {
            const query = document.getElementById('query-input').value.trim();
            const stars = document.getElementById('star-filter').value;
            
            if (!query) return;

            document.getElementById('loading').classList.remove('hidden');
            document.getElementById('results-container').innerHTML = '';

            try {
                const response = await fetch('/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ query: query, stars: stars })
                });
                const results = await response.json();
                renderResults(results);
            } catch (err) {
                console.error(err);
            } finally {
                document.getElementById('loading').classList.add('hidden');
            }
        }

        function renderResults(results) {
    const container = document.getElementById('results-container');
    
    // Explicitly handle empty results
    if (results.length === 0) {
        container.innerHTML = `
            <div class="text-center p-8 bg-slate-50 rounded-lg border border-dashed border-slate-300">
                <p class="text-slate-500 font-medium">No reviews found matching your criteria.</p>
                <p class="text-sm text-slate-400 mt-1">Try changing the star filter or using a different keyword.</p>
            </div>
        `;
        return;
    }

            let html = '<h2 class="text-lg font-semibold text-slate-600 mb-4">Search Results:</h2>';
            results.forEach(res => {
                const stars = '★'.repeat(res.stars) + '☆'.repeat(5 - res.stars);
                const langColors = {'en': 'bg-blue-100 text-blue-800', 'de': 'bg-yellow-100 text-yellow-800', 'fr': 'bg-purple-100 text-purple-800', 'es': 'bg-red-100 text-red-800'};
                const badgeClass = langColors[res.language] || 'bg-gray-100';

                html += `
                    <div class="bg-white p-5 rounded-lg shadow-sm border border-slate-100 hover:shadow-md transition">
                        <div class="flex justify-between items-start mb-2">
                            <div class="flex items-center gap-2">
                                <span class="px-2 py-1 rounded text-xs font-bold uppercase ${badgeClass}">${res.language}</span>
                                <span class="text-yellow-500 text-sm tracking-widest">${stars}</span>
                            </div>
                            <span class="text-xs text-slate-400 bg-slate-50 px-2 py-1 rounded">Match: ${res.score}</span>
                        </div>
                        <p class="text-slate-700 leading-relaxed">"${res.text}"</p>
                    </div>
                `;
            });
            container.innerHTML = html;
        }
    </script>
</body>
</html>
    """)

@app.route('/search', methods=['POST'])
def search_endpoint():
    data = request.get_json()
    query_text = data.get('query', '')
    star_filter = data.get('stars', 'all')
    
    if not query_text: return jsonify([])

    # 1. Encode Query
    query_vector = MODEL.encode([query_text])
    
    # 2. Search (Get more results than we need, so we can filter)
    # We fetch top 50 matches to ensure we have enough left after filtering
    k = 50 
    distances, indices = FAISS_INDEX.search(query_vector, k)
    
    # 3. Filter & Format
    results = []
    for i, idx in enumerate(indices[0]):
        meta = METADATA[idx]
        
        # FILTER LOGIC:
        if star_filter != 'all':
            if str(meta['stars']) != star_filter:
                continue # Skip if stars don't match
        
        results.append({
            "text": SENTENCES[idx],
            "language": meta['language'],
            "stars": meta['stars'],
            "score": f"{distances[0][i]:.2f}"
        })
        
        # Stop once you have 5 good matches
        if len(results) >= 5:
            break
    
    return jsonify(results)

if __name__ == '__main__':
    app.run(debug=True, port=5000)