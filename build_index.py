import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os
import json

def load_knowledge_base(directory="knowledge_base"):
    """Loads embeddings, sentences, and metadata."""
    embeddings_path = os.path.join(directory, "embeddings.npy")
    sentences_path = os.path.join(directory, "sentences.txt")
    metadata_path = os.path.join(directory, "metadata.json")
    
    if not os.path.exists(embeddings_path):
        print("Error: Files not found! Run step2_create_embeddings.py first.")
        return None, None, None

    print("Loading knowledge base...")
    embeddings = np.load(embeddings_path)
    
    with open(sentences_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f.readlines()]
        
    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
        
    print(f"Loaded {len(sentences)} items successfully.")
    return embeddings, sentences, metadata

def create_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    print(f"Creating FAISS index (Dimension: {dimension})...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def search(query_text, model, index, sentences, metadata, top_k=3):
    print(f"\nAnalyzing Query: '{query_text}'")
    
    # 1. Encode the query (English)
    query_embedding = model.encode([query_text])
    
    # 2. Search the vector space
    distances, indices = index.search(query_embedding, top_k)
    
    # 3. Format results
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "text": sentences[idx],
            "metadata": metadata[idx],
            "score": distances[0][i]
        })
    return results

def main():
    print("Starting Step 3: Building the Search Index...")
    
    # --- 1. Load Data ---
    embeddings, sentences, metadata = load_knowledge_base()
    if embeddings is None:
        return

    # --- 2. Build Index ---
    faiss_index = create_faiss_index(embeddings)
    
    # --- 3. Save Index ---
    index_path = "knowledge_base/faiss.index"
    faiss.write_index(faiss_index, index_path)
    print(f"Index saved to {index_path}")

    # --- 4. TEST: The 'Business Insight' Demo ---
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    print(f"\nLoading model '{model_name}' for testing...")
    model = SentenceTransformer(model_name)
    
    # TEST QUERY: A common complaint in English
    test_query = "The battery life is terrible"
    
    results = search(test_query, model, faiss_index, sentences, metadata, top_k=3)
    
    print("\n--- Test Results (Cross-Lingual Insights) ---")
    for res in results:
        lang = res['metadata']['language'].upper()
        stars = res['metadata']['stars']
        print(f"[{lang}] {stars} Stars | Score: {res['score']:.4f}")
        print(f"Review: \"{res['text'][:100]}...\"\n")
        
    print("Success! The system can now find complaints across languages.")

if __name__ == "__main__":
    main()