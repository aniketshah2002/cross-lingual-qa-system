# Step 3: Building the Search Index with FAISS
# --------------------------------------------
# FAISS is a library for efficient similarity search. It's perfect for
# finding the "nearest neighbors" in a large set of vectors.
#
# Before running, you need to install the FAISS library.
# We'll use the CPU version, which is easier to install.
#
# Open your terminal and run:
# pip install faiss-cpu
# --------------------------------------------

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import os

def load_knowledge_base(directory="knowledge_base"):
    """Loads the sentences and embeddings from files."""
    embeddings_path = os.path.join(directory, "embeddings.npy")
    sentences_path = os.path.join(directory, "sentences.txt")
    
    if not os.path.exists(embeddings_path) or not os.path.exists(sentences_path):
        print("Error: Knowledge base files not found!")
        print("Please run step2_create_embeddings.py first.")
        return None, None

    print("Loading knowledge base...")
    embeddings = np.load(embeddings_path)
    with open(sentences_path, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f.readlines()]
    print("Knowledge base loaded successfully.")
    return embeddings, sentences

def create_faiss_index(embeddings):
    """Creates a FAISS index for the given embeddings."""
    # The dimension of our vectors is the number of columns in our embeddings array.
    dimension = embeddings.shape[1]
    
    # We'll use a simple 'IndexFlatL2' index. This index performs an exact,
    # exhaustive search. For larger datasets, more complex indexes can be used
    # for a trade-off between speed and accuracy.
    print(f"Creating a FAISS index with dimension {dimension}...")
    index = faiss.IndexFlatL2(dimension)
    
    # Add our document embeddings to the index.
    index.add(embeddings)
    print(f"Index created. Total vectors in index: {index.ntotal}")
    return index

def search(query_text, model, index, sentences, top_k=3):
    """
    Searches the index for the most relevant sentences to the query.
    
    Args:
        query_text (str): The user's question.
        model: The SentenceTransformer model.
        index: The FAISS index.
        sentences (list): The list of original sentences.
        top_k (int): The number of results to return.
        
    Returns:
        list: A list of tuples (sentence, score).
    """
    print(f"\nSearching for top {top_k} results for query: '{query_text}'")
    
    # 1. Convert the query text to an embedding.
    query_embedding = model.encode([query_text])
    
    # 2. Search the FAISS index.
    # The search function returns two arrays:
    # D: Distances (scores) of the nearest neighbors.
    # I: Indices of the nearest neighbors.
    distances, indices = index.search(query_embedding, top_k)
    
    # 3. Format and return the results.
    results = []
    for i, idx in enumerate(indices[0]):
        results.append((sentences[idx], distances[0][i]))
        
    return results

def main():
    """
    Main function to build the FAISS index and run a test search.
    """
    print("Starting Step 3: Building the FAISS Search Index...")
    
    # --- 1. Load the Knowledge Base ---
    embeddings, sentences = load_knowledge_base()
    if embeddings is None:
        return

    # --- 2. Build and Save the FAISS Index ---
    faiss_index = create_faiss_index(embeddings)
    
    output_dir = "knowledge_base"
    index_path = os.path.join(output_dir, "faiss.index")
    print(f"Saving FAISS index to {index_path}...")
    faiss.write_index(faiss_index, index_path)
    print("Index saved successfully.")

    # --- 3. Run a Test Search ---
    # Load the multilingual model we used for embedding.
    # FIX: Corrected the model name from '_' to '-'.
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    print(f"\nLoading model '{model_name}' for the test search...")
    model = SentenceTransformer(model_name)
    
    # This is where the magic happens! We ask a question in ENGLISH.
    # The system will find the best answers in the GERMAN text.
    test_query = "Where is the nearest train station?"
    
    search_results = search(test_query, model, faiss_index, sentences, top_k=3)
    
    print("\n--- Test Search Results ---")
    for result_sentence, score in search_results:
        print(f"  Score: {score:.4f}")
        print(f"  German Sentence: {result_sentence}\n")
        
    print("Step 3 complete! We have a searchable index and have tested it.")


if __name__ == "__main__":
    main()
