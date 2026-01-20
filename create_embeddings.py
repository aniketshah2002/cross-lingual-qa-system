import numpy as np
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
import os
import json

def create_embeddings(model, documents):
    """
    Generates embeddings for a list of documents.
    """
    print(f"Generating embeddings for {len(documents)} documents...")
    embeddings = model.encode(documents, show_progress_bar=True, batch_size=64)
    print("Embeddings generated successfully.")
    return embeddings

def main():
    print("Starting Step 2: Generating Embeddings (BALANCED VERSION)...")

    # --- 1. Load the Data (Balanced) ---
    target_languages = ['en', 'de', 'fr', 'es']
    documents = [] 
    metadatas = [] 

    print("Loading datasets and balancing classes...")
    try:
        for lang in target_languages:
            print(f"Processing {lang}...")
            # Load the full dataset for this language
            dataset = load_dataset("mteb/amazon_reviews_multi", lang, split="train")
            
            # We want 50 reviews for EACH star rating (0-4 in dataset, which is 1-5 stars)
            # This ensures we have exactly 250 reviews per language
            for star_label in range(5): # 0, 1, 2, 3, 4
                
                # Filter the dataset to find reviews with this specific star rating
                # We use .filter() to find matching rows, then take the first 50
                filtered = dataset.filter(lambda x: x['label'] == star_label).select(range(50))
                
                for item in filtered:
                    review_text = item['text']
                    star_rating = item['label'] + 1 # Convert 0-4 to 1-5
                    
                    documents.append(review_text)
                    metadatas.append({
                        "language": lang,
                        "stars": star_rating
                    })
            print(f"  -> Added 250 balanced reviews for {lang}")
                
        print(f"\nTotal Loaded: {len(documents)} reviews.")

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- 2. Load the Model ---
    model_name = 'paraphrase-multilingual-MiniLM-L12-v2'
    print(f"Loading model: {model_name}...")
    model = SentenceTransformer(model_name)

    # --- 3. Generate Embeddings ---
    embeddings = create_embeddings(model, documents)

    # --- 4. Save Everything ---
    output_dir = "knowledge_base"
    os.makedirs(output_dir, exist_ok=True)

    embeddings_path = os.path.join(output_dir, "embeddings.npy")
    sentences_path = os.path.join(output_dir, "sentences.txt")
    metadata_path = os.path.join(output_dir, "metadata.json")

    print(f"Saving embeddings to {embeddings_path}")
    np.save(embeddings_path, embeddings)

    print(f"Saving sentences to {sentences_path}")
    with open(sentences_path, "w", encoding="utf-8") as f:
        for sentence in documents:
            clean_sentence = sentence.replace("\n", " ")
            f.write(clean_sentence + "\n")

    print(f"Saving metadata to {metadata_path}")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadatas, f)

    print("\nStep 2 complete! Balanced Knowledge Base created.")

if __name__ == "__main__":
    main()